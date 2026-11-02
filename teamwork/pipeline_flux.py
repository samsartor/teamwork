import torch
from torch import Tensor, nn
from typing import Any, Literal, Callable
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel, FluxSingleTransformerBlock
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig, Layout
from .adapter import adapt, TEAMWORK_PROFILES, Adapt, set_selection, shallowcopy_into
from .checkpoint import save_adapters
from .data import TeamBuilder, OutputImageType
from .batch_linear import BatchSelection
from .sampling import sample_timestep_u, flow_match_noising
from . import family_flux


class BatchFluxPipeline(TeamworkPipeline, FluxPipeline):
    """Batch-axis teamwork for FLUX.1 and FLUX.1 Kontext (built from either base pipeline)."""

    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt_embeds: None | Tensor = None
    empty_pooled_prompt_embeds: None | Tensor = None
    empty_text_ids: None | Tensor = None

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: FluxPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
        cls.check_layout(teamwork_config)
        pipeline = cls(**base_pipeline.components)  # type: ignore
        pipeline.teamwork_config = teamwork_config
        pipeline.transformer = adapt(
            pipeline.transformer,
            cfg=teamwork_config,
            device=pipeline.device,
            dtype=torch.bfloat16,
            requires_grad=training,
            override_profile=override_profile,
            state=state,
            infer_layers_from_state=False,
        )
        assert isinstance(pipeline.transformer, FluxTransformer2DModel)
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    @property
    def unwrapped_transformer(self):
        model = self.transformer
        while hasattr(model, 'module'):
            model = getattr(model, 'module')
        return model

    @property
    def in_channels(self) -> int:
        return self.unwrapped_transformer.config["in_channels"] // 4

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.unwrapped_transformer, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

    def vae_encode(self, x: Tensor) -> Tensor:
        return family_flux.vae_encode(self.vae, x)

    def vae_decode(self, x: Tensor) -> Tensor:
        return family_flux.vae_decode(self.vae, x)

    def make_selection(self, batch: TeamBuilder) -> BatchSelection:
        return BatchSelection.from_components(batch.components_index())

    def image_ids(self, lh: int, lw: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """RoPE ids of the image tokens in each row: one image per row."""
        return family_flux.latent_image_ids(lh, lw, device, dtype)

    @torch.no_grad
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder_2 = self.text_encoder_2.to(device)
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt,
            prompt,
            num_images_per_prompt=1,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.empty_pooled_prompt_embeds = pooled_prompt_embeds
        self.empty_text_ids = text_ids
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_2 = None
        self.tokenizer_2 = None

    def train_loss(
        self,
        batch: TeamBuilder,
        noise: torch.Tensor,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        # A fresh scheduler, so training timesteps don't depend on the inference schedule
        # that __call__ last set on self.scheduler.
        train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        assert isinstance(train_scheduler, FlowMatchEulerDiscreteScheduler)

        with torch.no_grad():
            timestep_u = sample_timestep_u(self.timestep_weight, batch.batch_size)
            sel = self.make_selection(batch)
            latents = batch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            )
            noising = flow_match_noising(
                train_scheduler, timestep_u, sel, latents, noise, self.timestep_weight,
                device=self.device, dtype=self.dtype,
            )
            timesteps = noising.timesteps

            # Guidance is only supported on some Flux models
            guidance = torch.full_like(timesteps, 3.5) if family_flux.has_guidance(self.unwrapped_transformer) else None

            _, _, lh, lw = latents.shape
            image_ids = self.image_ids(lh, lw, latents.device, latents.dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = family_flux.encode_prompt(
                self, prompt, latents.shape[0], latents.dtype,
            )

        # Concatenate extra channels if present
        model_latents = noising.noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([model_latents, extra], dim=1)

        set_selection(self.unwrapped_transformer, sel)

        model_pred = self.transformer(
            hidden_states=family_flux.patchify(model_latents),
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        ).sample
        model_pred = family_flux.unpatchify(model_pred, self.in_channels, lh, lw)

        return LossOutput(
            selection=sel,
            latents=latents,
            prediction=model_pred,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * noising.weighting,
            timestep_idx=noising.timestep_idx,
            type='signal',
        )

    @torch.no_grad
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str = "",
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: TeamBuilder | None = None,
        callback_on_step_end: Callable[..., dict | None] | None = None,
    ) -> dict[str, Any]:
        """
        `callback_on_step_end(pipe, i, t, {'latents': (count, c, lh, lw), 'selection': Components})`
        may return a dict with replacement `latents`.
        """
        device = self._execution_device
        if batch is None:
            batch = TeamBuilder.from_inputs(
                 self.teamwork_config.teammates,
                 device=device,
                 dtype=self.dtype,
                 images=images,
                 request=request,
                 width=width,
                 height=height,
                 attn_allow=self.teamwork_config.attn_allow,
            )

        # Initialize latents
        sel = self.make_selection(batch)
        clean_latents = batch.packed_encoded_images(
            self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
        )
        if noise is None:
            latents = torch.randn(
                clean_latents.shape,
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            latents = noise.to(device, self.dtype)

        _, _, lh, lw = latents.shape
        image_ids = self.image_ids(lh, lw, latents.device, latents.dtype)
        prompt_embeds, pooled_prompt_embeds, text_ids = family_flux.encode_prompt(
            self, prompt, latents.shape[0], latents.dtype,
        )

        timesteps, num_inference_steps = family_flux.inference_timesteps(
            self.scheduler, num_inference_steps, (lh // 2) * (lw // 2), device,
        )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        set_selection(self.transformer, sel)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(model_latents.shape[0])

                # Guidance is only supported on some Flux models
                guidance = (
                    torch.full_like(timestep, guidance_scale)
                    if family_flux.has_guidance(self.unwrapped_transformer)
                    else None
                )

                # Concatenate extra channels if present
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                noise_pred = self.transformer(
                    hidden_states=family_flux.patchify(model_latents),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                ).sample
                noise_pred = family_flux.unpatchify(noise_pred, self.in_channels, lh, lw)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_kwargs = { 'latents': latents, 'selection': sel }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["FLUX"] = [
    "transformer_blocks.*.norm1.linear",
    "single_transformer_blocks.*.norm.linear",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "single_transformer_blocks.*.attn.to_q",
    "single_transformer_blocks.*.attn.to_k",
    "single_transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "single_transformer_blocks.*.proj_mlp",
    "single_transformer_blocks.*.proj_out",
]


# ======================================================================================
# LEGACY JOINT ATTENTION (`FLUX_PLUSATTN`). Delete in phase 6, together with attn.py.
#
# Batch-axis everywhere except attention: every component keeps its own text copy, and
# the joint processors in attn.py scatter the components of each batch element into one
# dense sequence for attention. Replaced by the sequence-axis implementation.
# ======================================================================================

from .attn import FluxTeamworkJointAttention, LegacyJointSelection  # noqa: E402


class LegacyJointFluxPipeline(BatchFluxPipeline):
    layout: Layout = "sequence"

    def make_selection(self, batch: TeamBuilder) -> LegacyJointSelection:
        return LegacyJointSelection.from_builder(batch)

    def image_ids(self, lh: int, lw: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        # Per-teammate (not per-component) image ids, so the rotary embedding is
        # batch-invariant and matches the (L_text + T*L_img) layout of the joint processors.
        offsets = self.teamwork_config.teammate_image_ids
        assert offsets is not None  # resolved in TeamworkConfig.__post_init__
        return family_flux.offset_image_ids(super().image_ids(lh, lw, device, dtype), offsets)


class TeamworkFluxSingleTransformerBlock(FluxSingleTransformerBlock):
    def __init__(self, base: FluxSingleTransformerBlock, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.adapter = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        joint_attention_kwargs = joint_attention_kwargs or {}
        # TeamworkFluxJointAttnProcessor is refactored to cat img and txt branches internally
        attn_output, encoder_attn_output = self.attn(
            hidden_states=norm_hidden_states[:, text_seq_len:],
            encoder_hidden_states=norm_hidden_states[:, :text_seq_len],
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )
        attn_output = torch.cat([encoder_attn_output, attn_output], dim=1)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return encoder_hidden_states, hidden_states


TEAMWORK_PROFILES["FLUX_PLUSATTN"] = [
    *TEAMWORK_PROFILES["FLUX"],
    ("transformer_blocks.*.attn", FluxTeamworkJointAttention),
    ("single_transformer_blocks.*", TeamworkFluxSingleTransformerBlock),
    ("single_transformer_blocks.*.attn", FluxTeamworkJointAttention),
]
