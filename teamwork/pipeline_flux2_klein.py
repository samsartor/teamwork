
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Literal, Callable
from diffusers.pipelines.flux2.pipeline_flux2_klein import (
    Flux2KleinPipeline,
    retrieve_timesteps,
    compute_empirical_mu,
)
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Transformer2DModel,
    Flux2SingleTransformerBlock,
    Flux2Modulation,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from einops import rearrange
import numpy as np

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig
from .adapter import adapt, save_adapters, TEAMWORK_PROFILES, Adapt, adapter_modules, shallowcopy_into, AdapterMixin
from .batch import BatchBuilder, OutputImageType
from .attn import Flux2TeamworkJointAttention, teamwork_joint_attention


class Flux2TeamworkPipeline(TeamworkPipeline, Flux2KleinPipeline):
    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt_embeds: None | Tensor = None
    empty_text_ids: None | Tensor = None
    teamwork_joint_attn = False

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: Flux2KleinPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
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
        assert isinstance(pipeline.transformer, Flux2Transformer2DModel)
        if isinstance(
            pipeline.transformer.single_transformer_blocks[0],
            TeamworkFlux2SingleTransformerBlock,
        ):
            pipeline.teamwork_joint_attn = True
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
        x = x.to(self.vae.device, self.vae.dtype)
        x = self.vae.encode(x).latent_dist.sample()  # type: ignore
        x = (x - (self.vae.config.get("shift_factor", None) or 0)) * (
            self.vae.config.get("scaling_factor", None) or 1
        )
        return x

    def vae_decode(self, x: Tensor) -> Tensor:
        x = x / (self.vae.config.get("scaling_factor", None) or 1) + (
            self.vae.config.get("shift_factor", None) or 0
        )
        x = x.to(self.vae.device, self.vae.dtype)
        x = self.vae.decode(x, return_dict=False)[0]  # type: ignore
        return x

    @torch.no_grad
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
        # Flux2 Klein has a single text encoder (Qwen3) and no pooled embeddings.
        (prompt_embeds, text_ids) = self.encode_prompt(
            prompt,
            num_images_per_prompt=1,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.empty_text_ids = text_ids
        self.text_encoder = None
        self.tokenizer = None

    def _latent_image_ids(self, hh: int, ww: int, device: torch.device) -> Tensor:
        """4-axis (T, H, W, L) position ids for a packed latent grid of size (hh, ww).

        Mirrors Flux2KleinPipeline._prepare_latent_ids for a single image: T and L
        stay 0, H/W index the patch grid. Built in float32 so large grids stay exact.
        """
        ids = torch.zeros(hh * ww, 4, device=device, dtype=torch.float32)
        ids[:, 1] = torch.arange(hh, device=device).repeat_interleave(ww)
        ids[:, 2] = torch.arange(ww, device=device).repeat(hh)
        return ids

    def _joint_image_ids(self, image_ids: Tensor, num_teammates: int) -> Tensor:
        """Per-teammate image ids for the (L_text + T*L_img) joint-attention layout.

        Each teammate's image gets its own RoPE T-coord from the config's
        `teammate_image_ids` so that, with zero LoRA, the layout matches the base
        klein editing model (generated images at T=0, references at T=10, 20, ...).
        Naive offsets (0, 1, 2, ...) put a reference adjacent to the generation and
        yield garbage at init -- see scripts/test_edit_parity.py.
        """
        offsets = self.teamwork_config.teammate_image_ids
        assert offsets is not None  # resolved in TeamworkConfig.__post_init__
        parts = []
        for t in range(num_teammates):
            ids = image_ids.clone()
            ids[:, 0] += offsets[t]
            parts.append(ids)
        return torch.cat(parts, 0)

    def train_loss(
        self,
        batch: BatchBuilder,
        noise: torch.Tensor,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        assert isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
        assert self.scheduler.timesteps is not None
        assert self.scheduler.sigmas is not None

        with torch.no_grad():
            if self.timestep_weight == "unit" or self.timestep_weight == "sigma_sqrt":
                timestep_u = torch.rand([batch.batch_size])
            elif self.timestep_weight == "logit_normal":
                timestep_u = torch.nn.functional.sigmoid(torch.randn([batch.batch_size]))
            else:
                raise ValueError(f"Unknown timestep weight {self.timestep_weight}")

            sel = batch.selection()
            timestep_u = timestep_u[sel.batch_indices.cpu()]
            timestep_i = (timestep_u * len(self.scheduler.timesteps)).long()
            timesteps = (
                self.scheduler.timesteps[timestep_i]
                .to(device=self.device, dtype=self.dtype)
            )

            # Flux2 Klein is guidance-distilled (guidance_embeds=False); no guidance input.
            guidance = None

            # Figure out how much noise to add to the latents
            latents = batch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            )
            sigmas = self.scheduler.sigmas[timestep_i].reshape(batch.count, 1, 1, 1).to(
                dtype=self.dtype, device=self.device
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            if self.timestep_weight == "sigma_sqrt":
                weighting = (sigmas**-2.0).float()
            elif (
                self.timestep_weight == "logit_normal" or self.timestep_weight == "unit"
            ):
                weighting = 1.0
            else:
                raise ValueError(f"Unknown timestep weight {self.timestep_weight}")

            noisy_latents[sel.input_subindices] = latents[sel.input_subindices]

            b, _, lh, lw = latents.shape
            image_ids = self._latent_image_ids(lh // 2, lw // 2, latents.device)
            if self.teamwork_joint_attn:
                image_ids = self._joint_image_ids(image_ids, sel.num_teammates)

            # Get prompt embeds if needed
            if self.text_encoder is None:
                assert self.empty_prompt_embeds is not None
                assert self.empty_text_ids is not None
                prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
                text_ids = self.empty_text_ids
            else:
                (prompt_embeds, text_ids) = self.encode_prompt(
                    prompt,
                    num_images_per_prompt=latents.shape[0] if isinstance(prompt, str) else 1,
                )
            prompt_embeds = prompt_embeds.to(latents.dtype)
            text_ids = text_ids.to(latents.dtype)

        # Concatenate extra channels if present
        model_latents = noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([noisy_latents, extra], dim=1)

        # Update selection (and the text length the single blocks split on)
        for adapter in adapter_modules(self.unwrapped_transformer).values():
            adapter.selection = sel
            if isinstance(adapter, TeamworkFlux2SingleTransformerBlock):
                adapter.text_seq_len = prompt_embeds.shape[1]

        # Account for extra channels in the rearrange
        total_channels = model_latents.shape[1]
        model_input = rearrange(
            model_latents,
            "b c (h ph) (w pw) -> b (h w) (c ph pw)",
            b=b,
            c=total_channels,
            w=lw // 2,
            h=lh // 2,
            ph=2,
            pw=2,
        )
        model_pred = self.transformer(
            hidden_states=model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        ).sample
        model_pred = rearrange(
            model_pred,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            b=b,
            c=self.in_channels,
            w=lw // 2,
            h=lh // 2,
            ph=2,
            pw=2,
        )

        return LossOutput(
            selection=sel,
            latents=latents,
            prediction=model_pred,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * weighting,
            timestep_idx=timestep_i,
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
        batch: BatchBuilder | None = None,
        callback_on_step_end: Callable[[int, int, dict], None] | None = None,
    ) -> dict[str, Any]:
        device = self._execution_device
        if batch is None:
            batch = BatchBuilder.from_inputs(
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
        sel = batch.selection()
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

        b, _, lh, lw = latents.shape
        image_ids = self._latent_image_ids(lh // 2, lw // 2, latents.device)
        if self.teamwork_joint_attn:
            image_ids = self._joint_image_ids(image_ids, sel.num_teammates)

        # Get prompt embeds if needed
        if self.empty_prompt_embeds is not None:
            assert self.empty_text_ids is not None
            prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
            text_ids = self.empty_text_ids
        else:
            (prompt_embeds, text_ids) = self.encode_prompt(
                prompt,
                num_images_per_prompt=latents.shape[0],
            )
        prompt_embeds = prompt_embeds.to(latents.dtype)
        text_ids = text_ids.to(latents.dtype)

        # Set timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Update selection (and the text length the single blocks split on)
        for adapter in adapter_modules(self.transformer).values():
            adapter.selection = sel
            if isinstance(adapter, TeamworkFlux2SingleTransformerBlock):
                adapter.text_seq_len = prompt_embeds.shape[1]

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(model_latents.shape[0])

                # Flux2 Klein is guidance-distilled (guidance_embeds=False).
                guidance = None

                # Concatenate extra channels if present
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                # Account for extra channels in the rearrange
                total_channels = model_latents.shape[1]

                # Get model pred
                model_input = rearrange(
                    model_latents,
                    "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                    b=b,
                    c=total_channels,
                    w=lw // 2,
                    h=lh // 2,
                    ph=2,
                    pw=2,
                )
                noise_pred = self.transformer(
                    hidden_states=model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                ).sample
                noise_pred = rearrange(
                    noise_pred,
                    "b (h w) (c ph pw) -> b c (h ph) (w pw)",
                    b=b,
                    c=self.in_channels,
                    w=lw // 2,
                    h=lh // 2,
                    ph=2,
                    pw=2,
                )

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


# LoRA-only profile. Flux2 differs structurally from Flux1: the adaLN modulation
# linears are hoisted to the model level (one per stream, shared across blocks)
# instead of living in per-block norm1.linear, the single blocks fuse QKV+MLP into
# one projection (to_qkv_mlp_proj / to_out), and the feedforward uses linear_in/out.
# As in the Flux1 profile, we adapt the image-stream pathway and the shared single
# stream, leaving the text-stream (add_*_proj, ff_context, *_modulation_txt) at base.
TEAMWORK_PROFILES["FLUX2"] = [
    "double_stream_modulation_img.linear",
    "single_stream_modulation.linear",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.ff.linear_in",
    "transformer_blocks.*.ff.linear_out",
    "single_transformer_blocks.*.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.*.attn.to_out",
]

class TeamworkFlux2SingleTransformerBlock(Flux2SingleTransformerBlock, AdapterMixin):
    """Teamwork-enabled Flux2 single (parallel) block.

    Flux2's single block fuses QKV+MLP-in into one projection (`attn.to_qkv_mlp_proj`)
    and attn-out+MLP-out into another (`attn.to_out`), and the transformer hands it
    the full `[text, image]` sequence (text first) with `encoder_hidden_states=None`.
    To do cross-teammate attention we run the fused input projection per component,
    split out the QKV portion, route it through the masked joint attention (text
    pulled to per-batch, image scattered per teammate), run the SwiGLU MLP per
    component, then recombine and run the fused output projection. The image-stream
    modulation/gate/residual mirror the base block. `text_seq_len` is threaded in by
    the pipeline since the concatenated stream doesn't carry the split point.
    """

    def __init__(self, base: Flux2SingleTransformerBlock, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.adapter = nn.Parameter(torch.tensor(0.0))
        self.text_seq_len: int | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        temb_mod: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        split_hidden_states: bool = False,
        text_seq_len: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sel = self.selection
        assert sel is not None
        attn = self.attn

        # In the single stack the transformer concatenates text+image and passes
        # encoder_hidden_states=None; recover the split point either way.
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        if text_seq_len is None:
            text_seq_len = self.text_seq_len
        assert text_seq_len is not None, "text_seq_len must be threaded in by the pipeline"

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

        residual = hidden_states
        norm_hidden_states = self.norm(hidden_states)
        norm_hidden_states = (1 + mod_scale) * norm_hidden_states + mod_shift

        # Fused QKV + MLP-in projection (per-component LoRA), then split.
        proj = attn.to_qkv_mlp_proj(norm_hidden_states)
        qkv, mlp_hidden_states = torch.split(
            proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        # Split text/image along the sequence; text shares norm_q/norm_k with image.
        txt_q, img_q = query[:, :text_seq_len], query[:, text_seq_len:]
        txt_k, img_k = key[:, :text_seq_len], key[:, text_seq_len:]
        txt_v, img_v = value[:, :text_seq_len], value[:, text_seq_len:]

        img_out, txt_out = teamwork_joint_attention(
            img_q, img_k, img_v,
            txt_q, txt_k, txt_v,
            sel, image_rotary_emb,
        )
        attn_output = torch.cat([txt_out, img_out], dim=1).flatten(-2, -1).to(query.dtype)

        # SwiGLU MLP per component, then fused output projection.
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        attn_output = attn.to_out(torch.cat([attn_output, mlp_hidden_states], dim=-1))

        hidden_states = residual + mod_gate * attn_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        if split_hidden_states:
            return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
        return hidden_states


TEAMWORK_PROFILES["FLUX2_PLUSATTN"] = [
    *TEAMWORK_PROFILES["FLUX2"],
    ("transformer_blocks.*.attn", Flux2TeamworkJointAttention),
    ("single_transformer_blocks.*", TeamworkFlux2SingleTransformerBlock),
]
