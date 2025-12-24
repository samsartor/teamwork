import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Literal
from diffusers.pipelines.flux.pipeline_flux import (
    FluxPipeline,
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from einops import rearrange
import numpy as np

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig
from .adapter import adapt, save_adapters, TEAMWORK_PROFILES, Adapt, adapter_modules
from .batch import BatchBuilder, OutputImageType
from .attn import TeamworkJointAttention


class FluxTeamworkPipeline(TeamworkPipeline, FluxPipeline):
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
        )
        if training:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    @property
    def in_channels(self) -> int:
        return self.transformer.config["in_channels"] // 4

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.transformer, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

    def vae_encode(self, x: Tensor) -> Tensor:
        x = x.to(self.vae.device, self.vae.dtype)
        x = self.vae.encode(x).latent_dist.sample()  # type: ignore
        x = (x - (self.vae.config.get("shift_factor", None) or 0)) * self.vae.config[
            "scaling_factor"
        ]
        return x

    def vae_decode(self, x: Tensor) -> Tensor:
        x = x / self.vae.config["scaling_factor"] + (
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
        batch: BatchBuilder,
        noise: torch.Tensor,
        model: Any | None = None,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        assert isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)
        assert self.scheduler.timesteps is not None
        assert self.scheduler.sigmas is not None

        with torch.no_grad():
            if self.timestep_weight == "unit" or self.timestep_weight == "sigma_sqrt":
                timestep_u = torch.rand([])
            elif self.timestep_weight == "logit_normal":
                timestep_u = torch.nn.functional.sigmoid(torch.randn([]))
            else:
                raise ValueError(f"Unknown timestep weight {self.timestep_weight}")

            timestep_i = (timestep_u * len(self.scheduler.timesteps)).long()
            timesteps = (
                self.scheduler.timesteps[timestep_i]
                .to(device=self.device, dtype=self.dtype)
                .repeat(batch.count)
            )

            # Guidance is only supported on some Flux models
            if isinstance(
                self.transformer.time_text_embed,
                CombinedTimestepGuidanceTextProjEmbeddings,
            ):
                guidance = torch.full_like(timesteps, 3.5)
            else:
                guidance = None

            # Figure out how much noise to add to the latents
            sel = batch.selection()
            latents = batch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            )
            sigmas = self.scheduler.sigmas[timestep_i].to(
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
            image_ids = self._prepare_latent_image_ids(
                b,  # this is ignored
                lh // 2,
                lw // 2,
                latents.device,
                latents.dtype,
            )

            # Get prompt embeds if needed
            if self.text_encoder is None:
                assert self.empty_prompt_embeds is not None
                assert self.empty_pooled_prompt_embeds is not None
                assert self.empty_text_ids is not None
                prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
                pooled_prompt_embeds = self.empty_pooled_prompt_embeds.repeat(
                    latents.shape[0], 1
                )
                text_ids = self.empty_text_ids
            else:
                (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
                    prompt,
                    prompt,
                    num_images_per_prompt=latents.shape[0] if isinstance(prompt, str) else 1,
                )
            prompt_embeds = prompt_embeds.to(latents.dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(latents.dtype)
            text_ids = text_ids.to(latents.dtype)

        # Concatenate extra channels if present
        model_latents = noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([noisy_latents, extra], dim=1)

        # Update selection
        for adapter in adapter_modules(self.transformer).values():
            adapter.selection = sel

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
        if model is None:
            model = self.transformer
        model_pred = model(
            hidden_states=model_input,
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
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
        latents_pred = model_pred * (-sigmas) + noisy_latents

        return LossOutput(
            prediction=latents_pred[sel.output_subindices],
            target=latents[sel.output_subindices],
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor)[sel.output_subindices].unsqueeze(1) * weighting,
            type='signal',
        )

    @torch.no_grad
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 3.5,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: BatchBuilder | None = None,
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
        image_ids = self._prepare_latent_image_ids(
            b,  # this is ignored
            lh // 2,
            lw // 2,
            latents.device,
            latents.dtype,
        )

        # Get prompt embeds if needed
        if (
            self.empty_prompt_embeds is not None
            and self.empty_pooled_prompt_embeds is not None
        ):
            assert self.empty_text_ids is not None
            prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
            pooled_prompt_embeds = self.empty_pooled_prompt_embeds.repeat(
                latents.shape[0], 1
            )
            text_ids = self.empty_text_ids
        else:
            (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
                prompt,
                prompt,
                num_images_per_prompt=latents.shape[0],
            )
        prompt_embeds = prompt_embeds.to(latents.dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(latents.dtype)
        text_ids = text_ids.to(latents.dtype)

        # Set timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Update selection
        for adapter in adapter_modules(self.transformer).values():
            adapter.selection = sel

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                model_latents = latents
                model_latents[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(model_latents.shape[0])

                # Guidance is only supported on some Flux models
                if isinstance(
                    self.transformer.time_text_embed,
                    CombinedTimestepGuidanceTextProjEmbeddings,
                ):
                    guidance = torch.full_like(timestep, guidance_scale)
                else:
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
                    pooled_projections=pooled_prompt_embeds,
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

TEAMWORK_PROFILES["FLUX_PLUSATTN"] = [
    *TEAMWORK_PROFILES["FLUX"],
    ("transformer_blocks.*.attn", TeamworkJointAttention),
]
