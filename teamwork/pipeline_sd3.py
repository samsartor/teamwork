import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Literal
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig
from .adapter import adapt, TEAMWORK_PROFILES, Adapt, set_selection
from .checkpoint import save_adapters
from .data import TeamBuilder, OutputImageType
from .batch_linear import BatchSelection
from .sampling import sample_timestep_u, flow_match_noising
from . import family_sd3


class BatchSD3Pipeline(TeamworkPipeline, StableDiffusion3Pipeline):
    teamwork_config: TeamworkConfig
    timestep_weight: str = "sigma_sqrt"
    empty_prompt_embeds: None | Tensor = None
    empty_pooled_prompt_embeds: None | Tensor = None

    @classmethod
    @torch.no_grad()
    def from_base_pipeline(
        cls,
        base_pipeline: StableDiffusion3Pipeline,
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
        )
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    @property
    def in_channels(self) -> int:
        return self.transformer.config["in_channels"]

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.transformer, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

    def vae_encode(self, x: Tensor) -> Tensor:
        return family_sd3.vae_encode(self.vae, x)

    def vae_decode(self, x: Tensor) -> Tensor:
        return family_sd3.vae_decode(self.vae, x)

    @torch.no_grad
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder_2 = self.text_encoder_2.to(device)
            self.text_encoder_3 = self.text_encoder_3.to(device)
        (prompt_embeds, _, pooled_prompt_embeds, _) = self.encode_prompt(
            prompt,
            prompt,
            prompt,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.empty_pooled_prompt_embeds = pooled_prompt_embeds
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.text_encoder_3 = None
        self.tokenizer_3 = None

    def train_loss(
        self,
        batch: TeamBuilder,
        noise: torch.Tensor,
        model: Any | None = None,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        # A fresh scheduler, so training timesteps don't depend on the inference schedule
        # that __call__ last set on self.scheduler.
        train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        assert isinstance(train_scheduler, FlowMatchEulerDiscreteScheduler)

        with torch.no_grad():
            timestep_u = sample_timestep_u(self.timestep_weight, batch.batch_size)
            sel = BatchSelection.from_components(batch.components_index())
            latents = batch.packed_encoded_images(
                self.vae_encode,
                self.transformer.config["in_channels"],
                1 / self.vae_scale_factor,
            )
            noising = flow_match_noising(
                train_scheduler, timestep_u, sel, latents, noise, self.timestep_weight,
                device=self.device, dtype=self.dtype,
            )
            noisy_latents = noising.noisy_latents
            prompt_embeds, pooled_prompt_embeds = family_sd3.encode_prompt(
                self, prompt, latents.shape[0], latents.dtype,
            )

        # Concatenate extra channels if present
        model_input = noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_input = torch.cat([noisy_latents, extra], dim=1)

        set_selection(self.transformer, sel)

        if model is None:
            model = self.transformer
        model_pred = model(
            hidden_states=model_input,
            timestep=noising.timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        ).sample
        latents_pred = model_pred * (-noising.sigmas) + noisy_latents

        return LossOutput(
            selection=sel,
            latents=latents,
            prediction=latents_pred,
            target=latents,
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
        negative_prompt="",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: TeamBuilder | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
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
            )

        # Initialize latents
        sel = BatchSelection.from_components(batch.components_index())
        clean_latents = batch.packed_encoded_images(
            self.vae_encode,
            self.transformer.config["in_channels"],
            1 / self.vae_scale_factor,
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

        # Get prompt embeds if needed
        neg_embeds = None
        pooled_neg_embeds = None
        if self.empty_prompt_embeds is None and guidance_scale > 1.0:
            (prompt_embeds, neg_embeds, pooled_prompt_embeds, pooled_neg_embeds) = (
                self.encode_prompt(
                    prompt,
                    prompt,
                    prompt,
                    negative_prompt=negative_prompt,
                    num_images_per_prompt=latents.shape[0],
                    do_classifier_free_guidance=True,
                )
            )
            prompt_embeds = prompt_embeds.to(latents.dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(latents.dtype)
        else:
            prompt_embeds, pooled_prompt_embeds = family_sd3.encode_prompt(
                self, prompt, latents.shape[0], latents.dtype,
            )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        set_selection(self.transformer, sel)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = latents
                latent_model_input[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(latent_model_input.shape[0])

                # Concatenate extra channels if present
                model_input = latent_model_input
                if extra is not None:
                    model_input = torch.cat([latent_model_input, extra], dim=1)

                # Get model pred
                pos_noise_pred = self.transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                ).sample
                if guidance_scale > 1.0:
                    assert neg_embeds is not None
                    assert pooled_neg_embeds is not None
                    neg_noise_pred = self.transformer(
                        hidden_states=model_input,
                        timestep=timestep,
                        encoder_hidden_states=neg_embeds,
                        pooled_projections=pooled_neg_embeds,
                    ).sample
                    noise_pred = neg_noise_pred + guidance_scale * (
                        pos_noise_pred - neg_noise_pred
                    )
                else:
                    noise_pred = pos_noise_pred

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["SD3"] = [
    "transformer_blocks.*.norm1.linear",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
]
