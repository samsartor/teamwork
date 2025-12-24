from typing import Any, Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    StableDiffusion3Pipeline,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .pipelines import TeamworkPipeline, LossOutput
from .adapter import TEAMWORK_PROFILES, save_entire, Adapt
from .config import TeamworkConfig
from .batch import BatchBuilder, OutputImageType

DEFAULT_COMPONENT_PROMPTS = {
    "image.in": "Image",
    "albedo.out": "Albedo (basecolor)",
    "normals.out": "Camera-space Normal",
    "roughness.out": "Roughness",
    "metalness.out": "Metallicness",
    "inverseshading.out": "Shading (lighting and reflection)",
}


class StableDiffusion3RGB2XPipeline(TeamworkPipeline, StableDiffusion3Pipeline):
    teamwork_config: TeamworkConfig
    component_prompt_embeds: None | Tensor = None
    component_pooled_prompt_embeds: None | Tensor = None
    component_prompts: dict[str, str]
    timestep_weight: str = "sigma_sqrt"

    @classmethod
    def from_base_config(
        cls,
        base_pipeline: StableDiffusion3Pipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
    ):
        assert override_profile is None, "rgb2x doesn't use teamwork"
        pipeline = cls(**base_pipeline.components)
        pipeline.teamwork_config = teamwork_config
        pipeline.component_prompts = DEFAULT_COMPONENT_PROMPTS
        pipeline._expand_head()
        if state is not None:
            pipeline.transformer.load_state_dict(state)
        if training:
            pipeline.transformer.requires_grad_(True)
        return pipeline

    @property
    def in_channels(self) -> int:
        return self.transformer.config["in_channels"]

    def _expand_head(self):
        head = self.model.pos_embed.proj
        new_head = nn.Conv2d(
            head.in_channels * 2,
            head.out_channels,
            head.kernel_size,
            head.stride,
            head.padding,
        )
        new_head.weight.zero_()
        new_head.weight[:, : head.in_channels, :, :].copy_(head.weight)
        self.model.pos_embed.proj = new_head

    def save_adapters(self, safetensors_path: str):
        save_entire(
            self.transformer,
            safetensors_path,
            self.teamwork_config,
            extra_metadata={
                "component_prompts": "\n".join(
                    self.component_prompts[t] for t in self.teamwork_config.teammates
                ),
            },
        )

    def load_extra_metadata(self, metadata: dict[str, str]):
        if "component_prompts" in metadata:
            self.component_prompts = {
                t: p
                for (t, p) in zip(
                    self.teamwork_config.teammates,
                    metadata["component_prompts"].split("\n"),
                )
            }

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
    def empty_prompts(self):
        for t in self.teamwork_config.teammates:
            if t not in self.component_prompts:
                print(f"WARNING: no RGBX prompt for {t}")
        prompts = [
            self.component_prompts.get(t, "") for t in self.teamwork_config.teammates
        ]
        (prompt_embeds, _, pooled_prompt_embeds, _) = self.encode_prompt(
            prompts,
            prompts,
            prompts,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.component_prompt_embeds = prompt_embeds
        self.component_pooled_prompt_embeds = pooled_prompt_embeds
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_2 = None
        self.tokenizer_2 = None
        self.text_encoder_3 = None
        self.tokenizer_3 = None

    def train_loss(
        self,
        batch: BatchBuilder,
        noise: torch.Tensor,
        model: Any | None = None,
        prompt: str = "",
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

            # Split into control signal and outputs
            ibatch, obatch = batch.split_io()
            assert ibatch.count == 1, (
                "this rgb2x implementation only supports a single input"
            )
            control_img = ibatch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            ).repeat(obatch.count, 1, 1, 1)
            noise = noise[batch.selection().output_subindices]

            # Figure out how much noise to add to the latents
            timestep_i = (timestep_u * len(self.scheduler.timesteps)).long()
            timesteps = (
                self.scheduler.timesteps[timestep_i]
                .to(device=self.device, dtype=self.dtype)
                .repeat(obatch.count)
            )

            sel = obatch.selection()
            latents = obatch.packed_encoded_images(
                self.vae_encode,
                self.transformer.config["in_channels"],
                1 / self.vae_scale_factor,
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

            # Add condition
            noisy_cond_latents = torch.cat([noisy_latents, control_img], 1)

            # Add extra channels if present
            extra = obatch.packed_scaled_extra(1 / self.vae_scale_factor)
            if extra is not None:
                noisy_cond_latents = torch.cat([noisy_cond_latents, extra], 1)

            # Get prompt embeds if needed
            if self.text_encoder is None:
                assert self.component_prompt_embeds is not None
                assert self.component_pooled_prompt_embeds is not None
                prompt_embeds = self.component_prompt_embeds[sel.teammate_indices]
                pooled_prompt_embeds = self.component_pooled_prompt_embeds[
                    sel.teammate_indices
                ]
            else:
                prompts = [
                    f"{self.component_prompts[c.component]} {prompt}"
                    for c in obatch.components
                ]
                (prompt_embeds, _, pooled_prompt_embeds, _) = self.encode_prompt(
                    prompts,
                    prompts,
                    prompts,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

        # Predict the noise residual
        if model is None:
            model = self.transformer
        model_pred = model(
            hidden_states=noisy_cond_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
        ).sample
        latents_pred = model_pred * (-sigmas) + noisy_latents

        return LossOutput(
            prediction=latents_pred[sel.output_subindices],
            target=latents[sel.output_subindices],
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor)[sel.output_subindices].unsqueeze(1) * weighting,
            type='signal',
        )

    @torch.no_grad()
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str = "",
        num_inference_steps: int = 50,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: BatchBuilder | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
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

        # Split into control signal and outputs
        ibatch, obatch = batch.split_io()
        assert ibatch.count == ibatch.batch_size, (
            "this rgb2x implementation only supports a single input"
        )
        control_img = ibatch.packed_encoded_images(
            self.vae_encode, self.unet.config["in_channels"], 1 / self.vae_scale_factor
        ).repeat(obatch.count // ibatch.count, 1, 1, 1)

        # Initialize latents
        sel = obatch.selection()
        clean_latents = obatch.packed_encoded_images(
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

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Get prompt embeds if needed
        if self.text_encoder is None:
            assert self.component_prompt_embeds is not None
            assert self.component_pooled_prompt_embeds is not None
            prompt_embeds = self.component_prompt_embeds[sel.teammate_indices]
            pooled_prompt_embeds = self.component_pooled_prompt_embeds[
                sel.teammate_indices
            ]
        else:
            prompts = [
                f"{self.component_prompts[c.component]} {prompt}"
                for c in obatch.components
            ]
            (prompt_embeds, _, pooled_prompt_embeds, _) = self.encode_prompt(
                prompts,
                prompts,
                prompts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # Get extra channels if present
        extra = obatch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = latents
                model_input = torch.cat([latent_model_input, control_img], dim=1)

                # Add extra channels if present
                if extra is not None:
                    model_input = torch.cat([model_input, extra], dim=1)

                timestep = t.expand(model_input.shape[0])

                # Get model pred
                noise_pred = self.transformer(
                    hidden_states=model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                ).sample

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                progress_bar.update()

        outputs = obatch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["SD3_RGB2X"] = []
