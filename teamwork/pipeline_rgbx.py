from typing import Any, Literal
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
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


class StableDiffusionRGB2XPipeline(TeamworkPipeline, StableDiffusionPipeline):
    teamwork_config: TeamworkConfig
    component_prompt_embeds: None | Tensor = None
    component_prompts: dict[str, str]

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: StableDiffusionPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
    ) -> "StableDiffusionRGB2XPipeline":
        assert override_profile is None, "rgb2x doesn't use teamwork"
        pipeline = cls(**base_pipeline.components)
        pipeline.teamwork_config = teamwork_config
        pipeline.component_prompts = DEFAULT_COMPONENT_PROMPTS
        pipeline._expand_head()
        if state is not None:
            pipeline.unet.load_state_dict(state)
        if training:
            pipeline.unet.requires_grad_(True)
        return pipeline

    @property
    def in_channels(self) -> int:
        return self.unet.config["in_channels"]

    def _expand_head(self):
        head = self.model.conv_in
        new_head = nn.Conv2d(
            head.in_channels * 2,
            head.out_channels,
            head.kernel_size,
            head.stride,
            head.padding,
        )
        new_head.weight.zero_()
        new_head.weight[:, : head.in_channels, :, :].copy_(head.weight)
        self.model.conv_in = new_head

    def save_adapters(self, safetensors_path: str):
        save_entire(
            self.unet,
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
        x = x * self.vae.config.scaling_factor
        return x

    def vae_decode(self, x: Tensor) -> Tensor:
        x = x / self.vae.config.scaling_factor
        x = x.to(self.vae.device, self.vae.dtype)
        x = self.vae.decode(x, return_dict=False)[0]  # type: ignore
        return x

    @torch.no_grad
    def empty_prompts(self):
        prompts = [
            self.component_prompts.get(t, "") for t in self.teamwork_config.teammates
        ]
        (prompt_embeds, _) = self.encode_prompt(
            prompts,
            self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.component_prompt_embeds = prompt_embeds
        self.text_encoder = None
        self.tokenizer = None

    def train_loss(
        self,
        batch: BatchBuilder,
        noise: torch.Tensor,
        model: Any | None = None,
        prompt: str = "",
    ) -> Tensor:
        with torch.no_grad():
            # Split into control signal and outputs
            ibatch, obatch = batch.split_io()
            assert ibatch.count == 1, (
                "this rgb2x implementation only supports a single input"
            )
            control_img = ibatch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            ).repeat(obatch.count, 1, 1, 1)
            noise = noise[batch.selection().output_subindices]

            # Sample a random timestep for each image
            scheduler_config: Any = self.scheduler.config
            timesteps: Any = torch.randint(
                0,
                scheduler_config.num_train_timesteps,
                [1],
                device=self.device,
                dtype=torch.long,
            ).repeat(obatch.count)

            # Add noise to the latents according to the noise magnitude at each timestep
            sel = obatch.selection()
            latents = obatch.packed_encoded_images(
                self.vae_encode,
                self.unet.config["in_channels"],
                1 / self.vae_scale_factor,
            )
            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if scheduler_config.prediction_type == "epsilon":
                target = noise
                pred_type = 'noise'
            elif scheduler_config.prediction_type == "v_prediction":
                target = self.scheduler.get_velocity(latents, noise, timesteps)
                pred_type = 'velocity'
            else:
                raise ValueError(
                    f"Unknown prediction type {scheduler_config.prediction_type}"
                )

            # Add condition
            noisy_cond_latents = torch.cat([noisy_latents, control_img], 1)

            # Add extra channels if present
            extra = obatch.packed_scaled_extra(1 / self.vae_scale_factor)
            if extra is not None:
                noisy_cond_latents = torch.cat([noisy_cond_latents, extra], 1)

            # Get prompt embeds if needed
            if self.text_encoder is None:
                assert self.component_prompt_embeds is not None
                prompt_embeds = self.component_prompt_embeds[sel.teammate_indices]
            else:
                prompts = [
                    f"{self.component_prompts[c.component]} {prompt}"
                    for c in obatch.components
                ]
                (prompt_embeds, _) = self.encode_prompt(
                    prompts,
                    self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )

        # Predict the noise residual
        if model is None:
            model = self.unet
        model_pred = model(
            sample=noisy_cond_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        return LossOutput(
            prediction=model_pred.float()[sel.output_subindices],
            target=target.float()[sel.output_subindices],
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor)[sel.output_subindices].unsqueeze(1),
            type=pred_type,
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
            self.vae_encode, self.unet.config["in_channels"], 1 / self.vae_scale_factor
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
        latents *= self.scheduler.init_noise_sigma

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Get prompt embeds if needed
        if self.text_encoder is None:
            assert self.component_prompt_embeds is not None
            prompt_embeds = self.component_prompt_embeds[sel.teammate_indices]
        else:
            prompts = [
                f"{self.component_prompts[c.component]} {prompt}"
                for c in obatch.components
            ]
            (prompt_embeds, _) = self.encode_prompt(
                prompts,
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # Get extra channels if present
        extra = obatch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                model_input = torch.cat([latent_model_input, control_img], dim=1)

                # Add extra channels if present
                if extra is not None:
                    model_input = torch.cat([model_input, extra], dim=1)

                # Get model pred
                noise_pred = self.unet(
                    model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                progress_bar.update()

        outputs = obatch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["SD2_RGB2X"] = []
