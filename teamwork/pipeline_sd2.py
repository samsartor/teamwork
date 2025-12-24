from typing import Any, Literal
import torch
import torch.nn.functional as F
from torch import Tensor
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    StableDiffusionPipeline,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig
from .adapter import adapt, save_adapters, TEAMWORK_PROFILES, Adapt, adapter_modules
from .batch import BatchBuilder, OutputImageType
from .attn import TeamworkJointAttention


class StableDiffusion2TeamworkPipeline(TeamworkPipeline, StableDiffusionPipeline):
    teamwork_config: TeamworkConfig
    empty_prompt_embeds: None | Tensor = None

    @classmethod
    @torch.no_grad()
    def from_base_pipeline(
        cls,
        base_pipeline: StableDiffusionPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
    ):
        pipeline = cls(**base_pipeline.components)  # type: ignore
        pipeline.teamwork_config = teamwork_config
        pipeline.unet = adapt(
            pipeline.unet,
            cfg=teamwork_config,
            device=pipeline.device,
            dtype=torch.bfloat16,
            requires_grad=training,
            override_profile=override_profile,
            state=state,
        )
        if training:
            pipeline.unet.enable_gradient_checkpointing()
        return pipeline

    @property
    def in_channels(self) -> int:
        return self.unet.config["in_channels"]

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.unet, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

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
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
        (prompt_embeds, _) = self.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.text_encoder = None
        self.tokenizer = None

    def train_loss(
        self,
        batch: BatchBuilder,
        noise: torch.Tensor,
        model: Any | None = None,
        prompt: str = "",
    ) -> LossOutput:
        with torch.no_grad():
            # Sample a random timestep for each image
            train_scheduler = DDPMScheduler.from_config(self.scheduler.config)
            assert isinstance(train_scheduler, DDPMScheduler)
            scheduler_config: Any = train_scheduler.config
            timesteps: Any = torch.randint(
                0,
                scheduler_config.num_train_timesteps,
                [1],
                device=self.device,
                dtype=torch.long,
            ).repeat(batch.count)

            # Add noise to the latents according to the noise magnitude at each timestep
            sel = batch.selection()
            latents = batch.packed_encoded_images(
                self.vae_encode,
                self.unet.config["in_channels"],
                1 / self.vae_scale_factor,
            )
            noisy_latents = train_scheduler.add_noise(latents, noise, timesteps)
            noisy_latents[sel.input_subindices] = latents[sel.input_subindices]

            # Get the target for loss depending on the prediction type
            if scheduler_config.prediction_type == "epsilon":
                target = noise
                pred_type = 'noise'
            elif scheduler_config.prediction_type == "v_prediction":
                target = train_scheduler.get_velocity(latents, noise, timesteps)
                pred_type = 'velocity'
            else:
                raise ValueError(
                    f"Unknown prediction type {scheduler_config.prediction_type}"
                )

            # Get prompt embeds if needed
            if self.text_encoder is None:
                prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
            else:
                (prompt_embeds, _) = self.encode_prompt(
                    prompt,
                    device=self.device,
                    num_images_per_prompt=latents.shape[0],
                    do_classifier_free_guidance=False,
                )

        # Concatenate extra channels if present
        model_input = noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_input = torch.cat([noisy_latents, extra], dim=1)

        # Update selection
        for adapter in adapter_modules(self.unet).values():
            adapter.selection = sel

        # Predict the noise residual
        if model is None:
            model = self.unet
        model_pred = model(
            sample=model_input,
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
        negative_prompt="",
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
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

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # Initialize latents
        sel = batch.selection()
        clean_latents = batch.packed_encoded_images(
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

        # Get prompt embeds if needed
        neg_embeds = None
        if self.empty_prompt_embeds is not None:
            prompt_embeds = self.empty_prompt_embeds.repeat(latents.shape[0], 1, 1)
        elif guidance_scale > 1.0:
            (prompt_embeds, neg_embeds) = self.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=latents.shape[0],
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        else:
            (prompt_embeds, _) = self.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=latents.shape[0],
                do_classifier_free_guidance=False,
            )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Update selection
        for adapter in adapter_modules(self.unet).values():
            adapter.selection = sel

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for t in timesteps:
                latent_model_input = self.scheduler.scale_model_input(latents, t)
                latent_model_input[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(latent_model_input.shape[0])

                # Concatenate extra channels if present
                model_input = latent_model_input
                if extra is not None:
                    model_input = torch.cat([latent_model_input, extra], dim=1)

                # Get model pred
                pos_noise_pred = self.unet(
                    sample=model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )[0]
                if guidance_scale > 1.0:
                    assert neg_embeds is not None
                    neg_noise_pred = self.unet(
                        sample=model_input,
                        timestep=timestep,
                        encoder_hidden_states=neg_embeds,
                        return_dict=False,
                    )[0]
                    noise_pred = neg_noise_pred + guidance_scale * (
                        pos_noise_pred - neg_noise_pred
                    )
                else:
                    noise_pred = pos_noise_pred

                # Scheduler step
                latents = self.scheduler.step(
                    noise_pred, t.item(), latents, return_dict=False
                )[0]

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["SD2"] = [
    "*.*.attentions.*.transformer_blocks.*.attn1.to_q",
    "*.*.attentions.*.transformer_blocks.*.attn1.to_k",
    "*.*.attentions.*.transformer_blocks.*.attn1.to_v",
    "*.*.attentions.*.transformer_blocks.*.attn1.to_out.0",
    "*.*.attentions.*.transformer_blocks.*.ff.net.0.proj",
    "*.*.attentions.*.transformer_blocks.*.ff.net.2",
    "*.*.resnets.*.conv1",
    "*.*.resnets.*.conv2",
    "*.*.resnets.*.conv_shortcut",
]

TEAMWORK_PROFILES["SD2_PLUSATTN"] = [
    *TEAMWORK_PROFILES["SD2"],
    ("*.*.attentions.*.transformer_blocks.*.attn1", TeamworkJointAttention),
]
