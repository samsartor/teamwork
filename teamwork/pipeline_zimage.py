import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Literal
from diffusers.pipelines.z_image.pipeline_z_image import (
    ZImagePipeline,
    retrieve_timesteps,
    calculate_shift,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
import numpy as np

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig
from .adapter import adapt, save_adapters, TEAMWORK_PROFILES, Adapt, adapter_modules
from .batch import BatchBuilder, OutputImageType


class ZImageTeamworkPipeline(TeamworkPipeline, ZImagePipeline):
    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt_embeds: None | list[Tensor] = None

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: ZImagePipeline,
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
        )
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
        return self.unwrapped_transformer.config["in_channels"]

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.unwrapped_transformer, safetensors_path, self.teamwork_config)

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
        prompt_embeds, _ = self.encode_prompt(
            prompt,
            device=device or self._execution_device,
            do_classifier_free_guidance=False,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.text_encoder = None
        self.tokenizer = None

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

            # Get prompt embeds if needed
            if self.text_encoder is None:
                assert self.empty_prompt_embeds is not None
                # Repeat embeddings for each item in batch
                prompt_embeds = [
                    self.empty_prompt_embeds[0] for _ in range(latents.shape[0])
                ]
            else:
                prompt_embeds, _ = self.encode_prompt(
                    prompt if isinstance(prompt, str) else prompt[0],
                    device=self.device,
                    do_classifier_free_guidance=False,
                )
                if isinstance(prompt, str):
                    # Repeat for batch
                    prompt_embeds = prompt_embeds * latents.shape[0]

        # Concatenate extra channels if present
        model_latents = noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([noisy_latents, extra], dim=1)

        # Update selection
        for adapter in adapter_modules(self.unwrapped_transformer).values():
            adapter.selection = sel

        # Z-Image uses a list-based input format
        b, _, lh, lw = model_latents.shape

        # Add channel dimension and convert to list
        model_latents_unsqueezed = model_latents.unsqueeze(2)
        latent_model_input_list = list(model_latents_unsqueezed.unbind(dim=0))

        # Normalize timesteps to [0, 1] range
        timestep_normalized = (1000 - timesteps) / 1000

        model_out_list = self.transformer(
            latent_model_input_list,
            timestep_normalized,
            prompt_embeds,
            return_dict=False,
        )[0]

        # Stack outputs back to tensor
        model_pred = torch.stack([t.float() for t in model_out_list], dim=0)
        model_pred = model_pred.squeeze(2)

        # Z-Image negates the output
        model_pred = -model_pred

        # Only use the original channels for prediction
        model_pred = model_pred[:, :self.in_channels]

        return LossOutput(
            selection=sel,
            latents=latents,
            prediction=model_pred,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * weighting,
            timestep_idx=timestep_i,
            type='flow',
        )

    @torch.no_grad
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str = "",
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
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
                dtype=torch.float32,  # Z-Image uses float32 for latents
                generator=generator,
            )
        else:
            latents = noise.to(device, torch.float32)

        b, _, lh, lw = latents.shape
        image_seq_len = (lh // 2) * (lw // 2)

        # Get prompt embeds if needed
        do_classifier_free_guidance = guidance_scale > 1.0
        if self.empty_prompt_embeds is not None:
            prompt_embeds = [self.empty_prompt_embeds[0] for _ in range(latents.shape[0])]
            if do_classifier_free_guidance:
                negative_prompt_embeds = prompt_embeds.copy()
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device=device,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            # Repeat for batch
            prompt_embeds = prompt_embeds * latents.shape[0]
            if do_classifier_free_guidance:
                negative_prompt_embeds = negative_prompt_embeds * latents.shape[0]

        # Set timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            mu=mu,
        )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        # Update selection
        for adapter in adapter_modules(self.transformer).values():
            adapter.selection = sel

        # Store CFG settings
        self._guidance_scale = guidance_scale
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]

                timestep = t.expand(model_latents.shape[0])
                # Normalize timesteps to [0, 1] range
                timestep_normalized = (1000 - timestep) / 1000
                t_norm = timestep_normalized[0].item()

                # Handle cfg truncation
                current_guidance_scale = guidance_scale
                if (
                    do_classifier_free_guidance
                    and cfg_truncation is not None
                    and float(cfg_truncation) <= 1
                ):
                    if t_norm > cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

                # Concatenate extra channels if present
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                # Prepare model input
                if apply_cfg:
                    latents_typed = model_latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep_normalized.repeat(2)
                else:
                    latent_model_input = model_latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep_normalized

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                model_out_list = self.transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    return_dict=False,
                )[0]

                if apply_cfg:
                    # Perform CFG
                    actual_batch_size = b
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if cfg_normalization and float(cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                # Only use original channels
                noise_pred = noise_pred[:, :self.in_channels]

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["ZIMAGE"] = [
    "layers.*.adaLN_modulation.0",
    "layers.*.attention.to_q",
    "layers.*.attention.to_k",
    "layers.*.attention.to_v",
    "layers.*.attention.to_out",
    "layers.*.feed_forward.w1",
    "layers.*.feed_forward.w2",
    "layers.*.feed_forward.w3",
]
