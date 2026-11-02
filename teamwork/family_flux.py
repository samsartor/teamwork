"""
FLUX.1 (and FLUX.1 Kontext) helpers shared by the batch and sequence implementations.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from einops import rearrange
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, retrieve_timesteps, calculate_shift
from diffusers.models.embeddings import CombinedTimestepGuidanceTextProjEmbeddings

# FLUX.1 latents follow the SD3 VAE convention (shift, then scale).
from .family_sd3 import vae_encode, vae_decode

__all__ = [
    "vae_encode",
    "vae_decode",
    "patchify",
    "unpatchify",
    "latent_image_ids",
    "offset_image_ids",
    "has_guidance",
    "encode_prompt",
    "inference_timesteps",
]


def patchify(latents: Tensor) -> Tensor:
    """`(b, c, h, w)` latents to `(b, h/2 * w/2, c*4)` tokens, channels ordered `(c ph pw)`."""
    return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def unpatchify(tokens: Tensor, channels: int, lh: int, lw: int) -> Tensor:
    """Inverse of `patchify` for a `(lh, lw)` latent grid."""
    return rearrange(
        tokens, "b (h w) (c ph pw) -> b c (h ph) (w pw)", c=channels, h=lh // 2, w=lw // 2, ph=2, pw=2,
    )


def latent_image_ids(lh: int, lw: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """3-axis RoPE ids `(lh/2 * lw/2, 3)` of one patchified image, as FluxPipeline builds them."""
    return FluxPipeline._prepare_latent_image_ids(1, lh // 2, lw // 2, device, dtype)


def offset_image_ids(image_ids: Tensor, offsets: list[int]) -> Tensor:
    """
    RoPE ids of several images in one sequence: a copy of `image_ids` per entry of `offsets`,
    with the offset added to the first (frame) axis, concatenated along the sequence.
    Shared by FLUX.1 and FLUX.2 Klein.
    """
    parts = []
    for offset in offsets:
        ids = image_ids.clone()
        ids[:, 0] += offset
        parts.append(ids)
    return torch.cat(parts, 0)


def has_guidance(transformer: Any) -> bool:
    """Whether the transformer takes a distilled guidance input (FLUX.1 dev, not schnell)."""
    return isinstance(transformer.time_text_embed, CombinedTimestepGuidanceTextProjEmbeddings)


def encode_prompt(
    pipe: Any, prompt: str | list[str], count: int, dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    `(prompt_embeds, pooled_prompt_embeds, text_ids)` for `count` rows, from the cached
    empty prompt (see `empty_prompts`) if there is one, otherwise from the text encoders.
    """
    if pipe.empty_prompt_embeds is not None and pipe.empty_pooled_prompt_embeds is not None:
        assert pipe.empty_text_ids is not None
        prompt_embeds = pipe.empty_prompt_embeds.repeat(count, 1, 1)
        pooled_prompt_embeds = pipe.empty_pooled_prompt_embeds.repeat(count, 1)
        text_ids = pipe.empty_text_ids
    else:
        (prompt_embeds, pooled_prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            prompt,
            num_images_per_prompt=count if isinstance(prompt, str) else 1,
        )
    return prompt_embeds.to(dtype), pooled_prompt_embeds.to(dtype), text_ids.to(dtype)


def inference_timesteps(
    scheduler: Any, num_inference_steps: int, image_seq_len: int, device: torch.device,
) -> tuple[Tensor, int]:
    """FluxPipeline's shifted sampling schedule for `image_seq_len` packed image tokens."""
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,  # type: ignore
    )
    return timesteps, num_inference_steps
