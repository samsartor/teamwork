"""
FLUX.2 Klein helpers shared by the batch and sequence implementations.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor
from einops import repeat
from diffusers.pipelines.flux2.pipeline_flux2_klein import retrieve_timesteps, compute_empirical_mu

# Klein patchifies latents and offsets image ids the same way as FLUX.1.
from .family_flux import patchify, unpatchify, offset_image_ids

__all__ = [
    "patchify",
    "unpatchify",
    "offset_image_ids",
    "legacy_latents",
    "latent_bn_stats",
    "vae_encode",
    "vae_decode",
    "latent_image_ids",
    "encode_prompt",
    "inference_timesteps",
]


def legacy_latents(version: str) -> bool:
    """Version 1.0 checkpoints were trained on raw, sampled VAE latents."""
    return version == "1.0"


def latent_bn_stats(vae: Any, lh: int, lw: int, x: Tensor) -> tuple[Tensor, Tensor]:
    """The VAE's BN mean/std, laid out on the unpatchified (b, 32, lh, lw) grid.

    Klein normalizes the 2x2-patchified latents with `vae.bn`, whose 128 channels are
    ordered (c ph pw), matching Flux2KleinPipeline._patchify_latents and `patchify`.
    """
    bn = vae.bn
    assert bn.running_mean is not None and bn.running_var is not None
    std = torch.sqrt(bn.running_var + vae.config["batch_norm_eps"])
    layout = "(c ph pw) -> 1 c (h ph) (w pw)"
    mean = repeat(bn.running_mean, layout, ph=2, pw=2, h=lh // 2, w=lw // 2)
    std = repeat(std, layout, ph=2, pw=2, h=lh // 2, w=lw // 2)
    return mean.to(x.device, x.dtype), std.to(x.device, x.dtype)


def vae_encode(vae: Any, x: Tensor, version: str, generator: torch.Generator | None = None) -> Tensor:
    """
    Images in [-1, 1] to latents: BN-normalized posterior means, or 1.0's raw samples (drawn
    from `generator`, or the global RNG if None).
    """
    x = x.to(vae.device, vae.dtype)
    if legacy_latents(version):
        return vae.encode(x).latent_dist.sample(generator)
    x = vae.encode(x).latent_dist.mode()
    mean, std = latent_bn_stats(vae, x.shape[2], x.shape[3], x)
    return (x - mean) / std


def vae_decode(vae: Any, x: Tensor, version: str) -> Tensor:
    x = x.to(vae.device, vae.dtype)
    if not legacy_latents(version):
        mean, std = latent_bn_stats(vae, x.shape[2], x.shape[3], x)
        x = x * std + mean
    x = vae.decode(x, return_dict=False)[0]
    return x


def latent_image_ids(lh: int, lw: int, device: torch.device) -> Tensor:
    """4-axis (T, H, W, L) RoPE ids `(lh/2 * lw/2, 4)` of one patchified image.

    Mirrors Flux2KleinPipeline._prepare_latent_ids for a single image: T and L
    stay 0, H/W index the patch grid. Built in float32 so large grids stay exact.
    """
    hh, ww = lh // 2, lw // 2
    ids = torch.zeros(hh * ww, 4, device=device, dtype=torch.float32)
    ids[:, 1] = torch.arange(hh, device=device).repeat_interleave(ww)
    ids[:, 2] = torch.arange(ww, device=device).repeat(hh)
    return ids


def encode_prompt(pipe: Any, prompt: str | list[str], count: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """
    `(prompt_embeds, text_ids)` for `count` rows, from the cached empty prompt (see
    `empty_prompts`) if there is one, otherwise from the Qwen3 text encoder.
    Klein has no pooled embeddings. The ids stay float32: Klein's text positions run to 511,
    which bf16 can't represent exactly.
    """
    if pipe.empty_prompt_embeds is not None:
        assert pipe.empty_text_ids is not None
        prompt_embeds = pipe.empty_prompt_embeds.repeat(count, 1, 1)
        text_ids = pipe.empty_text_ids
    else:
        (prompt_embeds, text_ids) = pipe.encode_prompt(
            prompt,
            num_images_per_prompt=count if isinstance(prompt, str) else 1,
        )
    return prompt_embeds.to(dtype), text_ids.to(torch.float32)


def inference_timesteps(
    scheduler: Any, num_inference_steps: int, image_seq_len: int, device: torch.device,
) -> tuple[Tensor, int]:
    """Flux2KleinPipeline's empirically shifted sampling schedule for `image_seq_len` packed image tokens."""
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu,  # type: ignore
    )
    return timesteps, num_inference_steps
