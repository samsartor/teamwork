"""
Stable Diffusion 3 helpers shared by the batch and sequence implementations.
"""

from typing import Any

import torch
from torch import Tensor


def vae_encode(vae: Any, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """
    Images in [-1, 1] to shifted and scaled latents, sampled from the posterior with `generator`
    (or the global RNG if None).
    """
    x = x.to(vae.device, vae.dtype)
    x = vae.encode(x).latent_dist.sample(generator)
    x = (x - (vae.config.get("shift_factor", None) or 0)) * vae.config["scaling_factor"]
    return x


def vae_decode(vae: Any, x: Tensor) -> Tensor:
    x = x / vae.config["scaling_factor"] + (vae.config.get("shift_factor", None) or 0)
    x = x.to(vae.device, vae.dtype)
    x = vae.decode(x, return_dict=False)[0]
    return x


def encode_prompt(pipe: Any, prompt: str | list[str], count: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """
    `(prompt_embeds, pooled_prompt_embeds)` for `count` rows, from the cached empty prompt
    (see `empty_prompts`) if there is one, otherwise from the text encoders.
    """
    if pipe.empty_prompt_embeds is not None and pipe.empty_pooled_prompt_embeds is not None:
        prompt_embeds = pipe.empty_prompt_embeds.repeat(count, 1, 1)
        pooled_prompt_embeds = pipe.empty_pooled_prompt_embeds.repeat(count, 1)
    else:
        (prompt_embeds, _, pooled_prompt_embeds, _) = pipe.encode_prompt(
            prompt,
            prompt,
            prompt,
            num_images_per_prompt=count if isinstance(prompt, str) else 1,
            do_classifier_free_guidance=False,
        )
    return prompt_embeds.to(dtype), pooled_prompt_embeds.to(dtype)
