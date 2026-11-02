"""
Qwen-Image 2.1 helpers shared by the batch and sequence implementations: VAE encode/decode, latent
packing, the sampling and training schedules, prompt + condition-image encoding through Qwen3-VL,
RoPE, and the single-stream block forward (flex attention under a teamwork `BlockMask`).

How the base model reads one sample (`QwenImage21Transformer2DModel.forward`):
- Qwen3-VL encodes `user\\n<image1><|vision_start|><|image_pad|>…<|vision_end|>{prompt}<|im_end|>\\n
  <|im_start|>assistant\\n` (the system prompt is dropped). Each `<|image_pad|>` slot stands for a 2×2
  group of latent tokens; the transformer replaces the slots of each condition image with that image's
  `img_in(latents)` and appends the target image's tokens.
- Attention is block-causal: causal over the stream, each image block bidirectional.
- Text and condition tokens modulate from t=0, the target from the sampled t.
"""

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal, Sequence

import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from einops import einsum, rearrange
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21TransformerBlock,
    QwenImage21Transformer2DModel,
    apply_rotary_emb_qwen,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from .sequence_linear import ModulationRows, SequenceSelection, SequenceTeamworkModulation
from .sequence_mask import sequence_mask_mod

VAE_SCALE_FACTOR = 16  # pixels per latent token; the transformer reads latents unpatched
TOKENS_PER_SLOT = 4  # latent tokens per Qwen3-VL `<|image_pad|>` slot (2×2)
FLEX_BLOCK_SIZE = 128  # flex_attention's block size
# Streams are padded to a multiple of this inside attention, so the static-shape compiled kernel
# sees few distinct lengths as prompt lengths vary (one per bucket, not one per prompt length).
FLEX_LENGTH_BUCKET = 1024
assert FLEX_LENGTH_BUCKET % FLEX_BLOCK_SIZE == 0


def check_resolution(height: int, width: int):
    """Every image-pad slot covers 2×2 latent tokens of 16 pixels, so sides must be multiples of 32."""
    m = 2 * VAE_SCALE_FACTOR
    assert height % m == 0 and width % m == 0, f"Qwen-Image 2.1 needs sides divisible by {m}, got {height}x{width}"


# --------------------------------------------------------------------------------------
# VAE and latents
# --------------------------------------------------------------------------------------


def _latent_stats(vae: Any, like: Tensor) -> tuple[Tensor, Tensor]:
    channels = vae.config["z_dim"]
    mean = torch.tensor(vae.config["latents_mean"]).view(1, channels, 1, 1)
    std = torch.tensor(vae.config["latents_std"]).view(1, channels, 1, 1)
    return mean.to(like.device, like.dtype), std.to(like.device, like.dtype)


@torch.no_grad()
def vae_encode(vae: Any, x: Tensor) -> Tensor:
    """
    `(n, 3, H, W)` RGB in [-1, 1] -> `(n, z, H/16, W/16)` normalized posterior means, as
    `QwenImage21Pipeline._encode_vae_image`. The VAE reads RGBA: the base pipeline converts condition
    images to RGBA, so opaque images get an alpha channel of 1.
    """
    assert x.ndim == 4 and x.shape[1] == 3, f"expected (n, 3, H, W) RGB, got {tuple(x.shape)}"
    check_resolution(x.shape[2], x.shape[3])
    x = x.to(vae.device, vae.dtype)
    rgba = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
    z = vae.encode(rearrange(rgba, "n c h w -> n c 1 h w")).latent_dist.mode()
    z = rearrange(z, "n c 1 h w -> n c h w")
    mean, std = _latent_stats(vae, z)
    return (z - mean) / std


@torch.no_grad()
def vae_decode(vae: Any, z: Tensor) -> Tensor:
    """Inverse of `vae_encode`: `(n, z, h, w)` -> `(n, 3, 16h, 16w)` RGB in [-1, 1] (alpha dropped)."""
    assert z.ndim == 4 and z.shape[1] == vae.config["z_dim"], tuple(z.shape)
    z = z.to(vae.device, vae.dtype)
    mean, std = _latent_stats(vae, z)
    x = vae.decode(rearrange(z * std + mean, "n c h w -> n c 1 h w"), return_dict=False)[0]
    return rearrange(x, "n c 1 h w -> n c h w")[:, :3]


def pack(latents: Tensor) -> Tensor:
    """`(n, C, h, w)` -> `(n, h*w, C)` tokens in raster order (2.1 reads latents unpatched)."""
    return rearrange(latents, "n c h w -> n (h w) c")


def unpack(tokens: Tensor, h: int, w: int) -> Tensor:
    return rearrange(tokens, "n (h w) c -> n c h w", h=h, w=w)


# --------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------


def calculate_mu(scheduler_config: Any, image_seq_len: int) -> float:
    """The resolution-dependent shift `QwenImage21Pipeline` passes to the scheduler."""
    return calculate_shift(
        image_seq_len,
        scheduler_config.get("base_image_seq_len", 256),
        scheduler_config.get("max_image_seq_len", 4096),
        scheduler_config.get("base_shift", 0.5),
        scheduler_config.get("max_shift", 1.15),
    )


def inference_timesteps(
    scheduler: Any,
    num_inference_steps: int,
    image_seq_len: int,
    device: torch.device,
    sample_sigmas: Sequence[float] | None,
) -> tuple[Tensor, int]:
    """`QwenImage21Pipeline`'s sampling schedule for `image_seq_len` target tokens."""
    sigmas = sample_sigmas
    if sigmas is None:
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()
    mu = calculate_mu(scheduler.config, image_seq_len)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu  # type: ignore
    )
    scheduler.set_begin_index(0)
    return timesteps, num_inference_steps


def training_scheduler(scheduler_config: Any, image_seq_len: int) -> FlowMatchEulerDiscreteScheduler:
    """
    A fresh scheduler whose `timesteps`/`sigmas` are the base sampling schedule's warp (dynamic shift
    for `image_seq_len` plus the terminal stretch) over `num_train_timesteps` uniform steps, so
    `sampling.flow_match_noising`'s uniform index draws train where inference samples.
    """
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    n = scheduler.config["num_train_timesteps"]
    sigmas = np.linspace(1.0, 1 / n, n).tolist()
    scheduler.set_timesteps(sigmas=sigmas, mu=calculate_mu(scheduler.config, image_seq_len))
    return scheduler


# --------------------------------------------------------------------------------------
# Prompt and condition images through Qwen3-VL
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class QwenPrompt:
    """
    Qwen3-VL's last-layer hidden states for one prompt per batch element, as the base pipeline's
    `encode_prompt` returns them: system prompt dropped, right-padded to a common length. Condition
    images precede the prompt in the template, so their `<|image_pad|>` slots sit at the same
    positions for every batch element.
    """

    hidden: Tensor  # (B, S_vlm, C)
    valid: Tensor  # (B, S_vlm) bool, False on right padding
    image_pad: Tensor  # (S_vlm,) bool, True at `<|image_pad|>` slots (shared by the batch)

    def __post_init__(self):
        assert self.hidden.ndim == 3, tuple(self.hidden.shape)
        B, S, _ = self.hidden.shape
        assert self.valid.dtype == torch.bool and self.valid.shape == (B, S), (self.valid.dtype, tuple(self.valid.shape))
        assert self.image_pad.dtype == torch.bool and self.image_pad.shape == (S,), (
            self.image_pad.dtype, tuple(self.image_pad.shape)
        )
        assert self.valid.device == self.hidden.device and self.image_pad.device == self.hidden.device
        assert bool(self.valid[:, self.image_pad].all()), "image-pad slots must be valid"

    @property
    def batch_size(self) -> int:
        return self.hidden.shape[0]

    def to(self, device: torch.device, dtype: torch.dtype) -> "QwenPrompt":
        return QwenPrompt(
            self.hidden.to(device, dtype), self.valid.to(device), self.image_pad.to(device)
        )

    def select(self, indices: Tensor) -> "QwenPrompt":
        """One row per entry of `indices` (e.g. `Components.batch_indices`)."""
        return QwenPrompt(self.hidden[indices], self.valid[indices], self.image_pad)

    def text_hidden(self) -> Tensor:
        """`(B, S_text, C)`: the non-slot positions, in order. The slots are replaced by latents."""
        return self.hidden[:, ~self.image_pad]

    def text_valid(self) -> Tensor:
        return self.valid[:, ~self.image_pad]

    def pieces(self, image_length: int) -> list[tuple[Literal["text", "image"], int]]:
        """
        Stream pieces of the VLM part: `("text", n)` runs and `("image", k)` for the k-th condition
        image, each of which must have `image_length / 4` slots.
        """
        assert image_length % TOKENS_PER_SLOT == 0, image_length
        slots = image_length // TOKENS_PER_SLOT
        pad = self.image_pad.tolist()
        pieces: list[tuple[Literal["text", "image"], int]] = []
        position = 0
        num_images = 0
        while position < len(pad):
            end = position
            while end < len(pad) and pad[end] == pad[position]:
                end += 1
            if pad[position]:
                assert end - position == slots, (
                    f"condition image {num_images} has {end - position} slots, expected {slots} for "
                    f"{image_length} latent tokens"
                )
                pieces.append(("image", num_images))
                num_images += 1
            else:
                pieces.append(("text", end - position))
            position = end
        return pieces


class PatchwiseConv3d(nn.Conv3d):
    """
    A `Conv3d` with `kernel_size == stride` over inputs that are exactly one kernel each, computed
    as a linear over the flattened patch. Qwen3-VL's vision `patch_embed.proj` is called that way,
    and cuDNN's bf16 Conv3d takes ~20 s for a 1024² image there; this takes well under 1 ms. Same
    parameters and state_dict keys; only the accumulation order differs.
    """

    def forward(self, input: Tensor) -> Tensor:
        assert tuple(self.stride) == tuple(self.kernel_size), (self.stride, self.kernel_size)
        assert self.groups == 1 and all(p == 0 for p in self.padding) and all(d == 1 for d in self.dilation)  # type: ignore[union-attr]
        assert input.ndim == 5 and tuple(input.shape[1:]) == (self.in_channels, *self.kernel_size), (
            tuple(input.shape), self.in_channels, self.kernel_size
        )
        out = einsum(input, self.weight, "n c t h w, o c t h w -> n o")
        if self.bias is not None:
            out = out + self.bias
        return rearrange(out, "n o -> n o 1 1 1")


def use_patchwise_vision_embed(text_encoder: nn.Module) -> int:
    """
    Re-class (in place, sharing weights) every `visual.patch_embed.proj` Conv3d of a Qwen3-VL text
    encoder to `PatchwiseConv3d`. Idempotent; returns how many modules are patchwise afterwards.
    """
    count = 0
    for name, module in text_encoder.named_modules():
        if name.endswith("patch_embed.proj") and isinstance(module, nn.Conv3d):
            if type(module) is nn.Conv3d:
                module.__class__ = PatchwiseConv3d
            assert isinstance(module, PatchwiseConv3d), type(module)
            count += 1
    return count


def image_to_pil(x: Tensor) -> PIL.Image.Image:
    """`(3, H, W)` in [-1, 1] -> RGB PIL, rounding (so uint8 images round-trip exactly)."""
    assert x.ndim == 3 and x.shape[0] == 3, tuple(x.shape)
    u8 = ((x.detach().float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return PIL.Image.fromarray(rearrange(u8, "c h w -> h w c").cpu().numpy())


@torch.no_grad()
def encode_prompts(
    pipe: Any,
    prompts: Sequence[str],
    images: Sequence[Sequence[PIL.Image.Image]],
) -> QwenPrompt:
    """
    Encode one prompt (and its condition images, VLM order) per batch element with the base
    pipeline's own `_get_qwen_prompt_embeds`, so the t2i/ti2i templates, the system-prompt drop,
    and the pre-norm hidden states are exactly the base model's. Runs on the text encoder's device.
    Every element must have the same number and size of condition images.
    """
    assert len(prompts) == len(images) and len(prompts) > 0, (len(prompts), len(images))
    num_images = {len(i) for i in images}
    assert len(num_images) == 1, f"every batch element needs the same number of condition images: {num_images}"
    device = pipe.text_encoder.device
    rows: list[tuple[Tensor, Tensor, Tensor]] = []
    for prompt, condition in zip(prompts, images):
        embeds, mask, pad = pipe._get_qwen_prompt_embeds(
            [prompt], list(condition) if len(condition) > 0 else None, device=device
        )
        assert embeds.shape[0] == 1 and mask.shape == embeds.shape[:2] and pad.shape == embeds.shape[:2]
        assert bool(mask.bool().all()), "a single prompt has no padding"
        rows.append((embeds[0], pad[0].bool(), mask[0].bool()))
    S = max(e.shape[0] for e, _, _ in rows)
    hidden = torch.stack([F.pad(e, (0, 0, 0, S - e.shape[0])) for e, _, _ in rows])
    valid = torch.stack([F.pad(m, (0, S - m.shape[0])) for _, _, m in rows])
    pads = torch.stack([F.pad(p, (0, S - p.shape[0])) for _, p, _ in rows])
    assert bool((pads == pads[0]).all()), "condition-image slots differ across the batch"
    return QwenPrompt(hidden=hidden, valid=valid, image_pad=pads[0])


# --------------------------------------------------------------------------------------
# Transformer pieces
# --------------------------------------------------------------------------------------


def rotary(
    transformer: QwenImage21Transformer2DModel,
    image_mask: Tensor,  # (S,) bool, True at image tokens
    lh: int,
    lw: int,
) -> Tensor:
    """The base RoPE over a stream whose image blocks (all `lh × lw`) sit where `image_mask` is set."""
    num_images = int(image_mask.sum()) // (lh * lw)
    assert num_images * lh * lw == int(image_mask.sum()), (int(image_mask.sum()), lh, lw)
    return transformer.pos_embed([(1, lh, lw)] * num_images, image_mask, device=image_mask.device)


def time_embedding(transformer: QwenImage21Transformer2DModel, timesteps: Tensor, like: Tensor) -> Tensor:
    """`(n,)` timesteps in [0, 1] -> `(n, D)` temb, in `like`'s dtype as the base model computes it."""
    assert timesteps.ndim == 1, tuple(timesteps.shape)
    return transformer.time_text_embed(timesteps.to(like.dtype), like)


def modulation_linear(transformer: QwenImage21Transformer2DModel) -> nn.Module:
    """The shared `modulation` linear (`modulation = Sequential(SiLU, Linear)`)."""
    act, linear = transformer.modulation[0], transformer.modulation[1]
    assert isinstance(act, nn.SiLU), type(act)
    return linear


def modulation_rows(transformer: QwenImage21Transformer2DModel, text_temb: Tensor, teammate_temb: Tensor) -> ModulationRows:
    """
    Text and per-slot modulation rows. With `SequenceTeamworkModulation` these get the teammate
    LoRAs; any other module (the base linear, or a PEFT-wrapped one) is applied to both as is.
    """
    linear = modulation_linear(transformer)
    if isinstance(linear, SequenceTeamworkModulation):
        return linear.rows(F.silu(text_temb), F.silu(teammate_temb))
    return ModulationRows(text=linear(F.silu(text_temb)), teammates=linear(F.silu(teammate_temb)))


def base_modulation(transformer: QwenImage21Transformer2DModel, temb: Tensor) -> Tensor:
    """The un-adapted modulation of `temb` (whatever adapter the linear carries)."""
    linear = modulation_linear(transformer)
    assert isinstance(linear, nn.Linear), type(linear)
    return F.linear(F.silu(temb), linear.weight, linear.bias)


def output_projection(transformer: QwenImage21Transformer2DModel, image_tokens: Tensor, temb: Tensor) -> Tensor:
    """`norm_out` + `proj_out` of `(n, L, D)` image tokens, each row modulated by its `(n, D)` temb."""
    assert image_tokens.ndim == 3 and temb.shape == (image_tokens.shape[0], image_tokens.shape[2]), (
        tuple(image_tokens.shape), tuple(temb.shape)
    )
    return transformer.proj_out(transformer.norm_out(image_tokens, temb, None))


def padded_block_mask(selection: SequenceSelection) -> BlockMask:
    """
    `sequence_mask_mod(selection)` over the stream padded to a multiple of `FLEX_LENGTH_BUCKET`:
    padded keys are never attended to, padded queries attend to nothing (and are sliced off).

    `_compiled_flex_attention` is static-shape and guards on everything `mask_mod` closes over, so
    the closure holds only tensors whose shapes depend on the padded length (no Python ints such
    as the true length). Prompts of different lengths within one bucket then reuse one kernel.
    Fully padded blocks are skipped by the kernel, so a coarse bucket costs almost nothing.
    """
    layout = selection.layout
    S = layout.seq_len
    S_pad = -(-S // FLEX_LENGTH_BUCKET) * FLEX_LENGTH_BUCKET
    device = selection.present.device
    key_valid = (
        selection.key_valid
        if selection.key_valid is not None
        else torch.ones(selection.batch_size, S, dtype=torch.bool, device=device)
    )
    # The padding is trailing text whose keys are invalid; its queries are cut off by `real`.
    padded = replace(selection, layout=replace(layout, seq_len=S_pad), key_valid=F.pad(key_valid, (0, S_pad - S)))
    inner = sequence_mask_mod(padded)
    real = torch.arange(S_pad, device=device) < S  # (S_pad,)

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return real[q_idx] & inner(b, h, q_idx, kv_idx)

    return create_block_mask(mask_mod, B=selection.batch_size, H=None, Q_LEN=S_pad, KV_LEN=S_pad, device=device)


# The fused kernel; uncompiled flex_attention materializes the dense fp32 score matrix.
_compiled_flex_attention: Callable[..., Any] = torch.compile(flex_attention, dynamic=False)


def attention(attn: QwenImage21Attention, x: Tensor, rope: Tensor, block_mask: BlockMask) -> Tensor:
    """`QwenImage21Attention` over `(N, S, D)` streams under `block_mask` (from `padded_block_mask`)."""
    N, S, _ = x.shape
    S_pad = block_mask.seq_lengths[0]
    assert S_pad >= S and S_pad % FLEX_BLOCK_SIZE == 0, (S, S_pad)
    head_dim = attn.inner_dim // attn.heads
    query = rearrange(attn.to_q(x), "n s (h d) -> n s h d", d=head_dim)
    key = rearrange(attn.to_k(x), "n s (h d) -> n s h d", d=head_dim)
    value = rearrange(attn.to_v(x), "n s (h d) -> n s h d", d=head_dim)
    query = attn.norm_q(query).to(value.dtype)
    key = attn.norm_k(key).to(value.dtype)
    # diffusers annotates the return as a tuple; with use_real=False it is one tensor.
    query, key = apply_rotary_emb_qwen(query, rope, use_real=False), apply_rotary_emb_qwen(key, rope, use_real=False)
    assert isinstance(query, Tensor) and isinstance(key, Tensor)

    def heads_first(t: Tensor) -> Tensor:
        return F.pad(rearrange(t, "n s h d -> n h s d"), (0, 0, 0, S_pad - S)).contiguous()

    out = _compiled_flex_attention(heads_first(query), heads_first(key), heads_first(value), block_mask=block_mask)
    assert isinstance(out, Tensor)
    out = rearrange(out[:, :, :S], "n h s d -> n s (h d)").to(query.dtype)
    return attn.to_out[1](attn.to_out[0](out))


def block_forward(
    block: QwenImage21TransformerBlock,
    x: Tensor,  # (N, S, D)
    modulation: Tensor,  # (N, S, 4D): per-token [scale1, gate1, scale2, gate2]
    rope: Tensor,
    block_mask: BlockMask,
) -> Tensor:
    """`QwenImage21TransformerBlock.forward` with per-token modulation and a teamwork mask."""
    assert modulation.shape == (*x.shape[:2], 4 * x.shape[2]), (tuple(modulation.shape), tuple(x.shape))
    scale1, gate1, scale2, gate2 = modulation.chunk(4, dim=-1)
    x = x + gate1.tanh() * attention(block.attn, block.img_norm1(x) * (1 + scale1), rope, block_mask)
    x = x + gate2.tanh() * block.img_mlp(block.img_norm2(x) * (1 + scale2))
    if x.dtype == torch.float16:
        x = x.clip(-65504, 65504)
    return x


def per_token(rows: Tensor, token_row: Tensor) -> Tensor:
    """`(N, R, F)` modulation rows, `(S,)` row index per token -> `(N, S, F)`."""
    assert rows.ndim == 3 and token_row.ndim == 1 and token_row.dtype == torch.int64
    return rows[:, token_row]
