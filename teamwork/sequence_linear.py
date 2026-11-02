"""
Sequence-axis teamwork: each batch element is one token stream holding the text and one image
block per roster slot, and teammate LoRAs slice their image ranges out of the stream.

Pipelines keep latents in component layout `(count, ...)` (see `data.Components`). A family's
sequence forward converts at the transformer boundary:

```
layout = SequenceLayout.joint(text_length, image_length, outputs)           # or from_pieces(...)
selection = SequenceSelection.from_components(components, layout, attn_keep=..., text_teammate=...)
stream = to_sequence(img_in(latents), txt_in(text), components, layout)    # (B, S, D)
...  # blocks, with SequenceTeamworkLinear/Modulation adapters and sequence_mask.sequence_block_mask
image_tokens, text_tokens = from_sequence(stream, components, layout)
```
"""

from dataclasses import dataclass
from functools import cached_property
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import einsum, rearrange

from .adapter import AdapterMixin, TeamworkConfig, shallowcopy_into
from .data import Components
from .adapter import LoraParameters

MaskRule = Literal["joint", "block_causal"]
"""
How a family's base model attends over the stream (see `sequence_mask`):
- `"joint"` (FLUX.1, Klein, SD3): full attention; text is always visible.
- `"block_causal"` (Qwen-Image 2.1): causal over the stream, every image block bidirectional.
"""

TEXT_SLOT = -1


@dataclass(frozen=True)
class SequenceLayout:
    """
    Static token layout of one stream, the same for every batch element.

    Roster slot `t` occupies `[image_offsets[t], image_offsets[t] + image_length)`. Every other
    position is "text" (for Qwen: VLM text, padding included), filled in stream order from the
    family's text tokens. Offsets need not be sorted or evenly spaced, but ranges never overlap.
    """

    seq_len: int  # S
    image_length: int  # L
    image_offsets: tuple[int, ...]  # one per roster slot
    outputs: tuple[bool, ...]  # one per roster slot, whether the slot is an `.out` teammate
    rule: MaskRule

    def __post_init__(self):
        T = len(self.image_offsets)
        assert T >= 1, "a sequence layout needs at least one roster slot"
        assert len(self.outputs) == T, f"{len(self.outputs)} output flags for {T} slots"
        assert self.image_length > 0, self.image_length
        assert self.rule in ("joint", "block_causal"), self.rule
        spans = sorted((o, o + self.image_length) for o in self.image_offsets)
        assert spans[0][0] >= 0 and spans[-1][1] <= self.seq_len, (
            f"image ranges {spans} exceed the stream length {self.seq_len}"
        )
        for (_, prev_end), (start, _) in zip(spans, spans[1:]):
            assert prev_end <= start, f"image ranges overlap: {spans}"

    @staticmethod
    def joint(
        text_length: int,
        image_length: int,
        outputs: Sequence[bool],
        rule: MaskRule = "joint",
    ) -> "SequenceLayout":
        """`[text | img_0 | … | img_{T-1}]`, the FLUX.1/Klein/SD3 layout (uniform stride)."""
        T = len(outputs)
        return SequenceLayout(
            seq_len=text_length + T * image_length,
            image_length=image_length,
            image_offsets=tuple(text_length + t * image_length for t in range(T)),
            outputs=tuple(outputs),
            rule=rule,
        )

    @staticmethod
    def from_pieces(
        pieces: Sequence[tuple[Literal["text", "image"], int]],
        image_length: int,
        outputs: Sequence[bool],
        rule: MaskRule,
    ) -> "SequenceLayout":
        """
        Build a layout from stream-ordered pieces: `("text", length)` or `("image", slot)`.
        Every slot in `range(len(outputs))` must appear exactly once. E.g. Qwen ti2i with roster
        `(edited.out, source.in)`: `[("text", a), ("image", 1), ("text", b), ("image", 0)]`.
        """
        offsets: dict[int, int] = {}
        position = 0
        for kind, value in pieces:
            if kind == "text":
                assert value >= 0, value
                position += value
            elif kind == "image":
                assert value not in offsets, f"slot {value} appears twice"
                offsets[value] = position
                position += image_length
            else:
                raise ValueError(f"unknown piece kind {kind!r}")
        T = len(outputs)
        assert sorted(offsets) == list(range(T)), f"slots {sorted(offsets)} for a roster of {T}"
        return SequenceLayout(
            seq_len=position,
            image_length=image_length,
            image_offsets=tuple(offsets[t] for t in range(T)),
            outputs=tuple(outputs),
            rule=rule,
        )

    @property
    def num_teammates(self) -> int:
        return len(self.image_offsets)

    @property
    def text_length(self) -> int:
        return self.seq_len - self.num_teammates * self.image_length

    @cached_property
    def segments(self) -> tuple[tuple[int, int, int], ...]:
        """Stream-ordered `(start, end, slot)` runs; `slot == TEXT_SLOT` for text. No empty runs."""
        images = sorted((o, o + self.image_length, t) for t, o in enumerate(self.image_offsets))
        segments: list[tuple[int, int, int]] = []
        position = 0
        for start, end, slot in images:
            if start > position:
                segments.append((position, start, TEXT_SLOT))
            segments.append((start, end, slot))
            position = end
        if position < self.seq_len:
            segments.append((position, self.seq_len, TEXT_SLOT))
        return tuple(segments)

    @cached_property
    def text_ranges(self) -> tuple[tuple[int, int], ...]:
        return tuple((a, b) for a, b, slot in self.segments if slot == TEXT_SLOT)

    @cached_property
    def uniform_start(self) -> int | None:
        """`o0` if `image_offsets == range(o0, o0 + T·L, L)` (the view fast path), else None."""
        o0 = self.image_offsets[0]
        L = self.image_length
        if all(o == o0 + t * L for t, o in enumerate(self.image_offsets)):
            return o0
        return None

    def slot_of_token(self, device: torch.device | str) -> Tensor:
        """`(S,)` int: the roster slot owning each position, `TEXT_SLOT` for text."""
        slot = torch.full((self.seq_len,), TEXT_SLOT, dtype=torch.int64)
        for t, o in enumerate(self.image_offsets):
            slot[o : o + self.image_length] = t
        return slot.to(device)


@dataclass(frozen=True)
class SequenceSelection:
    """
    What sequence adapters and masks read. Absent `(b, t)` slots still occupy their range in the
    stream: their tokens start at zero, never contribute to attention or to the LoRA
    communication sum, and receive no LoRA delta.
    """

    layout: SequenceLayout
    present: Tensor  # (B, T): bool
    attn_keep: Tensor | None  # (T, T): bool with an all-True diagonal; None = all kept
    text_teammate: int | None  # roster slot whose adapter also applies to text tokens; None = base
    key_valid: Tensor | None = None  # (B, S): bool; False keys are never attended to (padded text)

    def __post_init__(self):
        T = self.layout.num_teammates
        assert self.present.dtype == torch.bool, self.present.dtype
        assert self.present.ndim == 2 and self.present.shape[1] == T, (
            f"present {tuple(self.present.shape)} for {T} slots"
        )
        if self.attn_keep is not None:
            assert self.attn_keep.dtype == torch.bool, self.attn_keep.dtype
            assert self.attn_keep.shape == (T, T), tuple(self.attn_keep.shape)
            assert self.attn_keep.device == self.present.device
            assert bool(self.attn_keep.diagonal().all()), "attn_keep diagonal must be True"
        if self.text_teammate is not None:
            assert 0 <= self.text_teammate < T, self.text_teammate
        if self.key_valid is not None:
            assert self.key_valid.dtype == torch.bool, self.key_valid.dtype
            assert self.key_valid.shape == (self.batch_size, self.layout.seq_len), (
                tuple(self.key_valid.shape)
            )
            assert self.key_valid.device == self.present.device

    @property
    def batch_size(self) -> int:
        return self.present.shape[0]

    @property
    def num_teammates(self) -> int:
        return self.layout.num_teammates

    @staticmethod
    def from_components(
        components: Components,
        layout: SequenceLayout,
        *,
        attn_keep: Tensor | None,
        text_teammate: int | None,
        key_valid: Tensor | None = None,
    ) -> "SequenceSelection":
        """`present[b, t]` iff some component has `(batch_indices, teammate_indices) == (b, t)`."""
        _check_components(components, layout)
        B, T = components.batch_size, components.num_teammates
        device = components.teammate_indices.device
        counts = torch.zeros(B, T, dtype=torch.int64, device=device).index_put(
            (components.batch_indices, components.teammate_indices),
            torch.ones_like(components.teammate_indices),
            accumulate=True,
        )
        assert int(counts.max()) <= 1, "a (batch, teammate) slot holds more than one component"
        return SequenceSelection(
            layout=layout,
            present=counts > 0,
            attn_keep=None if attn_keep is None else attn_keep.to(device),
            text_teammate=text_teammate,
            key_valid=key_valid,
        )


def _check_components(components: Components, layout: SequenceLayout):
    assert components.num_teammates == layout.num_teammates, (
        f"components have {components.num_teammates} teammates, layout has {layout.num_teammates}"
    )
    assert components.teammate_indices.shape == components.batch_indices.shape


def image_blocks(stream: Tensor, layout: SequenceLayout) -> Tensor:
    """`(B, S, D)` stream -> `(B, T, L, D)` image blocks (a view on the uniform path)."""
    assert stream.ndim == 3 and stream.shape[1] == layout.seq_len, (
        f"stream {tuple(stream.shape)} for seq_len {layout.seq_len}"
    )
    L, T = layout.image_length, layout.num_teammates
    o0 = layout.uniform_start
    if o0 is not None:
        return rearrange(stream[:, o0 : o0 + T * L], "b (t l) d -> b t l d", t=T)
    return torch.stack([stream[:, o : o + L] for o in layout.image_offsets], dim=1)


def text_tokens_of(stream: Tensor, layout: SequenceLayout) -> Tensor:
    """`(B, S, D)` stream -> `(B, S_text, D)` text tokens, in stream order."""
    pieces = [stream[:, a:b] for a, b in layout.text_ranges]
    if len(pieces) == 0:
        return stream[:, :0]
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=1)


def to_sequence(
    image_tokens: Tensor,  # (count, L, D), component layout
    text_tokens: Tensor,  # (B, S_text, D), in stream order
    components: Components,
    layout: SequenceLayout,
) -> Tensor:
    """Scatter component image tokens and family text tokens into `(B, S, D)` streams."""
    _check_components(components, layout)
    count = components.teammate_indices.shape[0]
    B, L = components.batch_size, layout.image_length
    assert image_tokens.ndim == 3 and image_tokens.shape[:2] == (count, L), (
        f"image tokens {tuple(image_tokens.shape)}, expected ({count}, {L}, D)"
    )
    D = image_tokens.shape[2]
    assert text_tokens.shape == (B, layout.text_length, D), (
        f"text tokens {tuple(text_tokens.shape)}, expected ({B}, {layout.text_length}, {D})"
    )
    assert text_tokens.dtype == image_tokens.dtype and text_tokens.device == image_tokens.device

    dense = image_tokens.new_zeros(B, layout.num_teammates, L, D).index_put(
        (components.batch_indices, components.teammate_indices), image_tokens
    )
    pieces: list[Tensor] = []
    text_position = 0
    for start, end, slot in layout.segments:
        if slot == TEXT_SLOT:
            pieces.append(text_tokens[:, text_position : text_position + end - start])
            text_position += end - start
        else:
            pieces.append(dense[:, slot])
    assert text_position == layout.text_length
    return torch.cat(pieces, dim=1)


def from_sequence(
    stream: Tensor,  # (B, S, D)
    components: Components,
    layout: SequenceLayout,
) -> tuple[Tensor, Tensor]:
    """Inverse of `to_sequence`: `(count, L, D)` component image tokens, `(B, S_text, D)` text."""
    _check_components(components, layout)
    assert stream.shape[0] == components.batch_size, (
        f"stream batch {stream.shape[0]}, components batch {components.batch_size}"
    )
    blocks = image_blocks(stream, layout)
    image_tokens = blocks[components.batch_indices, components.teammate_indices]
    return image_tokens, text_tokens_of(stream, layout)


def _lora_parameters(base: nn.Linear, cfg: TeamworkConfig) -> LoraParameters:
    T = len(cfg.teammates)
    return LoraParameters(
        torch.empty([T, base.in_features, cfg.lora_rank]),
        torch.empty([T, cfg.lora_rank, base.out_features]),
        torch.empty([T, base.out_features]) if base.bias is not None and cfg.use_bias else None,
    )


class SequenceTeamworkLinear(nn.Linear, AdapterMixin["SequenceSelection", LoraParameters]):
    """
    `out = base(x)` on every token of `(B, S, in)` streams, plus per-teammate LoRA on the image
    ranges. With `lora_communication`, slot `t`'s delta is `up_t(Σ_{present s} down_s(x_s))`,
    otherwise `up_t(down_t(x_t))`. Absent slots get no delta. Text tokens get
    `up_k(down_k(x)) + bias_k` for `k = text_teammate`, or nothing when it is None.

    Parameters and checkpoint keys (`adapter.down|up|bias`) match the batch-axis `TeamworkLinear`.

    Both LoRA matmuls run over the whole stream, so the layout (uniform stride or not) only
    affects rank-sized bookkeeping:
    - one GEMM computes every teammate's down projection for every token, `(B·S, in) × (in, T·r)`;
    - each token's owner r-block (the communicated sum, for image tokens) is routed into a
      `(B, S, K)` tensor, `K = T·r` plus one ownership column per teammate for the bias, padded
      to a multiple of 8 for tensor-core alignment;
    - one GEMM against the stacked `[up; bias]` gives the delta for the whole stream.
    Slicing the `in`/`out`-sized activations instead costs a full-size zero-fill + copy (slice
    backward) or clone (in-place add on an output view) per range in the backward pass, which
    measured slower than the T-fold redundant rank-r FLOPs for T=3.
    """

    selection: SequenceSelection | None = None

    def __init__(self, base: nn.Linear, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.communicate = cfg.lora_communication
        self.adapter = _lora_parameters(base, cfg)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        sel = self.selection
        if sel is None:
            return output
        layout = sel.layout
        B, T, L = sel.batch_size, layout.num_teammates, layout.image_length
        assert input.shape[:2] == (B, layout.seq_len), (
            f"input {tuple(input.shape)}, expected ({B}, {layout.seq_len}, {self.in_features})"
        )
        down, up, bias = self.adapter.down, self.adapter.up, self.adapter.bias
        assert down.shape[0] == T, f"adapter has {down.shape[0]} teammates, layout has {T}"
        r = down.shape[2]
        lora_dtype = down.dtype
        present = sel.present.to(lora_dtype)  # (B, T)

        hidden = einsum(input.to(lora_dtype), down, "b s i, t i r -> b s t r")
        slot_hidden = [hidden[:, o : o + L, t] for t, o in enumerate(layout.image_offsets)]
        if self.communicate:
            summed = sum(einsum(h, present[:, t], "b l r, b -> b l r") for t, h in enumerate(slot_hidden))
            assert isinstance(summed, Tensor)
            slot_hidden = [summed] * T

        num_bias = T if bias is not None else 0
        K = T * r + num_bias
        K_padded = -(-K // 8) * 8

        def routed(h: Tensor, owner: int, ownership: Tensor) -> Tensor:
            """`(B, n, r)` owner block + `(B, n, 1)` ownership -> `(B, n, K_padded)`."""
            row = F.pad(h, (owner * r, (T - 1 - owner) * r))
            if bias is None:
                return F.pad(row, (0, K_padded - T * r))
            column = F.pad(ownership, (owner, K_padded - T * r - 1 - owner))
            return torch.cat([row, column], dim=-1)

        k = sel.text_teammate
        pieces: list[Tensor] = []
        for start, end, slot in layout.segments:
            n = end - start
            if slot == TEXT_SLOT:
                if k is None:
                    pieces.append(hidden.new_zeros(B, n, K_padded))
                else:
                    pieces.append(routed(hidden[:, start:end, k], k, hidden.new_ones(B, n, 1)))
            else:
                p = present[:, slot, None, None]
                pieces.append(routed(slot_hidden[slot] * p, slot, p.expand(B, n, 1)))
        routing = torch.cat(pieces, dim=1)

        weight_rows = [rearrange(up, "t r o -> (t r) o")]
        if bias is not None:
            weight_rows.append(bias)
        weight_rows.append(up.new_zeros(K_padded - K, self.out_features))
        delta = einsum(routing, torch.cat(weight_rows, dim=0), "b s k, k o -> b s o")
        return output + delta.to(output.dtype)


@dataclass(frozen=True)
class ModulationRows:
    text: Tensor  # (B, out): base row, or text_teammate's adapted row
    teammates: Tensor  # (B, T, out): adapted row per roster slot; absent slots get the base row


class SequenceTeamworkModulation(nn.Linear, AdapterMixin["SequenceSelection", LoraParameters]):
    """
    Adapts a linear on `temb` (no token axis): FLUX `norm*.linear`, Klein `*_modulation_*`,
    Qwen `modulation`. Call `rows(text_temb, teammate_temb)`; the family's block applies
    `teammates[:, t]` to range `t` and `text` to the text tokens. Plain `forward` is the base
    linear and refuses to run while a selection is active (it would silently drop the LoRA).
    """

    selection: SequenceSelection | None = None

    def __init__(self, base: nn.Linear, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.communicate = cfg.lora_communication
        self.adapter = _lora_parameters(base, cfg)

    def forward(self, input: Tensor) -> Tensor:
        assert self.selection is None, (
            "SequenceTeamworkModulation must be called through .rows() while a selection is set"
        )
        return super().forward(input)

    def rows(self, text_temb: Tensor, teammate_temb: Tensor) -> ModulationRows:
        text = F.linear(text_temb, self.weight, self.bias)
        teammates = F.linear(teammate_temb, self.weight, self.bias)
        sel = self.selection
        if sel is None:
            return ModulationRows(text=text, teammates=teammates)
        B, T = sel.batch_size, sel.num_teammates
        assert text_temb.shape == (B, self.in_features), tuple(text_temb.shape)
        assert teammate_temb.shape == (B, T, self.in_features), tuple(teammate_temb.shape)
        down, up, bias = self.adapter.down, self.adapter.up, self.adapter.bias
        lora_dtype = down.dtype
        present = sel.present.to(lora_dtype)

        hidden = einsum(teammate_temb.to(lora_dtype), down, "b t i, t i r -> b t r")
        if self.communicate:
            hidden = einsum(einsum(hidden, present, "b t r, b t -> b r"), present, "b r, b t -> b t r")
        else:
            hidden = einsum(hidden, present, "b t r, b t -> b t r")
        delta = einsum(hidden, up, "b t r, t r o -> b t o")
        if bias is not None:
            delta = delta + einsum(present, bias, "b t, t o -> b t o")
        teammates = teammates + delta.to(teammates.dtype)

        k = sel.text_teammate
        if k is not None:
            text_hidden = einsum(text_temb.to(lora_dtype), down[k], "b i, i r -> b r")
            text_delta = einsum(text_hidden, up[k], "b r, r o -> b o")
            if bias is not None:
                text_delta = text_delta + bias[k]
            text = text + text_delta.to(text.dtype)

        return ModulationRows(text=text, teammates=teammates)
