"""
Token-ownership attention masks for sequence-axis teamwork, generalizing the joint-attention
`_teammate_block_mask` to arbitrary `SequenceLayout`s.

Rules shared by both `MaskRule`s:
- Absent `(b, t)` slots are never attended to, and their queries attend to nothing (flex_attention
  returns zeros for fully masked rows; those tokens are never read back).
- Teammate↔teammate edges between different slots also need `attn_keep[q_slot, k_slot]`.
- Keys with `key_valid[b, k] == False` (padded text) are never attended to.

`"joint"` (FLUX.1, Klein, SD3): otherwise full attention; text sees and is seen by every present
slot, so with full presence and no `attn_keep` this is the base model's unmasked attention.

`"block_causal"` (Qwen-Image 2.1): the base rule `q >= k or same image block` (text causal, each
image block bidirectional), plus:
- outputs see each other bidirectionally, wherever they sit in the stream;
- non-output tokens (text and inputs) never see outputs.
With roster `(edited.out, source.in)` laid out as base ti2i this is exactly the base mask.
"""

from typing import Callable

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from .sequence_linear import TEXT_SLOT, SequenceSelection

MaskMod = Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]


def sequence_mask_mod(selection: SequenceSelection) -> MaskMod:
    """flex_attention `mask_mod(b, h, q_idx, kv_idx)` for `selection`; vmappable and broadcastable."""
    layout = selection.layout
    device = selection.present.device
    T = layout.num_teammates
    slot = layout.slot_of_token(device)  # (S,)
    is_output = torch.tensor(layout.outputs, dtype=torch.bool, device=device)  # (T,)
    present = selection.present
    keep = (
        selection.attn_keep
        if selection.attn_keep is not None
        else torch.ones(T, T, dtype=torch.bool, device=device)
    )
    key_valid = (
        selection.key_valid
        if selection.key_valid is not None
        else torch.ones(selection.batch_size, layout.seq_len, dtype=torch.bool, device=device)
    )
    block_causal = layout.rule == "block_causal"

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        q_slot, k_slot = slot[q_idx], slot[kv_idx]
        q_img, k_img = q_slot != TEXT_SLOT, k_slot != TEXT_SLOT
        q_slot_c, k_slot_c = q_slot.clamp(min=0), k_slot.clamp(min=0)
        q_ok = ~q_img | present[b, q_slot_c]
        k_ok = (~k_img | present[b, k_slot_c]) & key_valid[b, kv_idx]
        edge_ok = ~q_img | ~k_img | keep[q_slot_c, k_slot_c]
        allowed = q_ok & k_ok & edge_ok
        if block_causal:
            same_block = q_img & (q_slot == k_slot)
            q_out = q_img & is_output[q_slot_c]
            k_out = k_img & is_output[k_slot_c]
            structure = ((q_idx >= kv_idx) | same_block | (q_out & k_out)) & (q_out | ~k_out)
            allowed = allowed & structure
        return allowed

    return mask_mod


def sequence_block_mask(selection: SequenceSelection) -> BlockMask:
    """`BlockMask` of `sequence_mask_mod` over `(B, S, S)`, on the selection's device."""
    return create_block_mask(
        sequence_mask_mod(selection),
        B=selection.batch_size,
        H=None,
        Q_LEN=selection.layout.seq_len,
        KV_LEN=selection.layout.seq_len,
        device=selection.present.device,
    )


@torch.no_grad()
def sequence_dense_mask(selection: SequenceSelection) -> Tensor:
    """
    Dense `(B, S, S)` bool reference, built segment by segment from the rules in the module
    docstring rather than per token. Meant for tests and debugging, not for the forward pass.
    """
    layout = selection.layout
    S, B = layout.seq_len, selection.batch_size
    present = selection.present.cpu()
    keep = None if selection.attn_keep is None else selection.attn_keep.cpu()
    mask = torch.zeros(B, S, S, dtype=torch.bool)
    for b in range(B):
        for qi, (qa, qe, qs) in enumerate(layout.segments):
            if qs != TEXT_SLOT and not bool(present[b, qs]):
                continue
            for ki, (ka, ke, ks) in enumerate(layout.segments):
                if ks != TEXT_SLOT and not bool(present[b, ks]):
                    continue
                if qs != TEXT_SLOT and ks != TEXT_SLOT and qs != ks:
                    if keep is not None and not bool(keep[qs, ks]):
                        continue
                if layout.rule == "joint":
                    block = torch.ones(qe - qa, ke - ka, dtype=torch.bool)
                else:
                    q_out = qs != TEXT_SLOT and layout.outputs[qs]
                    k_out = ks != TEXT_SLOT and layout.outputs[ks]
                    if k_out and not q_out:
                        continue
                    if (q_out and k_out) or ki < qi:
                        block = torch.ones(qe - qa, ke - ka, dtype=torch.bool)
                    elif ki == qi:
                        block = torch.ones(qe - qa, ke - ka, dtype=torch.bool)
                        if qs == TEXT_SLOT:
                            block = block.tril()
                    else:
                        continue
                mask[b, qa:qe, ka:ke] = block
    if selection.key_valid is not None:
        mask &= selection.key_valid.cpu()[:, None, :]
    return mask
