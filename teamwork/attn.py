"""
LEGACY JOINT ATTENTION (the `FLUX_PLUSATTN` profile). Delete in phase 6.

Everything outside attention runs batch-axis (every component is a batch row with its own
text copy), and these processors scatter the components into a dense
`(B, L_text + T*L_img)` sequence for `flex_attention`, then gather the result back.
The sequence-axis implementation (`sequence_linear.py`, `sequence_mask.py`) replaces it.
"""

from dataclasses import dataclass

import torch
from torch import nn, Tensor
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from diffusers.models.transformers.transformer_flux import FluxAttention, FluxAttnProcessor
from diffusers.models.embeddings import apply_rotary_emb

from .adapter import AdapterMixin, TeamworkConfig, shallowcopy_into
from .batch_linear import BatchSelection
from .data import TeamBuilder


@dataclass(frozen=True)
class LegacyJointSelection(BatchSelection):
    """`BatchSelection` plus the fields the legacy joint-attention modules read."""

    present_matrix: Tensor  # (batch, num_teammates): bool, whether each (batch, teammate) slot has a component
    first_component_per_batch: Tensor  # (batch,): int, the component whose text copy is canonical
    attn_keep: Tensor | None = None  # (num_teammates, num_teammates): bool, per-pair cross-teammate attention mask

    @classmethod
    def from_builder(cls, builder: TeamBuilder) -> "LegacyJointSelection":
        sel = BatchSelection.from_components(builder.components_index())
        present_matrix = torch.zeros(sel.batch_size, sel.num_teammates, dtype=torch.bool)
        first_component_per_batch = torch.full((sel.batch_size,), -1, dtype=torch.int64)
        for i, component in enumerate(builder.components):
            present_matrix[component.batch_idx, component.teammate] = True
            if first_component_per_batch[component.batch_idx] == -1:
                first_component_per_batch[component.batch_idx] = i
        return cls(
            teammate_indices=sel.teammate_indices,
            batch_indices=sel.batch_indices,
            input_subindices=sel.input_subindices,
            output_subindices=sel.output_subindices,
            num_teammates=sel.num_teammates,
            batch_size=sel.batch_size,
            batch_matrix=sel.batch_matrix,
            present_matrix=present_matrix.to(builder.device),
            first_component_per_batch=first_component_per_batch.to(builder.device),
            attn_keep=builder.attn_keep(),
        )


def _teammate_block_mask(
    present: Tensor,
    attn_keep: Tensor | None,
    L_text: int,
    L_img: int,
    T: int,
    device: torch.device,
):
    """
    Build a flex_attention BlockMask over `[text(L_text), img_t0(L_img), ..., img_t{T-1}(L_img)]`
    with a per-batch dim. Edges are kept iff:
      - both endpoints are present in the batch (text always counts as present), AND
      - either endpoint is text, or `attn_keep[q_team, k_team]` is True (when provided).
    """
    total_len = L_text + T * L_img
    B = present.shape[0]

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_text = q_idx < L_text
        k_is_text = kv_idx < L_text
        q_team = torch.clamp((q_idx - L_text) // L_img, 0, T - 1)
        k_team = torch.clamp((kv_idx - L_text) // L_img, 0, T - 1)
        q_ok = q_is_text | present[b, q_team]
        k_ok = k_is_text | present[b, k_team]
        if attn_keep is None:
            edge_ok = q_is_text | k_is_text | torch.tensor(True, device=device)
        else:
            edge_ok = q_is_text | k_is_text | attn_keep[q_team, k_team]
        return q_ok & k_ok & edge_ok

    return create_block_mask(
        mask_mod,
        B=B,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )


@torch.compile
def _masked_attention_impl(
    img_query: Tensor,  # (T_components, L_img, H, F)
    img_key: Tensor,
    img_value: Tensor,
    txt_query: Tensor,  # (B, L_text, H, F)
    txt_key: Tensor,
    txt_value: Tensor,
    image_rotary_emb: tuple[Tensor, Tensor] | None,  # (cos, sin), each (L_text + T*L_img, D)
    batch_indices: Tensor,    # (T_components,): int
    teammate_indices: Tensor, # (T_components,): int
    present: Tensor,          # (B, T): bool
    attn_keep: Tensor | None, # (T, T): bool
):
    """
    Run flex_attention over the dense (B, L_text + T*L_img, H, F) layout, scattering image
    components into per-(batch, teammate) slots and gathering the image output back into
    components form. Returns (img_out, txt_out).
    """
    B, T = present.shape
    Tc, L_img, H, Fd = img_query.shape
    L_text = txt_query.shape[1]

    def scatter_dense(x: Tensor) -> Tensor:
        dense = x.new_zeros(B, T, L_img, H, Fd)
        dense[batch_indices, teammate_indices] = x
        return dense.reshape(B, T * L_img, H, Fd)

    q = torch.cat([txt_query, scatter_dense(img_query)], dim=1)
    k = torch.cat([txt_key,   scatter_dense(img_key)],   dim=1)
    v = torch.cat([txt_value, scatter_dense(img_value)], dim=1)

    if image_rotary_emb is not None:
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        assert isinstance(q, Tensor)
        assert isinstance(k, Tensor)

    # flex_attention expects (B, H, S, F)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    block_mask = _teammate_block_mask(
        present, attn_keep, L_text, L_img, T, q.device
    )
    # On GPUs with <128 KiB dynamic smem per block (A40/A6000/4090/L40), the
    # autotuner's default flex_attention configs don't fit at head_dim=128 once
    # mask_mod captures attn_keep — every config OOMs and Triton bails with
    # "no valid triton configs". Force a smaller tile + single pipeline stage.
    props = torch.cuda.get_device_properties(q.device)
    if props.shared_memory_per_block_optin < 128 * 1024:
        kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    else:
        kernel_options = None
    out = flex_attention(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    assert isinstance(out, Tensor)
    out = out.transpose(1, 2)  # (B, S, H, F)

    txt_out = out[:, :L_text]
    img_dense = out[:, L_text:].reshape(B, T, L_img, H, Fd)
    img_out = img_dense[batch_indices, teammate_indices]
    return img_out, txt_out


class TeamworkFluxJointAttnProcessor(FluxAttnProcessor):
    def __init__(self):
        super().__init__()

    @torch.compile
    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(attn, FluxTeamworkJointAttention)
        assert encoder_hidden_states is not None
        assert attention_mask is None, (
            "TeamworkFluxJointAttnProcessor handles masking via the BlockMask "
            "(present_matrix + attn_keep); external attention_mask is not supported"
        )
        sel = attn.selection
        assert sel is not None

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attn.added_kv_proj_dim is not None:
            # double block
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
        else:
            # single block: text shares the same QKV projection as image
            encoder_query = attn.to_q(encoder_hidden_states)
            encoder_key = attn.to_k(encoder_hidden_states)
            encoder_value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))           # (Tc, L_img, H, F)
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))  # (Tc, L_text, H, F)
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        if attn.added_kv_proj_dim is not None:
            # double block: image and text use separate norms
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
        else:
            # single block: image and text share norm_q/norm_k. RMSNorm acts only on the
            # head dim, so applying it before vs. after the dense scatter is equivalent.
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            encoder_query = attn.norm_q(encoder_query)
            encoder_key = attn.norm_k(encoder_key)

        # One text QKV per batch element (B, L_text, H, F). Each component duplicates
        # the same per-batch text via the projection's per-teammate LoRA, so picking
        # the first component per batch matches the original [:1] semantics for B=1.
        fcb = sel.first_component_per_batch
        txt_q = encoder_query[fcb]
        txt_k = encoder_key[fcb]
        txt_v = encoder_value[fcb]

        img_out, txt_out = _masked_attention_impl(
            query, key, value,
            txt_q, txt_k, txt_v,
            image_rotary_emb,
            sel.batch_indices,
            sel.teammate_indices,
            sel.present_matrix,
            sel.attn_keep,
        )

        img_out = img_out.flatten(-2, -1).to(query.dtype)  # (Tc, L_img, dim)
        txt_out = txt_out.flatten(-2, -1).to(query.dtype)  # (B, L_text, dim)

        if attn.added_kv_proj_dim is not None:
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            txt_out = attn.to_add_out(txt_out)
            assert txt_out is not None

        # Broadcast text output from per-batch back to per-component, indexed by the
        # batch each component belongs to. Equivalent to repeat_interleave when B=1.
        encoder_hidden_states_out = txt_out[sel.batch_indices]

        return img_out, encoder_hidden_states_out  # type: ignore
        
 
class FluxTeamworkJointAttention(FluxAttention, AdapterMixin[LegacyJointSelection, nn.Parameter]):
    def __init__(self, base: FluxAttention, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.adapter = nn.Parameter(torch.tensor(0.0))
        self.set_processor(TeamworkFluxJointAttnProcessor()) # type: ignore
