import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0,
)
from diffusers.models.transformers.transformer_flux import FluxAttention, FluxAttnProcessor
from diffusers.models.embeddings import apply_rotary_emb
from einops import rearrange

from .adapter import AdapterMixin, TeamworkConfig, shallowcopy_into


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


class TeamworkJointAttention(Attention, AdapterMixin):
    def __init__(self, base: Attention, cfg: TeamworkConfig):
        shallowcopy_into(self, base)

        self.adapter = nn.Parameter(torch.tensor(0.0))

        if isinstance(base.processor, AttnProcessor2_0):
            self.set_processor(TeamworkAttnProcessor())  # type: ignore
        elif isinstance(self.processor, JointAttnProcessor2_0):
            self.set_processor(TeamworkJointAttnProcessor())  # type: ignore
        elif isinstance(self.processor, FluxAttnProcessor2_0):
            self.set_processor(TeamworkFluxAttnProcessor())  # type: ignore
        else:
            assert False, (
                f"no teamwork joint attention avalible for {self.processor.__class__}"
            )


class TeamworkAttnProcessor(AttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        temb: Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(attn, TeamworkJointAttention)
        assert attn.selection is None or attn.selection.attn_keep is None, (
            "TeamworkAttnProcessor does not implement attn_keep gating; "
            "use a flex_attention-based processor (e.g. TeamworkFluxJointAttnProcessor) instead"
        )
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        assert attn.to_k is not None
        key = attn.to_k(encoder_hidden_states)
        assert attn.to_v is not None
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        t, h, l, _ = query.shape
        _, _, s, _ = value.shape
        query = rearrange(query, "(b t) h l f -> b h (t l) f", b=1, t=t, h=h, l=l)
        key = rearrange(key, "(b t) h s f -> b h (t s) f", b=1, t=t, h=h, s=s)
        value = rearrange(value, "(b t) h s f -> b h (t s) f", b=1, t=t, h=h, s=s)
        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = rearrange(
            hidden_states, "b h (t l) f -> (b t) h l f", b=1, t=t, h=h, l=l
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        assert attn.to_out is not None
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class TeamworkJointAttnProcessor(JointAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        attention_mask: Tensor | None = None,
        *args,
        **kwargs,
    ):
        assert isinstance(attn, TeamworkJointAttention)
        assert attn.selection is None or attn.selection.attn_keep is None, (
            "TeamworkJointAttnProcessor does not implement attn_keep gating; "
            "use a flex_attention-based processor (e.g. TeamworkFluxJointAttnProcessor) instead"
        )
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        assert attn.to_k is not None
        key = attn.to_k(hidden_states)
        assert attn.to_v is not None
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # `context` projections.
        if encoder_hidden_states is not None:
            assert attn.add_q_proj is not None
            assert attn.add_k_proj is not None
            assert attn.add_v_proj is not None
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(
                    encoder_hidden_states_key_proj
                )

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        t, h, l, _ = query.shape
        query = rearrange(query, "(b t) h l f -> b h (t l) f", b=1, t=t, h=h, l=l)
        key = rearrange(key, "(b t) h l f -> b h (t l) f", b=1, t=t, h=h, l=l)
        value = rearrange(value, "(b t) h l f -> b h (t l) f", b=1, t=t, h=h, l=l)
        # print('communicating via attn', query.shape)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = rearrange(
            hidden_states, "b h (t l) f -> (b t) h l f", b=1, t=t, h=h, l=l
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                assert attn.to_add_out is not None
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        assert attn.to_out is not None
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

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
        
 
class FluxTeamworkJointAttention(FluxAttention, AdapterMixin):
    def __init__(self, base: Attention, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.adapter = nn.Parameter(torch.tensor(0.0))
        self.set_processor(TeamworkFluxJointAttnProcessor()) # type: ignore

