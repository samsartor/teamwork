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
from einops import rearrange

from .adapter import AdapterMixin, TeamworkConfig, shallowcopy_into


def _teammate_block_mask(
    attn_keep: Tensor,
    L_text: int,
    L_img: int,
    T: int,
    device: torch.device,
):
    """
    Build a flex_attention BlockMask over `[text, img_t0, img_t1, ..., img_tT-1]`.
    Text rows/cols are always unmasked; image-image edges follow `attn_keep[q_team, k_team]`.
    """
    total_len = L_text + T * L_img

    def mask_mod(b, h, q_idx, kv_idx):
        q_is_text = q_idx < L_text
        k_is_text = kv_idx < L_text
        q_team = torch.clamp((q_idx - L_text) // L_img, 0, T - 1)
        k_team = torch.clamp((kv_idx - L_text) // L_img, 0, T - 1)
        keep_edge = attn_keep[q_team, k_team]
        return q_is_text | k_is_text | keep_edge

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        device=device,
    )


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

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from diffusers.models.transformers.transformer_flux import _get_qkv_projections
        from diffusers.models.embeddings import apply_rotary_emb
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        
        assert isinstance(attn, FluxTeamworkJointAttention)
        assert encoder_hidden_states is not None

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        if attn.added_kv_proj_dim is not None:
            # double block
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
        else:
            # double single block
            encoder_query = attn.to_q(encoder_hidden_states)
            encoder_key = attn.to_k(encoder_hidden_states)
            encoder_value = attn.to_v(encoder_hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        assert attn.selection is not None
        if attn.selection.batch_matrix.shape[1] != 1:
            raise NotImplementedError('all of teamwork, batching, and joint attention')
        t, l, h, _ = query.shape
        query = rearrange(query, "(b t) l h f -> b (t l) h f", b=1, t=t, h=h, l=l)
        key   = rearrange(key,   "(b t) l h f -> b (t l) h f", b=1, t=t, h=h, l=l)
        value = rearrange(value, "(b t) l h f -> b (t l) h f", b=1, t=t, h=h, l=l)
    
        if attn.added_kv_proj_dim is not None:
            # double block, normalize separately
            query = attn.norm_q(query)
            key = attn.norm_k(key)
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
    
        query = torch.cat([encoder_query[:1, ...], query], dim=1)
        key = torch.cat([encoder_key[:1, ...], key], dim=1)
        value = torch.cat([encoder_value[:1, ...], value], dim=1)

        if attn.added_kv_proj_dim is None:
            # sngle blouck, normalize together  
            query = attn.norm_q(query)
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
            assert isinstance(query, Tensor)
            assert isinstance(key, Tensor)

        attn_keep = attn.selection.attn_keep
        if attn_keep is not None:
            assert attention_mask is None, (
                "attn_keep dropout and attention_mask cannot be combined"
            )
            L_text = encoder_hidden_states.shape[1]
            block_mask = _teammate_block_mask(
                attn_keep, L_text, l, t, query.device
            )
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            hidden_states = flex_attention(q, k, v, block_mask=block_mask)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
            )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
        )
        hidden_states = rearrange(hidden_states, "b (t l) f -> (b t) l f", b=1, t=t, l=l)
        if attn.added_kv_proj_dim is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            assert encoder_hidden_states is not None
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(hidden_states.shape[0], 0)

        return hidden_states, encoder_hidden_states # type: ignore
        
 
class FluxTeamworkJointAttention(FluxAttention, AdapterMixin):
    def __init__(self, base: Attention, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.adapter = nn.Parameter(torch.tensor(0.0))
        self.set_processor(TeamworkFluxJointAttnProcessor()) # type: ignore

