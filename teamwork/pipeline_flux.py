from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from typing import Any, Literal, Callable
from einops import rearrange, repeat
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig, Layout
from .adapter import adapt, TEAMWORK_PROFILES, Adapt, set_selection, shallowcopy_into
from .checkpoint import save_adapters
from .data import TeamBuilder, OutputImageType, Components
from .batch_linear import BatchSelection
from .sequence_linear import (
    SequenceLayout,
    SequenceSelection,
    SequenceTeamworkLinear,
    SequenceTeamworkModulation,
    from_sequence,
    to_sequence,
)
from .sequence_mask import sequence_block_mask
from .sampling import sample_timestep_u, flow_match_noising
from . import family_flux


class _FluxPipelineCommon(TeamworkPipeline, FluxPipeline):
    """What the batch and sequence FLUX.1 pipelines share: VAE, prompts, and checkpoint IO."""

    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt_embeds: None | Tensor = None
    empty_pooled_prompt_embeds: None | Tensor = None
    empty_text_ids: None | Tensor = None

    @property
    def unwrapped_transformer(self):
        model = self.transformer
        while hasattr(model, 'module'):
            model = getattr(model, 'module')
        return model

    @property
    def in_channels(self) -> int:
        return self.unwrapped_transformer.config["in_channels"] // 4

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.unwrapped_transformer, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

    def vae_encode(self, x: Tensor) -> Tensor:
        return family_flux.vae_encode(self.vae, x)

    def vae_decode(self, x: Tensor) -> Tensor:
        return family_flux.vae_decode(self.vae, x)

    @torch.no_grad
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
            self.text_encoder_2 = self.text_encoder_2.to(device)
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt,
            prompt,
            num_images_per_prompt=1,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.empty_pooled_prompt_embeds = pooled_prompt_embeds
        self.empty_text_ids = text_ids
        self.text_encoder = None
        self.tokenizer = None
        self.text_encoder_2 = None
        self.tokenizer_2 = None


class BatchFluxPipeline(_FluxPipelineCommon):
    """Batch-axis teamwork for FLUX.1 and FLUX.1 Kontext (built from either base pipeline)."""

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: FluxPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
        cls.check_layout(teamwork_config)
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
            infer_layers_from_state=False,
        )
        assert isinstance(pipeline.transformer, FluxTransformer2DModel)
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    def train_loss(
        self,
        batch: TeamBuilder,
        noise: torch.Tensor,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        # A fresh scheduler, so training timesteps don't depend on the inference schedule
        # that __call__ last set on self.scheduler.
        train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        assert isinstance(train_scheduler, FlowMatchEulerDiscreteScheduler)

        with torch.no_grad():
            timestep_u = sample_timestep_u(self.timestep_weight, batch.batch_size)
            sel = BatchSelection.from_components(batch.components_index())
            latents = batch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            )
            noising = flow_match_noising(
                train_scheduler, timestep_u, sel, latents, noise, self.timestep_weight,
                device=self.device, dtype=self.dtype,
            )
            timesteps = noising.timesteps

            # Guidance is only supported on some Flux models
            guidance = torch.full_like(timesteps, 3.5) if family_flux.has_guidance(self.unwrapped_transformer) else None

            _, _, lh, lw = latents.shape
            image_ids = family_flux.latent_image_ids(lh, lw, latents.device, latents.dtype)
            prompt_embeds, pooled_prompt_embeds, text_ids = family_flux.encode_prompt(
                self, prompt, latents.shape[0], latents.dtype,
            )

        # Concatenate extra channels if present
        model_latents = noising.noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([model_latents, extra], dim=1)

        set_selection(self.unwrapped_transformer, sel)

        model_pred = self.transformer(
            hidden_states=family_flux.patchify(model_latents),
            timestep=timesteps / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        ).sample
        model_pred = family_flux.unpatchify(model_pred, self.in_channels, lh, lw)

        return LossOutput(
            selection=sel,
            latents=latents,
            prediction=model_pred,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * noising.weighting,
            timestep_idx=noising.timestep_idx,
            type='signal',
        )

    @torch.no_grad
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str = "",
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: TeamBuilder | None = None,
        callback_on_step_end: Callable[..., dict | None] | None = None,
    ) -> dict[str, Any]:
        """
        `callback_on_step_end(pipe, i, t, {'latents': (count, c, lh, lw), 'selection': Components})`
        may return a dict with replacement `latents`.
        """
        device = self._execution_device
        if batch is None:
            batch = TeamBuilder.from_inputs(
                 self.teamwork_config.teammates,
                 device=device,
                 dtype=self.dtype,
                 images=images,
                 request=request,
                 width=width,
                 height=height,
                 attn_allow=self.teamwork_config.attn_allow,
            )

        # Initialize latents
        sel = BatchSelection.from_components(batch.components_index())
        clean_latents = batch.packed_encoded_images(
            self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
        )
        if noise is None:
            latents = torch.randn(
                clean_latents.shape,
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            latents = noise.to(device, self.dtype)

        _, _, lh, lw = latents.shape
        image_ids = family_flux.latent_image_ids(lh, lw, latents.device, latents.dtype)
        prompt_embeds, pooled_prompt_embeds, text_ids = family_flux.encode_prompt(
            self, prompt, latents.shape[0], latents.dtype,
        )

        timesteps, num_inference_steps = family_flux.inference_timesteps(
            self.scheduler, num_inference_steps, (lh // 2) * (lw // 2), device,
        )

        # Get extra channels if present
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        set_selection(self.transformer, sel)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[sel.input_subindices] = clean_latents[
                    sel.input_subindices
                ]
                timestep = t.expand(model_latents.shape[0])

                # Guidance is only supported on some Flux models
                guidance = (
                    torch.full_like(timestep, guidance_scale)
                    if family_flux.has_guidance(self.unwrapped_transformer)
                    else None
                )

                # Concatenate extra channels if present
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                noise_pred = self.transformer(
                    hidden_states=family_flux.patchify(model_latents),
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                ).sample
                noise_pred = family_flux.unpatchify(noise_pred, self.in_channels, lh, lw)

                # Scheduler step
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_kwargs = { 'latents': latents, 'selection': sel }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]


TEAMWORK_PROFILES["FLUX"] = [
    "transformer_blocks.*.norm1.linear",
    "single_transformer_blocks.*.norm.linear",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "single_transformer_blocks.*.attn.to_q",
    "single_transformer_blocks.*.attn.to_k",
    "single_transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "single_transformer_blocks.*.proj_mlp",
    "single_transformer_blocks.*.proj_out",
]


# ======================================================================================
# SEQUENCE AXIS
#
# Each batch element is one stream. Double blocks keep FLUX's two streams: the image stream
# `[img_0 | … | img_{T-1}]` (adapted per roster slot) and the text stream (base weights), which
# attend jointly over `[text | img_0 | …]`. Single blocks run on `[text | img_0 | …]`, where the
# text gets `text_teammate`'s LoRA and modulation row, or none.
# ======================================================================================

# FLUX's AdaLN modulation linears, which act on `temb` (no token axis).
FLUX_MODULATION_SUFFIXES = (".norm1.linear", ".norm1_context.linear", ".norm.linear")


def sequence_flux_profile(profile: list[Adapt]) -> list[Adapt]:
    """
    Map a FLUX layer list to sequence adapters: modulation linears get
    `SequenceTeamworkModulation`, every other glob `SequenceTeamworkLinear`. Explicit
    `(glob, cls)` entries pass through. The globs (so the checkpoint keys) are unchanged.
    """
    resolved: list[Adapt] = []
    for entry in profile:
        if isinstance(entry, str):
            cls = SequenceTeamworkModulation if entry.endswith(FLUX_MODULATION_SUFFIXES) else SequenceTeamworkLinear
            resolved.append((entry, cls))
        else:
            resolved.append(entry)
    return resolved


# Automatic dynamic shapes: static compiles per resolution/batch would hit dynamo's recompile
# limit and silently fall back to the unfused flex_attention, which materializes S² scores.
_compiled_flex_attention = torch.compile(flex_attention)


def _attention(
    query: Tensor,  # (B, S, H, Dh), normed
    key: Tensor,
    value: Tensor,
    rotary: tuple[Tensor, Tensor],  # (cos, sin), each (S, Dh)
    block_mask: BlockMask,
) -> Tensor:
    """RoPE + masked flex attention over the whole stream: `(B, S, H·Dh)`."""
    assert query.shape == key.shape == value.shape, (query.shape, key.shape, value.shape)
    assert rotary[0].shape[0] == query.shape[1], f"rotary for {rotary[0].shape[0]} tokens, stream has {query.shape[1]}"
    # diffusers annotates `freqs_cis` as `tuple[Tensor]` and the result as a tuple.
    query = apply_rotary_emb(query, rotary, sequence_dim=1)  # type: ignore
    key = apply_rotary_emb(key, rotary, sequence_dim=1)  # type: ignore
    assert isinstance(query, Tensor) and isinstance(key, Tensor)
    kernel_options = None
    if query.device.type == "cuda":
        # The autotuner's default configs need >= 128 KiB of shared memory at head_dim=128.
        props = torch.cuda.get_device_properties(query.device)
        if props.shared_memory_per_block_optin < 128 * 1024:
            kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    out = _compiled_flex_attention(
        rearrange(query, "b s h d -> b h s d"),
        rearrange(key, "b s h d -> b h s d"),
        rearrange(value, "b s h d -> b h s d"),
        block_mask=block_mask,
        kernel_options=kernel_options,
    )
    assert isinstance(out, Tensor)
    return rearrange(out, "b h s d -> b s (h d)").to(query.dtype)


def _heads(x: Tensor, attn: FluxAttention) -> Tensor:
    return rearrange(x, "b s (h d) -> b s h d", d=attn.head_dim)


def _per_owner(
    x: Tensor,  # (B, S, D)
    layout: SequenceLayout,
    text_rows: tuple[Tensor, ...],  # each (B, D); unused when the layout has no text
    slot_rows: tuple[Tensor, ...],  # each (B, T, D)
    fn: Callable[..., Tensor],
) -> Tensor:
    """
    `fn(tokens, *rows)` with every token paired with its owner's rows, on a `[text | img_0 | …]`
    layout: the text takes `text_rows`, image range `t` takes `slot_rows[:][:, t]`.
    """
    S_text, T = layout.text_length, layout.num_teammates
    assert layout.uniform_start == S_text, f"not a [text | images] layout: {layout}"
    assert x.ndim == 3 and x.shape[1] == layout.seq_len, f"{tuple(x.shape)} for seq_len {layout.seq_len}"
    images = rearrange(x[:, S_text:], "b (t l) d -> b t l d", t=T)
    images = fn(images, *(rearrange(r, "b t d -> b t 1 d") for r in slot_rows))
    images = rearrange(images, "b t l d -> b (t l) d")
    if S_text == 0:
        return images
    text = fn(x[:, :S_text], *(rearrange(r, "b d -> b 1 d") for r in text_rows))
    return torch.cat([text, images], dim=1)


def _modulation_rows(linear: Any, text_temb: Tensor, teammate_temb: Tensor) -> tuple[Tensor, Tensor]:
    """`(text (B, out), teammates (B, T, out))` of an AdaLN linear on `silu(temb)`, adapted or not."""
    text_in, teammate_in = F.silu(text_temb), F.silu(teammate_temb)
    if isinstance(linear, SequenceTeamworkModulation):
        rows = linear.rows(text_in, teammate_in)
        return rows.text, rows.teammates
    assert isinstance(linear, torch.nn.Linear) and not isinstance(linear, SequenceTeamworkLinear), type(linear)
    return linear(text_in), linear(teammate_in)


def _scale_shift(x: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
    return x * (1 + scale) + shift


def _gated(x: Tensor, gate: Tensor) -> Tensor:
    return gate * x


def sequence_double_block(
    block: FluxTransformerBlock,
    image: Tensor,  # (B, T·L, D): [img_0 | … | img_{T-1}]
    text: Tensor,  # (B, S_text, D)
    text_temb: Tensor,  # (B, D)
    teammate_temb: Tensor,  # (B, T, D)
    rotary: tuple[Tensor, Tensor],  # over [text | images]
    block_mask: BlockMask,  # over [text | images]
    image_layout: SequenceLayout,  # [img_0 | …], no text
) -> tuple[Tensor, Tensor]:
    """`FluxTransformerBlock.forward` with modulation row `t` on image range `t`. Returns `(text, image)`."""
    attn = block.attn
    assert attn.added_kv_proj_dim is not None and not getattr(attn, "fused_projections", False)
    assert image_layout.text_length == 0
    _, mod = _modulation_rows(block.norm1.linear, text_temb, teammate_temb)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
    norm_image = _per_owner(block.norm1.norm(image), image_layout, (), (scale_msa, shift_msa), _scale_shift)
    norm_text, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = block.norm1_context(text, emb=text_temb)

    query = attn.norm_q(_heads(attn.to_q(norm_image), attn))
    key = attn.norm_k(_heads(attn.to_k(norm_image), attn))
    value = _heads(attn.to_v(norm_image), attn)
    text_query = attn.norm_added_q(_heads(attn.add_q_proj(norm_text), attn))
    text_key = attn.norm_added_k(_heads(attn.add_k_proj(norm_text), attn))
    text_value = _heads(attn.add_v_proj(norm_text), attn)
    out = _attention(
        torch.cat([text_query, query], dim=1),
        torch.cat([text_key, key], dim=1),
        torch.cat([text_value, value], dim=1),
        rotary,
        block_mask,
    )
    S_text = text.shape[1]
    text_out = attn.to_add_out(out[:, :S_text].contiguous())
    image_out = attn.to_out[1](attn.to_out[0](out[:, S_text:].contiguous()))

    image = image + _per_owner(image_out, image_layout, (), (gate_msa,), _gated)
    norm_image = _per_owner(block.norm2(image), image_layout, (), (scale_mlp, shift_mlp), _scale_shift)
    image = image + _per_owner(block.ff(norm_image), image_layout, (), (gate_mlp,), _gated)

    text = text + c_gate_msa.unsqueeze(1) * text_out
    norm_text = block.norm2_context(text) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    text = text + c_gate_mlp.unsqueeze(1) * block.ff_context(norm_text)
    if text.dtype == torch.float16:
        text = text.clip(-65504, 65504)
    return text, image


def sequence_single_block(
    block: FluxSingleTransformerBlock,
    stream: Tensor,  # (B, S, D): [text | img_0 | … | img_{T-1}]
    text_temb: Tensor,  # (B, D)
    teammate_temb: Tensor,  # (B, T, D)
    rotary: tuple[Tensor, Tensor],
    block_mask: BlockMask,
    layout: SequenceLayout,
) -> Tensor:
    """`FluxSingleTransformerBlock.forward` with the text row on the text and row `t` on range `t`."""
    attn = block.attn
    assert attn.added_kv_proj_dim is None and not getattr(attn, "fused_projections", False)
    text_mod, mod = _modulation_rows(block.norm.linear, text_temb, teammate_temb)
    shift, scale, gate = mod.chunk(3, dim=-1)
    text_shift, text_scale, text_gate = text_mod.chunk(3, dim=-1)
    norm = _per_owner(block.norm.norm(stream), layout, (text_scale, text_shift), (scale, shift), _scale_shift)

    mlp = block.act_mlp(block.proj_mlp(norm))
    out = _attention(
        attn.norm_q(_heads(attn.to_q(norm), attn)),
        attn.norm_k(_heads(attn.to_k(norm), attn)),
        _heads(attn.to_v(norm), attn),
        rotary,
        block_mask,
    )
    hidden = block.proj_out(torch.cat([out, mlp], dim=2))
    stream = stream + _per_owner(hidden, layout, (text_gate,), (gate,), _gated)
    if stream.dtype == torch.float16:
        stream = stream.clip(-65504, 65504)
    return stream


class SequenceFluxTransformer(FluxTransformer2DModel):
    """
    An adapted `FluxTransformer2DModel` (shallow copy) whose forward runs sequence-axis teamwork
    on component-layout tokens. Called without `components`, it is the base model's forward with
    every adapter disabled.
    """

    teamwork_config: TeamworkConfig

    @staticmethod
    def from_adapted(adapted: FluxTransformer2DModel, cfg: TeamworkConfig) -> "SequenceFluxTransformer":
        assert type(adapted) is FluxTransformer2DModel, type(adapted)
        model = object.__new__(SequenceFluxTransformer)
        shallowcopy_into(model, adapted)
        model.teamwork_config = cfg
        return model

    def layouts(self, text_length: int, image_length: int) -> tuple[SequenceLayout, SequenceLayout]:
        """`([text | img_0 | …], [img_0 | …])`: the single-block and double-block image layouts."""
        outputs = tuple(t.endswith(".out") for t in self.teamwork_config.teammates)
        return (
            SequenceLayout.joint(text_length, image_length, outputs),
            SequenceLayout.joint(0, image_length, outputs),
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,  # (count, L, C): patchified component tokens
        encoder_hidden_states: Tensor,  # (B, S_text, J), one prompt per batch element
        pooled_projections: Tensor,  # (B, P)
        timestep: Tensor,  # (B,) in [0, 1], one per batch element (all its components share it)
        img_ids: Tensor,  # (L, 3): RoPE ids of one image; slot t is offset by teammate_image_ids[t]
        txt_ids: Tensor,  # (S_text, 3)
        guidance: Tensor | None = None,  # (B,)
        components: Components | None = None,
        attn_keep: Tensor | None = None,  # (T, T) bool
        **base_kwargs: Any,
    ) -> Transformer2DModelOutput:
        if components is None:
            assert attn_keep is None, "attn_keep without components"
            set_selection(self, None)
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,  # type: ignore[arg-type]
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,  # type: ignore[arg-type]
                **base_kwargs,
            )  # type: ignore[return-value]
        assert not base_kwargs, f"unsupported in sequence teamwork: {sorted(base_kwargs)}"

        cfg = self.teamwork_config
        T = len(cfg.teammates)
        count, L, _ = hidden_states.shape
        B, S_text, _ = encoder_hidden_states.shape
        assert components.num_teammates == T and components.batch_size == B, (components.num_teammates, components.batch_size, T, B)
        assert components.teammate_indices.shape == (count,), (components.teammate_indices.shape, count)
        assert pooled_projections.ndim == 2 and pooled_projections.shape[0] == B, pooled_projections.shape
        assert timestep.shape == (B,), f"timestep {tuple(timestep.shape)}, expected ({B},)"
        assert guidance is None or guidance.shape == (B,), f"guidance {None if guidance is None else tuple(guidance.shape)}"
        assert img_ids.shape == (L, 3) and txt_ids.shape == (S_text, 3), (img_ids.shape, txt_ids.shape)
        assert cfg.teammate_image_ids is not None

        layout, image_layout = self.layouts(S_text, L)
        text_teammate = None if cfg.text_teammate is None else cfg.teammates.index(cfg.text_teammate)
        selection = SequenceSelection.from_components(
            components, layout, attn_keep=attn_keep, text_teammate=text_teammate,
        )
        # The double blocks' image stream has no text: their text runs on separate, unadapted weights.
        set_selection(self.transformer_blocks, replace(selection, layout=image_layout, text_teammate=None))
        set_selection(self.single_transformer_blocks, selection)
        block_mask = sequence_block_mask(selection)

        tokens = self.x_embedder(hidden_states)  # (count, L, D)
        timestep = timestep.to(tokens.dtype) * 1000
        if guidance is None:
            temb = self.time_text_embed(timestep, pooled_projections)
        else:
            temb = self.time_text_embed(timestep, guidance.to(tokens.dtype) * 1000, pooled_projections)
        teammate_temb = repeat(temb, "b d -> b t d", t=T)
        text = self.context_embedder(encoder_hidden_states)
        image = to_sequence(tokens, text[:, :0], components, image_layout)  # (B, T·L, D)

        ids = torch.cat([txt_ids, family_flux.offset_image_ids(img_ids, cfg.teammate_image_ids)], dim=0)
        rotary = self.pos_embed(ids)

        checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for block in self.transformer_blocks:
            args = (block, image, text, temb, teammate_temb, rotary, block_mask, image_layout)
            if checkpointing:
                text, image = self._gradient_checkpointing_func(sequence_double_block, *args)
            else:
                text, image = sequence_double_block(*args)

        stream = torch.cat([text, image], dim=1)
        for block in self.single_transformer_blocks:
            args = (block, stream, temb, teammate_temb, rotary, block_mask, layout)
            if checkpointing:
                stream = self._gradient_checkpointing_func(sequence_single_block, *args)
            else:
                stream = sequence_single_block(*args)

        # Every slot shares its batch element's temb, so the final norm can run on the whole stream.
        image = self.proj_out(self.norm_out(stream[:, S_text:], temb))
        output, _ = from_sequence(image, components, image_layout)
        return Transformer2DModelOutput(sample=output)


def element_timesteps(timesteps: Tensor, components: Components) -> Tensor:
    """`(count,)` per-component timesteps -> `(B,)`, asserting each element's components agree."""
    per_element = timesteps.new_full((components.batch_size,), float("nan")).index_put(
        (components.batch_indices,), timesteps,
    )
    assert torch.equal(per_element[components.batch_indices], timesteps), "components of one element disagree on t"
    return per_element


class SequenceFluxPipeline(_FluxPipelineCommon):
    """
    Sequence-axis teamwork for FLUX.1 and FLUX.1 Kontext (built from either base pipeline).
    With roster `(edited.out, source.in)`, `teammate_image_ids=(0, 1)`, and zero LoRA, this is
    base Kontext editing: `[text | target | context]` with the context at RoPE index 1.
    """

    layout: Layout = "sequence"

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: FluxPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
        cls.check_layout(teamwork_config)
        if override_profile is None:
            assert teamwork_config.profile is not None, "config has no teamwork profile"
            override_profile = TEAMWORK_PROFILES[teamwork_config.profile]
        pipeline = cls(**base_pipeline.components)  # type: ignore
        pipeline.teamwork_config = teamwork_config
        adapted = adapt(
            pipeline.transformer,
            cfg=teamwork_config,
            device=pipeline.device,
            dtype=torch.bfloat16,
            requires_grad=training,
            override_profile=sequence_flux_profile(override_profile),
            state=state,
            infer_layers_from_state=False,
        )
        assert isinstance(adapted, FluxTransformer2DModel)
        pipeline.transformer = SequenceFluxTransformer.from_adapted(adapted, teamwork_config)
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    def _transformer_inputs(self, prompt: str | list[str], batch_size: int, lh: int, lw: int, dtype: torch.dtype, device: torch.device):
        """`(prompt_embeds (B, S, J), pooled (B, P), text_ids (S, 3), img_ids (L, 3))`."""
        if isinstance(prompt, list):
            assert len(prompt) == batch_size, f"{len(prompt)} prompts for {batch_size} batch elements"
        prompt_embeds, pooled_prompt_embeds, text_ids = family_flux.encode_prompt(self, prompt, batch_size, dtype)
        img_ids = family_flux.latent_image_ids(lh, lw, device, dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids, img_ids

    def train_loss(
        self,
        batch: TeamBuilder,
        noise: torch.Tensor,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        """Like `BatchFluxPipeline.train_loss`, except a list `prompt` has one entry per batch element."""
        train_scheduler = FlowMatchEulerDiscreteScheduler.from_config(self.scheduler.config)
        assert isinstance(train_scheduler, FlowMatchEulerDiscreteScheduler)

        with torch.no_grad():
            timestep_u = sample_timestep_u(self.timestep_weight, batch.batch_size)
            components = batch.components_index()
            attn_keep = batch.attn_keep()
            latents = batch.packed_encoded_images(
                self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
            )
            noising = flow_match_noising(
                train_scheduler, timestep_u, components, latents, noise, self.timestep_weight,
                device=self.device, dtype=self.dtype,
            )
            timesteps = element_timesteps(noising.timesteps, components)
            guidance = torch.full_like(timesteps, 3.5) if family_flux.has_guidance(self.unwrapped_transformer) else None
            _, _, lh, lw = latents.shape
            prompt_embeds, pooled_prompt_embeds, text_ids, img_ids = self._transformer_inputs(
                prompt, components.batch_size, lh, lw, latents.dtype, latents.device,
            )

        model_latents = noising.noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([model_latents, extra], dim=1)

        model_pred = self.transformer(
            hidden_states=family_flux.patchify(model_latents),
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            timestep=timesteps / 1000,
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
            components=components,
            attn_keep=attn_keep,
        ).sample
        model_pred = family_flux.unpatchify(model_pred, self.in_channels, lh, lw)

        return LossOutput(
            selection=components,
            latents=latents,
            prediction=model_pred,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * noising.weighting,
            timestep_idx=noising.timestep_idx,
            type='signal',
        )

    @torch.no_grad
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str | list[str] = "",
        num_inference_steps: int = 40,
        guidance_scale: float = 3.5,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: TeamBuilder | None = None,
        callback_on_step_end: Callable[..., dict | None] | None = None,
    ) -> dict[str, Any]:
        """
        `callback_on_step_end(pipe, i, t, {'latents': (count, c, lh, lw), 'selection': Components})`
        may return a dict with replacement `latents`. A list `prompt` has one entry per batch element.
        """
        device = self._execution_device
        if batch is None:
            batch = TeamBuilder.from_inputs(
                 self.teamwork_config.teammates,
                 device=device,
                 dtype=self.dtype,
                 images=images,
                 request=request,
                 width=width,
                 height=height,
                 attn_allow=self.teamwork_config.attn_allow,
            )

        components = batch.components_index()
        attn_keep = batch.attn_keep()
        clean_latents = batch.packed_encoded_images(
            self.vae_encode, self.in_channels, 1 / self.vae_scale_factor
        )
        if noise is None:
            latents = torch.randn(
                clean_latents.shape,
                device=device,
                dtype=self.dtype,
                generator=generator,
            )
        else:
            latents = noise.to(device, self.dtype)

        _, _, lh, lw = latents.shape
        prompt_embeds, pooled_prompt_embeds, text_ids, img_ids = self._transformer_inputs(
            prompt, components.batch_size, lh, lw, latents.dtype, latents.device,
        )
        timesteps, num_inference_steps = family_flux.inference_timesteps(
            self.scheduler, num_inference_steps, (lh // 2) * (lw // 2), device,
        )
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[components.input_subindices] = clean_latents[components.input_subindices]
                timestep = t.expand(components.batch_size)
                guidance = (
                    torch.full_like(timestep, guidance_scale)
                    if family_flux.has_guidance(self.unwrapped_transformer)
                    else None
                )
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                noise_pred = self.transformer(
                    hidden_states=family_flux.patchify(model_latents),
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    timestep=timestep / 1000,
                    img_ids=img_ids,
                    txt_ids=text_ids,
                    guidance=guidance,
                    components=components,
                    attn_keep=attn_keep,
                ).sample
                noise_pred = family_flux.unpatchify(noise_pred, self.in_channels, lh, lw)

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_kwargs = { 'latents': latents, 'selection': components }
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs  # type: ignore[return-value]
        else:
            return outputs[0]
