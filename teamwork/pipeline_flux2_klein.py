import torch
from torch import Tensor
from typing import Any, Literal, Callable
from einops import rearrange, repeat
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2Modulation,
    Flux2ParallelSelfAttention,
    Flux2SingleTransformerBlock,
    Flux2Transformer2DModel,
    Flux2Transformer2DModelOutput,
    Flux2TransformerBlock,
)
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig, Layout
from .adapter import adapt, adapter_modules, TEAMWORK_PROFILES, Adapt, set_selection, shallowcopy_into
from .checkpoint import save_adapters
from .data import Components, TeamBuilder, OutputImageType
from .batch_linear import BatchSelection
from .sampling import sample_timestep_u, flow_match_noising, initial_noise
from .sequence_linear import (
    ModulationRows,
    SequenceLayout,
    SequenceSelection,
    SequenceTeamworkLinear,
    SequenceTeamworkModulation,
    from_sequence,
    to_sequence,
)
from .sequence_mask import sequence_block_mask
from . import family_flux2


class _Flux2KleinTeamwork(TeamworkPipeline, Flux2KleinPipeline):
    """Layout-independent parts of the FLUX.2 Klein teamwork pipelines."""

    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt_embeds: None | Tensor = None
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

    def vae_encode(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        return family_flux2.vae_encode(self.vae, x, self.teamwork_config.version, generator)

    def vae_decode(self, x: Tensor) -> Tensor:
        return family_flux2.vae_decode(self.vae, x, self.teamwork_config.version)

    @torch.no_grad
    def empty_prompts(self, device: torch.device | None = None):
        prompt = ""
        if device is not None:
            self.text_encoder = self.text_encoder.to(device)
        # Flux2 Klein has a single text encoder (Qwen3) and no pooled embeddings.
        (prompt_embeds, text_ids) = self.encode_prompt(
            prompt,
            num_images_per_prompt=1,
        )
        self.empty_prompt_embeds = prompt_embeds
        self.empty_text_ids = text_ids
        self.text_encoder = None
        self.tokenizer = None


class BatchFlux2KleinPipeline(_Flux2KleinTeamwork):
    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: Flux2KleinPipeline,
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
        assert isinstance(pipeline.transformer, Flux2Transformer2DModel)
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

            _, _, lh, lw = latents.shape
            image_ids = family_flux2.latent_image_ids(lh, lw, latents.device)
            prompt_embeds, text_ids = family_flux2.encode_prompt(self, prompt, latents.shape[0], latents.dtype)

        # Concatenate extra channels if present
        model_latents = noising.noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([model_latents, extra], dim=1)

        set_selection(self.unwrapped_transformer, sel)

        model_pred = self.transformer(
            hidden_states=family_flux2.patchify(model_latents),
            timestep=noising.timesteps / 1000,
            guidance=None,  # Flux2 Klein is guidance-distilled (guidance_embeds=False)
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=image_ids,
        ).sample
        model_pred = family_flux2.unpatchify(model_pred, self.in_channels, lh, lw)

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
    ) -> Any:  # one dict per batch element if `images` is a list, else the single dict
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
            lambda x: self.vae_encode(x, generator), self.in_channels, 1 / self.vae_scale_factor
        )
        if noise is None:
            latents = initial_noise(clean_latents.shape, generator, device, self.dtype)
        else:
            latents = noise.to(device, self.dtype)

        _, _, lh, lw = latents.shape
        image_ids = family_flux2.latent_image_ids(lh, lw, latents.device)
        prompt_embeds, text_ids = family_flux2.encode_prompt(self, prompt, latents.shape[0], latents.dtype)

        timesteps, num_inference_steps = family_flux2.inference_timesteps(
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

                # Concatenate extra channels if present
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                noise_pred = self.transformer(
                    hidden_states=family_flux2.patchify(model_latents),
                    timestep=timestep / 1000,
                    guidance=None,  # Flux2 Klein is guidance-distilled (guidance_embeds=False)
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=image_ids,
                ).sample
                noise_pred = family_flux2.unpatchify(noise_pred, self.in_channels, lh, lw)

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


# LoRA-only profile. Flux2 differs structurally from Flux1: the adaLN modulation
# linears are hoisted to the model level (one per stream, shared across blocks)
# instead of living in per-block norm1.linear, the single blocks fuse QKV+MLP into
# one projection (to_qkv_mlp_proj / to_out), and the feedforward uses linear_in/out.
# As in the Flux1 profile, we adapt the image-stream pathway and the shared single
# stream, leaving the text-stream (add_*_proj, ff_context, *_modulation_txt) at base.
# The sequence layout adapts the same layers (so checkpoint keys match), with
# `SequenceTeamworkModulation` for the `*_modulation_*` linears; see `sequence_profile`.
TEAMWORK_PROFILES["FLUX2"] = [
    "double_stream_modulation_img.linear",
    "single_stream_modulation.linear",
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.ff.linear_in",
    "transformer_blocks.*.ff.linear_out",
    "single_transformer_blocks.*.attn.to_qkv_mlp_proj",
    "single_transformer_blocks.*.attn.to_out",
]


# ======================================================================================
# SEQUENCE LAYOUT
#
# Every batch element is one token stream. The double-stream blocks see the image stream
# `[img_0 | … | img_{T-1}]` next to the separate text stream, the single-stream blocks see
# `[text | img_0 | … | img_{T-1}]`, and attention always runs over `[text | img_0 | …]`.
# With roster `(edited.out, source.in)`, RoPE T-coords `(0, 10)`, and zero LoRA this is
# exactly base Klein editing of `[text | target | reference]`.
# ======================================================================================


def sequence_profile(profile: list[Adapt]) -> list[Adapt]:
    """A layer-set profile (globs) as sequence adapters: `*_modulation*` linears act on `temb`."""
    return [
        (adapt, SequenceTeamworkModulation if "modulation" in adapt else SequenceTeamworkLinear)
        if isinstance(adapt, str)
        else adapt
        for adapt in profile
    ]


# Which stream each adapted module path sees: the image stream `[img_0 | …]` (double blocks,
# their image modulation, and the image-only embed/projection) or the full stream (single blocks).
_IMAGE_STREAM_PREFIXES = (
    "transformer_blocks.",
    "double_stream_modulation_img.",
    "double_stream_modulation_txt.",
    "x_embedder",
    "proj_out",
)
_FULL_STREAM_PREFIXES = ("single_transformer_blocks.", "single_stream_modulation.")
# The double blocks' text-stream linears have no image ranges, so the sequence adapters can't run there.
_TEXT_STREAM_PARTS = (".attn.add_q_proj", ".attn.add_k_proj", ".attn.add_v_proj", ".attn.to_add_out", ".ff_context.")

# Default (automatic) dynamic shapes: a static compile per resolution/batch/roster size would hit
# dynamo's recompile limit and silently fall back to the unfused, S²-memory implementation.
_COMPILED_FLEX = torch.compile(flex_attention)


def _flex_attention(query: Tensor, key: Tensor, value: Tensor, block_mask: BlockMask) -> Tensor:
    """`(B, S, H, F)` q/k/v (RoPE applied) -> `(B, S, H·F)` masked attention, in `query`'s dtype."""
    # On GPUs with <128 KiB dynamic smem per block (A40/A6000/4090/L40), the autotuner's default
    # flex_attention configs don't fit at head_dim=128 once mask_mod captures tensors. Force a
    # smaller tile and a single pipeline stage there.
    props = torch.cuda.get_device_properties(query.device)
    kernel_options = None
    if props.shared_memory_per_block_optin < 128 * 1024:
        kernel_options = {"BLOCK_M": 64, "BLOCK_N": 64, "num_stages": 1}
    # Contiguous copies: views guard on their base's strides, which differ between the double
    # and single blocks and would recompile flex_attention every time.
    q, k, v = (rearrange(x, "b s h f -> b h s f").contiguous() for x in (query, key, value))
    out = _COMPILED_FLEX(q, k, v, block_mask=block_mask, kernel_options=kernel_options)
    assert isinstance(out, Tensor)
    return rearrange(out, "b h s f -> b s (h f)").to(query.dtype)


def _blocks(stream: Tensor, T: int) -> Tensor:
    """Image stream `(B, T·L, D)` -> `(B, T, L, D)`."""
    return rearrange(stream, "b (t l) d -> b t l d", t=T)


def _stream(blocks: Tensor) -> Tensor:
    return rearrange(blocks, "b t l d -> b (t l) d")


def _image_mods(mod: Tensor, sets: int) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
    """Per-teammate modulation `(B, T, 3·sets·D)` -> shift/scale/gate sets, each `(B, T, 1, D)`."""
    return tuple(
        tuple(rearrange(p, "b t d -> b t 1 d") for p in params)  # type: ignore[misc]
        for params in Flux2Modulation.split(mod, sets)
    )


def _modulation_rows(modulation: Flux2Modulation, temb: Tensor, T: int) -> ModulationRows:
    """A shared Flux2 modulation as a text row and one row per roster slot."""
    x = modulation.act_fn(temb)
    if isinstance(modulation.linear, SequenceTeamworkModulation):
        return modulation.linear.rows(x, repeat(x, "b i -> b t i", t=T))
    base = modulation.linear(x)
    return ModulationRows(text=base, teammates=repeat(base, "b o -> b t o", t=T))


def _double_attention(
    attn: Flux2Attention,
    image: Tensor,  # (B, T·L, D), normed and modulated
    text: Tensor,  # (B, L_text, D)
    rotary: tuple[Tensor, Tensor],
    block_mask: BlockMask,
) -> tuple[Tensor, Tensor]:
    """Flux2AttnProcessor over `[text | img_0 | …]` with the teamwork block mask."""
    assert not attn.fused_projections, "fused QKV projections are not supported"
    query = attn.norm_q(attn.to_q(image).unflatten(-1, (-1, attn.head_dim)))
    key = attn.norm_k(attn.to_k(image).unflatten(-1, (-1, attn.head_dim)))
    value = attn.to_v(image).unflatten(-1, (-1, attn.head_dim))
    text_query = attn.norm_added_q(attn.add_q_proj(text).unflatten(-1, (-1, attn.head_dim)))
    text_key = attn.norm_added_k(attn.add_k_proj(text).unflatten(-1, (-1, attn.head_dim)))
    text_value = attn.add_v_proj(text).unflatten(-1, (-1, attn.head_dim))

    query = apply_rotary_emb(torch.cat([text_query, query], dim=1), rotary, sequence_dim=1)  # type: ignore[arg-type]  # diffusers annotates a 1-tuple
    key = apply_rotary_emb(torch.cat([text_key, key], dim=1), rotary, sequence_dim=1)  # type: ignore[arg-type]
    value = torch.cat([text_value, value], dim=1)
    assert isinstance(query, Tensor) and isinstance(key, Tensor)
    out = _flex_attention(query, key, value, block_mask)

    L_text = text.shape[1]
    text_out = attn.to_add_out(out[:, :L_text])
    image_out = attn.to_out[1](attn.to_out[0](out[:, L_text:]))
    return image_out, text_out


def _double_block_forward(
    block: Flux2TransformerBlock,
    image: Tensor,  # (B, T·L, D)
    text: Tensor,  # (B, L_text, D)
    image_mod: Tensor,  # (B, T, 6·D): one modulation row per roster slot
    text_mod: Tensor,  # (B, 6·D)
    rotary: tuple[Tensor, Tensor],
    block_mask: BlockMask,
) -> tuple[Tensor, Tensor]:
    """`Flux2TransformerBlock.forward` with slot `t`'s modulation on image range `t`."""
    T = image_mod.shape[1]
    (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = _image_mods(image_mod, 2)
    (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = Flux2Modulation.split(text_mod, 2)

    norm_image = _stream((1 + scale_msa) * _blocks(block.norm1(image), T) + shift_msa)
    norm_text = (1 + c_scale_msa) * block.norm1_context(text) + c_shift_msa
    attn_image, attn_text = _double_attention(block.attn, norm_image, norm_text, rotary, block_mask)

    image = image + _stream(gate_msa * _blocks(attn_image, T))
    norm_image = _stream(_blocks(block.norm2(image), T) * (1 + scale_mlp) + shift_mlp)
    image = image + _stream(gate_mlp * _blocks(block.ff(norm_image), T))

    text = text + c_gate_msa * attn_text
    norm_text = block.norm2_context(text) * (1 + c_scale_mlp) + c_shift_mlp
    text = text + c_gate_mlp * block.ff_context(norm_text)
    if text.dtype == torch.float16:
        text = text.clip(-65504, 65504)
    return text, image


def _single_attention(
    attn: Flux2ParallelSelfAttention,
    hidden: Tensor,  # (B, S, D), normed and modulated
    rotary: tuple[Tensor, Tensor],
    block_mask: BlockMask,
) -> Tensor:
    """Flux2ParallelSelfAttnProcessor with the teamwork block mask."""
    projected = attn.to_qkv_mlp_proj(hidden)
    qkv, mlp = torch.split(projected, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)
    query, key, value = (x.unflatten(-1, (-1, attn.head_dim)) for x in qkv.chunk(3, dim=-1))
    query = apply_rotary_emb(attn.norm_q(query), rotary, sequence_dim=1)  # type: ignore[arg-type]  # diffusers annotates a 1-tuple
    key = apply_rotary_emb(attn.norm_k(key), rotary, sequence_dim=1)  # type: ignore[arg-type]
    assert isinstance(query, Tensor) and isinstance(key, Tensor)
    out = _flex_attention(query, key, value, block_mask)
    return attn.to_out(torch.cat([out, attn.mlp_act_fn(mlp)], dim=-1))


def _single_block_forward(
    block: Flux2SingleTransformerBlock,
    hidden: Tensor,  # (B, L_text + T·L, D)
    text_length: int,
    text_mod: Tensor,  # (B, 3·D): the text row
    image_mod: Tensor,  # (B, T, 3·D): one row per roster slot
    rotary: tuple[Tensor, Tensor],
    block_mask: BlockMask,
) -> Tensor:
    """`Flux2SingleTransformerBlock.forward` with slot `t`'s modulation on image range `t`."""
    T = image_mod.shape[1]
    ((text_shift, text_scale, text_gate),) = Flux2Modulation.split(text_mod, 1)
    ((image_shift, image_scale, image_gate),) = _image_mods(image_mod, 1)

    normed = block.norm(hidden)
    normed = torch.cat([
        (1 + text_scale) * normed[:, :text_length] + text_shift,
        _stream((1 + image_scale) * _blocks(normed[:, text_length:], T) + image_shift),
    ], dim=1)
    out = _single_attention(block.attn, normed, rotary, block_mask)
    hidden = torch.cat([
        hidden[:, :text_length] + text_gate * out[:, :text_length],
        hidden[:, text_length:] + _stream(image_gate * _blocks(out[:, text_length:], T)),
    ], dim=1)
    if hidden.dtype == torch.float16:
        hidden = hidden.clip(-65504, 65504)
    return hidden


class SequenceFlux2Transformer(Flux2Transformer2DModel):
    """
    A FLUX.2 transformer adapted with sequence-axis teamwork (see `SequenceFlux2KleinPipeline`).

    `forward(..., components=Components, attn_keep=...)` takes component-layout image tokens
    `(count, L, C)`, scatters each batch element's components into one stream, and returns
    `(count, L, C_out)`. `img_ids` are the stream's `(T·L, 4)` RoPE ids (slot order), `timestep`
    is one value per batch element. Without `components` it is the base model: every adapter
    is deactivated and the arguments are those of `Flux2Transformer2DModel.forward`.
    """

    teamwork_outputs: tuple[bool, ...]  # one per roster slot
    text_teammate: int | None  # roster slot whose adapter also applies to the text tokens
    image_stream_adapters: list[SequenceTeamworkLinear | SequenceTeamworkModulation]
    full_stream_adapters: list[SequenceTeamworkLinear | SequenceTeamworkModulation]

    @classmethod
    def from_adapted(cls, adapted: Flux2Transformer2DModel, cfg: TeamworkConfig) -> "SequenceFlux2Transformer":
        """Wrap an `adapt()`ed transformer (sharing every module and parameter)."""
        model = object.__new__(cls)
        shallowcopy_into(model, adapted)
        model.teamwork_outputs = tuple(t.endswith(".out") for t in cfg.teammates)
        model.text_teammate = None if cfg.text_teammate is None else cfg.teammates.index(cfg.text_teammate)

        modulation_linears = {f"{n}.linear" for n, m in model.named_modules() if isinstance(m, Flux2Modulation)}
        image: list[SequenceTeamworkLinear | SequenceTeamworkModulation] = []
        full: list[SequenceTeamworkLinear | SequenceTeamworkModulation] = []
        for name, module in adapter_modules(model).items():
            assert isinstance(module, (SequenceTeamworkLinear, SequenceTeamworkModulation)), (
                f"{name}: {type(module).__name__} is not a sequence adapter"
            )
            assert isinstance(module, SequenceTeamworkModulation) == (name in modulation_linears), (
                f"{name}: {type(module).__name__}, but SequenceTeamworkModulation belongs exactly on the "
                f"Flux2Modulation linears {sorted(modulation_linears)}"
            )
            if any(part in name for part in _TEXT_STREAM_PARTS):
                raise NotImplementedError(f"{name}: the double blocks' text-stream linears can't be adapted")
            if name.startswith(_IMAGE_STREAM_PREFIXES):
                image.append(module)
            elif name.startswith(_FULL_STREAM_PREFIXES):
                full.append(module)
            else:
                raise NotImplementedError(f"{name}: no sequence-layout support for adapting this layer")
        model.image_stream_adapters = image
        model.full_stream_adapters = full
        return model

    def forward(  # type: ignore[override]
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor | None = None,
        timestep: Tensor | None = None,
        img_ids: Tensor | None = None,
        txt_ids: Tensor | None = None,
        guidance: Tensor | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        components: Components | None = None,
        attn_keep: Tensor | None = None,
        **kwargs,
    ) -> Any:
        if components is None:
            assert attn_keep is None, "attn_keep needs components"
            set_selection(self, None)
            return super().forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,  # type: ignore[arg-type]
                timestep=timestep,  # type: ignore[arg-type]
                img_ids=img_ids,  # type: ignore[arg-type]
                txt_ids=txt_ids,  # type: ignore[arg-type]
                guidance=guidance,  # type: ignore[arg-type]
                joint_attention_kwargs=joint_attention_kwargs,
                return_dict=return_dict,
                **kwargs,
            )
        assert not kwargs, f"not supported in the sequence layout: {sorted(kwargs)}"
        assert not joint_attention_kwargs, "joint_attention_kwargs are not supported in the sequence layout"
        assert encoder_hidden_states is not None and timestep is not None
        assert img_ids is not None and txt_ids is not None

        T, B = components.num_teammates, components.batch_size
        assert T == len(self.teamwork_outputs), f"{T} teammates, the adapters have {len(self.teamwork_outputs)}"
        count, L, C = hidden_states.shape
        L_text = encoder_hidden_states.shape[1]
        assert components.teammate_indices.shape == (count,), f"{count} image rows for {components.teammate_indices.shape} components"
        assert encoder_hidden_states.shape[0] == B, f"text batch {encoder_hidden_states.shape[0]}, components batch {B}"
        assert timestep.shape == (B,), f"timestep {tuple(timestep.shape)}, expected one per batch element ({B},)"
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        assert img_ids.shape[0] == T * L, f"img_ids {tuple(img_ids.shape)} for {T} slots of {L} tokens"
        assert txt_ids.shape[0] == L_text, f"txt_ids {tuple(txt_ids.shape)} for {L_text} text tokens"

        image_layout = SequenceLayout.joint(0, L, self.teamwork_outputs)
        layout = SequenceLayout.joint(L_text, L, self.teamwork_outputs)
        selection = SequenceSelection.from_components(
            components, layout, attn_keep=attn_keep, text_teammate=self.text_teammate,
        )
        image_selection = SequenceSelection(
            layout=image_layout,
            present=selection.present,
            attn_keep=selection.attn_keep,
            text_teammate=selection.text_teammate,
        )
        for module in self.image_stream_adapters:
            module.selection = image_selection
        for module in self.full_stream_adapters:
            module.selection = selection
        block_mask = sequence_block_mask(selection)

        # Image stream `[img_0 | … | img_{T-1}]`; absent slots are zeros and masked out.
        image = to_sequence(hidden_states, hidden_states.new_zeros(B, 0, C), components, image_layout)

        timestep = timestep.to(image.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(image.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)
        image_mod = _modulation_rows(self.double_stream_modulation_img, temb, T).teammates
        text_mod = _modulation_rows(self.double_stream_modulation_txt, temb, T).text
        single_mod = _modulation_rows(self.single_stream_modulation, temb, T)

        image = self.x_embedder(image)
        text = self.context_embedder(encoder_hidden_states)

        image_rotary = self.pos_embed(img_ids)
        text_rotary = self.pos_embed(txt_ids)
        rotary = (
            torch.cat([text_rotary[0], image_rotary[0]], dim=0),
            torch.cat([text_rotary[1], image_rotary[1]], dim=0),
        )

        checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing
        for block in self.transformer_blocks:
            assert isinstance(block, Flux2TransformerBlock)
            args = (block, image, text, image_mod, text_mod, rotary, block_mask)
            if checkpointing:
                text, image = self._gradient_checkpointing_func(_double_block_forward, *args)
            else:
                text, image = _double_block_forward(*args)

        hidden = torch.cat([text, image], dim=1)
        for block in self.single_transformer_blocks:
            assert isinstance(block, Flux2SingleTransformerBlock)
            args = (block, hidden, L_text, single_mod.text, single_mod.teammates, rotary, block_mask)
            if checkpointing:
                hidden = self._gradient_checkpointing_func(_single_block_forward, *args)
            else:
                hidden = _single_block_forward(*args)

        image = self.norm_out(hidden[:, L_text:], temb)
        output, _ = from_sequence(self.proj_out(image), components, image_layout)
        if not return_dict:
            return (output,)
        return Flux2Transformer2DModelOutput(sample=output)


def element_timesteps(timesteps: Tensor, components: Components) -> Tensor:
    """Per-component timesteps `(count,)` -> `(B,)`; every component of an element shares one."""
    per_element = timesteps.new_zeros(components.batch_size)
    per_element[components.batch_indices] = timesteps
    assert torch.equal(per_element[components.batch_indices], timesteps), (
        "components of one batch element have different timesteps"
    )
    return per_element


class SequenceFlux2KleinPipeline(_Flux2KleinTeamwork):
    """
    Sequence-axis teamwork on FLUX.2 Klein: each batch element's teammates share one token
    stream and attend to each other (gated by presence and `attn_keep`). Image ranges get
    their teammate's LoRA and modulation; the single text copy gets `text_teammate`'s, or
    runs at base when it is None. Teammate `t` sits at RoPE T-coord `teammate_image_ids[t]`.
    Every component of a batch element is modulated with that element's timestep, inputs too.
    """

    layout: Layout = "sequence"

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: Flux2KleinPipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
        cls.check_layout(teamwork_config)
        pipeline = cls(**base_pipeline.components)  # type: ignore
        pipeline.teamwork_config = teamwork_config
        if override_profile is None:
            assert teamwork_config.profile is not None, "the config names no profile"
            override_profile = TEAMWORK_PROFILES[teamwork_config.profile]
        adapted = adapt(
            pipeline.transformer,
            cfg=teamwork_config,
            device=pipeline.device,
            dtype=torch.bfloat16,
            requires_grad=training,
            override_profile=sequence_profile(override_profile),
            state=state,
            infer_layers_from_state=False,
        )
        assert isinstance(adapted, Flux2Transformer2DModel)
        pipeline.transformer = SequenceFlux2Transformer.from_adapted(adapted, teamwork_config)
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    def image_ids(self, lh: int, lw: int, device: torch.device) -> Tensor:
        """`(T·L, 4)` RoPE ids of the image stream: slot `t` at T-coord `teammate_image_ids[t]`."""
        offsets = self.teamwork_config.teammate_image_ids
        assert offsets is not None  # resolved in TeamworkConfig.__post_init__
        return family_flux2.offset_image_ids(family_flux2.latent_image_ids(lh, lw, device), offsets)

    def encode_element_prompts(self, prompt: str | list[str], batch_size: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        """`(prompt_embeds, text_ids)` with one text copy per batch element (`prompt` lists are per element)."""
        if isinstance(prompt, list):
            assert len(prompt) == batch_size, f"{len(prompt)} prompts for {batch_size} batch elements"
        prompt_embeds, text_ids = family_flux2.encode_prompt(self, prompt, batch_size, dtype)
        assert prompt_embeds.shape[0] == batch_size, f"{tuple(prompt_embeds.shape)} for {batch_size} batch elements"
        return prompt_embeds, text_ids

    def predict(
        self,
        model_latents: Tensor,  # (count, C, lh, lw), extra channels included
        timesteps: Tensor,  # (B,) in 0..1000
        prompt_embeds: Tensor,  # (B, L_text, J)
        text_ids: Tensor,
        components: Components,
        attn_keep: Tensor | None,
    ) -> Tensor:
        """The transformer's `(count, in_channels, lh, lw)` prediction for component-layout latents."""
        _, _, lh, lw = model_latents.shape
        prediction = self.transformer(
            hidden_states=family_flux2.patchify(model_latents),
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps / 1000,
            img_ids=self.image_ids(lh, lw, model_latents.device),
            txt_ids=text_ids,
            guidance=None,  # Flux2 Klein is guidance-distilled (guidance_embeds=False)
            components=components,
            attn_keep=attn_keep,
        ).sample
        return family_flux2.unpatchify(prediction, self.in_channels, lh, lw)

    def train_loss(
        self,
        batch: TeamBuilder,
        noise: torch.Tensor,
        prompt: str | list[str] = "",
    ) -> LossOutput:
        """`prompt`: one string, or one per batch element."""
        # A fresh scheduler, so training timesteps don't depend on the inference schedule
        # that __call__ last set on self.scheduler.
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
            prompt_embeds, text_ids = self.encode_element_prompts(prompt, components.batch_size, latents.dtype)

        model_latents = noising.noisy_latents
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)
        if extra is not None:
            model_latents = torch.cat([model_latents, extra], dim=1)

        model_pred = self.predict(
            model_latents, element_timesteps(noising.timesteps, components),
            prompt_embeds, text_ids, components, attn_keep,
        )
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
    ) -> Any:  # one dict per batch element if `images` is a list, else the single dict
        """
        `prompt`: one string, or one per batch element. `guidance_scale` is ignored (Klein is
        guidance-distilled). `callback_on_step_end(pipe, i, t, {'latents': (count, c, lh, lw),
        'selection': Components})` may return a dict with replacement `latents`.
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
            lambda x: self.vae_encode(x, generator), self.in_channels, 1 / self.vae_scale_factor
        )
        if noise is None:
            latents = initial_noise(clean_latents.shape, generator, device, self.dtype)
        else:
            latents = noise.to(device, self.dtype)
        assert latents.shape == clean_latents.shape, f"noise {tuple(latents.shape)}, latents {tuple(clean_latents.shape)}"

        _, _, lh, lw = latents.shape
        prompt_embeds, text_ids = self.encode_element_prompts(prompt, components.batch_size, latents.dtype)
        timesteps, num_inference_steps = family_flux2.inference_timesteps(
            self.scheduler, num_inference_steps, (lh // 2) * (lw // 2), device,
        )
        extra = batch.packed_scaled_extra(1 / self.vae_scale_factor)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                model_latents = latents
                model_latents[components.input_subindices] = clean_latents[components.input_subindices]
                if extra is not None:
                    model_latents = torch.cat([model_latents, extra], dim=1)

                noise_pred = self.predict(
                    model_latents, t.expand(components.batch_size),
                    prompt_embeds, text_ids, components, attn_keep,
                )

                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_kwargs = {'latents': latents, 'selection': components}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)

                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        else:
            return outputs[0]
