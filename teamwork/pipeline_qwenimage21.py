"""
Qwen-Image 2.1 teamwork pipelines.

Latents stay in component layout `(count, 64, h, w)` across training and sampling. The transformer
is the base `QwenImage21Transformer2DModel` re-classed (sharing every parameter) to a subclass whose
`forward` takes the teamwork batch, so DDP wrappers and forward hooks see one call per denoiser step:

    transformer(latents, timesteps, components, prompt, team, attn_keep).sample  # (count, 64, h, w)

`timesteps` are per component in [0, 1]; the forwards hard-code t=0 for inputs (and text), as the
base model modulates its condition tokens. Checkpoint keys are the module paths of the adapted
linears, identical in both layouts.

- `BatchQwenImage21Pipeline` (`layout="batch"`): one row per component,
  `[VLM prefix with the vlm_inputs' slots filled by their clean latents, own block]`. LoRA covers the
  own block only; the prefix runs at base. Teammates communicate only through LoRA.
- `SequenceQwenImage21Pipeline` (`layout="sequence"`): one stream per batch element,
  `user\\n<image1><vs>[IN vlm]<ve>…{prompt}<|im_end|>\\n<|im_start|>assistant\\n[IN non-vlm…][OUT…]`,
  under the base block-causal mask plus teamwork rules (`sequence_mask`, rule `"block_causal"`).
"""

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Literal, Sequence

import PIL.Image
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline

from .pipelines import TeamworkPipeline, LossOutput
from .config import TeamworkConfig, Layout
from .adapter import adapt, TEAMWORK_PROFILES, Adapt, AdapterMixin, set_selection, shallowcopy_into
from .checkpoint import save_adapters
from .data import TeamBuilder, Components, OutputImageType
from .batch_linear import BatchSelection, BatchTeamworkLinear, _batch_lora
from .sequence_linear import (
    TEXT_SLOT,
    SequenceLayout,
    SequenceSelection,
    SequenceTeamworkLinear,
    SequenceTeamworkModulation,
    from_sequence,
    to_sequence,
)
from .sampling import sample_timestep_u, flow_match_noising, initial_noise
from . import family_qwen21
from .family_qwen21 import QwenPrompt

# The linears on `temb` (no token axis): the one `modulation` shared by every block.
QWEN21_TEMB_LINEARS = ["modulation.1"]

# Mirrors FLUX2's coverage: the shared modulation plus every token linear of the (single-stream)
# blocks. `img_in`, `txt_in`, `norm_out`, and `proj_out` stay at base.
TEAMWORK_PROFILES["QWEN21"] = [
    *QWEN21_TEMB_LINEARS,
    "transformer_blocks.*.attn.to_q",
    "transformer_blocks.*.attn.to_k",
    "transformer_blocks.*.attn.to_v",
    "transformer_blocks.*.attn.to_out.0",
    "transformer_blocks.*.img_mlp.proj",
    "transformer_blocks.*.img_mlp.gate_layer",
    "transformer_blocks.*.img_mlp.out",
]


@dataclass(frozen=True)
class QwenTeam:
    """The roster facts the Qwen forwards read, from a `TeamworkConfig`."""

    vlm_slots: tuple[int, ...]  # roster slot of each Qwen3-VL condition image, in VLM order
    outputs: tuple[bool, ...]  # per roster slot
    text_teammate: int | None  # roster slot adapting the text (sequence layout), or None

    @staticmethod
    def from_config(cfg: TeamworkConfig) -> "QwenTeam":
        return QwenTeam(
            vlm_slots=tuple(cfg.teammates.index(name) for name in cfg.vlm_inputs),
            outputs=tuple(name.endswith(".out") for name in cfg.teammates),
            text_teammate=None if cfg.text_teammate is None else cfg.teammates.index(cfg.text_teammate),
        )


def input_timesteps_zeroed(timesteps: Tensor, components: Components) -> Tensor:
    """Qwen modulates condition (input) tokens from t=0: zero the inputs' entries of `(count,)`."""
    assert timesteps.shape == components.teammate_indices.shape, (
        tuple(timesteps.shape), tuple(components.teammate_indices.shape)
    )
    return timesteps.index_fill(0, components.input_subindices, 0.0)


def _stream_key_valid(text_valid: Tensor, layout: SequenceLayout) -> Tensor | None:
    """`(N, S)` key validity: text from `text_valid` `(N, S_text)` in stream order, images True."""
    N = text_valid.shape[0]
    assert text_valid.shape == (N, layout.text_length), (tuple(text_valid.shape), layout.text_length)
    if bool(text_valid.all()):
        return None
    pieces: list[Tensor] = []
    position = 0
    for start, end, slot in layout.segments:
        if slot == TEXT_SLOT:
            pieces.append(text_valid[:, position : position + end - start])
            position += end - start
        else:
            pieces.append(text_valid.new_ones(N, end - start))
    return torch.cat(pieces, dim=1)


def _selected_block(block: nn.Module, selection: Any, *args: Any) -> Tensor:
    # Set the selection inside the (possibly recomputed) function, so a checkpoint recompute in
    # backward always sees the selection of its own forward.
    set_selection(block, selection)
    return family_qwen21.block_forward(block, *args)  # type: ignore[arg-type]


def _run_blocks(
    transformer: QwenImage21Transformer2DModel,
    selection: Any,
    x: Tensor,
    modulation: Tensor,
    rope: Tensor,
    block_mask: Any,
) -> Tensor:
    use_checkpointing = transformer.gradient_checkpointing and torch.is_grad_enabled()
    for block in transformer.transformer_blocks:
        if use_checkpointing:
            x = checkpoint(_selected_block, block, selection, x, modulation, rope, block_mask, use_reentrant=False)  # type: ignore[assignment]
        else:
            x = _selected_block(block, selection, x, modulation, rope, block_mask)
    return x


def _check_inputs(latents: Tensor, timesteps: Tensor, components: Components, prompt: QwenPrompt):
    count = components.teammate_indices.shape[0]
    assert latents.ndim == 4 and latents.shape[0] == count, (tuple(latents.shape), count)
    assert timesteps.shape == (count,), (tuple(timesteps.shape), count)
    assert prompt.batch_size == components.batch_size, (prompt.batch_size, components.batch_size)
    assert prompt.hidden.device == latents.device, (prompt.hidden.device, latents.device)


# ======================================================================================
# Batch axis
# ======================================================================================


@dataclass(frozen=True)
class OwnBlockSelection(BatchSelection):
    """`BatchSelection` for rows `[prefix, own block]`: token LoRAs apply from `own_start` on."""

    own_start: int

    @staticmethod
    def build(components: Components, own_start: int) -> "OwnBlockSelection":
        base = BatchSelection.from_components(components)
        return OwnBlockSelection(
            teammate_indices=base.teammate_indices,
            batch_indices=base.batch_indices,
            input_subindices=base.input_subindices,
            output_subindices=base.output_subindices,
            num_teammates=base.num_teammates,
            batch_size=base.batch_size,
            batch_matrix=base.batch_matrix,
            own_start=own_start,
        )


class BatchQwenImage21Linear(BatchTeamworkLinear):
    """`BatchTeamworkLinear` on `(count, S, in)` rows whose LoRA covers only `[own_start:]`."""

    def forward(self, input: Tensor) -> Tensor:
        output = F.linear(input, self.weight, self.bias)
        sel = self.selection
        if sel is None:
            return output
        assert isinstance(sel, OwnBlockSelection), type(sel)
        assert input.ndim == 3, tuple(input.shape)
        s = sel.own_start
        delta = _batch_lora(self.adapter, sel, self.communicate, input[:, s:], torch.zeros_like(output[:, s:]))
        return torch.cat([output[:, :s], output[:, s:] + delta], dim=1)


class BatchQwenImage21Transformer(QwenImage21Transformer2DModel):
    """Batch-axis forward; holds exactly the base (adapted) modules. See the module docstring."""

    def forward(  # type: ignore[override]
        self,
        latents: Tensor,  # (count, C, lh, lw)
        timesteps: Tensor,  # (count,), in [0, 1]; inputs are zeroed here
        components: Components,
        prompt: QwenPrompt,  # one row per batch element
        team: QwenTeam,
        attn_keep: Tensor | None = None,
    ) -> Transformer2DModelOutput:
        assert attn_keep is None, "batch-axis teammates never attend to each other"
        _check_inputs(latents, timesteps, components, prompt)
        count, _, lh, lw = latents.shape
        L = lh * lw
        B, T, V = components.batch_size, components.num_teammates, len(team.vlm_slots)
        device = latents.device
        timesteps = input_timesteps_zeroed(timesteps, components)

        img = self.img_in(family_qwen21.pack(latents))  # (count, L, D)
        D = img.shape[2]
        txt = self.txt_in(prompt.text_hidden())[components.batch_indices]  # (count, S_text, D)

        # Row layout: the VLM part (condition image k is slot k), then the own block as slot V.
        pieces = [*prompt.pieces(L), ("image", V)]
        assert sum(1 for kind, _ in pieces if kind == "image") == V + 1, (
            f"the prompt holds {sum(1 for k, _ in pieces if k == 'image') - 1} condition images, "
            f"the config has {V} vlm_inputs"
        )
        layout = SequenceLayout.from_pieces(pieces, L, outputs=(False,) * V + (True,), rule="block_causal")
        own_start = layout.image_offsets[V]
        assert own_start + L == layout.seq_len, "the own block must end the row"

        # Every row of batch element b sees b's VLM inputs (clean latents; zeros where absent).
        dense = img.new_zeros(B, T, L, D).index_put((components.batch_indices, components.teammate_indices), img)
        present = torch.zeros(B, T, dtype=torch.bool, device=device).index_put(
            (components.batch_indices, components.teammate_indices), torch.ones_like(components.teammate_indices, dtype=torch.bool)
        )
        vlm_slots = torch.tensor(team.vlm_slots, dtype=torch.int64, device=device)
        rows = components.batch_indices
        pieces_x: list[Tensor] = []
        text_position = 0
        for start, end, slot in layout.segments:
            if slot == TEXT_SLOT:
                pieces_x.append(txt[:, text_position : text_position + end - start])
                text_position += end - start
            elif slot == V:
                pieces_x.append(img)
            else:
                pieces_x.append(dense[rows, team.vlm_slots[slot]])
        x = torch.cat(pieces_x, dim=1)

        row_present = torch.cat([present[rows][:, vlm_slots], present.new_ones(count, 1)], dim=1)
        key_valid = _stream_key_valid(prompt.text_valid()[rows], layout)
        mask_selection = SequenceSelection(layout, row_present, attn_keep=None, text_teammate=None, key_valid=key_valid)
        lora_selection = OwnBlockSelection.build(components, own_start)

        # Prefix tokens: base modulation at t=0. Own block: the row's (adapted) modulation at its t.
        set_selection(self.modulation, lora_selection)
        temb_rows = family_qwen21.time_embedding(self, timesteps, img)
        own_modulation = self.modulation(temb_rows)  # (count, 4D)
        prefix_modulation = family_qwen21.base_modulation(
            self, family_qwen21.time_embedding(self, timesteps.new_zeros(1), img)
        )
        token_row = torch.zeros(layout.seq_len, dtype=torch.int64, device=device)
        token_row[own_start:] = 1
        modulation = family_qwen21.per_token(
            torch.stack([prefix_modulation.expand(count, -1), own_modulation], dim=1), token_row
        )

        rope = family_qwen21.rotary(self, layout.slot_of_token(device) != TEXT_SLOT, lh, lw)
        block_mask = family_qwen21.padded_block_mask(mask_selection)
        x = _run_blocks(self, lora_selection, x, modulation, rope, block_mask)

        out = family_qwen21.output_projection(self, x[:, own_start:], temb_rows)
        return Transformer2DModelOutput(sample=family_qwen21.unpack(out, lh, lw))


# ======================================================================================
# Sequence axis
# ======================================================================================


def sequence_layout(prompt: QwenPrompt, team: QwenTeam, image_length: int) -> SequenceLayout:
    """`[VLM part with vlm_inputs' slots][IN non-vlm…][OUT…]`, roster order within each group."""
    pieces: list[tuple[Literal["text", "image"], int]] = []
    for kind, value in prompt.pieces(image_length):
        if kind == "text":
            pieces.append(("text", value))
        else:
            assert value < len(team.vlm_slots), f"the prompt holds more condition images than vlm_inputs {team.vlm_slots}"
            pieces.append(("image", team.vlm_slots[value]))
    T = len(team.outputs)
    rest = [t for t in range(T) if t not in team.vlm_slots]
    pieces += [("image", t) for t in rest if not team.outputs[t]]
    pieces += [("image", t) for t in rest if team.outputs[t]]
    return SequenceLayout.from_pieces(pieces, image_length, team.outputs, rule="block_causal")


class SequenceQwenImage21Transformer(QwenImage21Transformer2DModel):
    """Sequence-axis forward; holds exactly the base (adapted) modules. See the module docstring."""

    def forward(  # type: ignore[override]
        self,
        latents: Tensor,  # (count, C, lh, lw)
        timesteps: Tensor,  # (count,), in [0, 1]; inputs are zeroed here
        components: Components,
        prompt: QwenPrompt,  # one row per batch element
        team: QwenTeam,
        attn_keep: Tensor | None = None,
    ) -> Transformer2DModelOutput:
        _check_inputs(latents, timesteps, components, prompt)
        count, _, lh, lw = latents.shape
        L = lh * lw
        B, T = components.batch_size, components.num_teammates
        device = latents.device
        timesteps = input_timesteps_zeroed(timesteps, components)

        img = self.img_in(family_qwen21.pack(latents))  # (count, L, D)
        txt = self.txt_in(prompt.text_hidden())  # (B, S_text, D)
        layout = sequence_layout(prompt, team, L)
        selection = SequenceSelection.from_components(
            components,
            layout,
            attn_keep=attn_keep,
            text_teammate=team.text_teammate,
            key_valid=_stream_key_valid(prompt.text_valid(), layout),
        )
        x = to_sequence(img, txt, components, layout)

        # Modulation rows: text at t=0 (adapted iff text_teammate), slot t at its component's t
        # (absent slots get t=0; their tokens are masked out).
        slot_t = timesteps.new_zeros(B, T).index_put((components.batch_indices, components.teammate_indices), timesteps)
        text_temb = family_qwen21.time_embedding(self, timesteps.new_zeros(B), img)
        slot_temb = rearrange(
            family_qwen21.time_embedding(self, rearrange(slot_t, "b t -> (b t)"), img), "(b t) d -> b t d", t=T
        )
        set_selection(self.modulation, selection)
        rows = family_qwen21.modulation_rows(self, text_temb, slot_temb)
        slot = layout.slot_of_token(device)
        modulation = family_qwen21.per_token(torch.cat([rows.text[:, None], rows.teammates], dim=1), slot + 1)

        rope = family_qwen21.rotary(self, slot != TEXT_SLOT, lh, lw)
        block_mask = family_qwen21.padded_block_mask(selection)
        x = _run_blocks(self, selection, x, modulation, rope, block_mask)

        image_tokens, _ = from_sequence(x, components, layout)
        temb = family_qwen21.time_embedding(self, timesteps, img)
        out = family_qwen21.output_projection(self, image_tokens, temb)
        return Transformer2DModelOutput(sample=family_qwen21.unpack(out, lh, lw))


# ======================================================================================
# Pipelines
# ======================================================================================


def _as_class(module: nn.Module, cls: type[nn.Module]) -> nn.Module:
    """A shallow copy of `module` (sharing every parameter and submodule) of class `cls`."""
    assert isinstance(module, cls.__mro__[1]), f"{type(module)} is not a {cls.__mro__[1]}"
    copied = object.__new__(cls)
    shallowcopy_into(copied, module)
    return copied


class QwenImage21TeamworkPipeline(TeamworkPipeline, QwenImage21Pipeline):
    """
    Everything the two Qwen layouts share; subclasses only pick the transformer forward, the
    adapter classes, and whether `attn_keep` applies. Not constructed directly.
    """

    teamwork_config: TeamworkConfig
    timestep_weight: str = "unit"
    empty_prompt: QwenPrompt | None = None

    transformer_cls: ClassVar[type[QwenImage21Transformer2DModel]]
    token_adapter: ClassVar[type[AdapterMixin]]
    temb_adapter: ClassVar[type[AdapterMixin]]

    @classmethod
    def adapter_mapping(cls, profile: Sequence[Adapt]) -> list[Adapt]:
        """Profile globs -> `(glob, adapter class)`; explicit tuples pass through."""
        mapping: list[Adapt] = []
        for entry in profile:
            if isinstance(entry, tuple):
                mapping.append(entry)
            elif entry in QWEN21_TEMB_LINEARS:
                mapping.append((entry, cls.temb_adapter))
            else:
                mapping.append((entry, cls.token_adapter))
        return mapping

    @classmethod
    def from_base_pipeline(
        cls,
        base_pipeline: QwenImage21Pipeline,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
        grad_checkpointing: bool = True,
    ):
        cls.check_layout(teamwork_config)
        assert isinstance(base_pipeline, QwenImage21Pipeline), type(base_pipeline)
        assert isinstance(base_pipeline.transformer, QwenImage21Transformer2DModel)
        pipeline = cls(**base_pipeline.components, sample_sigmas=base_pipeline.config.get("sample_sigmas"))  # type: ignore
        pipeline.teamwork_config = teamwork_config
        if pipeline.text_encoder is not None:
            # Shared with `base_pipeline`, so this speeds up its encoder too (same weights).
            assert family_qwen21.use_patchwise_vision_embed(pipeline.text_encoder) == 1, "no Qwen3-VL patch_embed.proj"
        if override_profile is not None:
            profile = override_profile
        else:
            assert teamwork_config.profile is not None, "teamwork config has no profile"
            profile = TEAMWORK_PROFILES[teamwork_config.profile]
        transformer = adapt(
            pipeline.transformer,
            cfg=teamwork_config,
            device=pipeline.transformer.device,
            dtype=torch.bfloat16,
            requires_grad=training,
            override_profile=cls.adapter_mapping(profile),
            state=state,
            infer_layers_from_state=False,
        )
        pipeline.transformer = _as_class(transformer, cls.transformer_cls)  # type: ignore[assignment]
        if training and grad_checkpointing:
            pipeline.transformer.enable_gradient_checkpointing()
        return pipeline

    @property
    def unwrapped_transformer(self) -> QwenImage21Transformer2DModel:
        model = self.transformer
        while hasattr(model, "module"):
            model = getattr(model, "module")
        assert isinstance(model, self.transformer_cls), type(model)
        return model

    @property
    def in_channels(self) -> int:
        return self.unwrapped_transformer.config["in_channels"]

    @property
    def team(self) -> QwenTeam:
        return QwenTeam.from_config(self.teamwork_config)

    def save_adapters(self, safetensors_path: str):
        save_adapters(self.unwrapped_transformer, safetensors_path, self.teamwork_config)

    def load_extra_metadata(self, metadata: dict[str, str]):
        pass

    def vae_encode(self, x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        """Posterior means, so `generator` is unused (kept for the shared `vae_encode` signature)."""
        return family_qwen21.vae_encode(self.vae, x)

    def vae_decode(self, x: Tensor) -> Tensor:
        return family_qwen21.vae_decode(self.vae, x)

    def attn_keep(self, batch: TeamBuilder) -> Tensor | None:
        return None

    @torch.no_grad()
    def empty_prompts(self):
        """Encode the empty t2i prompt once, then drop the text encoder. Needs `vlm_inputs=[]`."""
        assert not self.teamwork_config.vlm_inputs, (
            "with vlm_inputs the encoding depends on the condition images, so it cannot be precomputed"
        )
        self.empty_prompt = family_qwen21.encode_prompts(self, [""], [[]])
        self.text_encoder = None  # type: ignore[assignment]

    def vlm_images(self, batch: TeamBuilder) -> list[list[PIL.Image.Image]]:
        """
        Per batch element, the `vlm_inputs` images in VLM order. A missing input (jagged presence)
        gets a mid-grey placeholder, so every element keeps the same layout; its latent slot is
        masked out of attention.
        """
        assert batch.resolution is not None
        H, W = batch.resolution
        images: list[list[PIL.Image.Image]] = []
        for b in range(batch.batch_size):
            row: list[PIL.Image.Image] = []
            for slot in self.team.vlm_slots:
                found = [c for c in batch.components if c.batch_idx == b and c.teammate == slot]
                assert len(found) <= 1
                if found and found[0].image is not None:
                    row.append(family_qwen21.image_to_pil(found[0].image))
                else:
                    row.append(PIL.Image.new("RGB", (W, H), (128, 128, 128)))
            images.append(row)
        return images

    def batch_prompts(self, prompt: str | Sequence[str], batch: TeamBuilder) -> list[str]:
        """One prompt per batch element, from a string, a per-element list, or a per-component list."""
        B = batch.batch_size
        if isinstance(prompt, str):
            return [prompt] * B
        if len(prompt) == B:
            return list(prompt)
        assert len(prompt) == batch.count, f"{len(prompt)} prompts for {B} batch elements / {batch.count} components"
        prompts: list[str | None] = [None] * B
        for component, p in zip(batch.components, prompt):
            assert prompts[component.batch_idx] in (None, p), "components of one batch element have different prompts"
            prompts[component.batch_idx] = p
        assert all(p is not None for p in prompts)
        return [p for p in prompts if p is not None]

    @torch.no_grad()
    def encode_team_prompt(
        self,
        batch: TeamBuilder,
        prompt: str | Sequence[str],
        prompt_embeds: QwenPrompt | None = None,
    ) -> QwenPrompt:
        """The VLM encoding for `batch` (prompts plus the vlm_inputs' images), on the batch's device."""
        prompts = self.batch_prompts(prompt, batch)
        if prompt_embeds is not None:
            encoded = prompt_embeds
        elif self.empty_prompt is not None:
            assert all(p == "" for p in prompts), "the text encoder was dropped by empty_prompts()"
            e = self.empty_prompt
            B = batch.batch_size
            encoded = QwenPrompt(e.hidden.expand(B, -1, -1), e.valid.expand(B, -1), e.image_pad)
        else:
            assert self.text_encoder is not None, "no text encoder; pass prompt_embeds"
            encoded = family_qwen21.encode_prompts(self, prompts, self.vlm_images(batch))
        assert encoded.batch_size == batch.batch_size, (encoded.batch_size, batch.batch_size)
        return encoded.to(batch.device, self.unwrapped_transformer.dtype)

    def train_loss(  # type: ignore[override]
        self,
        batch: TeamBuilder,
        noise: Tensor,
        prompt: str | Sequence[str] = "",
        prompt_embeds: QwenPrompt | None = None,
    ) -> LossOutput:
        assert batch.packed_extra() is None, "Qwen-Image 2.1 takes no extra input channels"
        components = batch.components_index()
        with torch.no_grad():
            latents = batch.packed_encoded_images(self.vae_encode, self.in_channels, 1 / self.vae_scale_factor)
            _, _, lh, lw = latents.shape
            train_scheduler = family_qwen21.training_scheduler(self.scheduler.config, lh * lw)
            timestep_u = sample_timestep_u(self.timestep_weight, batch.batch_size)
            noising = flow_match_noising(
                train_scheduler, timestep_u, components, latents, noise, self.timestep_weight,
                device=latents.device, dtype=latents.dtype,
            )
            encoded = self.encode_team_prompt(batch, prompt, prompt_embeds)
            attn_keep = self.attn_keep(batch)

        prediction = self.transformer(
            noising.noisy_latents, noising.timesteps / 1000, components, encoded, self.team, attn_keep,
        ).sample

        return LossOutput(
            selection=components,
            latents=latents,
            prediction=prediction,
            target=noise - latents,
            weight=batch.packed_scaled_weights(1 / self.vae_scale_factor).unsqueeze(1) * noising.weighting,
            timestep_idx=noising.timestep_idx,
            type="signal",
        )

    @torch.no_grad()
    def __call__(  # type: ignore[override]
        self,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        prompt: str | list[str] = "",
        num_inference_steps: int = 40,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: OutputImageType = "pil",
        batch: TeamBuilder | None = None,
        callback_on_step_end: Callable[..., dict | None] | None = None,
        prompt_embeds: QwenPrompt | None = None,
        sigmas: list[float] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """
        `callback_on_step_end(pipe, i, t, {'latents': (count, c, lh, lw), 'selection': Components})`
        may return a dict with replacement `latents`.
        """
        device = self._execution_device
        if batch is None:
            batch = TeamBuilder.from_inputs(
                self.teamwork_config.teammates,
                device=device,
                dtype=self.unwrapped_transformer.dtype,
                images=images,
                request=request,
                width=width,
                height=height,
                attn_allow=self.teamwork_config.attn_allow,
            )
        assert batch.packed_extra() is None, "Qwen-Image 2.1 takes no extra input channels"
        components = batch.components_index()
        clean_latents = batch.packed_encoded_images(self.vae_encode, self.in_channels, 1 / self.vae_scale_factor)
        if noise is None:
            latents = initial_noise(clean_latents.shape, generator, clean_latents.device, clean_latents.dtype)
        else:
            assert noise.shape == clean_latents.shape, (tuple(noise.shape), tuple(clean_latents.shape))
            latents = noise.to(clean_latents.device, clean_latents.dtype)
        _, _, lh, lw = latents.shape

        encoded = self.encode_team_prompt(batch, prompt, prompt_embeds)
        attn_keep = self.attn_keep(batch)
        team = self.team
        timesteps, num_inference_steps = family_qwen21.inference_timesteps(
            self.scheduler, num_inference_steps, lh * lw, latents.device,
            sigmas if sigmas is not None else self.config.get("sample_sigmas"),
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents[components.input_subindices] = clean_latents[components.input_subindices]
                # As the base pipeline: the timestep is rounded to the latents' dtype before /1000.
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                noise_pred = self.transformer(latents, timestep / 1000, components, encoded, team, attn_keep).sample
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                if callback_on_step_end is not None:
                    callback_outputs = callback_on_step_end(self, i, t, {"latents": latents, "selection": components})
                    if callback_outputs is not None:
                        latents = callback_outputs.pop("latents", latents)
                progress_bar.update()

        outputs = batch.unpack_decoded_images(latents, self.vae_decode, output_type=output_type)
        if isinstance(images, list):
            return outputs
        return outputs[0]


class BatchQwenImage21Pipeline(QwenImage21TeamworkPipeline):
    layout: Layout = "batch"
    transformer_cls = BatchQwenImage21Transformer
    token_adapter = BatchQwenImage21Linear
    temb_adapter = BatchTeamworkLinear


class SequenceQwenImage21Pipeline(QwenImage21TeamworkPipeline):
    layout: Layout = "sequence"
    transformer_cls = SequenceQwenImage21Transformer
    token_adapter = SequenceTeamworkLinear
    temb_adapter = SequenceTeamworkModulation

    def attn_keep(self, batch: TeamBuilder) -> Tensor | None:
        """`attn_allow` plus attention dropout, from the global RNG."""
        return batch.attn_keep()
