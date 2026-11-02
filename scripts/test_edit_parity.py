"""
Editing parity for FLUX.2 Klein sequence-axis teamwork, at the transformer level.

With zero LoRA, sequence teamwork with roster `(edited.out, source.in)` and RoPE T-coords
`(0, 10)` (Flux2 `_prepare_image_ids`, scale=10) must reproduce base Klein editing, which
runs `[text | target(T=0) | reference(T=10)]` in one sequence. The only expected difference
is bf16 noise (flex_attention vs SDPA, per-range modulation broadcasting).

For scale, the same comparison is printed for a random non-zero LoRA (what a real adapter
changes) and for naive T-coords (0, 1), which put the reference next to the target.

Run from the repo root (CUDA, ~20 GB):

    uv run python scripts/test_edit_parity.py
"""

import sys

import torch
from torch import Tensor

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from teamwork.adapter import TEAMWORK_PROFILES, adapt, adapter_parameters
from teamwork.config import TeamworkConfig
from teamwork.data import Components
from teamwork import family_flux2
from teamwork.pipeline_flux2_klein import SequenceFlux2Transformer, sequence_profile

REPO = "black-forest-labs/FLUX.2-klein-9B"
KLEIN_REF_SCALE = 10  # Flux2 _prepare_image_ids: reference images sit at T = 10, 20, ...
MAX_RELATIVE = 2e-2  # bf16 noise; the naive and random-LoRA cases are far above this
LH = LW = 64  # 512x512 pixels
L_TEXT = 512  # Klein's prompt length


def txt_ids(length: int, device: torch.device) -> Tensor:
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device)
    return ids


def base_edit_prediction(
    base: Flux2Transformer2DModel, gen: Tensor, ref: Tensor, text: Tensor, timestep: Tensor, ref_first: bool = False,
) -> Tensor:
    """
    Base editing: `[gen(T=0) | ref(T=10)]` in one sequence; returns the gen prediction.
    `ref_first` runs `[ref(T=10) | gen(T=0)]`, which is mathematically identical (attention is
    permutation-equivariant given the RoPE ids), so its difference is the bf16 noise floor.
    """
    L = gen.shape[1]
    offsets = [KLEIN_REF_SCALE, 0] if ref_first else [0, KLEIN_REF_SCALE]
    ids = family_flux2.offset_image_ids(family_flux2.latent_image_ids(LH, LW, gen.device), offsets)
    out = base(
        hidden_states=torch.cat([ref, gen] if ref_first else [gen, ref], dim=1),
        encoder_hidden_states=text,
        timestep=timestep,
        img_ids=ids,
        txt_ids=txt_ids(text.shape[1], gen.device),
        guidance=None,
        return_dict=False,
    )[0]
    return out[:, L:] if ref_first else out[:, :L]


def sequence_edit_prediction(
    model: SequenceFlux2Transformer, cfg: TeamworkConfig, gen: Tensor, ref: Tensor, text: Tensor, timestep: Tensor,
) -> Tensor:
    """Sequence teamwork; components in TeamBuilder order (source.in provided, then edited.out requested)."""
    device = gen.device
    components = Components(
        teammate_indices=torch.tensor([1, 0], device=device),
        batch_indices=torch.tensor([0, 0], device=device),
        input_subindices=torch.tensor([0], device=device),
        output_subindices=torch.tensor([1], device=device),
        num_teammates=2,
        batch_size=1,
    )
    assert cfg.teammate_image_ids is not None
    ids = family_flux2.offset_image_ids(family_flux2.latent_image_ids(LH, LW, device), cfg.teammate_image_ids)
    out = model(
        hidden_states=torch.cat([ref, gen], dim=0),
        encoder_hidden_states=text,
        timestep=timestep,
        img_ids=ids,
        txt_ids=txt_ids(text.shape[1], device),
        guidance=None,
        components=components,
    ).sample
    return out[1:2]


def sequence_model(base: Flux2Transformer2DModel, cfg: TeamworkConfig) -> SequenceFlux2Transformer:
    assert cfg.profile is not None
    adapted = adapt(
        base, cfg, device=base.device, dtype=torch.bfloat16,
        override_profile=sequence_profile(TEAMWORK_PROFILES[cfg.profile]),
    )
    assert isinstance(adapted, Flux2Transformer2DModel)
    return SequenceFlux2Transformer.from_adapted(adapted, cfg).eval()


def compare(name: str, prediction: Tensor, reference: Tensor, gate: bool) -> bool:
    diff = (prediction.float() - reference.float())
    max_abs = diff.abs().max().item()
    relative = (diff.norm() / reference.float().norm()).item()
    ok = relative <= MAX_RELATIVE
    status = ("PASS" if ok else "FAIL") if gate else "INFO"
    print(f"[{status}] {name}: max-abs {max_abs:.3e}, relative {relative:.3e} (scale {reference.abs().max().item():.3e})")
    return ok or not gate


@torch.no_grad()
def main():
    if not torch.cuda.is_available():
        print("sequence teamwork requires CUDA; skipping.")
        sys.exit(0)
    device = torch.device("cuda")
    base = Flux2Transformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16).to(device).eval()
    in_channels = base.config["in_channels"]
    joint_dim = base.config["joint_attention_dim"]

    g = torch.Generator(device).manual_seed(0)
    L = (LH // 2) * (LW // 2)
    gen = torch.randn(1, L, in_channels, device=device, dtype=torch.bfloat16, generator=g)
    ref = torch.randn(1, L, in_channels, device=device, dtype=torch.bfloat16, generator=g)
    text = torch.randn(1, L_TEXT, joint_dim, device=device, dtype=torch.bfloat16, generator=g)
    timestep = torch.full((1,), 0.5, device=device, dtype=torch.bfloat16)
    expected = base_edit_prediction(base, gen, ref, text, timestep)
    compare("bf16 noise floor: base editing with the images reordered", base_edit_prediction(base, gen, ref, text, timestep, ref_first=True), expected, gate=False)

    def config(**kwargs) -> TeamworkConfig:
        return TeamworkConfig(
            teammates=["edited.out", "source.in"], lora_rank=16, base_model=REPO, title="edit-parity",
            profile="FLUX2", layout="sequence", text_teammate="edited.out", **kwargs,
        )

    ok = True
    cfg = config()
    assert cfg.teammate_image_ids == [0, KLEIN_REF_SCALE], cfg.teammate_image_ids
    model = sequence_model(base, cfg)  # LoRA up and bias start at zero
    ok &= compare("zero LoRA, T-coords (0, 10) vs base editing", sequence_edit_prediction(model, cfg, gen, ref, text, timestep), expected, gate=True)

    lora = torch.Generator().manual_seed(1)
    for name, param in adapter_parameters(model).items():
        if name.endswith(".up"):
            param.data = (0.02 * torch.randn(param.shape, generator=lora)).to(param)
    compare("random LoRA (up ~ N(0, 0.02)) vs base editing", sequence_edit_prediction(model, cfg, gen, ref, text, timestep), expected, gate=False)
    del model

    naive = config(teammate_image_ids=[0, 1])
    compare("zero LoRA, naive T-coords (0, 1) vs base editing", sequence_edit_prediction(sequence_model(base, naive), naive, gen, ref, text, timestep), expected, gate=False)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
