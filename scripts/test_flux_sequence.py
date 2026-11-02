"""
Structural checks of the FLUX.1 sequence-axis transformer on a tiny random model (CUDA, fp32).

1. Zero LoRA, full presence: equals the base transformer on `[text | img_0 | img_1]` with
   per-slot RoPE offsets (flex vs SDPA tolerance).
2. Roster of one with `text_teammate` set and non-zero LoRA: equals the batch-axis transformer,
   which adapts every token of its single row (text included in the single blocks).
3. Jagged presence with non-zero LoRA and communication: an absent slot changes nothing, so
   each batch element equals running it alone on a roster without the absent teammate.
4. Gradient checkpointing gives the same loss and adapter gradients.

Usage: python scripts/test_flux_sequence.py
"""

import torch
from torch import Tensor, nn
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from teamwork.adapter import TEAMWORK_PROFILES, adapt, adapter_parameters, set_selection
from teamwork.batch_linear import BatchSelection
from teamwork.config import TeamworkConfig
from teamwork.data import Components
from teamwork import family_flux
from teamwork.pipeline_flux import SequenceFluxTransformer, sequence_flux_profile

DEVICE = torch.device("cuda")
TEXT_LEN, TEXT_DIM, POOLED_DIM, IN_CH, LH, LW = 8, 32, 32, 16, 8, 8
L = (LH // 2) * (LW // 2)
FAILURES: list[str] = []


def check(name: str, a: Tensor, b: Tensor, tol: float):
    diff = (a - b).abs().max().item()
    scale = b.abs().max().item()
    ok = diff <= tol * max(scale, 1.0)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max-abs-diff={diff:.3e} scale={scale:.3e}")
    if not ok:
        FAILURES.append(name)


def tiny_base() -> FluxTransformer2DModel:
    g = torch.Generator().manual_seed(0)
    model = FluxTransformer2DModel(
        patch_size=1, in_channels=IN_CH, num_layers=2, num_single_layers=2,
        attention_head_dim=16, num_attention_heads=2, joint_attention_dim=TEXT_DIM,
        pooled_projection_dim=POOLED_DIM, guidance_embeds=True, axes_dims_rope=(4, 6, 6),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                param.copy_(0.1 * torch.randn(param.shape, generator=g))
            elif param.ndim == 1:
                param.copy_(1.0 + 0.1 * torch.randn(param.shape, generator=g))
            else:
                param.copy_(torch.randn(param.shape, generator=g) / param.shape[-1] ** 0.5)
    return model.to(DEVICE).eval().requires_grad_(False)


def config(teammates: list[str], layout: str, text_teammate: str | None, communicate: bool = True) -> TeamworkConfig:
    return TeamworkConfig(
        teammates=teammates, lora_rank=4, base_model="tiny-flux", title="test", profile="FLUX",
        lora_communication=communicate, layout=layout, text_teammate=text_teammate,  # type: ignore[arg-type]
    )


def sequence_model(base: FluxTransformer2DModel, cfg: TeamworkConfig, state: dict[str, Tensor] | None = None) -> SequenceFluxTransformer:
    adapted = adapt(
        base, cfg, device=DEVICE, dtype=torch.float32, state=state, infer_layers_from_state=False,
        override_profile=sequence_flux_profile(TEAMWORK_PROFILES["FLUX"]),
    )
    assert isinstance(adapted, FluxTransformer2DModel)
    return SequenceFluxTransformer.from_adapted(adapted, cfg)


def random_state(model: nn.Module, seed: int, zero_up: bool = False) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    state = {}
    for name, param in adapter_parameters(model).items():
        value = 0.2 * torch.randn(param.shape, generator=g)
        if zero_up and (name.endswith(".up") or name.endswith(".bias")):
            value.zero_()
        state[name] = value.to(DEVICE)
    return state


def components(pairs: list[tuple[int, int]], T: int, B: int, outputs: set[int]) -> Components:
    """`pairs` of (batch, teammate), one per component."""
    idx = lambda xs: torch.tensor(xs, dtype=torch.int64, device=DEVICE)
    return Components(
        teammate_indices=idx([t for _, t in pairs]),
        batch_indices=idx([b for b, _ in pairs]),
        input_subindices=idx([i for i, (_, t) in enumerate(pairs) if t not in outputs]),
        output_subindices=idx([i for i, (_, t) in enumerate(pairs) if t in outputs]),
        num_teammates=T,
        batch_size=B,
    )


def inputs(B: int, count: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g).to(DEVICE)
    return dict(
        hidden_states=r(count, L, IN_CH),
        encoder_hidden_states=r(B, TEXT_LEN, TEXT_DIM),
        pooled_projections=r(B, POOLED_DIM),
        timestep=torch.rand(B, generator=g).to(DEVICE),
        guidance=torch.full((B,), 3.5, device=DEVICE),
        img_ids=family_flux.latent_image_ids(LH, LW, DEVICE, torch.float32),
        txt_ids=torch.zeros(TEXT_LEN, 3, device=DEVICE),
    )


@torch.no_grad()
def test_zero_lora_base(base: FluxTransformer2DModel):
    cfg = config(["edited.out", "source.in"], "sequence", "edited.out")
    model = sequence_model(base, cfg)
    model = sequence_model(base, cfg, random_state(model, 1, zero_up=True))
    x = inputs(1, 2, seed=2)
    comps = components([(0, 0), (0, 1)], T=2, B=1, outputs={0})
    out = model(**x, components=comps).sample
    assert cfg.teammate_image_ids == [0, 1]
    ref = base(
        hidden_states=torch.cat([x["hidden_states"][0:1], x["hidden_states"][1:2]], dim=1),
        encoder_hidden_states=x["encoder_hidden_states"],
        pooled_projections=x["pooled_projections"],
        timestep=x["timestep"],
        guidance=x["guidance"],
        img_ids=family_flux.offset_image_ids(x["img_ids"], [0, 1]),
        txt_ids=x["txt_ids"],
    ).sample
    check("zero LoRA == base [text | img_0 | img_1]", out, torch.cat([ref[:, :L], ref[:, L:]], dim=0), 1e-4)


@torch.no_grad()
def test_single_teammate_matches_batch(base: FluxTransformer2DModel):
    seq_cfg = config(["edited.out"], "sequence", "edited.out")
    seq = sequence_model(base, seq_cfg)
    state = random_state(seq, 3)
    seq = sequence_model(base, seq_cfg, state)
    batch_cfg = config(["edited.out"], "batch", None)
    batch_model = adapt(base, batch_cfg, device=DEVICE, dtype=torch.float32, state=state, infer_layers_from_state=False)
    x = inputs(1, 1, seed=4)
    comps = components([(0, 0)], T=1, B=1, outputs={0})
    out = seq(**x, components=comps).sample
    set_selection(batch_model, BatchSelection.from_components(comps))
    ref = batch_model(**x).sample
    check("roster of one, text_teammate set == batch layout", out, ref, 1e-4)


@torch.no_grad()
def test_absent_slot(base: FluxTransformer2DModel):
    for text_teammate in ["edited.out", None]:
        cfg3 = config(["edited.out", "source.in", "mask.in"], "sequence", text_teammate)
        cfg2 = config(["edited.out", "source.in"], "sequence", text_teammate)
        model3 = sequence_model(base, cfg3)
        state3 = random_state(model3, 5)
        model3 = sequence_model(base, cfg3, state3)
        state2 = {k: v[:2] if v.ndim > 0 else v for k, v in state3.items()}
        model2 = sequence_model(base, cfg2, state2)

        # Element 0 has all three teammates, element 1 has no mask.in (components out of order).
        pairs = [(0, 1), (1, 0), (0, 0), (1, 1), (0, 2)]
        x = inputs(2, len(pairs), seed=6)
        out = model3(**x, components=components(pairs, T=3, B=2, outputs={0})).sample

        def element(b: int, model: SequenceFluxTransformer, T: int) -> tuple[Tensor, list[int]]:
            rows = [i for i, (bb, _) in enumerate(pairs) if bb == b]
            sub = {
                k: (v[rows] if k == "hidden_states" else v[b : b + 1] if k in ("encoder_hidden_states", "pooled_projections", "timestep", "guidance") else v)
                for k, v in x.items()
            }
            comps = components([(0, pairs[i][1]) for i in rows], T=T, B=1, outputs={0})
            return model(**sub, components=comps).sample, rows

        ref0, rows0 = element(0, model3, 3)
        check(f"jagged batch, element 0 alone (text_teammate={text_teammate})", out[rows0], ref0, 1e-4)
        ref1, rows1 = element(1, model2, 2)
        check(f"absent mask.in == roster without it (text_teammate={text_teammate})", out[rows1], ref1, 1e-4)


def test_checkpointing(base: FluxTransformer2DModel):
    cfg = config(["edited.out", "source.in", "mask.in"], "sequence", "edited.out")
    model = sequence_model(base, cfg)
    state = random_state(model, 7)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    x = inputs(2, len(pairs), seed=8)
    comps = components(pairs, T=3, B=2, outputs={0})
    keep = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.bool, device=DEVICE)
    results = []
    for checkpointing in [False, True]:
        model = sequence_model(base, cfg, state)
        for p in adapter_parameters(model).values():
            p.requires_grad_(True)
        if checkpointing:
            model.enable_gradient_checkpointing()
        loss = model(**x, components=comps, attn_keep=keep).sample.square().mean()
        loss.backward()
        grads = torch.cat([p.grad.flatten() for p in adapter_parameters(model).values() if p.grad is not None])
        results.append((loss.detach(), grads))
    check("gradient checkpointing: loss", results[1][0], results[0][0], 1e-5)
    check("gradient checkpointing: adapter grads", results[1][1], results[0][1], 1e-4)


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = tiny_base()
    test_zero_lora_base(base)
    test_single_teammate_matches_batch(base)
    test_absent_slot(base)
    test_checkpointing(base)
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} failed: {FAILURES}")
    print("all FLUX sequence checks passed")


if __name__ == "__main__":
    main()
