"""
Parity smoke test for batch-axis teamwork adapters.

A core teamwork invariant: with every LoRA up-matrix (and bias) at zero, the
adapted transformer must behave like the base transformer. This script checks
that at the transformer level (no VAE / text encoder needed) by running a base
diffusers transformer and its teamwork-adapted copy on identical synthetic
inputs. Attention is untouched, so the LoRA adds exactly zero and the outputs
must match *bit for bit*. The sequence-layout parity checks against base editing
are `test_edit_parity.py` (Klein) and `test_flux_sequence.py` (FLUX.1).

Run from the repo root:

    uv run python scripts/test_parity.py                # all cases
    uv run python scripts/test_parity.py flux2-lora      # one case by name
"""

import sys
import torch

from teamwork.config import TeamworkConfig
from teamwork.adapter import adapt, adapter_modules
from teamwork.batch_linear import BatchSelection
from teamwork.data import Components

# Importing the pipeline modules registers their TEAMWORK_PROFILES.
import teamwork.pipeline_flux  # noqa: F401  registers FLUX
import teamwork.pipeline_flux2_klein  # noqa: F401  registers FLUX2


def single_batch_selection(teammate_indices, num_teammates, device):
    """A selection for one batch element with one output component per teammate."""
    n = len(teammate_indices)
    return BatchSelection.from_components(Components(
        teammate_indices=torch.tensor(teammate_indices, dtype=torch.int64, device=device),
        batch_indices=torch.zeros(n, dtype=torch.int64, device=device),
        input_subindices=torch.empty(0, dtype=torch.int64, device=device),
        output_subindices=torch.arange(n, dtype=torch.int64, device=device),
        num_teammates=num_teammates,
        batch_size=1,
    ))


def make_ids(t_offset, h, w, txt_len, n_axes, device, dtype):
    """Build (img_ids, txt_ids) position-id grids with `n_axes` columns.

    Flux1 uses 3 axes (t, row, col); Flux2 uses 4 (t, row, col, l). Only the
    shape/values need to match between base and adapted, so a plain row/col grid
    is sufficient for the parity check.
    """
    rows = torch.arange(h, device=device)
    cols = torch.arange(w, device=device)
    grid = torch.cartesian_prod(rows, cols)  # (h*w, 2)
    img_ids = torch.zeros(h * w, n_axes, device=device, dtype=dtype)
    img_ids[:, 0] = t_offset
    img_ids[:, 1] = grid[:, 0]
    img_ids[:, 2] = grid[:, 1]
    # axis 3 (Flux2 "l") stays 0

    txt_ids = torch.zeros(txt_len, n_axes, device=device, dtype=dtype)
    if n_axes == 4:
        # Flux2 text ids carry the sequence index in the last axis.
        txt_ids[:, 3] = torch.arange(txt_len, device=device)
    return img_ids, txt_ids


# Per-model description of how to load and feed the transformer.
CASES = {
    "flux1-lora": dict(
        repo="black-forest-labs/FLUX.1-Kontext-dev",
        model_cls="FluxTransformer2DModel",
        profile="FLUX",
        n_axes=3,
        in_channels=64,
        joint_dim=4096,
        pooled_dim=768,
        guidance=3.5,
        teammates=["image.in", "albedo.out"],
        num_components=2,
    ),
    "flux2-lora": dict(
        repo="black-forest-labs/FLUX.2-klein-4B",
        model_cls="Flux2Transformer2DModel",
        profile="FLUX2",
        n_axes=4,
        in_channels=128,
        joint_dim=7680,
        pooled_dim=None,
        guidance=None,
        teammates=["image.in", "albedo.out"],
        num_components=2,
    ),
}


def load_transformer(case, device, dtype):
    if case["model_cls"] == "FluxTransformer2DModel":
        from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel as M
    else:
        from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel as M
    return M.from_pretrained(case["repo"], subfolder="transformer", torch_dtype=dtype).to(device)


def run_case(name, device, base_dtype, h=16, w=16, txt_len=32):
    case = CASES[name]

    # A case may pin fp32; the env override forces fp32 globally.
    dtype = torch.float32 if (case.get("dtype") == "fp32" or base_dtype == torch.float32) else base_dtype

    base = load_transformer(case, device, dtype)
    base.eval()

    cfg = TeamworkConfig(
        teammates=case["teammates"],
        lora_rank=8,
        base_model=case["repo"],
        title="parity-test",
        profile=case["profile"],
    )
    adapted = adapt(base, cfg, device=device, dtype=dtype, requires_grad=False)
    adapted.eval()

    # Belt-and-suspenders: force every up-matrix and bias to zero so the adapter
    # is the identity, even if some state were loaded.
    for m in adapter_modules(adapted).values():
        ad = m.adapter
        if hasattr(ad, "up"):
            torch.nn.init.zeros_(ad.up)
        if getattr(ad, "bias", None) is not None:
            torch.nn.init.zeros_(ad.bias)

    sel = single_batch_selection(
        list(range(case["num_components"])), len(case["teammates"]), device
    )
    for m in adapter_modules(adapted).values():
        m.selection = sel

    B = case["num_components"]
    torch.manual_seed(0)
    hidden = torch.randn(B, h * w, case["in_channels"], device=device, dtype=dtype)
    enc = torch.randn(B, txt_len, case["joint_dim"], device=device, dtype=dtype)
    timestep = torch.full((B,), 0.5, device=device, dtype=dtype)
    img_ids, txt_ids = make_ids(0, h, w, txt_len, case["n_axes"], device, dtype)

    kwargs = dict(
        hidden_states=hidden,
        encoder_hidden_states=enc,
        timestep=timestep,
        img_ids=img_ids,
        txt_ids=txt_ids,
        return_dict=False,
    )
    if case["pooled_dim"] is not None:
        kwargs["pooled_projections"] = torch.randn(
            B, case["pooled_dim"], device=device, dtype=dtype
        )
    if case["guidance"] is not None:
        kwargs["guidance"] = torch.full((B,), case["guidance"], device=device, dtype=dtype)

    with torch.no_grad():
        out_base = base(**kwargs)[0]
        out_adapted = adapted(**kwargs)[0]

    diff = (out_base.float() - out_adapted.float()).abs().max().item()
    scale = out_base.float().abs().max().item()
    ok = diff == 0.0
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} ({case['profile']}, exact): "
          f"max-abs-diff={diff:.3e}, out-scale={scale:.3e}")

    del base, adapted, out_base, out_adapted, hidden, enc, kwargs
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ok


def main():
    if not torch.cuda.is_available():
        print("the parity cases need a CUDA GPU; skipping.")
        sys.exit(0)
    device = torch.device("cuda")
    # bf16 fits the big Flux1 (12B) on one GPU and keeps the adapter/model dtypes
    # matched so the lora-only round-trip cast is a no-op (stays bit-exact). Pass
    # TEAMWORK_PARITY_FP32=1 to run in fp32.
    import os
    dtype = torch.float32 if os.environ.get("TEAMWORK_PARITY_FP32") == "1" else torch.bfloat16

    names = sys.argv[1:] or list(CASES.keys())
    results = {}
    for name in names:
        if name not in CASES:
            print(f"unknown case {name!r}; known: {', '.join(CASES)}")
            sys.exit(2)
        results[name] = run_case(name, device, dtype)

    failed = [n for n, ok in results.items() if ok is False]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
