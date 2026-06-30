"""
Editing-parity test for FLUX.2 klein joint attention.

Teamwork with cross-teammate attention is supposed to reduce to the *base editing
model* when the LoRA up-matrices are zero: an output teammate that attends to an
input (reference) teammate should make the same prediction the base model makes
when given that reference as an edit condition. The plain parity test only covers
a single teammate (which trivially equals base self-attention); this one covers
the real 2-teammate editing setup (one input + one output).

The comparison is at the transformer level. Base editing puts the generated image
at RoPE T-coord 0 and the reference at T=10 (Flux2 `_prepare_image_ids`, scale=10),
concatenated in one sequence. Teamwork puts each teammate's image at
`T += teammate * offset` and lets the output teammate attend to the input teammate
via the masked joint attention. With zero LoRA the two should agree iff the
reference lands on the T-coord the base model expects -- so we sweep `offset` to
localize the issue and confirm the fix.

Run from the repo root:

    uv run python scripts/test_edit_parity.py
"""

import sys
import torch

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

from teamwork.config import TeamworkConfig
from teamwork.adapter import adapt, adapter_modules
from teamwork.batch import Selection
from teamwork.pipeline_flux2_klein import klein_teammate_t_offsets  # registers FLUX2 / FLUX2_PLUSATTN

REPO = "black-forest-labs/FLUX.2-klein-4B"  # same architecture as 9B, faster to load
IN_CHANNELS = 128
JOINT_DIM = 7680
KLEIN_REF_SCALE = 10  # Flux2 _prepare_image_ids: reference images sit at T = 10, 20, ...


def img_ids(hh, ww, t, device):
    ids = torch.zeros(hh * ww, 4, device=device, dtype=torch.float32)
    ids[:, 0] = t
    ids[:, 1] = torch.arange(hh, device=device).repeat_interleave(ww)
    ids[:, 2] = torch.arange(ww, device=device).repeat(hh)
    return ids


def txt_ids(l, device):
    ids = torch.zeros(l, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(l, device=device)
    return ids


def base_edit_prediction(transformer, gen, ref, text, hh, ww, device):
    """Base editing: [gen(T=0), ref(T=10)] in one sequence; return gen prediction."""
    L_img = hh * ww
    hidden = torch.cat([gen, ref], dim=1)  # (1, 2*L_img, C)
    ids = torch.cat([img_ids(hh, ww, 0, device), img_ids(hh, ww, KLEIN_REF_SCALE, device)], dim=0)
    out = transformer(
        hidden_states=hidden,
        encoder_hidden_states=text,
        timestep=torch.full((1,), 0.5, device=device, dtype=torch.float32),
        img_ids=ids,
        txt_ids=txt_ids(text.shape[1], device),
        guidance=None,
        return_dict=False,
    )[0]
    return out[:, :L_img]


def teamwork_edit_prediction(adapted, gen, ref, text, hh, ww, offsets, device):
    """Zero-LoRA teamwork with teammates [edited.out(0), source.in(1)].

    Components: comp0 = source.in (input/ref, teammate 1), comp1 = edited.out
    (output/gen, teammate 0), mirroring BatchBuilder (provide-before-request).
    `offsets[k]` is teammate k's RoPE T-coord. Returns the edited.out prediction.
    """
    L_img = hh * ww
    L_text = text.shape[1]
    T = 2

    # comp0 = source.in (teammate 1), comp1 = edited.out (teammate 0)
    hidden = torch.stack([ref.squeeze(0), gen.squeeze(0)], dim=0)  # (2, L_img, C)
    teammate_indices = [1, 0]

    sel = Selection(
        enabled=True,
        teammate_indices=torch.tensor(teammate_indices, dtype=torch.int64, device=device),
        input_subindices=torch.tensor([0], dtype=torch.int64, device=device),
        output_subindices=torch.tensor([1], dtype=torch.int64, device=device),
        batch_indices=torch.zeros(2, dtype=torch.int64, device=device),
        batch_matrix=torch.ones(2, 1, dtype=torch.bool, device=device),
        num_teammates=T,
        present_matrix=torch.ones(1, T, dtype=torch.bool, device=device),
        first_component_per_batch=torch.zeros(1, dtype=torch.int64, device=device),
        attn_keep=None,
    )
    for m in adapter_modules(adapted).values():
        m.selection = sel
        if hasattr(m, "text_seq_len"):
            m.text_seq_len = L_text

    # Per-teammate image ids: teammate k -> T = offsets[k].
    ids = torch.cat(
        [img_ids(hh, ww, offsets[k], device) for k in range(T)], dim=0
    )  # (T*L_img, 4)
    text2 = text.repeat(2, 1, 1)

    out = adapted(
        hidden_states=hidden,
        encoder_hidden_states=text2,
        timestep=torch.full((2,), 0.5, device=device, dtype=torch.float32),
        img_ids=ids,
        txt_ids=txt_ids(L_text, device),
        guidance=None,
        return_dict=False,
    )[0]
    # comp1 is edited.out (the output teammate)
    return out[1:2]


def main():
    if not torch.cuda.is_available():
        print("teamwork attention requires CUDA; skipping.")
        sys.exit(0)
    device = torch.device("cuda")
    dtype = torch.float32
    hh = ww = 16
    L_img = hh * ww
    L_text = 32

    base = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=dtype
    ).to(device).eval()

    cfg = TeamworkConfig(
        teammates=["edited.out", "source.in"], lora_rank=8,
        base_model=REPO, title="edit-parity", profile="FLUX2_PLUSATTN",
    )
    adapted = adapt(base, cfg, device=device, dtype=dtype, requires_grad=False).eval()
    for m in adapter_modules(adapted).values():
        if hasattr(m.adapter, "up"):
            torch.nn.init.zeros_(m.adapter.up)
        if getattr(m.adapter, "bias", None) is not None:
            torch.nn.init.zeros_(m.adapter.bias)

    torch.manual_seed(0)
    gen = torch.randn(1, L_img, IN_CHANNELS, device=device, dtype=dtype)
    ref = torch.randn(1, L_img, IN_CHANNELS, device=device, dtype=dtype)
    text = torch.randn(1, L_text, JOINT_DIM, device=device, dtype=dtype)

    with torch.no_grad():
        ref_pred = base_edit_prediction(base, gen, ref, text, hh, ww, device)
        scale = ref_pred.abs().max().item()
        tol = 1e-3 * max(scale, 1.0)
        print(f"base editing prediction scale = {scale:.3e}; tol = {tol:.2e}\n")

        # The pipeline's role-based offsets are the real regression gate; the naive
        # (0, 1, ...) offsets are kept to document why they don't work.
        pipeline_offsets = klein_teammate_t_offsets(cfg.teammates)
        cases = [
            ([0, 1], "naive (0, 1)"),
            (pipeline_offsets, f"pipeline {tuple(pipeline_offsets)}"),
        ]
        for offsets, label in cases:
            tw_pred = teamwork_edit_prediction(adapted, gen, ref, text, hh, ww, offsets, device)
            diff = (ref_pred.float() - tw_pred.float()).abs().max().item()
            status = "PASS" if diff <= tol else "FAIL"
            print(f"[{status}] offsets={label}: max-abs-diff vs base editing = {diff:.3e}")


if __name__ == "__main__":
    main()
