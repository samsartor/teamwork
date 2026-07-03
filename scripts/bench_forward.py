"""Time the FLUX.2-klein transformer forward at sd.cpp's 1024px seq length.

base:  plain Flux2Transformer2DModel, seq = text(512) + T*L_img
tw:    teamwork-adapted, T=3 teammates, offsets 0/10/20 (overpainting layout)

This isolates the per-denoise-step transformer cost, which is what sd.cpp's
s/it measures (Euler, cfg=1 -> one forward per step).
"""
import sys, time
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from teamwork.config import TeamworkConfig
from teamwork.adapter import adapt, adapter_modules
from teamwork.batch import Selection
import teamwork.pipeline_flux2_klein  # noqa: F401

REPO = "black-forest-labs/FLUX.2-klein-4B"
IN_CHANNELS = 128
JOINT_DIM = 7680


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


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hh = ww = 64          # 1024px -> 128 latent -> 64x64 patchified
    L_img = hh * ww       # 4096
    L_text = 512
    T = 3

    base = Flux2Transformer2DModel.from_pretrained(
        REPO, subfolder="transformer", torch_dtype=dtype
    ).to(device).eval()

    torch.manual_seed(0)
    text = torch.randn(1, L_text, JOINT_DIM, device=device, dtype=dtype)

    # ---- base forward: [gen(T=0), ref1(T=10), ref2(T=20)] one sequence ----
    hidden_b = torch.randn(1, T * L_img, IN_CHANNELS, device=device, dtype=dtype)
    ids_b = torch.cat([img_ids(hh, ww, k * 10, device) for k in range(T)], dim=0)

    def run_base():
        with torch.no_grad():
            base(hidden_states=hidden_b, encoder_hidden_states=text,
                 timestep=torch.full((1,), 0.5, device=device, dtype=dtype),
                 img_ids=ids_b, txt_ids=txt_ids(L_text, device),
                 guidance=None, return_dict=False)

    t_base = timeit(run_base)
    print(f"[torch] base forward  (seq={L_text + T*L_img}): {t_base*1000:7.1f} ms/it")

    # ---- teamwork-adapted forward ----
    cfg = TeamworkConfig(
        teammates=["edited.out", "source.in", "mask.in"], lora_rank=64,
        base_model=REPO, title="bench", profile="FLUX2_PLUSATTN",
    )
    adapted = adapt(base, cfg, device=device, dtype=dtype, requires_grad=False).eval()

    teammate_indices = [0, 1, 2]
    sel = Selection(
        enabled=True,
        teammate_indices=torch.tensor(teammate_indices, dtype=torch.int64, device=device),
        input_subindices=torch.tensor([1, 2], dtype=torch.int64, device=device),
        output_subindices=torch.tensor([0], dtype=torch.int64, device=device),
        batch_indices=torch.zeros(T, dtype=torch.int64, device=device),
        batch_matrix=torch.ones(T, 1, dtype=torch.bool, device=device),
        num_teammates=T,
        present_matrix=torch.ones(1, T, dtype=torch.bool, device=device),
        first_component_per_batch=torch.zeros(1, dtype=torch.int64, device=device),
        attn_keep=None,
    )
    for m in adapter_modules(adapted).values():
        m.selection = sel
        if hasattr(m, "text_seq_len"):
            m.text_seq_len = L_text

    hidden_tw = torch.randn(T, L_img, IN_CHANNELS, device=device, dtype=dtype)
    ids_tw = torch.cat([img_ids(hh, ww, k * 10, device) for k in range(T)], dim=0)
    text_tw = text.repeat(T, 1, 1)

    def run_tw():
        with torch.no_grad():
            adapted(hidden_states=hidden_tw, encoder_hidden_states=text_tw,
                    timestep=torch.full((T,), 0.5, device=device, dtype=dtype),
                    img_ids=ids_tw, txt_ids=txt_ids(L_text, device),
                    guidance=None, return_dict=False)

    try:
        t_tw = timeit(run_tw)
        print(f"[torch] teamwork fwd  (T={T}): {t_tw*1000:7.1f} ms/it")
    except Exception as e:
        print(f"[torch] teamwork fwd FAILED: {e}")


if __name__ == "__main__":
    main()
