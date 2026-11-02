"""
End-to-end editing parity: zero-LoRA teamwork pipeline vs the base diffusers editing pipeline.

`test_edit_parity.py` checks the transformer alone. This one runs both full pipelines
(prompt encoding, VAE encode/normalization, sampling schedule, decode) from the same
initial noise and compares the final images, so it also catches latent-space and
schedule mismatches. With every LoRA up-matrix at zero, the sequence layout with teammates
[edited.out, source.in] must reproduce base editing of `source`.

Run from the repo root:

    uv run python scripts/test_pipeline_parity.py           # all cases
    uv run python scripts/test_pipeline_parity.py klein     # one case
"""

import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from teamwork.data import TeamBuilder
from teamwork.config import TeamworkConfig

DEMO = Path(__file__).parent.parent / "demo" / "pink_input.png"
PROMPT = "make the sphere bright green"
MIN_PSNR = 35.0  # bf16 flex-attention vs sdpa differences stay well above this


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    return float(10 * np.log10(255**2 / max(((x - y) ** 2).mean(), 1e-12)))


def teamwork_edit(pipe, cfg: TeamworkConfig, src: Image.Image, edit_noise: torch.Tensor, **kwargs) -> Image.Image:
    """Run the teamwork pipeline with `edit_noise` as the edited teammate's initial noise."""
    device = torch.device("cuda")
    batch = TeamBuilder.from_inputs(
        cfg.teammates, device, torch.bfloat16, images={"source": src}, attn_allow=cfg.attn_allow
    )
    noise = torch.zeros(batch.count, *edit_noise.shape, device=device, dtype=torch.bfloat16)
    noise[[c.component for c in batch.components].index("edited")] = edit_noise
    return pipe({"source": src}, prompt=PROMPT, noise=noise, batch=batch, **kwargs)["edited"]


def case_klein() -> bool:
    from diffusers import Flux2KleinPipeline
    from teamwork.pipeline_flux2_klein import SequenceFlux2KleinPipeline

    repo, steps, res = "black-forest-labs/FLUX.2-klein-9B", 4, 512
    base = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16).to("cuda")
    src = Image.open(DEMO).convert("RGB").resize((res, res))
    g = torch.Generator("cuda").manual_seed(0)
    edit_noise = torch.randn(32, res // 8, res // 8, device="cuda", dtype=torch.bfloat16, generator=g)
    base_noise = rearrange(edit_noise, "c (h ph) (w pw) -> 1 (c ph pw) h w", ph=2, pw=2)
    ref = base(image=src, prompt=PROMPT, num_inference_steps=steps, guidance_scale=1.0, latents=base_noise).images[0]

    cfg = TeamworkConfig(
        teammates=["edited.out", "source.in"], lora_rank=4, base_model=repo, title="parity", profile="FLUX2",
        layout="sequence", text_teammate="edited.out",
    )
    pipe = SequenceFlux2KleinPipeline.from_base_pipeline(base, cfg)
    value = psnr(teamwork_edit(pipe, cfg, src, edit_noise, num_inference_steps=steps), ref)
    ok = value >= MIN_PSNR
    print(f"[{'PASS' if ok else 'FAIL'}] klein: zero-LoRA teamwork vs base editing PSNR = {value:.1f} dB")
    return ok


CASES = {"klein": case_klein}


def main():
    if not torch.cuda.is_available():
        print("teamwork attention requires CUDA; skipping.")
        sys.exit(0)
    names = sys.argv[1:] or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"unknown cases {unknown}; known: {', '.join(CASES)}")
        sys.exit(2)
    results = {}
    for name in names:
        results[name] = CASES[name]()
        torch.cuda.empty_cache()
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
