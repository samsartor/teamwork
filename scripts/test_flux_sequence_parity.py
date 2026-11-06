"""
Real-weight zero-LoRA parity of `SequenceFluxPipeline` against base FLUX.1 Kontext (bf16, CUDA).

With roster `(edited.out, source.in)`, `teammate_image_ids=(0, 1)`, and zero LoRA, the sequence
stream `[text | edited | source]` is Kontext's `[text | target | context]`, with the context at
RoPE index 1, so outputs must match up to bf16 / attention-kernel noise.

    python scripts/test_flux_sequence_parity.py transformer [--lora-checkpoint X.safetensors]
    python scripts/test_flux_sequence_parity.py pipeline --image in.png --prompt "..." [--strip out.png]

`transformer` compares one forward on random inputs. With `--lora-checkpoint`, it also reports
how far a trained LoRA (that checkpoint's `edited.out` and `source.in` slices) moves the output,
for scale. `pipeline` runs a whole edit with both pipelines from the same noise and reports the
PSNR. Kontext encodes the context image with the posterior mode, while teamwork samples it; the
check uses the mode for both (and also reports the sampled PSNR).
"""

from dataclasses import replace
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from PIL import Image
from torch import Tensor
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline

from teamwork.adapter import TEAMWORK_PROFILES, adapt, adapter_parameters
from teamwork.checkpoint import load_safetensors
from teamwork.config import TeamworkConfig
from teamwork.data import Components
from teamwork import family_flux
from teamwork.pipeline_flux import SequenceFluxPipeline, SequenceFluxTransformer, sequence_flux_profile

app = typer.Typer(pretty_exceptions_show_locals=False)
REPO = "black-forest-labs/FLUX.1-Kontext-dev"
ROSTER = ["edited.out", "source.in"]


def config() -> TeamworkConfig:
    cfg = TeamworkConfig(
        teammates=ROSTER, lora_rank=64, base_model=REPO, title="parity", profile="FLUX",
        layout="sequence", text_teammate="edited.out",
    )
    assert cfg.teammate_image_ids == [0, 1], cfg.teammate_image_ids
    return cfg


def roster_slices(path: Path) -> tuple[TeamworkConfig, dict[str, Tensor]]:
    """`config()` with the checkpoint's LoRA hyperparameters, and its adapter state restricted to `ROSTER`."""
    ckpt_cfg, state, _ = load_safetensors(path)
    index = torch.tensor([ckpt_cfg.teammates.index(t) for t in ROSTER])
    cfg = replace(
        config(), lora_rank=ckpt_cfg.lora_rank, lora_communication=ckpt_cfg.lora_communication,
        use_bias=ckpt_cfg.use_bias,
    )
    return cfg, {k: v[index] for k, v in state.items() if v.ndim > 0}


def report(name: str, value: Tensor, reference: Tensor):
    value, reference = value.float(), reference.float()
    diff = (value - reference).abs()
    rel = (diff.norm() / reference.norm()).item()
    print(
        f"{name}: max-abs={diff.max().item():.4e} mean-abs={diff.mean().item():.4e} "
        f"rel-l2={rel:.4e} (ref max-abs={reference.abs().max().item():.3e})"
    )
    return rel


@app.command()
@torch.no_grad()
def transformer(
    lora_checkpoint: Annotated[Path | None, typer.Option()] = None,
    resolution: Annotated[int, typer.Option()] = 1024,
    max_rel: Annotated[float, typer.Option(help="fail above this relative L2 difference")] = 2e-2,
):
    device = torch.device("cuda")
    base = FluxTransformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=torch.bfloat16).to(device)
    base.requires_grad_(False)
    cfg = config()

    lh = lw = resolution // 8
    L = (lh // 2) * (lw // 2)
    g = torch.Generator(device).manual_seed(0)
    bf = dict(device=device, dtype=torch.bfloat16)
    target = torch.randn(1, L, 64, generator=g, **bf)
    context = torch.randn(1, L, 64, generator=g, **bf)
    text = torch.randn(1, 512, 4096, generator=g, **bf)
    pooled = torch.randn(1, 768, generator=g, **bf)
    timestep = torch.full((1,), 0.6, **bf)
    guidance = torch.full((1,), 2.5, **bf)
    img_ids = family_flux.latent_image_ids(lh, lw, device, torch.bfloat16)
    txt_ids = torch.zeros(512, 3, **bf)

    reference = base(
        hidden_states=torch.cat([target, context], dim=1),
        encoder_hidden_states=text, pooled_projections=pooled, timestep=timestep, guidance=guidance,
        img_ids=family_flux.offset_image_ids(img_ids, [0, 1]), txt_ids=txt_ids,
    ).sample
    ref_target, ref_context = reference[:, :L], reference[:, L:]

    idx = lambda xs: torch.tensor(xs, dtype=torch.int64, device=device)
    components = Components(
        teammate_indices=idx([0, 1]), batch_indices=idx([0, 0]), input_subindices=idx([1]),
        output_subindices=idx([0]), num_teammates=2, batch_size=1,
    )

    def sequence(state: dict[str, Tensor] | None) -> Tensor:
        adapted = adapt(
            base, cfg, device=device, dtype=torch.bfloat16, state=state, infer_layers_from_state=False,
            override_profile=sequence_flux_profile(TEAMWORK_PROFILES["FLUX"]),
        )
        assert isinstance(adapted, FluxTransformer2DModel)
        if state is None:
            assert all(not p.any() for k, p in adapter_parameters(adapted).items() if not k.endswith(".down"))
        model = SequenceFluxTransformer.from_adapted(adapted, cfg)
        return model(
            hidden_states=torch.cat([target, context], dim=0),
            encoder_hidden_states=text, pooled_projections=pooled, timestep=timestep, guidance=guidance,
            img_ids=img_ids, txt_ids=txt_ids, components=components,
        ).sample

    out = sequence(None)
    rel = report("zero LoRA vs base Kontext, target", out[0:1], ref_target)
    report("zero LoRA vs base Kontext, context", out[1:2], ref_context)
    if lora_checkpoint is not None:
        cfg, state = roster_slices(lora_checkpoint)
        trained = sequence(state)
        report("trained LoRA vs base Kontext, target (for scale)", trained[0:1], ref_target)
    if rel > max_rel:
        raise SystemExit(f"FAIL: rel-l2 {rel:.3e} > {max_rel}")
    print("PASS")


def psnr(a: Image.Image, b: Image.Image) -> float:
    x = np.asarray(a.convert("RGB"), dtype=np.float64)
    y = np.asarray(b.convert("RGB"), dtype=np.float64)
    mse = float(np.mean((x - y) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


@app.command()
@torch.no_grad()
def pipeline(
    image: Annotated[Path, typer.Option()],
    prompt: Annotated[str, typer.Option()],
    steps: Annotated[int, typer.Option()] = 28,
    guidance: Annotated[float, typer.Option()] = 2.5,
    seed: Annotated[int, typer.Option()] = 0,
    strip: Annotated[Path | None, typer.Option()] = None,
    min_psnr: Annotated[float, typer.Option()] = 30.0,
):
    device = torch.device("cuda")
    source = Image.open(image).convert("RGB")
    width, height = source.size
    assert width % 16 == 0 and height % 16 == 0, source.size
    base = FluxKontextPipeline.from_pretrained(REPO, torch_dtype=torch.bfloat16).to(device)
    base.set_progress_bar_config(disable=True)

    lh, lw = height // 8, width // 8
    noise = torch.randn(1, 16, lh, lw, generator=torch.Generator(device).manual_seed(seed), device=device, dtype=torch.bfloat16)
    reference = base(
        image=source, prompt=prompt, height=height, width=width, guidance_scale=guidance,
        num_inference_steps=steps, latents=FluxKontextPipeline._pack_latents(noise, 1, 16, lh, lw),
        _auto_resize=False,
    ).images[0]

    pipe = SequenceFluxPipeline.from_base_pipeline(base, config())
    pipe.set_progress_bar_config(disable=True)
    sampled_encode = pipe.vae_encode

    def mode_encode(x: Tensor, generator: torch.Generator | None = None) -> Tensor:
        latents = pipe.vae.encode(x.to(pipe.vae.device, pipe.vae.dtype)).latent_dist.mode()
        return (latents - pipe.vae.config["shift_factor"]) * pipe.vae.config["scaling_factor"]

    results = {}
    for name, encode in [("mode", mode_encode), ("sampled", sampled_encode)]:
        pipe.vae_encode = encode  # type: ignore[method-assign]
        # Components are [source.in, edited.out]; only the output's noise matters.
        out = pipe(
            {"source": source}, prompt=prompt, num_inference_steps=steps, guidance_scale=guidance,
            noise=torch.cat([torch.zeros_like(noise), noise], dim=0),
            generator=torch.Generator(device).manual_seed(seed),
        )["edited"]
        results[name] = out
        print(f"pipeline zero LoRA vs FluxKontextPipeline ({name} context latents): PSNR {psnr(out, reference):.2f} dB")

    if strip is not None:
        tiles = [source, reference, results["mode"], results["sampled"]]
        canvas = Image.new("RGB", (width * len(tiles), height))
        for i, tile in enumerate(tiles):
            canvas.paste(tile.resize((width, height)), (i * width, 0))
        canvas.save(strip)
        print(f"strip (source | Kontext | teamwork mode | teamwork sampled): {strip}")
    if psnr(results["mode"], reference) < min_psnr:
        raise SystemExit(f"FAIL: PSNR below {min_psnr} dB")
    print("PASS")


if __name__ == "__main__":
    app()
