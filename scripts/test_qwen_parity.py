"""
Zero-LoRA parity of the Qwen-Image 2.1 teamwork pipelines against the base model.

    python scripts/test_qwen_parity.py tiny                 # tiny random model, fp32, no weights
    python scripts/test_qwen_parity.py real [--resolution 512] [--steps 20]
    python scripts/test_qwen_parity.py train [--layout sequence|batch] [--resolution 1024] [--text-teammate edited.out]

`tiny`: transformer outputs of both layouts vs `QwenImage21Transformer2DModel.forward` (base ti2i,
and base t2i with a right-padded batch of two prompts). Expected ~1e-7 (flex vs SDPA in fp32).

`real` (bf16, `Qwen/Qwen-Image-2.1`):
  (a) batch, roster (edited.out, source.in, mask.in), vlm_inputs=[source.in] vs base ti2i
  (b) batch, roster (edited.out, mask.in), vlm_inputs=[] vs base t2i
  (c) sequence, roster (edited.out, source.in), vlm_inputs=[source.in] vs base ti2i
For each: the first step's transformer output (max-abs / relative diff), and the full `__call__`
PSNR against `QwenImage21Pipeline` (no KV cache) from the same noise. For scale, it also reports
base KV-cache-on vs -off (bf16 rounding only), the base model's own flex vs SDPA attention
processors at step 0 (the bf16 kernel noise floor), and the effect of a small random LoRA.

`train`: a training smoke with roster (edited.out, source.in, mask.in), vlm_inputs=[source.in].
"""

import math
from typing import Annotated, Any

import numpy as np
import PIL.Image
import torch
from torch import Tensor
from torch._dynamo.utils import counters as dynamo_counters
import typer
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import QwenImage21Pipeline
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21
from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
    QwenImage21Transformer2DModel,
)
from transformers import Qwen3VLProcessor

from teamwork.adapter import adapter_parameters
from teamwork.config import TeamworkConfig
from teamwork.data import TeamBuilder
from teamwork.pipelines import automatic_pipeline
from teamwork.pipeline_qwenimage21 import (
    BatchQwenImage21Pipeline,
    QwenImage21TeamworkPipeline,
    SequenceQwenImage21Pipeline,
)
from teamwork import family_qwen21
from teamwork.family_qwen21 import QwenPrompt, pack, unpack

app = typer.Typer(pretty_exceptions_show_locals=False)
MODEL = "Qwen/Qwen-Image-2.1"
PROMPT = "Turn the sky into a vivid orange sunset and add a small red kite."
T2I_PROMPT = "A red kite flying over a green hill under a blue sky, photograph."


def teamwork_pipe(base: Any, roster: list[str], layout: str, vlm_inputs: tuple[str, ...]) -> QwenImage21TeamworkPipeline:
    cfg = TeamworkConfig(
        teammates=roster, lora_rank=16, base_model=MODEL, title="parity", profile="QWEN21",
        layout=layout, vlm_inputs=vlm_inputs,  # type: ignore[arg-type]
    )
    pipe = automatic_pipeline(base, cfg).from_base_pipeline(base, cfg)
    assert isinstance(pipe, QwenImage21TeamworkPipeline)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def diff_report(name: str, ours: Tensor, ref: Tensor) -> str:
    d = (ours.float() - ref.float()).abs()
    rel = d.norm() / ref.float().norm()
    return f"{name}: max-abs {d.max().item():.3e}, mean-abs {d.mean().item():.3e}, rel-L2 {rel.item():.3e} (ref max-abs {ref.abs().max().item():.3e})"


def psnr(a: Tensor, b: Tensor) -> float:
    """Images in [0, 1]."""
    mse = (a.float() - b.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


# --------------------------------------------------------------------------------------
# tiny
# --------------------------------------------------------------------------------------


@app.command()
def tiny():
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    Z, res = 8, 64
    tr = QwenImage21Transformer2DModel(
        in_channels=Z, out_channels=Z, num_layers=2, attention_head_dim=16, num_attention_heads=2,
        context_in_dim=32, axes_dims_rope=(4, 6, 6),
    )
    for p in tr.parameters():
        p.data = torch.randn(p.shape, generator=g) / math.sqrt(p.shape[-1]) if p.ndim >= 2 else 0.1 * torch.randn(p.shape, generator=g)
    vae = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=8, z_dim=Z, num_res_blocks=1, latents_mean=[0.1] * Z, latents_std=[1.5] * Z)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL, subfolder="scheduler")
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    base = QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae, text_encoder=None,  # type: ignore[arg-type]
        processor=Qwen3VLProcessor.from_pretrained(MODEL, subfolder="processor"), transformer=tr,
    ).to(dev)
    lh = lw = res // 16
    L = lh * lw
    failures = []

    def check(name: str, ours: Tensor, ref: Tensor, tol: float = 1e-5):
        print(diff_report(name, ours, ref))
        if (ours - ref).abs().max().item() > tol:
            failures.append(name)

    # ti2i: [3 text][source slots][5 text][target]
    S = 3 + L // 4 + 5
    pad = torch.zeros(S, dtype=torch.bool, device=dev)
    pad[3 : 3 + L // 4] = True
    hidden = torch.randn(1, S, 32, generator=g).to(dev)
    prompt = QwenPrompt(hidden, torch.ones(1, S, dtype=torch.bool, device=dev), pad)
    src = torch.rand(3, res, res, generator=g) * 2 - 1
    noise = torch.randn(1, Z, lh, lw, generator=g).to(dev)
    with torch.no_grad():
        src_latents = family_qwen21.vae_encode(vae, src[None].to(dev))
        ref = tr(
            hidden_states=torch.cat([pack(src_latents), pack(noise)], 1), encoder_hidden_states=hidden,
            timestep=torch.tensor([0.6], device=dev), img_shapes=[[(1, lh, lw)] * 2],
            img_mask=torch.cat([pad, pad.new_ones(L // 4)])[None],
        ).sample[:, -L:]
    ref = unpack(ref, lh, lw)
    for layout, roster in [("batch", ["edited.out", "source.in", "mask.in"]), ("sequence", ["edited.out", "source.in"])]:
        pipe = teamwork_pipe(base, roster, layout, ("source.in",))
        for p in pipe.unwrapped_transformer.parameters():
            p.data = p.data.float()
        batch = TeamBuilder(roster, dev, torch.float32)
        batch.provide("source", src)
        if "mask.in" in roster:
            batch.provide("mask", torch.rand(3, res, res, generator=g) * 2 - 1)
        batch.request("edited")
        components = batch.components_index()
        edited = int(components.output_subindices[0])
        with torch.no_grad():
            latents = batch.packed_encoded_images(pipe.vae_encode, Z, 1 / 16)
            latents[edited] = noise[0]
            ours = pipe.transformer(
                latents, torch.full((batch.count,), 0.6, device=dev), components, prompt, pipe.team,
            ).sample[edited : edited + 1]
        check(f"tiny ti2i {layout} {roster}", ours, ref)

    # t2i, B=2, element 1 right-padded
    S = 9
    hidden = torch.randn(2, S, 32, generator=g).to(dev)
    valid = torch.ones(2, S, dtype=torch.bool, device=dev)
    valid[1, 6:] = False
    prompt = QwenPrompt(hidden, valid, torch.zeros(S, dtype=torch.bool, device=dev))
    noise = torch.randn(2, Z, lh, lw, generator=g).to(dev)
    timesteps = torch.tensor([0.6, 0.3], device=dev)
    with torch.no_grad():
        ref = tr(
            hidden_states=pack(noise), encoder_hidden_states=hidden, encoder_hidden_states_mask=valid,
            timestep=timesteps, img_shapes=[[(1, lh, lw)]] * 2,
            img_mask=torch.cat([valid.new_zeros(2, S), valid.new_ones(2, L // 4)], 1),
        ).sample[:, -L:]
    ref = unpack(ref, lh, lw)
    for layout, roster in [("batch", ["edited.out", "mask.in"]), ("sequence", ["edited.out"])]:
        pipe = teamwork_pipe(base, roster, layout, ())
        for p in pipe.unwrapped_transformer.parameters():
            p.data = p.data.float()
        batch = TeamBuilder(roster, dev, torch.float32)
        for b in range(2):
            batch.request("edited", width=res, batch_idx=b)
        with torch.no_grad():
            ours = pipe.transformer(noise, timesteps, batch.components_index(), prompt, pipe.team).sample
        check(f"tiny t2i B=2 padded {layout} {roster}", ours, ref)

    if failures:
        print(f"FAIL: {failures}")
        raise typer.Exit(1)
    print("all tiny parity checks pass")


# --------------------------------------------------------------------------------------
# real
# --------------------------------------------------------------------------------------


def test_image(res: int) -> PIL.Image.Image:
    """A deterministic photo-like test image (gradients and shapes)."""
    y, x = np.mgrid[0:res, 0:res] / res
    img = np.stack([0.3 + 0.5 * y, 0.5 + 0.3 * np.sin(6 * x), 0.8 - 0.4 * y], axis=-1)
    img[(x - 0.5) ** 2 + (y - 0.6) ** 2 < 0.04] = [0.2, 0.6, 0.2]
    img[int(0.7 * res) :, :] *= 0.6
    return PIL.Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))


def mask_image(res: int) -> PIL.Image.Image:
    m = np.zeros((res, res, 3), dtype=np.uint8)
    m[: res // 2] = 255
    return PIL.Image.fromarray(m)


@torch.no_grad()
def base_prompt(base: QwenImage21Pipeline, prompt: str, image: PIL.Image.Image | None) -> tuple[Tensor, Tensor]:
    """Base `encode_prompt` output for one prompt: `(embeds, image_pad_mask)`."""
    images = None if image is None else [image.convert("RGBA")]
    embeds, mask, pad = base.encode_prompt(prompt=prompt, image=images)  # type: ignore[arg-type]
    assert mask is None
    return embeds, pad


@torch.no_grad()
def run_case(
    base: QwenImage21Pipeline,
    name: str,
    layout: str,
    roster: list[str],
    vlm_inputs: tuple[str, ...],
    images: dict[str, PIL.Image.Image],
    prompt: str,
    res: int,
    steps: int,
    seed: int,
):
    device = base._execution_device
    pipe = teamwork_pipe(base, roster, layout, vlm_inputs)
    source = images.get("source") if vlm_inputs else None
    lh = lw = res // 16
    L = lh * lw

    noise = torch.randn(1, 64, lh, lw, generator=torch.Generator().manual_seed(seed)).to(device, torch.bfloat16)

    # The teamwork batch at the first step.
    batch = TeamBuilder(roster, device, torch.bfloat16)
    for k, img in images.items():
        batch.provide(k, img)
    batch.request("edited", width=res)
    components = batch.components_index()
    edited = int(components.output_subindices[0])
    encoded = pipe.encode_team_prompt(batch, prompt)

    # Prompt encoding is the base pipeline's own.
    ref_embeds, ref_pad = base_prompt(base, prompt, source)
    assert torch.equal(encoded.hidden, ref_embeds) and torch.equal(encoded.image_pad, ref_pad[0]), f"{name}: prompt encoding differs"

    timesteps, _ = family_qwen21.inference_timesteps(base.scheduler, steps, L, device, None)
    t = timesteps[0].expand(1).to(torch.bfloat16) / 1000
    clean = batch.packed_encoded_images(pipe.vae_encode, 64, 1 / 16)
    latents = clean.clone()
    latents[edited] = noise[0]
    ours = pipe.transformer(latents, t.expand(batch.count), components, encoded, pipe.team).sample[edited : edited + 1]

    cond = [] if source is None else [pack(family_qwen21.vae_encode(base.vae, batch_image(source, device)))]
    img_mask = torch.cat([ref_pad, ref_pad.new_ones(1, L // 4)], 1)
    def base_step0() -> Tensor:
        out = base.transformer(
            hidden_states=torch.cat([*cond, pack(noise)], 1), encoder_hidden_states=ref_embeds, timestep=t,
            img_shapes=[[(1, lh, lw)] * (len(cond) + 1)], img_mask=img_mask,
        ).sample[:, -L:]
        return unpack(out, lh, lw)

    ref = base_step0()
    print(diff_report(f"({name}) step-0 transformer, zero LoRA", ours, ref))
    # bf16 noise floor: the base model itself with its flex processor instead of multi-pass SDPA.
    base.transformer.set_attn_processor(QwenImage21FlexAttnProcessor())
    print(diff_report(f"({name}) step-0 base flex processor vs base SDPA processor (reference)", base_step0(), ref))
    base.transformer.set_attn_processor(QwenImage21AttnProcessor())

    # A small random LoRA, for scale.
    params = adapter_parameters(pipe.unwrapped_transformer)
    saved = {k: p.detach().clone() for k, p in params.items()}
    g = torch.Generator().manual_seed(7)
    for k, p in params.items():
        if k.endswith(".up"):
            p.data = (0.01 * torch.randn(p.shape, generator=g)).to(p)
    lora = pipe.transformer(latents, t.expand(batch.count), components, encoded, pipe.team).sample[edited : edited + 1]
    print(diff_report(f"({name}) step-0 transformer, random LoRA (up~N(0,0.01²), down at init)", lora, ref))
    for k, p in params.items():
        p.data = saved[k]

    # Full sampling from the same noise.
    count_noise = torch.zeros(batch.count, 64, lh, lw, device=device, dtype=torch.bfloat16)
    count_noise[edited] = noise[0]
    ours_img = pipe(images, prompt=prompt, num_inference_steps=steps, noise=count_noise, output_type="pt", batch=batch)
    assert isinstance(ours_img, dict)
    ours_rgb = (ours_img["edited"].float() + 1) / 2
    base_kwargs: dict[str, Any] = {
        "prompt": prompt, "height": res, "width": res, "num_inference_steps": steps, "latents": pack(noise),
        "output_type": "pt", "output_resolution": res,
    }
    if source is not None:
        base_kwargs["image"] = source
    base.set_progress_bar_config(disable=True)
    ref_nokv = base(**base_kwargs, use_kv_cache=False).images[0, :3].float()  # type: ignore[union-attr]
    ref_kv = base(**base_kwargs, use_kv_cache=True).images[0, :3].float()  # type: ignore[union-attr]
    print(
        f"({name}) {steps}-step __call__ PSNR vs base (no KV cache): {psnr(ours_rgb, ref_nokv):.2f} dB; "
        f"base KV cache on vs off: {psnr(ref_kv, ref_nokv):.2f} dB"
    )
    tag = name.split()[0]
    PIL.Image.fromarray((ours_rgb.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f"/tmp/qwen_parity_{tag}_ours.png")
    PIL.Image.fromarray((ref_nokv.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)).save(f"/tmp/qwen_parity_{tag}_base.png")
    del pipe


def batch_image(image: PIL.Image.Image, device: torch.device) -> Tensor:
    x = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255 * 2 - 1
    return x.permute(2, 0, 1)[None].to(device, torch.bfloat16)


@app.command()
def real(
    resolution: Annotated[int, typer.Option()] = 512,
    steps: Annotated[int, typer.Option()] = 20,
    only: Annotated[str | None, typer.Option(help="a, b, or c")] = None,
):
    base = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda")
    assert isinstance(base, QwenImage21Pipeline)
    source, mask = test_image(resolution), mask_image(resolution)
    cases = {
        "a": ("a batch ti2i", "batch", ["edited.out", "source.in", "mask.in"], ("source.in",), {"source": source, "mask": mask}, PROMPT),
        "b": ("b batch t2i", "batch", ["edited.out", "mask.in"], (), {"mask": mask}, T2I_PROMPT),
        "c": ("c sequence ti2i", "sequence", ["edited.out", "source.in"], ("source.in",), {"source": source}, PROMPT),
    }
    for key, (name, layout, roster, vlm, images, prompt) in cases.items():
        if only is not None and key not in only:
            continue
        run_case(base, name, layout, roster, vlm, images, prompt, resolution, steps, seed=1)
        torch.cuda.empty_cache()
    print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


@app.command()
def train(
    layout: Annotated[str, typer.Option()] = "sequence",
    resolution: Annotated[int, typer.Option()] = 1024,
    steps: Annotated[int, typer.Option()] = 3,
    text_teammate: Annotated[str | None, typer.Option()] = None,
    attn_dropout: Annotated[float, typer.Option()] = 0.0,
):
    """
    Training smoke (real weights, bf16): roster (edited.out, source.in, mask.in), vlm_inputs=[source.in],
    B=1, `steps` x (train_loss + backward + AdamW step) with per-block gradient checkpointing.
    The prompt is encoded once with Qwen3-VL on the GPU, which then moves to the CPU, so peak
    memory is the transformer's training footprint.
    """
    device = torch.device("cuda")
    base = QwenImage21Pipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
    assert isinstance(base, QwenImage21Pipeline)
    base.transformer.requires_grad_(False)
    base.vae.requires_grad_(False)
    base.text_encoder.requires_grad_(False)
    base.to(device)
    roster = ["edited.out", "source.in", "mask.in"]
    cfg = TeamworkConfig(
        teammates=roster, lora_rank=64, base_model=MODEL, title="smoke", profile="QWEN21",
        layout=layout, vlm_inputs=("source.in",), text_teammate=text_teammate,  # type: ignore[arg-type]
    )
    pipe_cls = SequenceQwenImage21Pipeline if layout == "sequence" else BatchQwenImage21Pipeline
    assert automatic_pipeline(base, cfg) is pipe_cls
    pipe = pipe_cls.from_base_pipeline(base, cfg, training=True, grad_checkpointing=True)
    assert isinstance(pipe, QwenImage21TeamworkPipeline)
    transformer = pipe.unwrapped_transformer
    adapters = adapter_parameters(transformer)
    trainable = {id(p) for p in transformer.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in adapters.values()}, "only adapter parameters may require grad"
    print(f"{layout}: {len(adapters)} adapter tensors, {sum(p.numel() for p in adapters.values()) / 1e6:.1f}M params")

    source = test_image(resolution)
    edited = PIL.Image.fromarray(255 - np.array(source))
    mask = mask_image(resolution)

    def make_batch() -> TeamBuilder:
        batch = TeamBuilder(roster, device, torch.bfloat16, dropout_prob=attn_dropout)
        batch.provide("edited", edited)
        batch.provide("source", source)
        batch.provide("mask", mask)
        return batch

    # Prompts of varying length (cycled), encoded up front so Qwen3-VL can leave the GPU.
    filler = "with soft warm light, gentle film grain, and a slightly hazy horizon over the distant hills"
    prompts = [PROMPT] + [f"{PROMPT} Keep everything else unchanged, {', '.join([filler] * k)}." for k in range(1, 6)]
    encodings = [pipe.encode_team_prompt(make_batch(), p) for p in prompts]
    print("prompt text lengths:", [e.hidden.shape[1] for e in encodings])
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    optim = torch.optim.AdamW(adapters.values(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    lh = resolution // 16
    for step in range(steps):
        k = step % len(prompts)
        batch = make_batch()
        noise = torch.randn(batch.count, 64, lh, lh, device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = pipe.train_loss(batch, noise, prompt=prompts[k], prompt_embeds=encodings[k])
        loss = out.loss
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        assert torch.isfinite(loss), f"loss {loss.item()}"
        missing = [k for k, p in adapters.items() if p.grad is None]
        assert not missing, f"adapters without grad: {missing[:5]}"
        assert all(bool(torch.isfinite(p.grad).all()) for p in adapters.values() if p.grad is not None)
        assert all(p.grad is None for p in transformer.parameters() if not p.requires_grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(list(adapters.values()), max_norm=1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        graphs = dynamo_counters["stats"]["unique_graphs"]
        print(
            f"step {step}: text {encodings[k].hidden.shape[1]} tokens, loss {loss.item():.4f}, "
            f"grad norm {grad_norm.item():.4f}, timestep {int(out.timestep_idx[0])}, "
            f"{start.elapsed_time(end) / 1000:.1f}s, compiled graphs {graphs}, "
            f"allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
        )
    print(f"{layout} {resolution}x{resolution}: peak training memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    app()
