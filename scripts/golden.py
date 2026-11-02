"""
Golden-output regression checks for the teamwork pipelines.

Every case builds a tiny random-init diffusers denoiser + VAE from a config (no real
weights), adapts it with a non-zero random teamwork LoRA (random `down`, `up`, and
`bias`, loaded through the checkpoint path), and records in fp32:

- `transformer/{loss,call}/{i}`: every denoiser output, captured with a forward hook
  at the pipeline's own call sites (one in `train_loss`, one per `__call__` step);
- `loss/*`: the `train_loss` LossOutput tensors, with fixed images, noise, and RNG seed;
- `call/*`: the final latents of a 2-step `__call__`.

Batches have B=2 with jagged presence (batch element 1 has no `mask.in`). Sequence-layout
cases, where teammates attend to each other, also use a static `attn_allow` and attention
dropout, so `attn_keep` gates attention in both `train_loss` and `__call__`. They run with
the text adapted by `edited.out` and at base (`text_teammate` set and None).

Usage:
    python scripts/golden.py record [--goldens-dir scripts/goldens]
    python scripts/golden.py check [--atol 0.0] [--only klein-FLUX2-seq]

`check` is bit-exact (`torch.equal`) unless `--atol` is given.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Annotated, Any, Callable

import safetensors.torch
import torch
from torch import Tensor, nn
import typer
import diffusers.utils.logging

from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from teamwork.adapter import adapter_parameters
from teamwork.data import TeamBuilder
from teamwork.config import TeamworkConfig
from teamwork.pipeline_flux import BatchFluxPipeline, SequenceFluxPipeline
from teamwork.pipelines import TeamworkPipeline
from teamwork.pipeline_flux2_klein import BatchFlux2KleinPipeline, SequenceFlux2KleinPipeline
from teamwork.pipeline_sd2 import BatchSD2Pipeline
from teamwork.pipeline_sd3 import BatchSD3Pipeline
from teamwork.pipeline_sdxl import BatchSDXLPipeline

app = typer.Typer(pretty_exceptions_show_locals=False)
diffusers.utils.logging.set_verbosity_error()

DEFAULT_DIR = Path(__file__).parent / "goldens"
ROSTER = ["edited.out", "source.in", "mask.in"]
RESOLUTION = 16  # pixels; every tiny VAE downsamples 2x
TEXT_LEN = 8
TEXT_DIM = 32
POOLED_DIM = 32
LORA_RANK = 4
NUM_STEPS = 2
JOINT_ATTN_ALLOW = "110;111;011"
JOINT_DROPOUT = 0.5


@dataclass(frozen=True)
class Family:
    pipeline_cls: type[TeamworkPipeline]
    build_base: Callable[[], Any]  # tiny random diffusers base pipeline, fp32, on CPU
    denoiser_attr: str  # "transformer" or "unet"
    prompt_embeds: dict[str, Callable[[torch.Generator], Tensor]]  # injected instead of text encoders
    sequence_cls: type[TeamworkPipeline] | None = None  # for `layout="sequence"` cases


@dataclass(frozen=True)
class Case:
    family: str
    profile: str
    communicate: bool
    version: str | None = None  # None: TeamworkConfig's default
    layout: str = "batch"
    text_teammate: str | None = None  # sequence layout only

    @property
    def name(self) -> str:
        name = f"{self.family}-{self.profile}"
        if self.layout == "sequence":
            name += "-seq"
        name += f"-{'comm' if self.communicate else 'nocomm'}"
        if self.layout == "sequence":
            name += f"-{'text' if self.text_teammate is not None else 'notext'}"
        if self.version is not None:
            name += f"-v{self.version}"
        return name

    @property
    def joint(self) -> bool:
        """Teammates attend to each other, so `attn_allow` and attention dropout apply."""
        return self.layout == "sequence"


def randomize_(module: nn.Module, generator: torch.Generator):
    """Re-init every parameter from `generator`, independent of diffusers' (unseeded) init,
    so that no layer is zero-initialized (zeros hide bugs)."""
    for name, param in module.named_parameters():
        if param.ndim >= 2:
            fan_in = math.prod(param.shape[1:])
            param.data = torch.randn(param.shape, generator=generator) / math.sqrt(fan_in)
        elif name.endswith("bias"):
            param.data = 0.02 * torch.randn(param.shape, generator=generator)
        else:  # norm scales and other 1-d weights
            param.data = 1.0 + 0.1 * torch.randn(param.shape, generator=generator)


def tiny_vae(latent_channels: int, generator: torch.Generator, **config) -> AutoencoderKL:
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=latent_channels,
        norm_num_groups=8,
        sample_size=RESOLUTION,
        **config,
    )
    randomize_(vae, generator)
    return vae


def tiny_unet(generator: torch.Generator, **config) -> UNet2DConditionModel:
    unet = UNet2DConditionModel(
        sample_size=RESOLUTION // 2,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(16, 32),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=TEXT_DIM,
        attention_head_dim=8,
        norm_num_groups=8,
        **config,
    )
    randomize_(unet, generator)
    return unet


def build_sd2():
    g = torch.Generator().manual_seed(10)
    return StableDiffusionPipeline(
        vae=tiny_vae(4, g, scaling_factor=0.5),
        text_encoder=None,  # type: ignore
        tokenizer=None,  # type: ignore
        unet=tiny_unet(g, use_linear_projection=True),
        scheduler=DDIMScheduler(
            beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012, steps_offset=1, clip_sample=False,
        ),  # type: ignore
        safety_checker=None,  # type: ignore
        feature_extractor=None,  # type: ignore
        requires_safety_checker=False,
    )


def build_sdxl():
    g = torch.Generator().manual_seed(11)
    unet = tiny_unet(
        g,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=8 * 6 + POOLED_DIM,
        transformer_layers_per_block=1,
    )
    return StableDiffusionXLPipeline(
        vae=tiny_vae(4, g, scaling_factor=0.5),
        text_encoder=None,  # type: ignore
        text_encoder_2=None,  # type: ignore
        tokenizer=None,  # type: ignore
        tokenizer_2=None,  # type: ignore
        unet=unet,
        scheduler=EulerDiscreteScheduler(beta_schedule="scaled_linear", beta_start=0.00085, beta_end=0.012),  # type: ignore
        add_watermarker=False,
    )


def build_sd3():
    g = torch.Generator().manual_seed(12)
    transformer = SD3Transformer2DModel(
        sample_size=RESOLUTION // 2,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=TEXT_DIM,
        caption_projection_dim=32,
        pooled_projection_dim=POOLED_DIM,
        pos_embed_max_size=16,
    )
    randomize_(transformer, g)
    return StableDiffusion3Pipeline(
        transformer=transformer,
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.0),
        vae=tiny_vae(4, g, scaling_factor=0.5, shift_factor=0.1),
        text_encoder=None,  # type: ignore
        tokenizer=None,  # type: ignore
        text_encoder_2=None,  # type: ignore
        tokenizer_2=None,  # type: ignore
        text_encoder_3=None,  # type: ignore
        tokenizer_3=None,  # type: ignore
    )


def flow_scheduler() -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler(
        shift=3.0,
        use_dynamic_shifting=True,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
    )


def build_flux():
    g = torch.Generator().manual_seed(13)
    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=TEXT_DIM,
        pooled_projection_dim=POOLED_DIM,
        guidance_embeds=True,
        axes_dims_rope=(4, 6, 6),
    )
    randomize_(transformer, g)
    return FluxPipeline(
        scheduler=flow_scheduler(),
        vae=tiny_vae(4, g, scaling_factor=0.5, shift_factor=0.1),
        text_encoder=None,  # type: ignore
        tokenizer=None,  # type: ignore
        text_encoder_2=None,  # type: ignore
        tokenizer_2=None,  # type: ignore
        transformer=transformer,
    )


def build_klein():
    g = torch.Generator().manual_seed(14)
    transformer = Flux2Transformer2DModel(
        patch_size=1,
        in_channels=16,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=TEXT_DIM,
        timestep_guidance_channels=32,
        axes_dims_rope=(4, 4, 4, 4),
        guidance_embeds=False,
    )
    randomize_(transformer, g)
    vae = AutoencoderKLFlux2(
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
        sample_size=RESOLUTION,
    )
    randomize_(vae, g)
    # Non-trivial BN stats, so the 1.1 latent normalization is not an identity.
    assert vae.bn.running_mean is not None and vae.bn.running_var is not None
    vae.bn.running_mean.copy_(0.1 * torch.randn(vae.bn.running_mean.shape, generator=g))
    vae.bn.running_var.copy_(0.5 + torch.rand(vae.bn.running_var.shape, generator=g))
    return Flux2KleinPipeline(
        scheduler=flow_scheduler(),
        vae=vae,
        text_encoder=None,  # type: ignore
        tokenizer=None,  # type: ignore
        transformer=transformer,
    )


def embeds(g: torch.Generator) -> Tensor:
    return torch.randn(1, TEXT_LEN, TEXT_DIM, generator=g)


def pooled(g: torch.Generator) -> Tensor:
    return torch.randn(1, POOLED_DIM, generator=g)


FAMILIES: dict[str, Family] = {
    "sd2": Family(BatchSD2Pipeline, build_sd2, "unet", {"empty_prompt_embeds": embeds}),
    "sdxl": Family(
        BatchSDXLPipeline, build_sdxl, "unet",
        {"empty_prompt_embeds": embeds, "empty_pooled_prompt_embeds": pooled},
    ),
    "sd3": Family(
        BatchSD3Pipeline, build_sd3, "transformer",
        {"empty_prompt_embeds": embeds, "empty_pooled_prompt_embeds": pooled},
    ),
    "flux": Family(
        BatchFluxPipeline, build_flux, "transformer",
        {
            "empty_prompt_embeds": embeds,
            "empty_pooled_prompt_embeds": pooled,
            "empty_text_ids": lambda g: torch.zeros(TEXT_LEN, 3),
        },
        sequence_cls=SequenceFluxPipeline,
    ),
    "klein": Family(
        BatchFlux2KleinPipeline, build_klein, "transformer",
        {
            "empty_prompt_embeds": embeds,
            "empty_text_ids": lambda g: Flux2KleinPipeline._prepare_text_ids(torch.zeros(1, TEXT_LEN, TEXT_DIM)),
        },
        sequence_cls=SequenceFlux2KleinPipeline,
    ),
}

CASES: list[Case] = [
    Case(family, profile, communicate)
    for family, profile in [
        ("sd2", "SD2"),
        ("sdxl", "SDXL"),
        ("sd3", "SD3"),
        ("flux", "FLUX"),
        ("klein", "FLUX2"),
    ]
    for communicate in [True, False]
] + [Case("klein", "FLUX2", True, version="1.0")] + [
    Case(family, profile, communicate, layout="sequence", text_teammate=text_teammate)
    for family, profile in [("flux", "FLUX"), ("klein", "FLUX2")]
    for communicate in [True, False]
    for text_teammate in ["edited.out", None]
]


def make_config(case: Case) -> TeamworkConfig:
    # The version is left at the default for all but the legacy case, so the goldens
    # recorded at 1.1 also check that newer versions keep 1.1's behavior.
    kwargs: dict[str, Any] = {}
    if case.version is not None:
        kwargs["version"] = case.version
    if case.layout == "sequence":
        kwargs["layout"] = "sequence"
        kwargs["text_teammate"] = case.text_teammate
    return TeamworkConfig(
        teammates=ROSTER,
        lora_rank=LORA_RANK,
        base_model=f"tiny-{case.family}",
        title=f"golden {case.name}",
        profile=case.profile,
        lora_communication=case.communicate,
        attn_allow=JOINT_ATTN_ALLOW if case.joint else None,
        **kwargs,
    )


def random_adapter_state(pipe: Any, denoiser_attr: str) -> dict[str, Tensor]:
    """Random values for every saved adapter key, exactly representable in bf16.

    The pipelines adapt in bf16, so routing fp32 values through a bf16 round trip
    keeps the load lossless; `make_pipeline` then casts the adapters back to fp32.
    """
    g = torch.Generator().manual_seed(1234)
    state = {}
    for name, param in sorted(adapter_parameters(getattr(pipe, denoiser_attr)).items()):
        scale = 0.05 if name.endswith(".bias") else 0.2
        value = scale * torch.randn(param.shape, generator=g)
        state[name] = value.to(torch.bfloat16).to(torch.float32)
    assert len(state) > 0
    return state


def make_pipeline(case: Case, device: torch.device) -> tuple[Any, Family]:
    family = FAMILIES[case.family]
    torch.manual_seed(0)  # anything left on the global RNG is per-case, not order-dependent
    base = family.build_base().to(device)
    cfg = make_config(case)
    pipeline_cls = family.sequence_cls if case.layout == "sequence" else family.pipeline_cls
    assert pipeline_cls is not None, f"{case.family} has no pipeline for {case.name}"
    blank = pipeline_cls.from_base_pipeline(base, cfg)
    state = random_adapter_state(blank, family.denoiser_attr)
    del blank
    pipe = pipeline_cls.from_base_pipeline(base, cfg, state=state)
    denoiser = getattr(pipe, family.denoiser_attr)
    loaded = adapter_parameters(denoiser)
    assert set(loaded) == set(state), f"loaded keys differ: {set(loaded) ^ set(state)}"
    for name, param in loaded.items():
        param.data = param.data.to(torch.float32)
        assert torch.equal(param.data.cpu(), state[name]), f"{name} did not load losslessly"
    g = torch.Generator().manual_seed(99)
    for attr, make in family.prompt_embeds.items():
        setattr(pipe, attr, make(g).to(device))
    pipe.set_progress_bar_config(disable=True)
    return pipe, family


def fixed_images(seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        name: torch.rand(3, RESOLUTION, RESOLUTION, generator=g) * 2 - 1
        for name in ["edited", "source", "mask"]
    }


def run_case(case: Case, device: torch.device) -> tuple[dict[str, Tensor], dict[str, str]]:
    # Compiled kernels (flex_attention) depend on what was compiled before; start every case fresh
    # so that `--only` subsets reproduce the full run.
    torch._dynamo.reset()
    pipe, family = make_pipeline(case, device)
    tensors: dict[str, Tensor] = {}
    errors: dict[str, str] = {}

    phase = "none"
    calls: dict[str, int] = {}

    def hook(module: nn.Module, args: Any, output: Any):
        sample = output.sample if hasattr(output, "sample") else output[0]
        assert isinstance(sample, Tensor)
        idx = calls.get(phase, 0)
        calls[phase] = idx + 1
        tensors[f"transformer/{phase}/{idx}"] = sample.detach().clone()

    getattr(pipe, family.denoiser_attr).register_forward_hook(hook)

    images = [fixed_images(1), fixed_images(2)]

    # train_loss: element 0 has every teammate, element 1 is missing mask.in.
    # Like training, the output is provided first.
    phase = "loss"
    torch.manual_seed(1)
    batch = TeamBuilder(
        ROSTER, device, torch.float32,
        dropout_prob=JOINT_DROPOUT if case.joint else 0.0,
        attn_allow=pipe.teamwork_config.attn_allow,
    )
    for batch_idx, present in enumerate([["edited", "source", "mask"], ["edited", "source"]]):
        for name in present:
            batch.provide(name, images[batch_idx][name], batch_idx=batch_idx)
    g = torch.Generator().manual_seed(3)
    latent_res = RESOLUTION // pipe.vae_scale_factor
    noise = torch.randn(batch.count, pipe.in_channels, latent_res, latent_res, generator=g).to(device)
    kept: list[Tensor | None] = []
    if case.layout == "sequence":
        # Record the attn_keep that train_loss draws (the selection is just Components).
        draw_attn_keep = batch.attn_keep
        batch.attn_keep = lambda: kept.append(draw_attn_keep()) or kept[-1]  # type: ignore[method-assign]
    try:
        out = pipe.train_loss(batch, noise)
        for field in ["latents", "prediction", "target", "weight", "timestep_idx"]:
            value = getattr(out, field)
            if isinstance(value, Tensor):
                tensors[f"loss/{field}"] = value.detach().clone()
        tensors["loss/loss"] = out.loss.detach().clone()
        if case.joint:
            assert len(kept) == 1, f"train_loss drew attn_keep {len(kept)} times"
            attn_keep = kept[0]
            assert attn_keep is not None, "attention dropout should drop at least one edge"
            tensors["loss/attn_keep"] = attn_keep.to(torch.uint8)
    except Exception as e:
        errors["loss"] = f"{type(e).__name__}: {e}"

    # __call__: element 0 provides source and mask, element 1 only source.
    phase = "call"
    torch.manual_seed(2)
    call_images = [
        {"source": images[0]["source"], "mask": images[0]["mask"]},
        {"source": images[1]["source"]},
    ]
    batch = TeamBuilder.from_inputs(
        ROSTER, device, torch.float32, images=call_images, attn_allow=pipe.teamwork_config.attn_allow,
    )
    g = torch.Generator().manual_seed(4)
    noise = torch.randn(batch.count, pipe.in_channels, latent_res, latent_res, generator=g).to(device)
    try:
        outputs = pipe(call_images, noise=noise, num_inference_steps=NUM_STEPS, output_type="latents", batch=batch)
        assert isinstance(outputs, list) and len(outputs) == 2
        for batch_idx, result in enumerate(outputs):
            assert set(result) == {"edited"}, result.keys()
            tensors[f"call/{batch_idx}/edited"] = result["edited"].detach().clone()
        if case.joint:
            assert batch.attn_keep() is not None, "attn_allow should gate attention"
    except Exception as e:
        errors["call"] = f"{type(e).__name__}: {e}"

    tensors = {k: v.contiguous().cpu() for k, v in tensors.items()}
    return tensors, errors


def setup_determinism():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selected(only: str | None) -> list[Case]:
    cases = [c for c in CASES if only is None or only in c.name]
    assert cases, f"no case matches {only!r}; cases: {[c.name for c in CASES]}"
    return cases


def coverage(tensors: dict[str, Tensor], errors: dict[str, str]) -> str:
    parts = []
    for kind, prefix in [("transformer", "transformer/"), ("loss", "loss/"), ("call", "call/")]:
        n = sum(1 for k in tensors if k.startswith(prefix))
        parts.append(f"{kind}={n}")
    for kind, err in errors.items():
        parts.append(f"{kind}-error={err.split(':')[0]}")
    return " ".join(parts)


@app.command()
def record(
    goldens_dir: Annotated[Path, typer.Option()] = DEFAULT_DIR,
    device: Annotated[str, typer.Option()] = "cuda",
    only: Annotated[str | None, typer.Option(help="substring filter on case names")] = None,
):
    """Run every case and overwrite its golden file."""
    setup_determinism()
    goldens_dir.mkdir(parents=True, exist_ok=True)
    for case in selected(only):
        tensors, errors = run_case(case, torch.device(device))
        metadata = {f"error.{k}": v for k, v in errors.items()}
        safetensors.torch.save_file(tensors, str(goldens_dir / f"{case.name}.safetensors"), metadata)
        print(f"recorded {case.name}: {coverage(tensors, errors)}")
        for kind, err in errors.items():
            print(f"    {kind} raised {err}")


@app.command()
def check(
    goldens_dir: Annotated[Path, typer.Option()] = DEFAULT_DIR,
    device: Annotated[str, typer.Option()] = "cuda",
    only: Annotated[str | None, typer.Option(help="substring filter on case names")] = None,
    atol: Annotated[float | None, typer.Option(help="compare with allclose instead of torch.equal")] = None,
):
    """Re-run every case and compare against its golden file."""
    setup_determinism()
    failures = []
    for case in selected(only):
        path = goldens_dir / f"{case.name}.safetensors"
        assert path.exists(), f"missing golden {path}; run `record` first"
        golden = safetensors.torch.load_file(str(path))
        with safetensors.safe_open(str(path), framework="pt") as f:
            golden_errors = {k.removeprefix("error."): v for k, v in (f.metadata() or {}).items() if k.startswith("error.")}
        tensors, errors = run_case(case, torch.device(device))

        problems = []
        for kind, err in golden_errors.items():
            if kind not in errors:
                problems.append(f"{kind} raised when recorded ({err}) but now succeeds; re-record")
            elif errors[kind].split(":")[0] != err.split(":")[0]:
                problems.append(f"{kind} error changed: {err} -> {errors[kind]}")
        for kind, err in errors.items():
            if kind not in golden_errors:
                problems.append(f"{kind} now raises {err}")
        for key in sorted(set(golden) | set(tensors)):
            if key not in tensors:
                problems.append(f"{key} missing")
                continue
            if key not in golden:
                problems.append(f"{key} not in golden")
                continue
            a, b = golden[key], tensors[key]
            if a.shape != b.shape or a.dtype != b.dtype:
                problems.append(f"{key}: {a.dtype}{list(a.shape)} -> {b.dtype}{list(b.shape)}")
                continue
            same = torch.equal(a, b) if atol is None else torch.allclose(a.float(), b.float(), atol=atol, rtol=0.0, equal_nan=True)
            if not same:
                diff = (a.float() - b.float()).abs().max().item()
                problems.append(f"{key}: max abs diff {diff:.3e}")

        status = "PASS" if not problems else "FAIL"
        print(f"{status} {case.name}: {coverage(tensors, errors)}")
        for problem in problems:
            print(f"    {problem}")
        if problems:
            failures.append(case.name)

    if failures:
        print(f"{len(failures)} case(s) failed: {', '.join(failures)}")
        raise typer.Exit(1)
    mode = "bit-exact" if atol is None else f"atol={atol}"
    print(f"all {len(selected(only))} cases match ({mode})")


if __name__ == "__main__":
    app()
