import gc
import math
import os
from enum import Enum
from pathlib import Path
from random import randint, random, sample
from typing import Annotated

import torch
from torch import nn
import numpy as np
import rerun as rr
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image
import typer
import gfxds
from gfxds import Loader
from prodigyopt import Prodigy
from tqdm import tqdm

from teamwork import TeamworkConfig, TeamworkPipeline, adapter
from teamwork.checkpoint import load_safetensors
from teamwork.pipelines import automatic_pipeline
from teamwork.utils import pdb_check, pdb_exception
from teamwork.data import TeamBuilder

app = typer.Typer(pretty_exceptions_show_locals=False)


class Task(Enum):
    MATFUSION = "matfusion"
    INTERIORVERSE = "interiorverse"
    HETEROGENEOUS = "heterogeneous"

    def teammates(self) -> list[str]:
        match self:
            case Task.MATFUSION:
                return [
                    "image.in",
                    "halfway.in",
                    "diffuse.out",
                    "specular.out",
                    "roughness.out",
                    "normals.out",
                ]
            case Task.INTERIORVERSE:
                return [
                    "image.in",
                    "diffuse.out",
                    "specular.out",
                    "roughness.out",
                    "normals.out",
                    "depth.out",
                    "albedo.out",
                    "inverseshading.out",
                ]
            case Task.HETEROGENEOUS:
                return [
                    "image.in",
                    "diffuse.out",
                    "specular.out",
                    "roughness.out",
                    "normals.out",
                    "depth.out",
                    "albedo.out",
                    "inverseshading.out",
                    "diffuseshading.out",
                    "residual.out",
                ]
        raise ValueError()

    def train_dataset(self):
        match self:
            case Task.MATFUSION:
                return "train/matfusion/rast"
            case Task.INTERIORVERSE:
                return "train/interiorverse/85"
            case Task.HETEROGENEOUS:
                return {
                    "kind": "Multi",
                    "datasets": [
                        {"id": "train/rooms", "weight": 6.0},
                        {"id": "train/pbrjumble/v0", "weight": 2.0},
                        {"id": "train/infinigen/preview", "weight": 0.5},
                        {"id": "train/matfusion/env", "weight": 1.0},
                        {"id": "train/matfusion/flash", "weight": 0.5},
                    ],
                }

    def val_dataset(self):
        match self:
            case Task.MATFUSION:
                return "test/matfusion/rast"
            case Task.INTERIORVERSE:
                return "val/interiorverse/85"
            case Task.HETEROGENEOUS:
                return {
                    "kind": "Multi",
                    "datasets": [
                        {"id": "val/interiorverse/85", "weight": 1.0},
                        {"id": "val/hypersim", "weight": 1.0},
                        {
                            "id": "test/photos",
                            "dataset": {
                                "kind": "ImageFolder",
                                "dir": "datasets:test_photos",
                                "component": "image",
                                "split": "test",
                                "resize": [512, 512],
                            },
                            "weight": 1.0,
                        },
                    ],
                }


@app.command()
def main(
    base_model: Annotated[str, typer.Option()],
    task: Annotated[Task, typer.Option()],
    output_checkpoint: Annotated[Path, typer.Option()],
    title: Annotated[str, typer.Option()],
    profile: Annotated[str, typer.Option()],
    datasets: Annotated[Path, typer.Option()],
    num_samples: int = 64_000,
    accumulation: int = 16,
    batch_size: int = 1,
    lora_rank: int = 64,
    eval_every: int = 500,
    save_every: int = 8_000,
    offset_noise: float = 0.1,
    resume_checkpoint: Path | None = None,
    resume_step: int = -1,
    rerun_connect: str | None = "rerun+http://localhost:9876/proxy",
    dropout_prob: float = 0.0,
    resize: int | None = None,
):
    args = locals()
    print(args)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = TeamworkConfig(
        base_model=base_model,
        title=title,
        teammates=task.teammates(),
        lora_rank=lora_rank,
        lora_communication=True,
        profile=profile,
    )
    base_pipeline = AutoPipelineForText2Image.from_pretrained(
        config.base_model, torch_dtype=dtype
    ).to(device)
    resume_state = None
    if resume_checkpoint is not None:
        assert resume_step >= 0, "please provide --resume-step"
        _, resume_state, _ = load_safetensors(resume_checkpoint)
    pipe = automatic_pipeline(base_pipeline, config).from_base_pipeline(
        base_pipeline=base_pipeline,
        teamwork_config=config,
        state=resume_state,
        training=True,
    )
    assert isinstance(pipe, TeamworkPipeline)

    pipe.set_progress_bar_config(disable=True)
    pipe.empty_prompts()
    gc.collect()
    torch.cuda.empty_cache()
    print(pipe.teamwork_config)

    pdb_check()

    # Get the dataset
    train_loader = Loader(
        datasets,
        task.train_dataset(),
        limit=num_samples,
        offset=max(resume_step, 0),
        replacement=True,  # type: ignore
        concurrent=8,
        resize=(resize, resize) if resize else None,
    )
    train_loader.start()
    print(f"Found {len(train_loader)} training samples")
    test_loader = Loader(
        datasets,
        task.val_dataset(),
        limit=(num_samples // eval_every + 1),
        replacement=True,  # type: ignore
        concurrent=1,
        resize=(resize, resize) if resize else None,
    )
    test_loader.start()

    # Setup the optimizer
    if hasattr(pipe, "transformer"):
        model = pipe.transformer
    else:
        model = pipe.unet
    assert isinstance(model, nn.Module)
    trainable_modules = adapter.adapter_modules(model)
    trainable_params = adapter.adapter_parameters(model).values()

    trainable_param_count = 0
    for p in trainable_params:
        trainable_param_count += np.array([*p.shape]).prod()
    all_param_count = 0
    for p in model.parameters():
        all_param_count += np.array([*p.shape]).prod()
    model_stats = f"""model stats:
- {len(config.teammates)} teammates
- {len(trainable_modules)} adapter modules
- {len(trainable_params)} trainable tensors
- {trainable_param_count} trainable params
- {all_param_count} total params"""
    print(model_stats)

    # Setup logging
    if rerun_connect is not None:
        rr.init(title, spawn=False)
        rr.connect_grpc(rerun_connect)
        rr.set_time("step", sequence=0)
        rr.log(
            "args", rr.TextDocument("\n".join(f"--{k} {v}" for k, v in args.items()))
        )
        rr.log("stats", rr.TextDocument(model_stats))

    base_lr = 1.0
    optimizer = Prodigy(
        trainable_params,
        lr=base_lr,
        safeguard_warmup=True,
        use_bias_correction=True,
        weight_decay=1e-4,  # type: ignore
    )

    smooth_loss_count = 0
    smooth_loss = 0.0

    # Main training loop
    end_step = num_samples // batch_size
    for step_idx in tqdm(range(resume_step, end_step - 1)):
        try:
            # Evaluate the model
            if step_idx % eval_every == 0:
                eval_sample = next(test_loader)
                generated = pipe({c: r.image for (c, r) in eval_sample.images.items()})
                if rr.is_enabled():
                    for c, img in generated.items():
                        rr.log(f"eval/est/{c}", rr.Image(img))
                    for c, r in eval_sample.images.items():
                        rr.log(f"eval/gt/{c}", rr.Image(r.image))

            if step_idx % save_every == 0:
                pipe.save_adapters(output_checkpoint)

            # Get a batch of training data
            batch = TeamBuilder(pipe.teamwork_config.teammates, device, dtype)
            for batch_idx in range(batch_size):
                train_sample = next(train_loader)
                weight = (
                    train_sample.images.pop("weight").image
                    if "weight" in train_sample.images
                    else None
                )               
                if random() < dropout_prob and len(train_sample.images) >= 2:
                    train_sample.images = sample(train_sample.images, k=randint(1, len(train_sample.images) - 1))
                for c, r in train_sample.images.items():
                    batch.provide(c, r.image, weight=weight, cropinfo=r.crop, batch_idx=batch_idx)

            assert batch.resolution is not None
            h, w = batch.resolution
            lh = h // pipe.vae_scale_factor
            lw = w // pipe.vae_scale_factor
            lc = pipe.in_channels

            # Noise
            noise = torch.randn((batch.count, lc, lh, lw), device=device, dtype=dtype)
            noise += offset_noise * torch.randn(
                (*noise.shape[:2], 1, 1), device=device, dtype=dtype
            )

            # Backpropagate
            if rr.is_enabled():
                rr.set_time("step", sequence=step_idx)
                
            loss = pipe.train_loss(batch, noise).loss
            log_loss = loss.clone().detach().item()
            del batch
            if loss.isfinite():
                (loss / accumulation).backward()

                smooth_loss_count = min(smooth_loss_count + 1, 500)
                smooth_loss_t = 1 / smooth_loss_count
                smooth_loss = (
                    smooth_loss * (1.0 - smooth_loss_t) + log_loss * smooth_loss_t
                )

            # Log the loss
            if rr.is_enabled():
                rr.log("loss", rr.Scalars(log_loss))
                rr.log("smooth_loss", rr.Scalars(smooth_loss))

            # Take an optimizer step every args.accumulation steps
            if step_idx % accumulation == accumulation - 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                if rr.is_enabled():
                    rr.log("grad_norm", rr.Scalars(grad_norm.cpu().item()))

                lr = (
                    base_lr * math.cos(step_idx / end_step * math.pi) * 0.5 + 0.5
                )
                optimizer.defaults["lr"] = lr
                if rr.is_enabled():
                    rr.log("lr", rr.Scalars(lr))

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            pdb_check()
            step_idx += 1
        except Exception as e:
            pdb_exception(e)

    pipe.save_adapters(output_checkpoint)


if __name__ == "__main__":
    app()
