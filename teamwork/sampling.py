"""
Sampling helpers shared by the batch and sequence implementations: initial noise, timestep
sampling, flow-match noising, and loss weighting. Latents are in component layout `(count, C, h, w)`.
"""

from dataclasses import dataclass
from typing import Literal, get_args

import torch
from torch import Tensor
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from .data import Components

TimestepWeight = Literal["unit", "sigma_sqrt", "logit_normal"]


def initial_noise(
    shape: torch.Size | tuple[int, ...], generator: torch.Generator | None, device: torch.device, dtype: torch.dtype,
) -> Tensor:
    """
    Gaussian noise drawn on the generator's device, then moved to `device`. CUDA `randn` output for
    a seed depends on the GPU's SM count, so pass a CPU generator to get the same noise on every GPU.
    """
    draw_device = device if generator is None else generator.device
    return torch.randn(shape, generator=generator, device=draw_device, dtype=dtype).to(device)


def sample_timestep_u(timestep_weight: str, batch_size: int) -> Tensor:
    """One `u` in [0, 1) per batch element, on the CPU from the global RNG."""
    if timestep_weight == "unit" or timestep_weight == "sigma_sqrt":
        return torch.rand([batch_size])
    elif timestep_weight == "logit_normal":
        return torch.nn.functional.sigmoid(torch.randn([batch_size]))
    else:
        raise ValueError(f"Unknown timestep weight {timestep_weight}, expected one of {get_args(TimestepWeight)}")


def loss_weighting(timestep_weight: str, sigmas: Tensor) -> Tensor | float:
    if timestep_weight == "sigma_sqrt":
        return sigmas.pow(-2.0).float()
    elif timestep_weight == "logit_normal" or timestep_weight == "unit":
        return 1.0
    else:
        raise ValueError(f"Unknown timestep weight {timestep_weight}, expected one of {get_args(TimestepWeight)}")


@dataclass
class FlowMatchNoising:
    timestep_idx: Tensor  # (count,): int64 on the CPU, index into the training schedule
    timesteps: Tensor  # (count,): the schedule's timesteps (0..1000)
    sigmas: Tensor  # (count, 1, 1, 1)
    noisy_latents: Tensor  # (count, C, h, w), with input components left clean
    weighting: Tensor | float  # broadcastable against (count, C, h, w)


def flow_match_noising(
    scheduler: FlowMatchEulerDiscreteScheduler,
    timestep_u: Tensor,
    components: Components,
    latents: Tensor,
    noise: Tensor,
    timestep_weight: str,
    device: torch.device,
    dtype: torch.dtype,
) -> FlowMatchNoising:
    """
    Noise every component of a batch element to that element's timestep (from `timestep_u`,
    see `sample_timestep_u`) along the flow-match path; input components stay clean.
    """
    count = latents.shape[0]
    assert timestep_u.shape == (components.batch_size,), f"{timestep_u.shape} != ({components.batch_size},)"
    assert components.batch_indices.shape == (count,), f"{components.batch_indices.shape} != ({count},)"
    assert noise.shape == latents.shape, f"{noise.shape} != {latents.shape}"

    schedule_timesteps, schedule_sigmas = scheduler.timesteps, scheduler.sigmas
    assert isinstance(schedule_timesteps, Tensor) and isinstance(schedule_sigmas, Tensor)
    timestep_u = timestep_u[components.batch_indices.cpu()]
    timestep_idx = (timestep_u * len(schedule_timesteps)).long()
    timesteps = schedule_timesteps[timestep_idx].to(device=device, dtype=dtype)
    sigmas = schedule_sigmas[timestep_idx].reshape(count, 1, 1, 1).to(dtype=dtype, device=device)
    noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    weighting = loss_weighting(timestep_weight, sigmas)
    noisy_latents[components.input_subindices] = latents[components.input_subindices]
    return FlowMatchNoising(
        timestep_idx=timestep_idx,
        timesteps=timesteps,
        sigmas=sigmas,
        noisy_latents=noisy_latents,
        weighting=weighting,
    )
