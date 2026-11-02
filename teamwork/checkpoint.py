"""
Teamwork checkpoint IO: safetensors save/load and strict adapter-state loading.

A checkpoint stores the adapter parameters of an adapted model under their module
paths (e.g. `transformer_blocks.0.attn.to_q.adapter.down`) plus the TeamworkConfig as
metadata. Loading is strict: every adapter parameter of the adapted model must be in the
checkpoint and every checkpoint key must be an adapter parameter. The one exemption is
the legacy scalar `<module>.adapter` markers written for the joint-attention modules.
"""

from pathlib import Path
import re

import safetensors
import safetensors.torch
import torch
from torch import nn

from .adapter import adapter_parameters
from .config import TeamworkConfig, config_to_metadata, metadata_to_config

ADAPTER_KEY = re.compile(r"(.+)\.adapter(\..+)?")


def is_legacy_marker(key: str, tensor: torch.Tensor) -> bool:
    """The scalar `<module>.adapter` placeholders of the joint-attention modules."""
    return tensor.ndim == 0 and key.endswith(".adapter")


def adapted_layers_in_state(state: dict[str, torch.Tensor]) -> set[str]:
    """Module paths that a checkpoint adapts (every adapted layer has a child "adapter")."""
    layers = set()
    for key in state.keys():
        m = ADAPTER_KEY.fullmatch(key)
        if m is not None:
            layers.add(m.group(1))
    return layers


@torch.no_grad()
def load_adapter_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Load adapter parameters into an already adapted model, strictly.

    Raises if an adapter parameter of `model` is missing from `state`, if `state` has a
    key that is not an adapter parameter of `model`, or if a shape differs. Legacy
    scalar markers are exempt both ways: present ones are loaded into the module that
    still has a marker, and others are ignored.
    """
    expected = adapter_parameters(model)
    missing = sorted(
        k for k, p in expected.items() if k not in state and not is_legacy_marker(k, p)
    )
    unexpected = sorted(
        k for k, t in state.items() if k not in expected and not is_legacy_marker(k, t)
    )
    if missing or unexpected:
        raise KeyError(
            f"teamwork checkpoint does not match the adapted model: "
            f"{len(missing)} missing keys {missing[:8]}, "
            f"{len(unexpected)} unexpected keys {unexpected[:8]}"
        )
    for key, tensor in state.items():
        if key not in expected:
            continue  # legacy marker of a module that no longer has one
        if expected[key].shape != tensor.shape:
            raise ValueError(
                f"teamwork checkpoint key {key} has shape {tuple(tensor.shape)}, "
                f"but the adapted model expects {tuple(expected[key].shape)}"
            )
        module_path, attr = key.rsplit(".", 1)
        setattr(
            model.get_submodule(module_path),
            attr,
            nn.Parameter(tensor.to(device=device, dtype=dtype)),
        )


def save_adapters(
    module: nn.Module,
    path: str | Path,
    cfg: TeamworkConfig,
    extra_metadata: dict[str, str] | None = None,
):
    """
    Save teamwork adapter parameters to a safetensors file.

    Args:
        module: The module containing teamwork adapters
        path: Path to save the safetensors file
        cfg: The TeamworkConfig used to create the adapters
        extra_metadata: Optional additional metadata to store
    """

    state_dict = {}
    for name, parameter in adapter_parameters(module).items():
        state_dict[name] = parameter.detach()

    metadata = config_to_metadata(cfg)
    if extra_metadata:
        metadata.update(extra_metadata)

    return safetensors.torch.save_file(state_dict, str(path), metadata)


def save_entire(
    module: nn.Module,
    path: str | Path,
    cfg: TeamworkConfig,
    extra_metadata: dict[str, str] | None = None,
):
    """
    Save the entire model to a safetensors file, but with teamwork metadata.

    Args:
        module: The module containing teamwork adapters
        path: Path to save the safetensors file
        cfg: The TeamworkConfig used to create the adapters
        extra_metadata: Optional additional metadata to store
    """

    state_dict = {}
    for name, parameter in module.named_parameters():
        state_dict[name] = parameter.detach()

    metadata = config_to_metadata(cfg)
    if extra_metadata:
        metadata.update(extra_metadata)

    return safetensors.torch.save_file(state_dict, str(path), metadata)


def load_safetensors(
    path: str | Path,
) -> tuple[TeamworkConfig, dict[str, torch.Tensor], dict[str, str]]:
    with safetensors.safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        cfg = metadata_to_config(metadata)

        # Load parameters
        checkpoint = {}
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)

        return cfg, checkpoint, metadata
