"""
Everything needed to take some base model and apply teamwork by inserting adapter layers.

This is superficially similar to the `peft` library, but simpler and more suited to the
sort of arbitrary monkeypatching that teamwork requires.

Each TEAMWORK_PROFILE is a list of layer globs or of tuples `(glob, AdapterClass)`. For examples,
see the pipeline_*.py files which include profiles for a number of common models and tasks.
"""

from collections import OrderedDict
import torch
import safetensors.torch
from torch import nn
from typing import Optional, Tuple, Any, TypeAlias, TypeVar, Generic
from pathlib import Path
import re
from copy import copy
from collections import defaultdict

from .config import TeamworkConfig, config_to_metadata, metadata_to_config
from .batch import Selection

Params = TypeVar('Params')


class AdapterMixin(Generic[Params]):
    """
    Implemented by adapted layers. See TeamworkLinear in linear.py

    Anything trainable must be in the child module/param called "adapter".
    Even if there are no trainable parameters, it is still a good idea to include
    something so the adapter is visible in checkpoint files.

    The `selection` field will contain information about the currently active teammates.

    In general an implementation should look like this:
    ```
    class SomeLayer(BaseLayer):
        def __init__(self, base: BaseLayer, cfg: TeamworkConfig):
            shallowcopy_into(self, base)
            self.adapter = ...

        def forward(self, input: Tensor) -> Tensor:
            output = super().forward(input)
            ...
            return output
    ```
    """

    adapter: Params
    selection: Selection | None = None


Adapt: TypeAlias = str | tuple[str, type[AdapterMixin]]

TEAMWORK_PROFILES: dict[str, list[Adapt]] = {}


def adapt_single(
    module: nn.Module,
    adapt: Adapt,
    cfg: TeamworkConfig,
) -> nn.Module:
    try:
        from diffusers.quantizers.gguf.utils import GGUFLinear
    except ModuleNotFoundError:
        GGUFLinear = None

    if isinstance(adapt, str):
        if GGUFLinear is not None and isinstance(module, GGUFLinear):
            from .linear import TeamworkGGUFLinear

            return TeamworkGGUFLinear(module, cfg)
        elif isinstance(module, nn.Linear):
            from .linear import TeamworkLinear

            return TeamworkLinear(module, cfg)
        elif isinstance(module, nn.Conv2d):
            from .linear import TeamworkConv2d

            return TeamworkConv2d(module, cfg)
        else:
            raise NotImplementedError(
                f"automatic adapter for {type(module)} (via {adapt})"
            )
    else:
        return adapt[1](module, cfg)  # type: ignore


def resolve_paths_to_adapt(
    model: nn.Module,
    profile: list[Adapt],
    known: set[str] | None = None,
) -> dict[str, Adapt]:
    # Create a regex from all of the different globs in profile.
    source = ""
    for idx, adapt in enumerate(profile):
        if idx != 0:
            source += "|"
        source += "("
        if isinstance(adapt, str):
            glob = adapt
        else:
            glob, _ = adapt
        for c in glob:
            if c == "*":
                source += r"[^.]+"
            else:
                source += re.escape(c)
        source += ")"
    regex = re.compile(source)

    # Now iterate over all module paths in the model and figure out which
    # should be adated.
    resolved = dict()
    for name, _ in model.named_modules():
        m = regex.fullmatch(name) if len(profile) > 0 else None

        if known is None:
            should = m is not None
        else:
            should = name in known

        if should:
            chosen = "*"
            if m is not None:
                # Figure out the first Adapt which matched via the capture groups
                chosen = next(a for a, g in zip(profile, m.groups()) if g is not None)
            resolved[name] = chosen

    return resolved


def shallowcopy_into(this: nn.Module, original: nn.Module):
    """
    Make `this` the same as `original`, sharing all the same tensors.
    Overwritten submodules or parameters will not appear in original!
    This is the staring point for adapting any layer.
    """

    items = copy(original.__dict__)
    items["_modules"] = copy(items["_modules"])
    items["_parameters"] = copy(items["_parameters"])
    items["_buffers"] = copy(items["_buffers"])
    this.__dict__.update(items)


def shallowcopy(original: nn.Module) -> nn.Module:
    copied = object.__new__(type(original))
    shallowcopy_into(copied, original)
    return copied


@torch.no_grad()
def adapt(
    model: nn.Module,
    cfg: TeamworkConfig,
    device: torch.device,
    dtype: torch.dtype,
    override_profile: list[Adapt] | None = None,
    state: dict[str, torch.Tensor] | None = None,
    requires_grad: bool = False,
) -> nn.Module:
    """
    This function returns a new model with teamwork enabled, given a base model.

    Args:
        model: the base model to adapt, which will remain unmodified by this function
        profile: a list of globs indicating which layers should be adapted
        cfg: configuration options
        device: all adapted layers will be moved to this device
    """

    original = dict(model.named_modules())
    original[""] = model
    adapted: dict[str, nn.Module] = dict()

    # This writes any given module (or param) into the correct place
    # in the module tree _without changing the base model_, by making
    # shallow copies of all parent modules and only then modifying them.

    def create_parent_copy(name: str) -> tuple[Any, str] | None:
        if name == "":
            return None

        *parent_parts, attr = name.split(".")
        parent_name = ".".join(parent_parts)
        if parent_name not in adapted:
            if parent_name in original:
                # The parent module is in the original model, make sure we have a copy
                replace_module(parent_name, shallowcopy(original[parent_name]))
            else:
                # The parent module must itself be an adapter created earlier.
                # Get it from the adapted model, no need to copy
                grand = create_parent_copy(parent_name)
                assert grand is not None
                grandparent, grandattr = grand
                try:
                    parent = grandparent[int(grandattr)]
                except ValueError:
                    parent = getattr(grandparent, grandattr)
                adapted[parent_name] = parent
        return adapted[parent_name], attr

    def replace_module(name: str, module: nn.Module | nn.Parameter):
        if isinstance(module, nn.Module):
            adapted[name] = module

        created = create_parent_copy(name)
        if created is not None:
            parent, attr = created
            try:
                parent[int(attr)] = module
            except ValueError:
                setattr(parent, attr, module)

    # If a state dictionary is given, then it should be used to establish
    # which layers are modified (instead of using the profile). We take
    # advantage of every adapted layer having a child called "adapter".
    known = None
    if state is not None:
        known = set()
        adapter_pattern = re.compile(r"(.+).adapter(\..+)?")
        for path in state.keys():
            m = adapter_pattern.fullmatch(path)
            if m is not None:
                known.add(m.group(1))

    # Now we iterate over all the layers we should adapt, and update them.
    if override_profile is not None:
        profile = override_profile
    else:
        assert cfg.profile is not None, "cfg has no teamwork_profile"
        profile = TEAMWORK_PROFILES[cfg.profile]
    profile_counts = defaultdict(lambda: 0)
    for name, adapt in resolve_paths_to_adapt(model, profile, known=known).items():
        new = adapt_single(original[name], adapt, cfg)
        new.adapter = new.adapter.to(device=device, dtype=dtype)
        new.adapter.requires_grad_(requires_grad)
        replace_module(name, new)
        profile_counts[adapt] += 1

    # And insert parameters from the state if any.
    if state is not None:
        for path, tensor in state.items():
            replace_module(path, nn.Parameter(tensor.to(device=device, dtype=dtype)))

    # Finally check that everything in the profile did get applied. It is almost
    # always a bug/typo if one of the Adapt entries got skipped.
    for adapt in profile:
        if profile_counts[adapt] == 0:
            print(f"WARNING: teamwork adapter {adapt} was not applied to any layers")

    # Return the root adapted object.
    return adapted.get("", model)


def adapter_modules(module: nn.Module) -> OrderedDict[str, AdapterMixin]:
    return OrderedDict(
        (k, m) for (k, m) in module.named_modules() if isinstance(m, AdapterMixin)
    )


def adapter_parameters(module: nn.Module) -> OrderedDict[str, nn.Parameter]:
    return OrderedDict(
        (f"{mk}.{pk}", p)
        for (mk, m) in adapter_modules(module).items()
        if isinstance(m, nn.Module)
        for (pk, p) in m.named_parameters()
        if pk.startswith("adapter.")
    )


def save_adapters(
    module: nn.Module,
    path: str | Path,
    cfg: TeamworkConfig,
    extra_metadata: Optional[dict[str, str]] = None,
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
    extra_metadata: Optional[dict[str, str]] = None,
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

    return safetensors.torch.save_file(state_dict, path, metadata)


def load_safetensors(
    path: str | Path,
) -> Tuple[TeamworkConfig, dict[str, torch.Tensor], dict[str, str]]:
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        cfg = metadata_to_config(metadata)

        # Load parameters
        checkpoint = {}
        for key in f.keys():
            checkpoint[key] = f.get_tensor(key)

        return cfg, checkpoint, metadata
