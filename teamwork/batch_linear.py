"""
The batch-axis implementation of teamwork's key idea: low-rank coordination and adaptation.

Every component (one teammate's image in one batch element) is a row of the batch
dimension. Each row gets its own teammate's LoRA, and with `lora_communication` the LoRA
hidden states are summed over the rows of each batch element, which is the only way
teammates communicate.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import einsum

from .adapter import AdapterMixin, LoraParameters, shallowcopy_into
from .config import TeamworkConfig
from .data import Components


@dataclass(frozen=True)
class BatchSelection(Components):
    """The components index plus the one thing batch adapters need to communicate."""

    batch_matrix: Tensor  # (components, batch): bool, batch element of each component

    @classmethod
    def from_components(cls, components: Components) -> "BatchSelection":
        batch_matrix = F.one_hot(components.batch_indices, components.batch_size).bool()
        return cls(
            teammate_indices=components.teammate_indices,
            batch_indices=components.batch_indices,
            input_subindices=components.input_subindices,
            output_subindices=components.output_subindices,
            num_teammates=components.num_teammates,
            batch_size=components.batch_size,
            batch_matrix=batch_matrix,
        )


def _new_lora(cfg: TeamworkConfig, in_features: int, out_features: int, has_bias: bool) -> LoraParameters:
    num_teammates = len(cfg.teammates)
    return LoraParameters(
        torch.empty([num_teammates, in_features, cfg.lora_rank]),
        torch.empty([num_teammates, cfg.lora_rank, out_features]),
        torch.empty([num_teammates, out_features]) if has_bias and cfg.use_bias else None,
    )


def _batch_lora(
    adapter: LoraParameters,
    sel: BatchSelection,
    communicate: bool,
    input: Tensor,  # (components, ..., in)
    output: Tensor,  # (components, ..., out), the base layer's output
) -> Tensor:
    """Add the per-row LoRA (and bias) to the base output of a linear layer."""
    output_dtype = output.dtype
    input = input.to(adapter.down.dtype)
    output = output.to(adapter.down.dtype)
    down = torch.index_select(adapter.down, 0, sel.teammate_indices)
    up = torch.index_select(adapter.up, 0, sel.teammate_indices)

    if communicate:
        hidden = einsum(
            input, down, sel.batch_matrix.to(input.dtype),
            "t ... i, t i r, t b -> b ... r",
        )
        output += einsum(
            hidden, up, sel.batch_matrix.to(hidden.dtype),
            "b ... r, t r o, t b -> t ... o",
        )
    else:
        hidden = einsum(input, down, "t ... i, t i r -> t ... r")
        output += einsum(hidden, up, "t ... r, t r o -> t ... o")

    if adapter.bias is not None:
        residual = torch.index_select(adapter.bias, 0, sel.teammate_indices)
        while len(residual.shape) < len(output.shape):
            residual = residual.unsqueeze(1)
        output += residual

    return output.to(output_dtype)


class BatchTeamworkLinear(nn.Linear, AdapterMixin[BatchSelection, LoraParameters]):
    def __init__(self, base: nn.Linear, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.communicate = cfg.lora_communication
        self.adapter = _new_lora(cfg, self.in_features, self.out_features, self.bias is not None)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.selection is None:
            return output
        return _batch_lora(self.adapter, self.selection, self.communicate, input, output)


class BatchTeamworkConv2d(nn.Conv2d, AdapterMixin[BatchSelection, LoraParameters]):
    def __init__(self, base: nn.Conv2d, cfg: TeamworkConfig):
        shallowcopy_into(self, base)
        self.communicate = cfg.lora_communication
        self.adapter = _new_lora(cfg, self.in_channels, self.out_channels, self.bias is not None)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        sel = self.selection
        if sel is None:
            return output
        output_dtype = output.dtype
        input = input.to(self.adapter.down.dtype)
        output = output.to(self.adapter.down.dtype)
        down = torch.index_select(self.adapter.down, 0, sel.teammate_indices)
        up = torch.index_select(self.adapter.up, 0, sel.teammate_indices)

        if self.communicate:
            hidden = einsum(
                input, down, sel.batch_matrix.to(input.dtype),
                "t i h w, t i r, t b -> b h w r",
            )
            output += einsum(
                hidden, up, sel.batch_matrix.to(hidden.dtype),
                "b h w r, t r o, t b -> t o h w",
            )
        else:
            hidden = einsum(input, down, "t i h w, t i r -> t h w r")
            output += einsum(hidden, up, "t h w r, t r o -> t o h w")

        if self.adapter.bias is not None:
            output += torch.index_select(self.adapter.bias, 0, sel.teammate_indices).unsqueeze(2).unsqueeze(3)

        return output.to(output_dtype)


try:
    from diffusers.quantizers.gguf.utils import GGUFLinear

    class BatchTeamworkGGUFLinear(GGUFLinear, AdapterMixin[BatchSelection, LoraParameters]):
        def __init__(self, base: GGUFLinear, cfg: TeamworkConfig):
            shallowcopy_into(self, base)
            self.communicate = cfg.lora_communication
            self.adapter = _new_lora(cfg, self.in_features, self.out_features, self.bias is not None)

        def forward(self, input: Tensor) -> Tensor:
            output = super().forward(input)
            if self.selection is None:
                return output
            return _batch_lora(self.adapter, self.selection, self.communicate, input, output)

except ModuleNotFoundError:
    pass
