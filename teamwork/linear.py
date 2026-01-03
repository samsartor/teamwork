"""
Implements the key idea of Teamwork: low-rank coordination and adaptation.
"""

import torch
from torch import nn, Tensor
from einops import einsum

from .adapter import AdapterMixin, TeamworkConfig, shallowcopy_into


class LoraParameters(nn.Module):
    def __init__(self, down: Tensor, up: Tensor, bias: Tensor | None):
        super().__init__()

        self.down = nn.Parameter(down)
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        self.up = nn.Parameter(up)
        nn.init.zeros_(self.up)
        if bias is not None:
            self.bias = nn.Parameter(bias)
            nn.init.zeros_(self.bias)
        else:
            self.bias = None


class TeamworkLinear(nn.Linear, AdapterMixin[LoraParameters]):
    def __init__(self, base: nn.Linear, cfg: TeamworkConfig):
        shallowcopy_into(self, base)

        self.communicate = cfg.lora_communication
        self.adapter = LoraParameters(
            torch.empty([len(cfg.teammates), self.in_features, cfg.lora_rank]),
            torch.empty([len(cfg.teammates), cfg.lora_rank, self.out_features]),
            torch.empty([len(cfg.teammates), self.out_features])
            if self.bias is not None and cfg.use_bias
            else None,
        )

    def sel(self, x: Tensor) -> Tensor:
        assert self.selection is not None
        return torch.index_select(x, 0, self.selection.teammate_indices)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        output_dtype = output.dtype
        input = input.to(self.adapter.down.dtype)
        output = output.to(self.adapter.down.dtype)

        if self.communicate:
            hidden = einsum(
                input,
                self.sel(self.adapter.down),
                self.selection.batch_matrix.to(input.dtype),
                "t ... i, t i r, t b -> b ... r",
            )
            output += einsum(
                hidden,
                self.sel(self.adapter.up),
                self.selection.batch_matrix.to(hidden.dtype),
                "b ... r, t r o, t b -> t ... o",
            )
        else:
            hidden = einsum(
                input,
                self.sel(self.adapter.down),
                "t ... i, t i r -> t ... r",
            )
            output += einsum(
                hidden,
                self.sel(self.adapter.up),
                "t ... r, t r o -> t ... o",
            )

        if self.adapter.bias is not None:
            residual = self.sel(self.adapter.bias)
            while len(residual.shape) < len(output.shape):
                residual = residual.unsqueeze(1)
            output += residual

        return output.to(output_dtype)


class TeamworkConv2d(nn.Conv2d, AdapterMixin[LoraParameters]):
    def __init__(self, base: nn.Conv2d, cfg: TeamworkConfig):
        shallowcopy_into(self, base)

        self.communicate = cfg.lora_communication
        self.adapter = LoraParameters(
            torch.empty([len(cfg.teammates), self.in_channels, cfg.lora_rank]),
            torch.empty([len(cfg.teammates), cfg.lora_rank, self.out_channels]),
            torch.empty([len(cfg.teammates), self.out_channels])
            if self.bias is not None and cfg.use_bias
            else None,
        )

    def sel(self, x: Tensor) -> Tensor:
        assert self.selection is not None
        return torch.index_select(x, 0, self.selection.teammate_indices)

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        output_dtype = output.dtype
        input = input.to(self.adapter.down.dtype)
        output = output.to(self.adapter.down.dtype)

        if self.communicate:
            hidden = einsum(
                input,
                self.sel(self.adapter.down),
                self.selection.batch_matrix.to(input.dtype),
                "t i h w, t i r, t b -> b h w r"
            )
            output += einsum(
                hidden,
                self.sel(self.adapter.up),
                self.selection.batch_matrix.to(hidden.dtype),
                "b h w r, t r o, t b -> t o h w"
            )
        else:
            hidden = einsum(
                input,
                self.sel(self.adapter.down),
                "t i h w, t i r -> t h w r",
            )
            output += einsum(
                hidden,
                self.sel(self.adapter.up),
                "t h w r, t r o -> t o h w",
            )

        if self.adapter.bias is not None:
            output += self.sel(self.adapter.bias).unsqueeze(2).unsqueeze(3)

        return output.to(output_dtype)


try:
    from diffusers.quantizers.gguf.utils import GGUFLinear

    class TeamworkGGUFLinear(GGUFLinear, AdapterMixin[LoraParameters]):
        def __init__(self, base: GGUFLinear, cfg: TeamworkConfig):
            shallowcopy_into(self, base)

            self.communicate = cfg.lora_communication
            self.adapter = LoraParameters(
                torch.empty([len(cfg.teammates), self.in_features, cfg.lora_rank]),
                torch.empty([len(cfg.teammates), cfg.lora_rank, self.out_features]),
                torch.empty([len(cfg.teammates), self.out_features])
                if self.bias is not None and cfg.use_bias
                else None,
            )

        def sel(self, x: Tensor) -> Tensor:
            assert self.selection is not None
            return torch.index_select(x, 0, self.selection.teammate_indices)

        def forward(self, input: Tensor) -> Tensor:
            output = super().forward(input)
            output_dtype = output.dtype
            input = input.to(self.adapter.down.dtype)
            output = output.to(self.adapter.down.dtype)

            if self.communicate:
                hidden = einsum(
                    input,
                    self.sel(self.adapter.down),
                    self.selection.batch_matrix.to(input.dtype),
                    "t ... i, t i r, t b -> b ... r",
                )
                output += einsum(
                    hidden,
                    self.sel(self.adapter.up),
                    self.selection.batch_matrix.to(hidden.dtype),
                    "b ... r, t r o, t b -> t ... o",
                )
            else:
                hidden = einsum(
                    input,
                    self.sel(self.adapter.down),
                    "t ... i, t i r -> t ... r",
                )
                output += einsum(
                    hidden,
                    self.sel(self.adapter.up),
                    "t ... r, t r o -> t ... o",
                )

            if self.adapter.bias is not None:
                residual = self.sel(self.adapter.bias)
                while len(residual.shape) < len(output.shape):
                    residual = residual.unsqueeze(1)
                output += residual

            return output.to(output_dtype)

except ModuleNotFoundError:
    pass
