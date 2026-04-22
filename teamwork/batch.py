from dataclasses import dataclass
import torch
from torch import Tensor
import numpy as np
from random import randint
from typing import Callable, TypeAlias, Any, Literal
import PIL.Image


@dataclass(frozen=True)
class Selection:
    enabled: bool
    teammate_indices: Tensor # (components,): int
    input_subindices: Tensor  # (input_components,): int
    output_subindices: Tensor  # (output_components,): int
    batch_indices: Tensor # (components): int
    batch_matrix: Tensor # (components, batch): bool
    attn_keep: Tensor | None = None # (components, components): bool, per-pair cross-teammate attention mask


@dataclass
class ComponentInBatch:
    teammate: int
    batch_idx: int
    component: str
    image: Tensor | None
    extra: Tensor | None
    weight: Tensor | None
    cropinfo: Tensor | None
    output: bool


def image_pt2np(image_pt: Tensor):
    image_pt = image_pt * 0.5 + 0.5  # Scale to [0, 1] range
    image_np = image_pt.cpu().float().permute(1, 2, 0).numpy()
    image_np = image_np.clip(0, 1)
    return image_np


def image_np2pil(image_np: np.ndarray):
    if image_np.dtype != np.uint8:
        image_np = (image_np.clip(0, 1) * 255).astype(np.uint8)
    return PIL.Image.fromarray(image_np)


def image_pt2pil(image_pt: Tensor):
    return image_np2pil(image_pt2np(image_pt))


OutputImageType: TypeAlias = Literal["pil", "np", "pt", "latents"]


class BatchBuilder:
    """
    Teamwork requires arranging input images and empty outputs into a batch, with information
    about which batch components corrispond to which model teammates. That is really a pain
    to do in each pipeline, so here is a shared helper class. It also handles all the usual
    image processing logic.
    """

    def __init__(
        self,
        teammates: list[str],
        device: torch.device,
        dtype: torch.dtype,
        dropout_prob: float = 0.0,
    ):
        self.teammates = teammates
        self.device = device
        self.dtype = dtype
        self.dropout_prob = dropout_prob
        self.components: list[ComponentInBatch] = []
        self.resolution: tuple[int, int] | None = None

    @classmethod
    def from_inputs(
        cls,
        teammates: list[str],
        device: torch.device,
        dtype: torch.dtype,
        images: dict[str, Any] | list[dict[str, Any]],
        request: list[str] | Literal["all"] = "all",
        height: int | None = None,
        width: int | None = None,
        dropout_prob: float = 0.0,
    ):
        batch = cls(teammates, device, dtype, dropout_prob=dropout_prob)
        if not isinstance(images, list):
            images = [images]
        for batch_idx, images_dict in enumerate(images):
            for name, image in images_dict.items():
                batch.provide(name, image, batch_idx=batch_idx)
            if request == "all":
                batch.request_all(width=width, height=height, batch_idx=batch_idx)
            else:
                batch.request(*request, width=width, height=height, batch_idx=batch_idx)
        return batch

    def provide(
        self,
        component: str,
        image: np.ndarray | Tensor | PIL.Image.Image,
        weight: np.ndarray | Tensor | None = None,
        extra: np.ndarray | Tensor | None = None,
        cropinfo: np.ndarray | None = None,
        kind: str | None = None,
        image_format: str = "auto",
        batch_idx: int = 0,
    ):
        """
        Provide a known image, either as input or to compute output loss.
        """

        if isinstance(image, PIL.Image.Image):
            image_np = np.array(image.convert("RGB"))
            image_tensor = torch.from_numpy(image_np)
            if image_format == "auto":
                image_format = "np"
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.expand_dims(image, 2)
            image_tensor = torch.from_numpy(image)
            if image_format == "auto":
                image_format = "np"
        elif isinstance(image, torch.Tensor):
            image_tensor = image
            if image_format == "auto":
                image_format = "pt"
        else:
            raise ValueError(
                f"image type {type(image)} is not a pil image, numpy array, or torch tensor"
            )

        if image_tensor.dtype == torch.uint8:
            image_tensor = image_tensor.to(torch.float32) / 255.0
        elif image_tensor.dtype == torch.uint16:
            image_tensor = image_tensor.to(torch.float32) / 65535.0
        else:
            pass

        if image_format == "np":
            image_tensor = image_tensor * 2 - 1
            image_tensor = image_tensor.permute(2, 0, 1)

        if image_tensor.shape[0] == 1:
            image_tensor = image_tensor.repeat(3, 1, 1)

        if image_tensor.shape[0] == 4:
            image_tensor = image_tensor[:3, :, :]

        image_tensor = image_tensor.nan_to_num()

        # Handle weight if present
        weight_tensor = None
        if weight is not None:
            if isinstance(weight, torch.Tensor):
                weight_tensor = weight.to(self.device, self.dtype)
            else:
                weight_tensor = torch.from_numpy(weight).to(self.device, self.dtype)
            if len(weight_tensor.shape) == 3:
                assert weight_tensor.shape[-1] == 1 or weight_tensor.shape[-1] == 3
                weight_tensor = weight_tensor[:, :, 1]

        # Handle concat channels if present
        concat_tensor = None
        if extra is not None:
            if isinstance(extra, torch.Tensor):
                concat_tensor = extra.to(self.device, self.dtype)
                assert len(concat_tensor.shape) == 3, (
                    f"concat tensor should have C,H,W (had shape {concat_tensor.shape})"
                )
            else:
                concat_tensor = torch.from_numpy(extra).to(self.device, self.dtype)
                assert len(concat_tensor.shape) == 3, (
                    f"concat ndarray should have H,W,C (had shape {concat_tensor.shape})"
                )
                concat_tensor = concat_tensor.permute(2, 0, 1)

        # Handle cropinfo if present
        cropinfo_tensor = None
        if cropinfo is not None:
            cropinfo_tensor = torch.from_numpy(cropinfo).to(self.device, self.dtype)

        _, h, w = image_tensor.shape
        if self.resolution is None:
            self.resolution = (h, w)
        else:
            assert self.resolution == (h, w), (
                f"component size {(h, w)} is inconsistant, was previously {self.resolution}"
            )

        if weight_tensor is not None:
            assert weight_tensor.shape == self.resolution, (
                f"weight shape {weight_tensor.shape} should be {self.resolution}"
            )

        if concat_tensor is not None:
            assert concat_tensor.shape[1:] == self.resolution, (
                f"concat shape {concat_tensor.shape} should have resolution {self.resolution}"
            )

        # Find matching teammate
        for idx, teammate in enumerate(self.teammates):
            teammate_component, teammate_kind = teammate.split(".")
            if component == teammate_component and (
                kind == teammate_kind or kind is None
            ):
                self.components.append(
                    ComponentInBatch(
                        teammate=idx,
                        batch_idx=batch_idx,
                        component=component,
                        image=image_tensor,
                        extra=concat_tensor,
                        weight=weight_tensor,
                        cropinfo=cropinfo_tensor,
                        output=(teammate_kind != "in"),
                    )
                )

    def request_all(self, width: int | None = None, height: int | None = None, batch_idx: int = 0):
        for teammate in self.teammates:
            teammate_component, teammate_kind = teammate.split(".")
            if teammate_kind == "out":
                self.request(teammate_component, width=width, height=height, batch_idx=batch_idx)

    def request(
        self,
        *components: str,
        width: int | None = None,
        height: int | None = None,
        batch_idx: int = 0,
    ):
        """
        Request that the model produce some outputs and include a placeholder. The resolution
        is infered from any previously provided input image, and otherwise.
        """

        if self.resolution is None:
            assert width is not None, "resolution must be known or provided"
            self.resolution = (height or width, width)
        if height is not None:
            assert self.resolution[0] == height
        if width is not None:
            assert self.resolution[1] == width

        for component in components:
            # Find matching teammate for output
            for idx, teammate in enumerate(self.teammates):
                teammate_component, teammate_kind = teammate.split(".")
                if teammate_component == component and teammate_kind == "out":
                    if any(
                        existing.teammate == idx and existing.batch_idx == batch_idx
                        for existing in self.components
                    ):
                        continue
                    self.components.append(
                        ComponentInBatch(
                            teammate=idx,
                            batch_idx=batch_idx,
                            component=component,
                            image=None,
                            extra=None,
                            weight=None,
                            cropinfo=None,
                            output=True,
                        )
                    )

    def split_io(self) -> tuple["BatchBuilder", "BatchBuilder"]:
        inputs = BatchBuilder(self.teammates, self.device, self.dtype, dropout_prob=self.dropout_prob)
        inputs.resolution = self.resolution
        outputs = BatchBuilder(self.teammates, self.device, self.dtype, dropout_prob=self.dropout_prob)
        outputs.resolution = self.resolution
        for component in self.components:
            if component.output:
                outputs.components.append(component)
            else:
                inputs.components.append(component)
        return (inputs, outputs)

    def crop_to_square(self):
        assert self.resolution is not None, "resolution must be known"
        h, w = self.resolution
        s = min(h, w)
        i = randint(0, h - s)
        j = randint(0, w - s)
        self.resolution = (s, s)
        for component in self.components:
            if component.image is not None:
                component.image = component.image[:, i : i + s, j : j + s]
            if component.weight is not None:
                component.weight = component.weight[i : i + s, j : j + s]
            if component.extra is not None:
                component.extra = component.extra[:, i : i + s, j : j + s]

    def packed_images(self) -> Tensor:
        return self.packed_encoded_images(lambda x: x, 3, scale=1.0)

    def packed_encoded_images(
        self,
        encode: Callable[[Tensor], Tensor],
        channels: int,
        scale: float,
    ) -> Tensor:
        to_pack = []
        for c in self.components:
            if c.image is not None:
                encoded_image = encode(c.image.unsqueeze(0)).squeeze(0)
                found_channels = encoded_image.shape[0]
                assert found_channels == channels, (
                    f"found {c.component} image with {found_channels} channels, expected {channels}"
                )
                to_pack.append(encoded_image.to(self.device, self.dtype))
            else:
                assert self.resolution is not None
                h, w = self.resolution
                zeros = torch.zeros(
                    (channels, int(h * scale), int(w * scale)),
                    device=self.device,
                    dtype=self.dtype,
                )
                to_pack.append(zeros)

        return (
            torch.stack(to_pack)
            if to_pack
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_extra(self) -> Tensor | None:
        extra = []
        for component in self.components:
            if component.extra is not None:
                extra.append(component.extra.to(self.device, self.dtype))
            else:
                assert not extra, "only some components had extra channels"
                return None

        return (
            torch.stack(extra)
            if extra
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_scaled_extra(self, scale: float) -> Tensor | None:
        extra = self.packed_extra()
        if extra is None or extra.numel() == 0:
            return extra

        assert self.resolution is not None
        h, w = self.resolution
        target_size = (int(h * scale), int(w * scale))

        # Interpolate (already has channel dimension)
        extra = torch.nn.functional.interpolate(
            extra,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        return extra

    def packed_weights(self) -> Tensor:
        weights = []
        for component in self.components:
            if component.weight is not None:
                weights.append(component.weight.to(self.device, self.dtype))
            else:
                # For missing weights, the weight defaults to one
                assert self.resolution is not None
                zeros = torch.ones(
                    self.resolution, device=self.device, dtype=self.dtype
                )
                weights.append(zeros)

        return (
            torch.stack(weights)
            if weights
            else torch.empty(0, device=self.device, dtype=self.dtype)
        )

    def packed_scaled_weights(self, scale: float) -> Tensor:
        weights = self.packed_weights()
        if weights.numel() == 0:
            return weights

        assert self.resolution is not None
        h, w = self.resolution
        target_size = (int(h * scale), int(w * scale))

        # Add channel dimension for interpolation, then remove it
        weights = weights.unsqueeze(1)  # (batch, 1, h, w)
        weights = torch.nn.functional.interpolate(
            weights,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        weights = weights.squeeze(1)  # (batch, h', w')

        return weights

    def packed_cropinfo(self, default: Tensor) -> Tensor:
        cropinfos = []
        for component in self.components:
            if component.cropinfo is not None:
                cropinfos.append(component.cropinfo.to(self.device, default.dtype))
            else:
                cropinfos.append(default.to(self.device))

        if len(cropinfos) == 0:
            return torch.empty((0, 6), device=self.device, dtype=default.dtype)

        return torch.stack(cropinfos)

    def selection(self) -> Selection:
        teammate_indices = []
        input_indices = []
        output_indices = []
        batch_indices = []
        batch_matrix = torch.zeros(self.count, self.batch_size, dtype=torch.bool)

        for i, component in enumerate(self.components):
            teammate_indices.append(component.teammate)
            if component.output:
                output_indices.append(i)
            else:
                input_indices.append(i)
            batch_indices.append(component.batch_idx)
            batch_matrix[i, component.batch_idx] = True

        attn_keep: Tensor | None = None
        if self.dropout_prob > 0 and self.count > 1:
            keep = torch.bernoulli(
                torch.full((self.count, self.count), 1.0 - self.dropout_prob)
            ).bool()
            keep.fill_diagonal_(True)
            attn_keep = keep.to(self.device)

        return Selection(
            enabled=True,
            teammate_indices=torch.tensor(
                teammate_indices, dtype=torch.int64, device=self.device
            ),
            input_subindices=torch.tensor(
                input_indices, dtype=torch.int64, device=self.device
            ),
            output_subindices=torch.tensor(
                output_indices, dtype=torch.int64, device=self.device
            ),
            batch_indices=torch.tensor(
                batch_indices, dtype=torch.int64, device=self.device
            ),
            batch_matrix=batch_matrix.to(self.device),
            attn_keep=attn_keep,
        )

    def unpack_decoded_images(
        self,
        packed: Tensor,
        decode: Callable[[Tensor], Tensor],
        output_type: OutputImageType = 'pil',
        outputs_only: bool = True,
    ) -> list[dict[str, Any]]:
        unpacked = [dict() for _ in range(self.batch_size)]
        for idx, component in enumerate(self.components):
            if component.output or not outputs_only:
                match output_type:
                    case 'latents':
                        image = packed[idx]
                    case 'pt':
                        image = decode(packed[idx].unsqueeze(0)).squeeze(0)
                    case 'np':
                        image = image_pt2np(decode(packed[idx].unsqueeze(0)).squeeze(0))
                    case 'pil':
                        image = image_pt2pil(decode(packed[idx].unsqueeze(0)).squeeze(0))
                    case _:
                        raise ValueError(f'unkonwn output type "{output_type}"')
                unpacked[component.batch_idx][component.component] = image
        return unpacked

    def unpack_images(
        self,
        packed: Tensor,
        output_type: OutputImageType = 'pil',
        outputs_only: bool = True,
    ) -> list[dict[str, Tensor]]:
        assert output_type != 'latents', 'images are not encoded'
        return self.unpack_decoded_images(packed, lambda x: x, output_type=output_type, outputs_only=outputs_only)

    def unpacked_images(self, output_type: OutputImageType = 'pil', outputs_only: bool = False) -> list[dict[str, Tensor]]:
        return self.unpack_images(self, self.packed_images(), output_type=output_type, outputs_only=outputs_only)

    @property
    def count(self):
        return len(self.components)

    @property
    def batch_size(self):
        if self.count == 0:
            return 0
        else:
            return max(c.batch_idx for c in self.components) + 1
        
