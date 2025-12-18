from pathlib import Path
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Any, Literal
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .config import TeamworkConfig
from .adapter import Adapt, load_safetensors
from .batch import BatchBuilder, Selection

@dataclass
class LossOutput:
    prediction: Tensor
    target: Tensor
    weight: Tensor
    type: str

    @property
    def loss(self):
        return (F.mse_loss(
            self.prediction,
            self.target,
            reduction="none",
        ) * self.weight).mean()


class TeamworkPipeline(ABC):
    teamwork_config: TeamworkConfig

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        filename: str | None = None,
        base_pipeline: Any | None = None,
        training: bool = False,
        **kwargs,
    ):
        """
        Create a pipeline based on a single safetensors file.

        Args:
            path: the location on disk or the huggingface repo (if filename is provided)
            filename: the location in the huggingface repo to find the safetensors file
            base_pipeline: optionally reuse parts of an existing pipeline, instead of loading it from scratch
            training: configure the model for finetuning
            **kwargs: arguments to pass to `DiffusionPipeline.from_pretrained` if loading from scratch
        """
        if isinstance(path, str) and filename is not None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(path, filename)
        config, state, metadata = load_safetensors(path)
        if base_pipeline is None:
            from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image

            base_pipeline = AutoPipelineForText2Image.from_pretrained(
                config.base_model, **kwargs
            )
        if cls == TeamworkPipeline:
            cls = automatic_pipeline(base_pipeline, config)
        pipe = cls.from_base_pipeline(
            base_pipeline, config, state=state, training=training
        )
        pipe.load_extra_metadata(metadata)
        return pipe

    @classmethod
    @abstractmethod
    def from_base_pipeline(
        cls,
        base_pipeline: Any,
        teamwork_config: TeamworkConfig,
        override_profile: list[Adapt] | None = None,
        state: dict[str, torch.Tensor] | None = None,
        training: bool = False,
    ) -> "TeamworkPipeline":
        """
        Create a teamwork pipeline from a base diffusers pipeline, reusing as much of the base pipeline
        as possible without altering it. This is the starting point for training your own teamwork model.

        Args:
            base_pipeline: the diffusers pipeline to build on top of
            teamwork_config: everything needed to specify the teamwork adapters
            override_profile: change how different layers are adapted, eyond the default from teamwork_config
            state: parameters to use from a loaded checkpoint
            training: configure the model for finetuning
        """
        ...

    @abstractmethod
    def save_adapters(self, safetensors_path: str):
        """
        Save the trainable parameters to given path, along with all metadata.
        """
        ...

    @abstractmethod
    def load_extra_metadata(self, metadata: dict[str, str]):
        """
        Load any extra safetensors metadata unique to this pipeline, beyond what is in TeamworkConfig.
        """
        ...

    @abstractmethod
    def empty_prompts(self):
        """
        Precompute the default prompt's encodings, and then drop all text encoders.
        This is a useful memory save when the model does not use the input prompt.
        """
        ...

    @abstractmethod
    def train_loss(
        self,
        batch: BatchBuilder,
        noise: Tensor,
        model: Any | None = None,
    ) -> LossOutput:
        """
        Compute the diffusion loss on the given batch.

        Args:
            batch: the batch of data to train on
            noise: the noise to add (should be batch.size*c*h*w)
            model: model object to use if not the one attached to this pipeline (eg DistributedDataParallel)
            
            Implementations will usually take additional arguments such as prompt,
            and can have class properties such as timestep_weight.

        Returns:
            prediction: the predicted values returned from the model
            target: the expected values
            weight: a loss weighting term, either a scalar or full b*c*h*w tensor
            type: what are prediction and target (eg noise, signal, velocity)
            loss: the MSE between prediction and target, scaled by weight
        """
        ...

    @abstractmethod
    def __call__(
        self,
        images: dict[str, Any],
        request: list[str] | Literal["all"] = "all",
        noise: Tensor | None = None,
        height: int | None = None,
        width: int | None = None,
        output_type: Literal["pil", "np", "pt", "latents"] = "pil",
    ) -> dict[str, Any]:
        """
        Run the teamwork model.

        Args:
            images: images to supply to input teammates
            request: output teammates to activate
            noise: the initial noise to use, defaults to randomly generated
            height: the image height, defaults to the height of the input (if any) or to width (if given)
            width: the image width, defaults to the width of the input (if any)
            output_type: what to return

            Implementations will usually take additional arguments such as step count, prompt, etc.

        Returns: a dictionary of all generated output and provided input images
        """
        ...


def automatic_pipeline(base: Any, config: TeamworkConfig) -> type[TeamworkPipeline]:
    base_type = type(base)
    if "stable-diffusion-2" in config.base_model:
        if config.profile is not None and "RGBX" in config.profile:
            from .pipeline_rgbx import StableDiffusionRGB2XPipeline

            return StableDiffusionRGB2XPipeline
        else:
            from .pipeline_sd2 import StableDiffusion2TeamworkPipeline

            return StableDiffusion2TeamworkPipeline
    if base_type.__name__ == "StableDiffusion3Pipeline":
        if config.profile is not None and "RGBX" in config.profile:
            from .pipeline_rgbx_sd3 import StableDiffusion3RGB2XPipeline

            return StableDiffusion3RGB2XPipeline
        else:
            from .pipeline_sd3 import StableDiffusion3TeamworkPipeline

            return StableDiffusion3TeamworkPipeline
    if base_type.__name__ == "StableDiffusionXLPipeline":
        from .pipeline_sdxl import StableDiffusionXLTeamworkPipeline

        return StableDiffusionXLTeamworkPipeline
    if base_type.__name__ == "FluxPipeline":
        from .pipeline_flux import FluxTeamworkPipeline

        return FluxTeamworkPipeline
    raise NotImplementedError(
        f"automatic teamwork pipeline for {base_type} ({config.base_model})"
    )
