from dataclasses import dataclass, asdict, fields
import re
from typing import Any, Literal, get_args

TEAMMATE_FMT = re.compile(r"[a-z]+\.(in|out)")


def parse_attn_allow(s: str, num_teammates: int) -> list[list[bool]]:
    """
    Parse a row-major bool matrix from a serialized string like ``"100;011;011"``.

    Whitespace is tolerated within and between rows. The matrix must be square of
    side ``num_teammates`` and have an all-True diagonal (a teammate must always
    be allowed to attend to itself).
    """
    rows = [r.strip().replace(" ", "") for r in s.split(";")]
    rows = [r for r in rows if r]
    if len(rows) != num_teammates:
        raise ValueError(
            f"attn_allow must have {num_teammates} rows, got {len(rows)}: {s!r}"
        )
    matrix: list[list[bool]] = []
    for i, row in enumerate(rows):
        if len(row) != num_teammates:
            raise ValueError(
                f"attn_allow row {i} must have {num_teammates} chars, got {len(row)}: {row!r}"
            )
        if any(c not in "01" for c in row):
            raise ValueError(
                f"attn_allow row {i} must contain only '0'/'1', got {row!r}"
            )
        bools = [c == "1" for c in row]
        if not bools[i]:
            raise ValueError(
                f"attn_allow diagonal must be '1' at position {i}, got '0' in row {row!r}"
            )
        matrix.append(bools)
    return matrix


def default_teammate_image_ids(profile: str | None, layout: str, num_teammates: int) -> list[int]:
    """Default per-teammate RoPE image-id T-offset for the sequence layout.

    In the batch layout these offsets are unused, so keep every teammate aligned at 0.
    FLUX.2 Klein mirrors its reference-image convention (T = 0, 10, 20, ... via Flux2
    ``_prepare_image_ids`` scale) so zero-LoRA sequence teamwork reduces to base editing;
    other FLUX-style families use distinct adjacent coords (0, 1, 2, ...). Families with
    their own position rule (Qwen-Image 2.1) ignore these ids.
    """
    if layout != "sequence":
        return [0] * num_teammates
    if profile == "FLUX2":
        return list(range(0, 10 * num_teammates, 10))
    return list(range(num_teammates))


# 1.1: FLUX.2 Klein trains/samples on BN-normalized posterior-mean latents.
# 2.0: adds `layout`, `text_teammate`, and `vlm_inputs`. Klein latents are as in 1.1.
CURRENT_VERSION = "2.0"
KNOWN_VERSIONS = ("1.0", "1.1", "2.0")
# Versions whose metadata predates `layout`, migrated by `metadata_to_config`.
PRE_LAYOUT_VERSIONS = ("1.0", "1.1")

Layout = Literal["batch", "sequence"]


def is_joint_attention_profile(profile: str | None) -> bool:
    """The legacy `*_PLUSATTN` profile names (before 2.0, the only sequence-layout checkpoints)."""
    return profile is not None and profile.endswith("_PLUSATTN")


@dataclass(frozen=True)
class TeamworkConfig:
    """
    Configures the teamwork adaption of a model.

    Teamwork-Specific Attributes:
        teammates: list of all teammates (e.g. 'image.in', 'albedo.out') to support, in the order their parameters are stored
        profile: identifies the default set of layers to adapt and the adapters to use
        lora_rank: the default rank of teamwork LoRAs
        lora_communication: whether to communicate between teammates via LoRA layers, or keep the LoRAs separate (useful as a baseline)
        use_bias: whether to train per-teammate bias for linear layers with a bias
        attn_allow: row-major bool matrix (e.g. ``"100;011;011"``) of allowed cross-teammate
            attention edges, or None for "all allowed". Diagonal must be ``1``. Combined
            multiplicatively with attention dropout to produce the runtime ``attn_keep``.
        teammate_image_ids: per-teammate RoPE image-id T-offset used to place each
            teammate's image tokens in the sequence layout. One int per teammate.
            Defaults (resolved at construction and stored in the checkpoint) come from
            ``default_teammate_image_ids``; set explicitly to control which teammates act
            as generated (T=0) vs reference (T>0) images for models like klein.
        layout: how teammates are arranged for the model. ``"batch"``: each teammate is a row
            of the batch and teammates only communicate through LoRA. ``"sequence"``: the
            teammates of a batch element share one token stream and attend to each other.
            Pre-2.0 joint-attention (``*_PLUSATTN``) checkpoints migrate to it.
        text_teammate: the roster name whose adapter applies to the text tokens in the sequence
            layout, or None to leave text at the base model.
        vlm_inputs: the ``.in`` teammates fed to the model's vision-language text encoder.
        version: checkpoint format version. New configs get ``CURRENT_VERSION``; checkpoints
            saved before the field existed load as ``"1.0"``. Pipelines branch on it where a
            fix changed what trained weights expect (e.g. 1.0 FLUX.2 Klein checkpoints were
            trained on raw, sampled VAE latents instead of BN-normalized posterior means).

    General Attributes:
        base_model: The checkpoint or model name from which the adapter was or will be trained
        title: A human-readable name for the adapter
        resolution: The resolution the model was finetuned at (eg "1024x1024")
    """

    teammates: list[str]
    lora_rank: int

    base_model: str
    title: str
    resolution: str | None = None

    profile: str | None = None
    lora_communication: bool = True
    use_bias: bool = True
    attn_allow: str | None = None
    teammate_image_ids: list[int] | None = None
    layout: Layout = "batch"
    text_teammate: str | None = None
    vlm_inputs: tuple[str, ...] = ()
    version: str = CURRENT_VERSION

    def __post_init__(self):
        for teammate in self.teammates:
            if TEAMMATE_FMT.fullmatch(teammate) is None:
                raise ValueError(f"{teammate} does not match {TEAMMATE_FMT}")
        if self.attn_allow is not None:
            parse_attn_allow(self.attn_allow, len(self.teammates))
        # Resolve the default eagerly so the concrete list is stored in the
        # checkpoint (frozen dataclass, so set via object.__setattr__).
        if self.layout not in get_args(Layout):
            raise ValueError(f"unknown layout {self.layout!r}, expected one of {get_args(Layout)}")
        if self.teammate_image_ids is None:
            object.__setattr__(
                self,
                "teammate_image_ids",
                default_teammate_image_ids(self.profile, self.layout, len(self.teammates)),
            )
        elif len(self.teammate_image_ids) != len(self.teammates):
            raise ValueError(
                f"teammate_image_ids must have one entry per teammate "
                f"({len(self.teammates)}), got {len(self.teammate_image_ids)}"
            )
        if self.text_teammate is not None:
            if self.text_teammate not in self.teammates:
                raise ValueError(f"text_teammate {self.text_teammate!r} is not in the roster {self.teammates}")
            if self.layout != "sequence":
                raise ValueError(f"text_teammate requires the sequence layout, not {self.layout!r}")
        object.__setattr__(self, "vlm_inputs", tuple(self.vlm_inputs))
        if len(set(self.vlm_inputs)) != len(self.vlm_inputs):
            raise ValueError(f"vlm_inputs has duplicates: {self.vlm_inputs}")
        for teammate in self.vlm_inputs:
            if teammate not in self.teammates or not teammate.endswith(".in"):
                raise ValueError(f"vlm_inputs entry {teammate!r} is not an input (.in) teammate of {self.teammates}")
        if self.version not in KNOWN_VERSIONS:
            raise ValueError(f"unknown teamwork version {self.version!r}, expected one of {KNOWN_VERSIONS}")


def config_to_metadata(cfg: TeamworkConfig) -> dict[str, str]:
    """Convert a TeamworkConfig to metadata dictionary format"""
    metadata = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": f"{cfg.base_model}/teamwork",
        "modelspec.implementation": "samsartor/teamwork",
        "modelspec.title": cfg.title,
        "modelspec.type": "teamwork",
        "base_model": cfg.base_model,
    }
    if cfg.resolution is not None:
        metadata["modelspec.resolution"] = cfg.resolution
    for k, v in asdict(cfg).items():
        if k == "base_model" or k == "title" or k == "resolution":
            pass
        elif k == "teammates":
            metadata["teamwork.teammates"] = ",".join(v)
        elif k == "teammate_image_ids":
            metadata["teamwork.teammate_image_ids"] = ",".join(str(x) for x in v)
        elif k == "vlm_inputs":
            metadata["teamwork.vlm_inputs"] = ",".join(v)
        elif v is not None:
            metadata[f"teamwork.{k}"] = str(v)
    return metadata


def metadata_to_config(metadata: dict[str, str]) -> TeamworkConfig:
    """
    Convert metadata dictionary back to TeamworkConfig.

    Metadata written before 2.0 has no `layout`: the joint-attention (``*_PLUSATTN``)
    profiles migrate to the sequence layout with the first ``.out`` teammate adapting the
    text (as they were trained), and everything else to the batch layout.
    """
    cfg_dict: dict[str, Any] = {}
    if metadata.get("modelspec.type") != "teamwork":
        raise ValueError("checkpoint does not appear to be a teamwork model")
    cfg_dict["base_model"] = metadata.get("base_model", "unknown")
    cfg_dict["title"] = metadata.get("modelspec.title", "Teamwork Model")
    cfg_dict["resolution"] = metadata.get("modelspec.resolution")
    # Checkpoints from before the version field predate every format change.
    cfg_dict["version"] = metadata.get("teamwork.version", "1.0")
    known = {f.name for f in fields(TeamworkConfig)}
    for k, v in metadata.items():
        if k.startswith("teamwork."):
            k = k.removeprefix("teamwork.")
            if k not in known:
                continue
            if k == "teammates":
                cfg_dict[k] = v.split(",")
            elif k == "teammate_image_ids":
                cfg_dict[k] = [int(x) for x in v.split(",")] if v else []
            elif k == "vlm_inputs":
                cfg_dict[k] = tuple(v.split(",")) if v else ()
            elif k == "lora_rank":
                cfg_dict[k] = int(v)
            elif k in ["lora_communication", "use_bias"]:
                cfg_dict[k] = v.lower() == "true"
            else:
                cfg_dict[k] = v
    if "layout" not in cfg_dict:
        if cfg_dict["version"] not in PRE_LAYOUT_VERSIONS:
            raise ValueError(f"teamwork {cfg_dict['version']} metadata is missing teamwork.layout")
        assert "text_teammate" not in cfg_dict, "text_teammate without layout"
        if is_joint_attention_profile(cfg_dict.get("profile")):
            # The sequence layout is now a config field, not a profile: FLUX2_PLUSATTN -> FLUX2.
            cfg_dict["profile"] = cfg_dict["profile"].removesuffix("_PLUSATTN")
            cfg_dict["layout"] = "sequence"
            cfg_dict["text_teammate"] = next(t for t in cfg_dict["teammates"] if t.endswith(".out"))
        else:
            cfg_dict["layout"] = "batch"
    return TeamworkConfig(**cfg_dict)
