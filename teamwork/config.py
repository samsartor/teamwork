from dataclasses import dataclass, asdict, fields
import re
from typing import Any

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


def default_teammate_image_ids(profile: str | None, num_teammates: int) -> list[int]:
    """Default per-teammate RoPE image-id T-offset for the joint-attention layout.

    Without a joint-attention (``*PLUSATTN``) profile these offsets are unused, so
    keep every teammate aligned at 0. ``FLUX2_PLUSATTN`` mirrors klein's
    reference-image convention (T = 0, 10, 20, ... via Flux2 ``_prepare_image_ids``
    scale) so zero-LoRA joint attention reduces to base editing; other joint
    profiles use distinct adjacent coords (0, 1, 2, ...).
    """
    if profile is None or "PLUSATTN" not in profile:
        return [0] * num_teammates
    if profile == "FLUX2_PLUSATTN":
        return list(range(0, 10 * num_teammates, 10))
    return list(range(num_teammates))


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
            teammate's image tokens in the joint-attention layout. One int per teammate.
            Defaults (resolved at construction and stored in the checkpoint) come from
            ``default_teammate_image_ids``; set explicitly to control which teammates act
            as generated (T=0) vs reference (T>0) images for models like klein.

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

    def __post_init__(self):
        for teammate in self.teammates:
            if TEAMMATE_FMT.fullmatch(teammate) is None:
                raise ValueError(f"{teammate} does not match {TEAMMATE_FMT}")
        if self.attn_allow is not None:
            parse_attn_allow(self.attn_allow, len(self.teammates))
        # Resolve the default eagerly so the concrete list is stored in the
        # checkpoint (frozen dataclass, so set via object.__setattr__).
        if self.teammate_image_ids is None:
            object.__setattr__(
                self,
                "teammate_image_ids",
                default_teammate_image_ids(self.profile, len(self.teammates)),
            )
        elif len(self.teammate_image_ids) != len(self.teammates):
            raise ValueError(
                f"teammate_image_ids must have one entry per teammate "
                f"({len(self.teammates)}), got {len(self.teammate_image_ids)}"
            )


def config_to_metadata(cfg: TeamworkConfig) -> dict[str, str]:
    """Convert a TeamworkConfig to metadata dictionary format"""
    metadata = {
        "modelspec.sai_model_spec": "1.0.0",
        "modelspec.architecture": f"{cfg.base_model}/teamwork",
        "modelspec.implementation": "samsartor/teamwork",
        "modelspec.title": cfg.title,
        "modelspec.type": "teamwork",
        "base_model": cfg.base_model,
        "teamwork.version": "1.0",
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
        elif v is not None:
            metadata[f"teamwork.{k}"] = str(v)
    return metadata


def metadata_to_config(metadata: dict[str, str]) -> TeamworkConfig:
    """Convert metadata dictionary back to TeamworkConfig"""
    cfg_dict: dict[str, Any] = {}
    if metadata.get("modelspec.type") != "teamwork":
        raise ValueError("checkpoint does not appear to be a teamwork model")
    cfg_dict["base_model"] = metadata.get("base_model", "unknown")
    cfg_dict["title"] = metadata.get("modelspec.title", "Teamwork Model")
    cfg_dict["resolution"] = metadata.get("modelspec.resolution")
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
            elif k == "lora_rank":
                cfg_dict[k] = int(v)
            elif k in ["lora_communication", "use_bias"]:
                cfg_dict[k] = v.lower() == "true"
            else:
                cfg_dict[k] = v
    return TeamworkConfig(**cfg_dict)
