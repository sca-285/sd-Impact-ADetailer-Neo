from __future__ import annotations

from dataclasses import dataclass

from .settings import ORDINALS, max_tabs

UNIT_COUNT = 8  # hard cap; live count is settings.max_tabs()


@dataclass
class RunArgs:
    """Global controls above the tabs. Not written to infotext."""

    enable: bool = False
    save_before: bool = False
    preview: str = "Off"


@dataclass
class UnitArgs:
    enabled: bool = False
    model: str = "None"
    prompt: str = ""
    negative_prompt: str = "blurry, distorted, unnatural, unclear"
    confidence: float = 0.5
    filter_by: str = "Confidence"
    mask_k: int = 0
    min_ratio: float = 0.0
    max_ratio: float = 1.0
    x_offset: int = 0
    y_offset: int = 0
    dilate: int = 4
    feather: int = 12
    denoise: float = 0.5
    noise_mask: bool = True
    padding: int = 32
    use_wh: bool = False
    inpaint_w: int = 512
    inpaint_h: int = 512
    use_separate_steps: bool = True
    steps: int = 20
    use_separate_cfg: bool = True
    cfg: float = 4.0
    sampler: str = "DPM++ 2M"
    scheduler: str = "Karras"
    use_separate_sampler: bool = True
    use_separate_checkpoint: bool = False
    checkpoint: str = "Use same checkpoint"
    use_separate_vae: bool = False
    vae: str = "Use same VAE"
    crop_factor: float = 3.0
    guide_size: int = 768
    max_size: int = 1536
    guide_size_for_bbox: bool = True
    force_inpaint: bool = True
    cycle: int = 3
    drop_size: int = 10
    noise_mask_feather: int = 5
    use_sam: bool = False
    sam_model: str = "None"
    use_clip_filter: bool = False
    clip_model: str = "None"
    clip_threshold: float = 0.2
    classes: str = "face, hand, -ear"
    merge_mode: str = "None"
    cn_model: str = "None"
    cn_module: str = "None"
    cn_weight: float = 1.0
    cn_start: float = 0.0
    cn_end: float = 1.0
