from __future__ import annotations

import gradio as gr

from . import uistate
from .args import ORDINALS, UNIT_COUNT, RunArgs, UnitArgs
from .detect import list_detector_models
from .preview import MODES as PREVIEW_MODES
from .settings import max_tabs, remember_ui
from .modelswitch import SAME_CHECKPOINT, SAME_VAE, list_checkpoints, list_vaes

# The controls above the tabs, in the order build_ui returns them. Everything
# after these is the repeating per-tab block.
HEAD_KEYS = ["enable", "save_before", "preview"]

def _sam_choices() -> list[str]:
    try:
        from .sam import list_sam_models

        return list_sam_models()
    except Exception:
        return ["None"]


def _clip_choices() -> list[str]:
    try:
        from .clip_filter import list_clip_models

        return list_clip_models()
    except Exception:
        return ["None"]


FIELD_KEYS: list[tuple[str, str | None]] = [
    ("enabled", None),
    ("model", "model"),
    ("prompt", "prompt"),
    ("negative_prompt", "negative prompt"),
    ("confidence", "confidence"),
    ("filter_by", None),
    ("mask_k", "top k"),
    ("min_ratio", None),
    ("max_ratio", None),
    ("x_offset", None),
    ("y_offset", None),
    ("dilate", "dilation"),
    ("feather", "feather"),
    ("denoise", "denoising strength"),
    ("noise_mask", None),
    ("padding", None),
    ("use_wh", None),
    ("inpaint_w", None),
    ("inpaint_h", None),
    ("use_separate_steps", None),
    ("steps", "steps"),
    ("use_separate_cfg", None),
    ("cfg", "cfg scale"),
    ("use_separate_sampler", None),
    ("sampler", "sampler"),
    ("scheduler", "scheduler"),
    ("use_separate_checkpoint", None),
    ("checkpoint", "checkpoint"),
    ("use_separate_vae", None),
    ("vae", "vae"),
    ("crop_factor", "crop factor"),
    ("guide_size", "guide size"),
    ("max_size", "max size"),
    ("guide_size_for_bbox", None),
    ("force_inpaint", None),
    ("cycle", "cycle"),
    ("drop_size", None),
    ("noise_mask_feather", "noise mask feather"),
    ("use_sam", None),
    ("sam_model", "sam"),
    ("use_clip_filter", None),
    ("clip_model", "clip"),
    ("clip_threshold", None),
    ("classes", "classes"),
    ("merge_mode", "merge mode"),
    ("cn_model", "controlnet model"),
    ("cn_module", "controlnet module"),
    ("cn_weight", "controlnet weight"),
    ("cn_start", None),
    ("cn_end", None),
]

MERGE_MODES = ["None", "Merge", "Merge and invert"]


def _checkpoints() -> list[str]:
    try:
        names = list_checkpoints()
        return names or [SAME_CHECKPOINT]
    except Exception:
        return [SAME_CHECKPOINT]


def _vaes() -> list[str]:
    try:
        names = list_vaes()
        return names or [SAME_VAE, "None", "Automatic"]
    except Exception:
        return [SAME_VAE, "None", "Automatic"]


def _cn_choices() -> tuple[list[str], list[str], bool]:
    try:
        from .controlnet import available, list_models, list_modules

        return list_models(), list_modules(), available()
    except Exception:
        return ["None"], ["None"], False


def _samplers() -> list[str]:
    names = ["Use same sampler"]
    try:
        from modules import sd_samplers

        names.extend(sd_samplers.visible_sampler_names())
    except Exception:
        names.extend(["Euler a", "Euler", "DPM++ 2M", "DPM++ SDE", "DDIM"])
    return names


def _schedulers() -> list[str]:
    """Scheduler(name='karras', label='Karras', ...) on Forge, ReForge and Neo
    alike: `name` is the internal id, `label` is what the rest of the WebUI
    shows and what infotext carries. Reading `name` put lowercase ids in the
    dropdown and meant the documented "Karras" default never matched anything,
    so it silently fell back to "Use same scheduler" on every fork.
    get_sampler_and_scheduler accepts either spelling, so the label is safe."""
    names = ["Use same scheduler"]
    try:
        from modules import sd_schedulers

        for scheduler in sd_schedulers.schedulers:
            label = getattr(scheduler, "label", None) or getattr(scheduler, "name", None)
            if label and label not in names:
                names.append(str(label))
    except Exception:
        names.extend(["Automatic", "Karras", "Exponential", "SGM Uniform"])
    return names


def suffix(n: int, sep: str = " ") -> str:
    """What ADetailer appends to every label from the second tab on.

    The WebUI's ui-config.json keys each entry on the component's LABEL, never
    on its elem_id. Four tabs sharing the label "ADetailer detector" therefore
    collapse onto one key and overwrite each other. ADetailer's whole trick is
    this one line: tab 2 is labelled "ADetailer detector 2nd", so every tab
    lands on a key of its own."""
    return "" if n == 0 else sep + ORDINALS[n]


PREFERRED_DETECTORS = (
    "face_yolov8n.pt",
    "face_yolov8s.pt",
    "face_yolov8m.pt",
    "person_yolov8n-seg.pt",
    "person_yolov8m-seg.pt",
)


def _default_model(n: int, models: list[str]) -> str:
    if n != 0:
        return "None"
    for needle in PREFERRED_DETECTORS:
        hit = next((m for m in models if m == needle or m.endswith("/" + needle)), None)
        if hit:
            return hit
    return models[1] if len(models) > 1 else "None"


def _dropdown(choices, value, label, info=None, elem_id=None):
    kwargs = dict(choices=choices, value=value, label=label)
    if info:
        kwargs["info"] = info
    if elem_id:
        kwargs["elem_id"] = elem_id
    try:
        return gr.Dropdown(allow_custom_value=True, **kwargs)
    except TypeError:
        return gr.Dropdown(**kwargs)


def _unit_ui(n: int, models: list[str], prefix: str, saved: dict | None = None) -> tuple[list, object]:
    ordinal = ORDINALS[n]
    saved = saved or {}
    sfx = suffix(n)

    D = UnitArgs()

    def val(field, default=None, choices=None):
        if default is None:
            default = getattr(D, field)
        return uistate.pick(saved, field, default, choices)
    
    def eid(name: str) -> str:
        return f"{prefix}_{name}_{n}"
        
    default_model = _default_model(n, models)

    samplers = _samplers()
    schedulers = _schedulers()
    sam_names = _sam_choices()
    clip_names = _clip_choices()
    default_sampler = "DPM++ 2M" if "DPM++ 2M" in samplers else samplers[0]
    default_sched = "Karras" if "Karras" in schedulers else schedulers[0]

    with gr.Tab(ordinal):
        enabled = gr.Checkbox(label=f"Enable this tab ({ordinal})", value=val("enabled", n == 0), elem_id=eid("enabled"))
        model = _dropdown(
            models,
            val("model", default_model, models),
            "Detector" + sfx,
            "YOLO / YOLO-seg weights from models/adetailer.",
            elem_id=eid("model"))
        prompt = gr.Textbox(
            label="Prompt" + sfx,
            info="Empty = main prompt. [SEP] per box, [CLASS:face], [SKIP], [PROMPT].",
            lines=2,
            value=val("prompt"),
            elem_id=eid("prompt"))
        negative = gr.Textbox(
            label="Negative prompt" + sfx,
            value=val("negative_prompt"),
            lines=2,
            elem_id=eid("negative_prompt"))

        with gr.Accordion("Detection", open=False):
            classes = gr.Textbox(
                label="Classes" + sfx,
                value=val("classes"),
                lines=1,
                placeholder="face, hand, -ear",
                info="Empty = all classes. Prefix - to exclude.",
                elem_id=eid("classes"),
            )
            merge_mode = gr.Radio(
                choices=MERGE_MODES,
                value=val("merge_mode", D.merge_mode, MERGE_MODES),
                label="Merge detections" + sfx,
                info="None = one pass per box. Merge = one pass. Invert = everything else.",
                elem_id=eid("merge_mode"),
            )
            confidence = gr.Slider(0.0, 1.0, value=val("confidence"), step=0.01, label="Confidence" + sfx, elem_id=eid("confidence"))
            filter_by = gr.Radio(
                choices=["Area", "Confidence"],
                value=val("filter_by", D.filter_by, ["Area", "Confidence"]),
                label="Top-k sort" + sfx,
                elem_id=eid("filter_by")
            )
            mask_k = gr.Slider(0, 10, value=val("mask_k"), step=1, label="Keep top k (0 = all)" + sfx, elem_id=eid("mask_k"))
            drop_size = gr.Slider(
                0, 128, value=val("drop_size"), step=1,
                label="Min box size (px)" + sfx,
                elem_id=eid("drop_size")
            )
            min_ratio = gr.Slider(0.0, 1.0, value=val("min_ratio"), step=0.001, label="Min area ratio" + sfx, elem_id=eid("min_ratio"))
            max_ratio = gr.Slider(0.0, 1.0, value=val("max_ratio"), step=0.001, label="Max area ratio" + sfx, elem_id=eid("max_ratio"))

        with gr.Accordion("SAM / CLIP", open=False):
            gr.Markdown("SAM tightens the box mask. CLIP drops boxes that do not match the prompt. Put weights in `models/sam` and `models/clip`.")
            use_sam = gr.Checkbox(value=val("use_sam"), label="Refine mask with SAM / SAM2" + sfx, elem_id=eid("use_sam"))
            sam_model = _dropdown(
                sam_names, val("sam_model", "None", sam_names), "SAM weights" + sfx,
                elem_id=eid("sam_model"),
            )
            use_clip = gr.Checkbox(value=val("use_clip_filter"), label="Filter boxes with CLIP" + sfx, elem_id=eid("use_clip_filter"))
            clip_model = _dropdown(
                clip_names, val("clip_model", "None", clip_names), "CLIP weights" + sfx,
                elem_id=eid("clip_model"),
            )
            clip_threshold = gr.Slider(
                0.0, 1.0, value=val("clip_threshold"), step=0.01,
                label="CLIP threshold" + sfx, elem_id=eid("clip_threshold"),
            )

        with gr.Accordion("Mask Preprocessing", open=False):
            x_offset = gr.Slider(-200, 200, value=val("x_offset", 0), step=1, label="Mask x(→) offset" + sfx, elem_id=eid("x_offset"))
            y_offset = gr.Slider(-200, 200, value=val("y_offset", 0), step=1, label="Mask y(↑) offset" + sfx, elem_id=eid("y_offset"))
            dilate = gr.Slider(-64, 64, value=val("dilate", 4), step=1, label="Mask erosion (−) / dilation (+)" + sfx, elem_id=eid("dilate"))

        with gr.Accordion("Inpainting", open=False):
            with gr.Row():
                feather = gr.Slider(0, 64, value=val("feather"), step=1, label="Mask blur" + sfx, elem_id=eid("feather"))
                denoise = gr.Slider(0.0, 1.0, value=val("denoise"), step=0.01, label="Denoising strength" + sfx, elem_id=eid("denoise"))
            with gr.Row():
                noise_mask = gr.Checkbox(value=val("noise_mask"), label="Inpaint only masked" + sfx, elem_id=eid("noise_mask"))
                use_wh = gr.Checkbox(value=val("use_wh"), label="Use separate width/height" + sfx, elem_id=eid("use_wh"))
            noise_mask_feather = gr.Slider(
                0, 64, value=val("noise_mask_feather"), step=1,
                label="Noise mask feather" + sfx,
                info="Softens denoise at the mask edge. 0 = hard cut.",
                elem_id=eid("noise_mask_feather")
            )
            padding = gr.Slider(0, 256, value=val("padding"), step=4, label="Masked padding (px)" + sfx, elem_id=eid("padding"))
            with gr.Row():
                inpaint_w = gr.Slider(64, 2048, value=val("inpaint_w"), step=8, label="Inpaint width" + sfx, elem_id=eid("inpaint_w"))
                inpaint_h = gr.Slider(64, 2048, value=val("inpaint_h"), step=8, label="Inpaint height" + sfx, elem_id=eid("inpaint_h"))
            with gr.Row():
                use_steps = gr.Checkbox(value=val("use_separate_steps"), label="Use separate steps" + sfx, elem_id=eid("use_separate_steps"))
                use_cfg = gr.Checkbox(value=val("use_separate_cfg"), label="Use separate CFG" + sfx, elem_id=eid("use_separate_cfg"))
            with gr.Row():
                steps = gr.Slider(1, 80, value=val("steps"), step=1, label="Steps" + sfx, elem_id=eid("steps"))
                cfg = gr.Slider(1.0, 15.0, value=val("cfg"), step=0.1, label="CFG" + sfx, elem_id=eid("cfg"))
            use_sampler = gr.Checkbox(value=val("use_separate_sampler"), label="Use separate sampler" + sfx, elem_id=eid("use_separate_sampler"))
            with gr.Row():
                sampler = gr.Dropdown(choices=samplers, value=val("sampler", default_sampler, samplers), label="Sampler" + sfx, elem_id=eid("sampler"))
                scheduler = gr.Dropdown(choices=schedulers, value=val("scheduler", default_sched, schedulers), label="Scheduler" + sfx, elem_id=eid("scheduler"))
            checkpoints, vaes = _checkpoints(), _vaes()
            with gr.Row():
                use_ckpt = gr.Checkbox(
                    value=val("use_separate_checkpoint", False),
                    label="Use different checkpoint" + sfx,
                    elem_id=eid("use_separate_checkpoint"),
                )
                use_vae = gr.Checkbox(
                    value=val("use_separate_vae", False),
                    label="Use different VAE" + sfx,
                    elem_id=eid("use_separate_vae"),
                )
            with gr.Row():
                checkpoint = _dropdown(
                    checkpoints,
                    val("checkpoint", SAME_CHECKPOINT, checkpoints),
                    "Checkpoint" + sfx,
                    "Loaded once for this tab, then restored.",
                    elem_id=eid("checkpoint"),
                )
                vae = _dropdown(
                    vaes,
                    val("vae", SAME_VAE, vaes),
                    "VAE" + sfx,
                    "None = baked VAE. Restored with the checkpoint.",
                    elem_id=eid("vae"),
                )
            try:
                use_ckpt.change(lambda v: gr.update(interactive=bool(v)), use_ckpt, checkpoint)
                use_vae.change(lambda v: gr.update(interactive=bool(v)), use_vae, vae)
                checkpoint.interactive = bool(val("use_separate_checkpoint", False))
                vae.interactive = bool(val("use_separate_vae", False))
            except Exception:
                pass

        with gr.Accordion("ControlNet", open=False):
            cn_models, cn_modules, cn_ok = _cn_choices()
            gr.Markdown(
                "Only ControlNet runs on the detail pass. **Passthrough** reuses the main generation units."
                if cn_ok else
                "ControlNet not found (needs Forge / ReForge / Forge Neo builtin)."
            )
            cn_model = _dropdown(
                cn_models, val("cn_model", "None", cn_models),
                "ControlNet model" + sfx, elem_id=eid("cn_model"),
            )
            cn_module = _dropdown(
                cn_modules, val("cn_module", "None", cn_modules),
                "ControlNet preprocessor" + sfx,
                "None for tile / inpaint models that take the crop as-is.",
                elem_id=eid("cn_module"),
            )
            with gr.Row():
                cn_weight = gr.Slider(
                    0.0, 2.0, value=val("cn_weight", 1.0), step=0.05,
                    label="ControlNet weight" + sfx, elem_id=eid("cn_weight"),
                )
                cn_start = gr.Slider(
                    0.0, 1.0, value=val("cn_start", 0.0), step=0.01,
                    label="Guidance start" + sfx, elem_id=eid("cn_start"),
                )
                cn_end = gr.Slider(
                    0.0, 1.0, value=val("cn_end", 1.0), step=0.01,
                    label="Guidance end" + sfx, elem_id=eid("cn_end"),
                )

        with gr.Accordion("Impact loop", open=False):
            gr.Markdown("Crop around the box, scale to guide_size, sample, paste back. Same loop as Impact Pack FaceDetailer.")
            with gr.Row():
                crop_factor = gr.Slider(1.0, 6.0, value=val("crop_factor"), step=0.1, label="crop_factor" + sfx, elem_id=eid("crop_factor"))
                guide_size = gr.Slider(128, 2048, value=val("guide_size"), step=8, label="guide_size" + sfx, elem_id=eid("guide_size"))
                max_size = gr.Slider(256, 2048, value=val("max_size"), step=8, label="max_size" + sfx, elem_id=eid("max_size"))
            with gr.Row():
                guide_bbox = gr.Checkbox(value=val("guide_size_for_bbox"), label="guide_size_for = bbox" + sfx, elem_id=eid("guide_size_for_bbox"))
                force_inpaint = gr.Checkbox(value=val("force_inpaint"), label="force_inpaint" + sfx, elem_id=eid("force_inpaint"))
            cycle = gr.Slider(
                1, 10, value=val("cycle"), step=1,
                label="cycle" + sfx,
                info="Re-sample the same crop N times.",
                elem_id=eid("cycle")
            )

    widgets = [
        enabled, model, prompt, negative,
        confidence, filter_by, mask_k, min_ratio, max_ratio,
        x_offset, y_offset, dilate,
        feather, denoise, noise_mask, padding, use_wh, inpaint_w, inpaint_h,
        use_steps, steps, use_cfg, cfg, use_sampler, sampler, scheduler,
        use_ckpt, checkpoint, use_vae, vae,
        crop_factor, guide_size, max_size, guide_bbox, force_inpaint, cycle,
        drop_size, noise_mask_feather,
        use_sam, sam_model, use_clip, clip_model, clip_threshold,
        classes, merge_mode,
        cn_model, cn_module, cn_weight, cn_start, cn_end,
    ]
    return widgets, model


def tab_name(is_img2img: bool) -> str:
    return "img2img" if is_img2img else "txt2img"


def build_ui(is_img2img: bool = False) -> list:
    prefix = "script_img2img_impact_adetailer" if is_img2img else "script_txt2img_impact_adetailer"
    tab = tab_name(is_img2img)

    # What this tab was last generated with. txt2img and img2img are kept
    # apart, the same way their prompts are.
    saved = uistate.load(tab)
    saved_units = saved.get("units") or []

    models = list_detector_models()
    # The id and the class are what style.css hangs on: they keep the panel
    # inside its column even when a theme's own rules would push it out.
    panel = dict(elem_id=f"{prefix}_panel", elem_classes="iad-panel")
    try:
        accordion = gr.Accordion("Impact ADetailer", open=False, **panel)
    except TypeError:  # a gradio old enough not to know elem_classes
        accordion = gr.Accordion("Impact ADetailer", open=False, elem_id=panel["elem_id"])
    with accordion:
        with gr.Row():
            enable = gr.Checkbox(
                label="Enable Impact ADetailer",
                value=bool(saved.get("enable", False)),
                elem_id=f"{prefix}_enable",
            )
            rescan = gr.Button("🔄 Rescan detectors")
            reset = gr.Button("↺ Reset tabs")
        with gr.Row():
            save_before = gr.Checkbox(
                label="Save the image before detailing",
                value=bool(saved.get("save_before", False)),
                elem_id=f"{prefix}_save_before",
            )
            preview = gr.Radio(
                choices=PREVIEW_MODES,
                value=uistate.pick_choice(saved.get("preview"), "Off", PREVIEW_MODES),
                label="Process images",
                elem_id=f"{prefix}_preview",
            )
        widgets = [enable, save_before, preview]
        dropdowns = []
        with gr.Tabs():
            for n in range(max_tabs()):
                unit_saved = saved_units[n] if n < len(saved_units) else {}
                unit_widgets, model_dropdown = _unit_ui(n, models, prefix, unit_saved)
                widgets.extend(unit_widgets)
                dropdowns.append(model_dropdown)

        def _rescan():
            found = list_detector_models()
            return [gr.update(choices=found) for _ in dropdowns]

        def _reset():
            """Forget this tab's saved state and put the factory values back
            on screen, without needing a UI reload."""
            uistate.reset(tab)
            fresh = list_detector_models()
            head = RunArgs()
            values = [getattr(head, name) for name in HEAD_KEYS]
            for n in range(max_tabs()):
                for value in _factory_values(n, fresh):
                    values.append(value)
            return [gr.update(value=v) for v in values[: len(widgets)]]

        try:
            rescan.click(fn=_rescan, inputs=[], outputs=dropdowns)
            reset.click(fn=_reset, inputs=[], outputs=widgets)
        except Exception:
            pass

    if remember_ui():
        # ui_loadsave runs AFTER this function returns and does
        #     setattr(component, "value", value_from_ui_config)
        # on everything it tracks. That silently threw away the values restored
        # above, replacing them with whatever landed in ui-config.json on the
        # very first launch. Opting out hands the job to uistate alone.
        # Turn the option off and the WebUI takes over instead, now with one
        # key per tab thanks to the label suffix.
        for widget in widgets:
            try:
                widget.do_not_save_to_config = True
            except Exception:
                pass
    return widgets


def _factory_values(n: int, models: list[str]) -> list:
    """The values a freshly installed tab would show, in widget order."""
    unit = UnitArgs()
    unit.enabled = n == 0
    unit.model = _default_model(n, models)
    samplers, schedulers = _samplers(), _schedulers()
    unit.sampler = "DPM++ 2M" if "DPM++ 2M" in samplers else samplers[0]
    unit.scheduler = "Karras" if "Karras" in schedulers else schedulers[0]
    return [getattr(unit, name) for name, _ in FIELD_KEYS]


def infotext_key(unit_index: int, label: str) -> str:
    return f"Impact ADetailer {ORDINALS[unit_index]} {label}"


def infotext_fields(widgets: list) -> list:
    per = len(FIELD_KEYS)
    pairs = []
    # The head controls carry no infotext of their own: "save a copy" and "draw
    # the boxes" are things you do to a run, not settings that reproduce it.
    for slot, widget in enumerate(widgets[len(HEAD_KEYS):]):
        unit, offset = divmod(slot, per)
        if unit >= len(ORDINALS):
            break
        label = FIELD_KEYS[offset][1]
        if label:
            pairs.append((widget, infotext_key(unit, label)))
    return pairs


def infotext_params(units: list[UnitArgs]) -> dict:
    params: dict = {}
    for i, unit in enumerate(units):
        if not unit.enabled or unit.model in ("", "None"):
            continue
        for name, label in FIELD_KEYS:
            if not label:
                continue
            if name == "checkpoint" and not getattr(unit, "use_separate_checkpoint", False):
                continue
            if name == "vae" and not getattr(unit, "use_separate_vae", False):
                continue
            value = getattr(unit, name, None)
            if value is None or value == "":
                continue
            if value in ("Use same checkpoint", "Use same VAE"):
                continue
            params[infotext_key(i, label)] = value
    return params


def parse_args(args: tuple) -> tuple[RunArgs, list[UnitArgs]]:
    head = len(HEAD_KEYS)
    run = RunArgs(
        enable=bool(args[0]) if len(args) > 0 else False,
        save_before=bool(args[1]) if len(args) > 1 else False,
        preview=str(args[2] or "Off") if len(args) > 2 else "Off",
    )
    raw = list(args[head:])
    per = len(FIELD_KEYS)
    units: list[UnitArgs] = []
    n_units = min(UNIT_COUNT, max(max_tabs(), len(raw) // per))
    for i in range(n_units):
        c = raw[i * per : (i + 1) * per]
        if len(c) < per:
            units.append(UnitArgs(enabled=False))
            continue
        units.append(
            UnitArgs(
                enabled=bool(c[0]),
                model=str(c[1]),
                prompt=str(c[2] or ""),
                negative_prompt=str(c[3] or ""),
                confidence=float(c[4]),
                filter_by=str(c[5] or "Area"),
                mask_k=int(c[6]),
                min_ratio=float(c[7]),
                max_ratio=float(c[8]),
                x_offset=int(c[9]),
                y_offset=int(c[10]),
                dilate=int(c[11]),
                feather=int(c[12]),
                denoise=float(c[13]),
                noise_mask=bool(c[14]),
                padding=int(c[15]),
                use_wh=bool(c[16]),
                inpaint_w=int(c[17]),
                inpaint_h=int(c[18]),
                use_separate_steps=bool(c[19]),
                steps=int(c[20]),
                use_separate_cfg=bool(c[21]),
                cfg=float(c[22]),
                use_separate_sampler=bool(c[23]),
                sampler=str(c[24]),
                scheduler=str(c[25]),
                use_separate_checkpoint=bool(c[26]),
                checkpoint=str(c[27] or SAME_CHECKPOINT),
                use_separate_vae=bool(c[28]),
                vae=str(c[29] or SAME_VAE),
                crop_factor=float(c[30]),
                guide_size=int(c[31]),
                max_size=int(c[32]),
                guide_size_for_bbox=bool(c[33]),
                force_inpaint=bool(c[34]),
                cycle=int(c[35]),
                drop_size=int(c[36]),
                noise_mask_feather=int(c[37]),
                use_sam=bool(c[38]),
                sam_model=str(c[39] or "None"),
                use_clip_filter=bool(c[40]),
                clip_model=str(c[41] or "None"),
                clip_threshold=float(c[42]),
                classes=str(c[43] or ""),
                merge_mode=str(c[44] or "None"),
                cn_model=str(c[45] or "None"),
                cn_module=str(c[46] or "None"),
                cn_weight=float(c[47]),
                cn_start=float(c[48]),
                cn_end=float(c[49]),
            )
        )
    return run, units