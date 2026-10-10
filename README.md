# Impact ADetailer Neo

After-detailer for **Forge**, **ReForge** and **Forge Neo**.

*This extension is inspired on the method how FaceDetailer - Impact-Pack operate in ComfyUI.*

Detects regions with YOLO, crops them the way Impact Pack FaceDetailer does (`crop_factor` → `guide_size` → sample → paste), then writes the result back onto the image.

Not a drop-in replacement for [ADetailer](https://github.com/Bing-su/adetailer) or [ADetailer-Neo](https://github.com/Haoming02/sd-forge-adetailer). Those squash each region to a fixed width/height. This one keeps the crop aspect ratio.

## Visual Comparer

This extension provides an additional "ADetailer Comparer" panel located directly below the main image display area of the WebUI.   

- This panel displays the image with a vertical split-screen slider.   
- You can drag this slider left or right to directly compare the differences in details, lighting, and sharpness between the original image and the processed result 

![Comparer Interface](https://iili.io/nAHLfCG.png)

## Install

WebUI → Extensions → Install from URL → this repo → Apply and restart.

```bash
git clone https://github.com/sca-285/impact-adetailer.git extensions/impact-adetailer
```

`install.py` installs `ultralytics` if it is missing.

Forge Neo may print that an `adetailer` folder looks outdated. That is a name match, not a version check. Rename the folder to `impact-adetailer-neo` to hide it.

Disable any other ADetailer copy. Two panels will fight.

## Detectors

Put weights in `models/adetailer` or `models/ultralytics`. Extra folders: Settings → Impact ADetailer → Extra paths.

| Kind | Example | Classes box |
|---|---|---|
| Closed-set YOLO | `face_yolov8n.pt` | filters the model's own names |
| YOLO-seg | `person_yolov8m-seg.pt` | same, plus a silhouette mask |
| YOLO-World / YOLOE | filename must contain `-world` or `yoloe` | open vocabulary |

Optional weights:

* SAM / SAM2 → `models/sam`
* OpenCLIP → `models/clip`

Nothing is downloaded at runtime unless you type classes into a YOLO-World / YOLOE model and the text encoder is not cached.

## Defaults (Check your own please)

| Setting | Default |
|---|---|
| Classes | `face` |
| Confidence | 0.5 |
| Top-k sort | Confidence |
| Keep top k | 0 (all) |
| Min box size | 10 px |
| Mask blur | 12 |
| Denoise | 0.4 |
| Inpaint only masked | on |
| Noise mask feather | 5 |
| Masked padding | 32 px |
| Steps / CFG | 20 / 4 |
| Sampler | DPM++ 2M + Karras |
| crop_factor | 3 |
| guide_size | 512-768 |
| max_size | 1024-1536 |
| guide_size_for | bbox |
| force_inpaint | on |
| cycle | 1-3 |

**Reset tabs** restores these. `impact-adetailer-ui.json` overrides them until you reset.

## Prompt tokens

Empty prompt = main prompt.

```
face prompt [SEP] second face
[CLASS:face] a face [CLASS:hand] a hand
[SKIP]
[PROMPT]
```

## Impact loop

Same order as FaceDetailer:

1. Crop around the box × `crop_factor`.
2. Scale so the short side of the **bbox** (or the crop, if the checkbox is off) reaches `guide_size`.
3. Clamp the long side to `max_size`.
4. If scale would be ≤ 1 and `force_inpaint` is off, skip the region.
5. Sample `cycle` times. Each cycle goes through the VAE (Comfy stays in latent).

Set **Masked padding** to 0 to match FaceDetailer `crop_factor` exactly.

## Anatomy Inspector

A judgement layer in front of the tabs. A plain hand tab repaints every hand it
finds, good ones included, and has the same odds of drawing six fingers again.
The Inspector looks first:

1. **Inspect**: DWPose puts a skeleton and 21 points per hand on every person.
2. **Gate**: a tab set to **Repaint: Only flagged by Inspector** repaints only the
   boxes that sit on a hand the Inspector found wrong. The others are drawn grey
   on the detection sheet ("not flagged") and left exactly as they were.
3. **Guide**: with the ControlNet preprocessor **Hand depth (Inspector)**, a flagged
   hand is repainted against a clean five-finger depth map built from its landmarks.
4. **Report**: skeleton, hand capsules and findings over the comparer (toggle
   *Anatomy overlay* in the dock), a short score in the infotext
   (`Anatomy score: 87 (hand_shape x1)`), and one console line per image:
   `[Impact ADetailer] Inspector: score 74, 2 findings (hand_shape p0 left ≈6 fingers, head_ratio p1 ...)`.

### What it checks

| Finding | How | Fixable |
|---|---|---|
| `hand_shape` | The 21 landmarks are rendered as a five-finger capsule hand and laid over the real hand mask. Hand pixels no finger explains (extra finger, blob) and rendered fingers with no pixels under them (missing finger) are measured; fingertips on the mask outline are counted as a second opinion. "≈N fingers" is only claimed when both agree. | yes |
| `extra_hand` | More than two hand boxes inside one person, and no wrist reaches the extra one. Needs a hand detector. | no (a hand repaint does not remove an arm) |
| `fused_bodies` | Two skeletons that share a whole limb, or more faces than skeletons inside overlapping people. | no |
| `head_ratio` | Body height in heads, only for an upright person fully in frame, against the profile's range. | no |
| `limb_asymmetry` | Left against right upper arm, forearm, thigh, shin, only when both sides point the same way. | no |

### What it cannot do

* **Count fingers from landmarks.** Pose models always return 21 points, also for a
  six-finger hand. The count comes from the hand mask, so the finger check needs a
  **hand-seg YOLO** model (any `*hand*seg*.pt` in `models/adetailer`) or **SAM**
  weights in `models/sam`. Without either, hands are listed as "not checked".
* Tell foreshortening from a short bone. Length checks are warnings only.
* Joint hyperextension (not reliable from 2D points).
* Anime hands: landmarks are weaker there. The Anime profile is more lenient and
  caps hand confidence at 0.6.

### Models

DWPose, two onnx files:

| File | Link |
|---|---|
| `yolox_l.onnx` | https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx |
| `dw-ll_ucoco_384.onnx` | https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx |

If the ControlNet **dw_openpose_full** preprocessor has run once, they are already in
`models/ControlNetPreprocessor/openpose` (Forge, reForge and Forge Neo all use that
folder, or `--controlnet-preprocessor-models-dir`), and the Inspector uses them from
there. Otherwise download them by hand into `models/iadetailer/anatomy/`. The
**Download DWPose models** button does the same from Hugging Face, but large
files are often faster by hand.

* `onnxruntime` runs them (GPU when torch has CUDA 12, else CPU); `install.py` adds it
  if missing. Without it, `cv2.dnn` on the CPU is used, as the WebUI's own DWPose does.
* **MediaPipe** (optional, `pip install mediapipe`): sharper finger points on realistic
  images under the Realistic profile. Wheels do not exist for every Python; if it
  does not import, DWPose hand points are used and the console says so once.
* Hand masks, hand boxes and face boxes come from your own detectors. Settings →
  Impact ADetailer → *Anatomy Inspector: ...* picks them by name (empty = automatic,
  `None` = off).

The **Check models** button and the status line in the Inspector block show what was found.

### Profiles

| Profile | Head ratio | Hand checks |
|---|---|---|
| Realistic | 6.8 to 8.2 heads | full; MediaPipe preferred when installed |
| Anime | 6.0 to 8.5 heads | lenient thresholds, confidence capped at 0.6 |
| Stylised | not checked | lenient |

Every threshold lives in `iadetailer/anatomy/profiles.py`.

### Recommended hand tab

| Setting | Value |
|---|---|
| Detector | a hand model, e.g. `hand_yolov8s.pt` |
| Repaint | Only flagged by Inspector |
| Denoise | 0.45 - 0.6 |
| ControlNet model | an SDXL depth model, or Union in depth mode |
| ControlNet preprocessor | Hand depth (Inspector) |
| Weight / Guidance end | 0.6 / 0.6 (set automatically when the preprocessor is picked) |
| cycle | 1 |

### Notes

* One inspection per image, on the image as it left the sampler, before the tabs.
  The score describes that image.
* Inspector off = the same pixels as before, byte for byte.
* Enable, profile and each tab's Repaint are written to the infotext and come back
  with *Send to txt2img*. The score is a measurement and is not read back.
* The onnx sessions are freed with the other caches (UI reload, or
  *Unload detector models after each generation*).

### Tests

```bash
python -m pytest tests     # numpy, Pillow, opencv; gradio for tests/test_ui.py
```

## ControlNet / checkpoint / VAE

* **Passthrough** reuses the main generation's ControlNet units.
* A named model builds one unit on the crop only.
* Needs the Forge-family builtin (`lib_controlnet`). Plain A1111 + sd-webui-controlnet is not wired.
* **Use different checkpoint / VAE** loads once per tab and restores afterwards. ControlNet is skipped if the new checkpoint is a different family (SDXL ↔ SD1 / Flux / Anima).

## All tabs at once

**🎛️ All tabs at once (inpainting)**, above the tabs, changes the inpaint
settings of every tab in one go: checkpoint, VAE, steps, CFG, sampler,
scheduler, denoise, mask blur, only masked, noise feather, padding, width and
height, crop_factor, guide_size, max_size, guide for bbox, force_inpaint and
cycle.

* A change there is written straight into the tabs, so each tab shows what will
  run. Tabs can still be changed one by one afterwards.
* **Apply to**: **All tabs**, or **Enabled tabs only**.
* Setting steps, CFG, sampler, scheduler, checkpoint, VAE, width or height also
  ticks the tabs' matching *Use separate ...* box (a "Use same ..." choice
  unticks it).
* **Write checkpoint, VAE, steps, CFG and sampler into the tabs** copies those at once.
  Denoise, mask blur, noise feather, padding, crop_factor, guide_size and max_size are
  never copied by it: they depend on the size of the area, and a face's values erase
  the eyes. Move them one by one, with **Enabled tabs only** when the eye or hand tabs
  should keep their own.
* Mask blur and noise feather are kept under a quarter of the detected area's short
  side at run time (the console says when): a 12 px blur on an eye mask a few pixels
  thick used to paste the redrawn eyes back at half strength, as a smear.
* The panel starts from the 1st tab's values. It is only a remote control:
  nothing in it is saved or written to the image's parameters by itself.

## Settings

| Option | Default |
|---|---|
| Max tabs | 8 (needs Reload UI) |
| Sort boxes | Area, large to small |
| Same seed every tab | off |
| Detector device | Automatic |

## Credits

Thanks also to **Claude**, Anthropic's AI assistant, for help building this extension.

## License

AGPL-3.0. See `LICENSE` and `NOTICE`.
