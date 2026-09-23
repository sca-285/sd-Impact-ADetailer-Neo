# Impact ADetailer Neo

After-detailer for **Forge**, **ReForge** and **Forge Neo**.

*This extension is inspired on the method how FaceDetailer operate in ComfyUI.*

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
| Classes | `face, hand, -ear` |
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
| cycle | 1 |

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

## ControlNet / checkpoint / VAE

* **Passthrough** reuses the main generation's ControlNet units.
* A named model builds one unit on the crop only.
* Needs the Forge-family builtin (`lib_controlnet`). Plain A1111 + sd-webui-controlnet is not wired.
* **Use different checkpoint / VAE** loads once per tab and restores afterwards. ControlNet is skipped if the new checkpoint is a different family (SDXL ↔ SD1 / Flux / Anima).

## Settings

| Option | Default |
|---|---|
| Max tabs | 8 (needs Reload UI) |
| Sort boxes | Area, large to small |
| Same seed every tab | off |
| Detector device | Automatic |

## License

AGPL-3.0. See `LICENSE` and `NOTICE`.
