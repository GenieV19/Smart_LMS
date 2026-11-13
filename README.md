---
title: Lysosome Scoring Assistant
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# Lysosome scoring assistant (prototype)

Gradio app that proposes a cell-state label for each lysosome in a micrograph, then reports
per-image counts and proportions. It assists manual scoring; it does not replace it.

## What a user does
1. Pick the species (Geodia or Primnoa).
2. Upload one or more images (PNG, JPG, TIFF incl. 16-bit).
3. Press **Score**. Boxes appear on the images, with a table of counts and proportions per class.
4. Download:
   - `summary_*.csv`: one row per image, plus a pooled row. Includes species, model id + weights hash, threshold, tiling, so each number is traceable to the exact model.
   - `detections_*.csv`: one row per detected lysosome (class, confidence, box).
   - `review_package_*.zip`: annotated PNGs plus YOLO-format label files and `classes.txt`, so predictions can be imported into Roboflow as pre-annotations and corrected.

## Adding trained weights
Put files in `weights/` (or set `hf_repo` in `config.yaml` to pull them from a Hugging Face model repo):

| file | used when |
|---|---|
| `weights/geodia.pt` | species = Geodia |
| `weights/primnoa.pt` | species = Primnoa |
| `weights/base.pt` | fallback when the species file is missing |

With none present, the app runs in **DEMO mode** (an OpenCV blob finder with a single
"unclassified" class), clearly flagged on screen and in every CSV row. For RF-DETR set
`backend: rfdetr` and fill `class_names` in `config.yaml` (that backend is untested).

Keep `imgsz` in `config.yaml` equal to the training image size, and if you trained on tiles,
set the tile size in the app's Advanced panel to that tile size.

## Using a model hosted on Roboflow
Roboflow does not offer weight downloads for its NAS-trained RF-DETR models, so the app can
call the hosted model instead. Set `roboflow_model` (and `class_names`) for a species in
`config.yaml`, and provide `ROBOFLOW_API_KEY` (from app.roboflow.com/settings/api) as an
environment variable or a Space secret. Never commit the key. Each image (or each tile, if
tiling is on) is one API call, which uses Roboflow credits and sends the image to Roboflow.

## Run locally
```
pip install -r requirements.txt
python app.py        # opens http://127.0.0.1:7860
```

## Deployment
This repository is mirrored to the Hugging Face Space https://huggingface.co/spaces/GVick/lms
by `.github/workflows/sync-to-hf-space.yml` on every push to `main` (needs the `HF_TOKEN`
repository secret). Edit here, not in the Space: the sync overwrites the Space.
Weight files (`*.pt`, `*.pth`) are stored with Git LFS.

## Deploy to Hugging Face Spaces
Creating a Gradio Space on CPU hardware needs a paid HF plan (PRO, $9/month). Free personal
accounts can host up to 2 Gradio Spaces on **ZeroGPU**, which this app supports: when creating
the Space pick SDK **Gradio** and hardware **ZeroGPU**, then upload the contents of this folder
(README front-matter above configures it). Visitors share a daily GPU quota; logged-in HF users
get more than anonymous ones.
Weights over 10 MB need Git LFS, or keep them in a separate (optionally private) model repo
and set `hf_repo` + an `HF_TOKEN` Space secret. Free CPU Spaces work but are slow for
large images (expect several seconds per image); they also sleep when idle.

**Data note:** a public Space means anyone with the link can upload images. Uploads are not
stored by the app beyond the session's temp files, but make the Space private if the images
are unpublished.
