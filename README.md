---
title: Lysosome Scoring Assistant
emoji: 🔬
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
pinned: false
---

# Lysosome scoring assistant (prototype)

Web app that proposes a cell-state label (e.g. Stable / Unstable / Uncertain) for each cell in a
micrograph and reports per-image counts and proportions. It assists manual scoring; it does not
replace it.

**Everything runs in the visitor's browser** (onnxruntime-web). Images are never uploaded, and
hosting is a free Hugging Face *Static* Space: https://huggingface.co/spaces/GVick/lms

## What a user does
1. Pick the species (Geodia or Primnoa).
2. Add one or more images (PNG, JPG, TIFF incl. 16-bit).
3. Press **Score**. Boxes appear on the images, with a table of counts and proportions per class.
4. Download:
   - **Summary CSV**: one row per image plus a pooled row, with species, model file + hash,
     threshold and tiling, so each number is traceable to the exact model.
   - **Per-detection CSV**: one row per detected cell (class, confidence, box).
   - **Review package (ZIP)**: annotated PNGs, YOLO-format labels and `classes.txt`, to import
     into Roboflow as pre-annotations and correct.

## Adding a trained model
Train as usual (YOLO or RF-DETR), then from the repository root:

```
python tools/export_onnx.py --weights path/to/best.pt --species Geodia --imgsz 640
git add models/ && git commit -m "Add Geodia model" && git push
```

This converts the weights to ONNX, saves `models/geodia.onnx` and registers it in
`models/models.json`. The Space updates automatically on push. `--imgsz` must equal the training
image size. `--species base` registers a species-blind fallback model.

Model lookup per species: species model, then `base`, then the **demo** model. The demo model
was trained on synthetic dots, not cells; the app flags it on screen and in every CSV row.

Notes:
- Model files are public on a public Space (anyone can download the weights).
- The RF-DETR export/decoding path is written but **untested** until a real RF-DETR export exists.
- Ultralytics YOLO is AGPL-3.0; RF-DETR is Apache-2.0. Check which licence suits the project.
- Large models make the first page load slow (the model downloads once per visitor, then is cached).

## Running the Python version locally
`local_app/` holds the original Gradio app (same features, runs PyTorch weights directly):
```
cd local_app
pip install -r requirements.txt
python app.py        # opens http://127.0.0.1:7860
```

## Deployment
This repository is mirrored to the Space by `.github/workflows/sync-to-hf-space.yml` on every
push to `main` (needs the `HF_TOKEN` repository secret: a Hugging Face token with write access to
the Space). Edit here, not in the Space: the sync overwrites the Space. Model files (`*.onnx`,
`*.pt`, `*.pth`) are stored with Git LFS.

To test the web app locally, serve the folder (e.g. `python -m http.server`) and open
http://localhost:8000; opening `index.html` as a file does not work because the model is fetched.
