"""Lysosome scoring assistant: prototype Gradio app.

Run locally:   pip install -r requirements.txt && python app.py
On HF Spaces:  push this folder to a Gradio Space (README.md carries the Space config).
"""
from __future__ import annotations

import datetime as dt
import os
import tempfile
import zipfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd
import yaml

from detector import DemoBlobDetector, Detection, file_sha1, predict_tiled

# ZeroGPU (Hugging Face Spaces): the free tier for Gradio Spaces on free accounts.
# `spaces.GPU` attaches a GPU only while the decorated function runs. Outside Spaces the
# package is absent and this becomes a no-op, so the app runs the same locally.
try:
    import spaces

    gpu = spaces.GPU(duration=120)
except ImportError:
    def gpu(fn):
        return fn

APP_DIR = Path(__file__).parent
CFG = yaml.safe_load((APP_DIR / "config.yaml").read_text())
SPECIES = list(CFG["species"])
_MODEL_CACHE: dict[str, object] = {}

# Fixed, colour-blind-friendly palette (Okabe-Ito), RGB
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
           (0, 114, 178), (213, 94, 0), (204, 121, 167), (255, 255, 255)]
PALETTE_NAMES = ["orange", "sky blue", "green", "yellow", "blue", "vermilion", "pink", "white"]


# --------------------------------------------------------------------------- model loading
def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = (APP_DIR / path).resolve()
    if p.exists():
        return p
    if CFG.get("hf_repo"):
        try:
            from huggingface_hub import hf_hub_download

            return Path(hf_hub_download(CFG["hf_repo"], Path(path).name, token=os.environ.get("HF_TOKEN")))
        except Exception as e:  # noqa: BLE001 - fall through to next option
            print(f"[weights] could not fetch {path} from {CFG['hf_repo']}: {e}")
    return None


def _load(path: Path):
    if CFG.get("backend", "ultralytics") == "rfdetr":
        from detector import RFDETRDetector

        return RFDETRDetector(path, CFG.get("class_names") or [])
    from detector import UltralyticsDetector

    return UltralyticsDetector(path, imgsz=int(CFG.get("imgsz", 1024)))


def get_model(species: str):
    """Return (detector, note). Roboflow hosted > species weights > base weights > demo."""
    if species in _MODEL_CACHE:
        return _MODEL_CACHE[species]
    rf_model = CFG["species"][species].get("roboflow_model")
    if rf_model:
        from detector import RoboflowHostedDetector, RoboflowError

        try:
            det = RoboflowHostedDetector(rf_model, CFG["species"][species].get("class_names"))
            result = (det, f"Using **{species}** model hosted on Roboflow (images are sent to Roboflow).")
            _MODEL_CACHE[species] = result
            return result
        except RoboflowError as e:
            raise gr.Error(str(e))
    sp = _resolve(CFG["species"][species].get("weights"))
    base = _resolve(CFG.get("base_weights"))
    if sp:
        result = (_load(sp), f"Using **{species}-specific** model.")
    elif base:
        result = (_load(base), f"No {species}-specific weights found: using the **species-blind base** model.")
    else:
        result = (DemoBlobDetector(),
                  "⚠️ **DEMO MODE: no trained weights found.** Boxes come from a simple blob "
                  "finder, not a lysosome model. Do not use these numbers for scoring.")
    _MODEL_CACHE[species] = result
    return result


# --------------------------------------------------------------------------- image IO
def load_image(path: str) -> np.ndarray:
    """Read any common microscopy format to 8-bit RGB.

    16-bit images are stretched with 0.5/99.5 percentiles. This MUST match whatever
    preprocessing produced the training images, or accuracy will silently drop.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {Path(path).name}")
    if img.dtype != np.uint8:
        lo, hi = np.percentile(img, (0.5, 99.5))
        img = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def draw(img: np.ndarray, dets: list[Detection], names: list[str], show_labels: bool) -> np.ndarray:
    out = img.copy()
    t = max(1, round(min(img.shape[:2]) / 600))
    for d in dets:
        c = PALETTE[d.class_id % len(PALETTE)]
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, c, t)
        if show_labels:
            cv2.putText(out, f"{names[d.class_id][:12]} {d.conf:.2f}", (p1[0], max(p1[1] - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, c, t)
    return out


# --------------------------------------------------------------------------- main action
@gpu
def run(species, files, conf, tile, show_labels, progress=gr.Progress()):
    if not species:
        raise gr.Error("Pick a species first.")
    if not files:
        raise gr.Error("Upload at least one image.")
    det, note = get_model(species)
    names = det.class_names
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(tempfile.mkdtemp(prefix=f"lms-{stamp}-"))
    gallery, summary_rows, det_rows = [], [], []
    zpath = outdir / f"review_package_{species}_{stamp}.zip"
    seen: dict[str, str] = {}
    dupes: list[str] = []

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("labels/classes.txt", "\n".join(names) + "\n")
        for i, f in enumerate(progress.tqdm(files, desc="Scoring")):
            path = f if isinstance(f, str) else f.name
            fname = Path(path).name
            digest = file_sha1(path, 16)
            if digest in seen:  # identical file uploaded twice would double-count in pooled totals
                dupes.append(f"{fname} (same as {seen[digest]})")
                continue
            seen[digest] = fname
            img = load_image(path)
            h, w = img.shape[:2]
            dets = predict_tiled(det, img, float(conf), int(tile or 0))

            counts = np.bincount([d.class_id for d in dets], minlength=len(names))
            total = int(counts.sum())
            row = {"image": fname, "species": species, "model": det.label,
                   "conf_threshold": conf, "tile": int(tile or 0), "total": total}
            for k, n in enumerate(names):
                row[f"n_{n}"] = int(counts[k])
            for k, n in enumerate(names):
                row[f"prop_{n}"] = round(counts[k] / total, 4) if total else None
            summary_rows.append(row)

            for d in dets:
                det_rows.append({"image": fname, "species": species, "model": det.label,
                                 "class": names[d.class_id], "conf": round(d.conf, 4),
                                 "x1": round(d.x1, 1), "y1": round(d.y1, 1),
                                 "x2": round(d.x2, 1), "y2": round(d.y2, 1)})

            ann = draw(img, dets, names, show_labels)
            caption = f"{fname}: {total} detected"
            gallery.append((ann, caption))

            # Review package: same-name files across species/batches are kept apart by index prefix
            stem = f"{i:03d}_{Path(fname).stem}"
            ok, buf = cv2.imencode(".png", cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
            zf.writestr(f"annotated/{stem}.png", buf.tobytes())
            yolo = "\n".join(
                f"{d.class_id} {((d.x1 + d.x2) / 2) / w:.6f} {((d.y1 + d.y2) / 2) / h:.6f} "
                f"{(d.x2 - d.x1) / w:.6f} {(d.y2 - d.y1) / h:.6f}"
                for d in dets
            )
            zf.writestr(f"labels/{stem}.txt", yolo + ("\n" if yolo else ""))

        summary = pd.DataFrame(summary_rows)
        dets_df = pd.DataFrame(det_rows, columns=["image", "species", "model", "class", "conf",
                                                  "x1", "y1", "x2", "y2"])
        # Pooled row across the batch
        if len(summary) > 1:
            pooled = {"image": "ALL (pooled)", "species": species, "model": det.label,
                      "conf_threshold": conf, "tile": int(tile or 0), "total": int(summary["total"].sum())}
            for n in names:
                pooled[f"n_{n}"] = int(summary[f"n_{n}"].sum())
            for n in names:
                pooled[f"prop_{n}"] = round(pooled[f"n_{n}"] / pooled["total"], 4) if pooled["total"] else None
            summary = pd.concat([summary, pd.DataFrame([pooled])], ignore_index=True)

        s_csv = outdir / f"summary_{species}_{stamp}.csv"
        d_csv = outdir / f"detections_{species}_{stamp}.csv"
        summary.to_csv(s_csv, index=False)
        dets_df.to_csv(d_csv, index=False)
        zf.write(s_csv, s_csv.name)
        zf.write(d_csv, d_csv.name)

    legend = ", ".join(f"{n} = {PALETTE_NAMES[k % len(PALETTE_NAMES)]}" for k, n in enumerate(names))
    info = f"{note}  \nModel: `{det.label}` · box colours: {legend}"
    if dupes:
        info += "  \n⚠️ Skipped duplicate uploads: " + "; ".join(dupes)
    shown = summary.drop(columns=["species", "model", "conf_threshold", "tile"])  # kept in CSV
    return info, gallery, shown, [str(s_csv), str(d_csv), str(zpath)]


# --------------------------------------------------------------------------- UI
with gr.Blocks(title="Lysosome scoring assistant") as demo:
    gr.Markdown(
        "# Lysosome scoring assistant\n"
        "Pick the species, upload micrographs, and press **Score**. The model proposes a "
        "label for each lysosome; treat the counts as a **first pass to check**, not a final score. "
        "Download the review package to correct boxes in Roboflow."
    )
    with gr.Row():
        with gr.Column(scale=1):
            species = gr.Radio(SPECIES, label="Species", value=None)
            files = gr.File(label="Images", file_count="multiple",
                            file_types=["image", ".tif", ".tiff"], type="filepath")
            conf = gr.Slider(0.05, 0.95, value=float(CFG.get("default_conf", 0.25)), step=0.05,
                             label="Confidence threshold",
                             info="Higher = fewer, more certain detections. Keep it the same across a study.")
            with gr.Accordion("Advanced", open=False):
                tile = gr.Number(value=int(CFG.get("default_tile", 0)), precision=0,
                                 label="Tile size (px, 0 = off)",
                                 info="For large images with small lysosomes. Use the size the model was trained on.")
                show_labels = gr.Checkbox(value=False, label="Show class labels on boxes")
            btn = gr.Button("Score", variant="primary")
        with gr.Column(scale=2):
            info = gr.Markdown()
            gallery = gr.Gallery(label="Detections", columns=2, height=520, object_fit="contain")
    summary = gr.Dataframe(label="Per-image counts and proportions", wrap=True)
    downloads = gr.File(label="Downloads (summary CSV, per-detection CSV, review package ZIP)",
                        file_count="multiple")

    btn.click(run, [species, files, conf, tile, show_labels], [info, gallery, summary, downloads])

if __name__ == "__main__":
    demo.launch()
