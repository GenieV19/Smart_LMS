"""Convert trained weights to ONNX for the browser app and register them in models/models.json.

Usage (run from the repository root, in the environment you trained in):

    # YOLO (Ultralytics) weights
    python tools/export_onnx.py --weights runs/detect/train/weights/best.pt --species Geodia --imgsz 640

    # species-blind base model
    python tools/export_onnx.py --weights base.pt --species base --imgsz 640

    # RF-DETR checkpoint (UNTESTED path; needs `pip install rfdetr`)
    python tools/export_onnx.py --weights checkpoint_best_ema.pth --species Geodia --format rfdetr \
        --imgsz 560 --names "Stable cell" "Uncertain" "Unstable cell"

Then commit models/<name>.onnx and models/models.json and push: the Space updates itself.
--imgsz must equal the image size used for training.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
MANIFEST = MODELS / "models.json"


def export_yolo(weights: Path, imgsz: int) -> tuple[Path, list[str]]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = [model.names[i] for i in sorted(model.names)]
    onnx = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False))
    return onnx, names


def export_rfdetr(weights: Path, imgsz: int, variant: str) -> Path:
    import rfdetr

    model = getattr(rfdetr, variant)(pretrain_weights=str(weights), resolution=imgsz)
    out_dir = weights.parent / "onnx_export"
    model.export(output_dir=str(out_dir))
    found = sorted(out_dir.glob("*.onnx"))
    if not found:
        raise SystemExit(f"RF-DETR export produced no .onnx in {out_dir}")
    return found[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--species", required=True, help='Species name as in models.json (e.g. Geodia), or "base"')
    ap.add_argument("--imgsz", type=int, default=640, help="training image size")
    ap.add_argument("--format", choices=["yolo", "rfdetr"], default="yolo")
    ap.add_argument("--names", nargs="+", help="class names in training order (required for rfdetr)")
    ap.add_argument("--rfdetr-variant", default="RFDETRBase")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text())
    if args.species != "base" and args.species not in manifest["species"]:
        raise SystemExit(f"Unknown species {args.species!r}; expected one of {list(manifest['species'])} or 'base'")

    if args.format == "yolo":
        onnx, names = export_yolo(args.weights, args.imgsz)
        if args.names:
            names = args.names
    else:
        if not args.names:
            raise SystemExit("--names is required for rfdetr")
        onnx, names = export_rfdetr(args.weights, args.imgsz, args.rfdetr_variant), args.names

    dest = MODELS / f"{args.species.lower()}.onnx"
    shutil.copyfile(onnx, dest)
    entry = {"file": f"models/{dest.name}", "format": args.format, "imgsz": args.imgsz,
             "names": names, "source_weights": args.weights.name}
    if args.species == "base":
        manifest["base"] = entry
    else:
        manifest["species"][args.species] = entry
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {dest.relative_to(ROOT)} ({dest.stat().st_size / 1e6:.1f} MB) and updated models/models.json")
    print("Classes:", names)


if __name__ == "__main__":
    main()
