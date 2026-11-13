"""Detector backends for the lysosome scoring app.

Every backend exposes the same interface:

    det = SomeDetector(...)
    boxes = det.predict(image_rgb: np.ndarray, conf: float) -> list[Detection]
    det.class_names -> list[str]
    det.label -> str   (human-readable description, written into every CSV row)

Backends:
  * UltralyticsDetector  - YOLO .pt weights (ultralytics package)
  * RFDETRDetector       - RF-DETR .pth weights (rfdetr package)  [UNTESTED: no weights yet]
 * RoboflowHostedDetector - calls a model hosted on Roboflow's serverless API (images leave
                           the machine; needs ROBOFLOW_API_KEY; uses Roboflow credits)
  * DemoBlobDetector     - no ML, OpenCV blob finder. Exists ONLY so the UI can be
                           exercised before trained weights exist. Its output is not a
                           lysosome classification and must not be used for scoring.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int


def file_sha1(path: str | Path, n: int = 10) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


# --------------------------------------------------------------------------- backends
class UltralyticsDetector:
    def __init__(self, weights: str | Path, imgsz: int = 1024, device: str | None = None):
        from ultralytics import YOLO  # imported lazily so the demo runs without torch

        self.model = YOLO(str(weights))
        self.imgsz = imgsz
        self.device = device
        names = self.model.names
        self.class_names = [names[i] for i in sorted(names)]
        self.label = f"YOLO:{Path(weights).name}@{file_sha1(weights)}"

    def predict(self, image_rgb: np.ndarray, conf: float) -> list[Detection]:
        # ultralytics expects BGR for numpy input
        res = self.model.predict(
            image_rgb[:, :, ::-1], conf=conf, imgsz=self.imgsz, device=self.device,
            verbose=False, max_det=3000,
        )[0]
        out = []
        if res.boxes is None:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, cls):
            out.append(Detection(float(x1), float(y1), float(x2), float(y2), float(c), int(k)))
        return out


class RFDETRDetector:
    """RF-DETR backend. UNTESTED until real weights exist; API per rfdetr>=1.2."""

    def __init__(self, weights: str | Path, class_names: list[str], variant: str = "RFDETRBase"):
        import rfdetr

        cls = getattr(rfdetr, variant)
        self.model = cls(pretrain_weights=str(weights))
        self.class_names = list(class_names)
        self.label = f"RFDETR:{Path(weights).name}@{file_sha1(weights)}"

    def predict(self, image_rgb: np.ndarray, conf: float) -> list[Detection]:
        from PIL import Image

        dets = self.model.predict(Image.fromarray(image_rgb), threshold=conf)
        return [
            Detection(*map(float, box), float(c), int(k))
            for box, c, k in zip(dets.xyxy, dets.confidence, dets.class_id)
        ]


class RoboflowError(RuntimeError):
    """Raised when the Roboflow hosted API cannot be reached or rejects a request."""


class RoboflowHostedDetector:
    """Object-detection model hosted on Roboflow (used when weights cannot be downloaded).

    Response shape (captured from a real call, 2026-11-13):
      {"image": {"width", "height"},
       "predictions": [{"x", "y", "width", "height", "confidence", "class", "class_id", ...}]}
    x/y are box centres in pixels of the image as sent.
    """

    def __init__(self, model_id: str, class_names: list[str] | None = None,
                 api_url: str = "https://serverless.roboflow.com", overlap: float = 0.5,
                 timeout: float = 60.0, retries: int = 3):
        import os

        self.api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise RoboflowError("ROBOFLOW_API_KEY is not set (add it as a Space secret or in .env).")
        self.model_id = model_id.strip("/")
        self.url = f"{api_url.rstrip('/')}/{self.model_id}"
        self.class_names = list(class_names or [])
        self.overlap, self.timeout, self.retries = overlap, timeout, retries
        self.label = f"Roboflow:{self.model_id}"

    def _class_index(self, name: str) -> int:
        if name not in self.class_names:  # unseen class: append so nothing is silently dropped
            self.class_names.append(name)
        return self.class_names.index(name)

    def _post(self, jpeg_b64: str, conf: float) -> dict:
        import time

        import requests

        last = None
        for attempt in range(self.retries):
            try:
                r = requests.post(
                    self.url,
                    params={"confidence": conf, "overlap": self.overlap},
                    data=jpeg_b64,
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout,
                )
                if r.status_code in (401, 403):
                    raise RoboflowError(f"Roboflow rejected the API key ({r.status_code}).")
                if r.status_code == 404:
                    raise RoboflowError(f"Roboflow model not found: {self.model_id}")
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                return r.json()
            except RoboflowError:
                raise
            except Exception as e:  # noqa: BLE001 - network/5xx: retry with backoff
                last = e
                time.sleep(2 ** attempt)
        raise RoboflowError(f"Roboflow request failed after {self.retries} tries: {last}")

    def predict(self, image_rgb: np.ndarray, conf: float) -> list[Detection]:
        import base64

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RoboflowError("Could not encode image for upload.")
        resp = self._post(base64.b64encode(buf.tobytes()).decode("ascii"), conf)
        preds = resp.get("predictions", resp)
        if isinstance(preds, dict):  # some deployments nest {"predictions": {"predictions": [...]}}
            preds = preds.get("predictions", [])
        out = []
        for p in preds or []:
            try:
                x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                c = float(p.get("confidence", 0.0))
                name = str(p.get("class", p.get("class_id", "?")))
            except (KeyError, TypeError, ValueError):
                continue
            out.append(Detection(x - w / 2, y - h / 2, x + w / 2, y + h / 2, c, self._class_index(name)))
        return out


class DemoBlobDetector:
    """Placeholder: finds dark/bright round blobs with OpenCV. NOT a trained model."""

    class_names = ["candidate (demo, unclassified)"]
    label = "DEMO-blob-detector (no trained weights)"

    def predict(self, image_rgb: np.ndarray, conf: float) -> list[Detection]:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        p = cv2.SimpleBlobDetector_Params()
        p.filterByArea = True
        p.minArea = 15
        p.maxArea = max(500.0, gray.size * 0.01)
        p.filterByCircularity = True
        p.minCircularity = 0.5
        p.filterByConvexity = False
        p.filterByInertia = False
        out = []
        for color in (0, 255):  # dark blobs, then bright blobs
            p.filterByColor = True
            p.blobColor = color
            for kp in cv2.SimpleBlobDetector_create(p).detect(gray):
                x, y, r = kp.pt[0], kp.pt[1], kp.size / 2
                # fake "confidence" from local contrast so the slider does something
                score = float(min(1.0, 0.3 + kp.size / 40))
                if score >= conf:
                    out.append(Detection(x - r, y - r, x + r, y + r, score, 0))
        return out


# --------------------------------------------------------------------------- tiling
def merge_tile_dets(dets: list[Detection], iou: float = 0.5, ios: float = 0.6) -> list[Detection]:
    """Class-agnostic greedy merge for tiled predictions.

    Class-agnostic because one lysosome seen in two tiles can get different labels, and
    per-class NMS would then count it twice. Besides IoU, a box is dropped when most of it
    lies inside a higher-scoring box (intersection over the smaller box > ios): a lysosome
    clipped by a tile edge yields a small partial box whose IoU with the full box is low.
    """
    if not dets:
        return dets
    b = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    order = np.argsort([-d.conf for d in dets])
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep, removed = [], np.zeros(len(dets), bool)
    for i in order:
        if removed[i]:
            continue
        keep.append(dets[i])
        xx1 = np.maximum(b[i, 0], b[:, 0]); yy1 = np.maximum(b[i, 1], b[:, 1])
        xx2 = np.minimum(b[i, 2], b[:, 2]); yy2 = np.minimum(b[i, 3], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        o_iou = inter / (area[i] + area - inter + 1e-9)
        o_ios = inter / (np.minimum(area[i], area) + 1e-9)
        removed |= (o_iou > iou) | (o_ios > ios)
    return keep


def predict_tiled(det, image_rgb: np.ndarray, conf: float, tile: int, overlap: float = 0.2,
                  iou: float = 0.5) -> list[Detection]:
    """Slice large images into overlapping tiles, predict each, merge with NMS.

    Small, dense objects in large micrographs are usually lost when the whole frame is
    downscaled to the model's input size; tiling at the training resolution avoids that.
    Note: a lysosome cut by a tile edge in all tiles that contain it can still be split;
    overlap should exceed the largest object diameter.
    """
    h, w = image_rgb.shape[:2]
    if tile <= 0 or (h <= tile and w <= tile):
        return det.predict(image_rgb, conf)
    step = max(1, int(tile * (1 - overlap)))
    ys = list(range(0, max(h - tile, 0) + 1, step))
    xs = list(range(0, max(w - tile, 0) + 1, step))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    all_dets: list[Detection] = []
    for y0 in ys:
        for x0 in xs:
            crop = image_rgb[y0:y0 + tile, x0:x0 + tile]
            for d in det.predict(np.ascontiguousarray(crop), conf):
                all_dets.append(Detection(d.x1 + x0, d.y1 + y0, d.x2 + x0, d.y2 + y0, d.conf, d.class_id))
    return merge_tile_dets(all_dets, iou)
