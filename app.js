/* Lysosome scoring assistant: browser version.
 * All inference runs locally with onnxruntime-web; no image is uploaded anywhere.
 * Models are listed in models/models.json (see tools/export_onnx.py).
 */
"use strict";

const ORT_VERSION = "1.20.1";
ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// Okabe-Ito, colour-blind friendly
const PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#FFFFFF"];
const $ = (id) => document.getElementById(id);

let MANIFEST = null;
const SESSIONS = {};          // key -> {session, entry, label}
let LAST = null;              // results of the last run, for exports

// ------------------------------------------------------------------ utilities
async function sha1Hex(buf, n = 10) {
  const h = await crypto.subtle.digest("SHA-1", buf);
  return [...new Uint8Array(h)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, n);
}
const stamp = () => new Date().toISOString().replace(/[-:]/g, "").replace("T", "-").slice(0, 15);
const csvCell = (v) => {
  if (v === null || v === undefined) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
const toCSV = (cols, rows) => [cols.join(","), ...rows.map((r) => cols.map((c) => csvCell(r[c])).join(","))].join("\n") + "\n";
function download(name, blob) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
const yieldUI = () => new Promise((r) => setTimeout(r, 0));

// ------------------------------------------------------------------ model loading
function resolveEntry(species) {
  const sp = MANIFEST.species?.[species];
  if (sp) return { entry: sp, key: `species:${species}`, note: `Using the <b>${species}-specific</b> model.` };
  if (MANIFEST.base) return { entry: MANIFEST.base, key: "base",
    note: `No ${species}-specific model yet: using the <b>species-blind base</b> model.` };
  return { entry: MANIFEST.demo, key: "demo", note:
    `<div class="warn">⚠️ <b>DEMO MODE: no trained model for ${species} yet.</b> This model was trained on ` +
    `synthetic dots, not cells. It only shows how the app works. Do not use these numbers for scoring.</div>` };
}

async function getModel(species) {
  const r = resolveEntry(species);
  if (!SESSIONS[r.key]) {
    $("progress").textContent = "Downloading model (first time only)…";
    const resp = await fetch(r.entry.file);
    if (!resp.ok) throw new Error(`Could not load model file ${r.entry.file} (${resp.status})`);
    const buf = await resp.arrayBuffer();
    const session = await ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
    const fname = r.entry.file.split("/").pop();
    SESSIONS[r.key] = { session, entry: r.entry, label: `${r.entry.demo ? "DEMO-" : ""}${fname}@${await sha1Hex(buf)}` };
  }
  return { ...SESSIONS[r.key], note: r.note };
}

// ------------------------------------------------------------------ image decoding
function percentileStretch(vals, lo = 0.005, hi = 0.995) {
  const sample = [];
  const step = Math.max(1, Math.floor(vals.length / 200000));
  for (let i = 0; i < vals.length; i += step) sample.push(vals[i]);
  sample.sort((a, b) => a - b);
  const a = sample[Math.floor(lo * (sample.length - 1))], b = sample[Math.floor(hi * (sample.length - 1))];
  const scale = 255 / Math.max(b - a, 1e-6);
  return (v) => Math.max(0, Math.min(255, Math.round((v - a) * scale)));
}

function decodeTiff(buf) {
  const ifds = UTIF.decode(buf);
  const ifd = ifds[0];
  UTIF.decodeImage(buf, ifd);
  const w = ifd.width, h = ifd.height;
  const bps = (ifd.t258 && ifd.t258[0]) || 8;
  const spp = (ifd.t277 && ifd.t277[0]) || 1;
  const rgba = new Uint8ClampedArray(w * h * 4);
  if (bps === 16) {
    // 16-bit: same rule as the Python app (0.5/99.5 percentile stretch per image)
    const little = new Uint8Array(buf, 0, 2)[0] === 0x49; // "II"
    const dv = new DataView(ifd.data.buffer, ifd.data.byteOffset, ifd.data.byteLength);
    const n = w * h * spp;
    const vals = new Uint16Array(n);
    for (let i = 0; i < n; i++) vals[i] = dv.getUint16(i * 2, little);
    const f = percentileStretch(vals);
    for (let p = 0; p < w * h; p++) {
      for (let c = 0; c < 3; c++) rgba[p * 4 + c] = f(vals[p * spp + (spp >= 3 ? c : 0)]);
      rgba[p * 4 + 3] = 255;
    }
  } else {
    rgba.set(UTIF.toRGBA8(ifd));
  }
  return new ImageData(rgba, w, h);
}

async function decodeImage(file, buf) {
  if (/\.tiff?$/i.test(file.name) || file.type === "image/tiff") return decodeTiff(buf);
  const bmp = await createImageBitmap(new Blob([buf], { type: file.type || "image/*" }));
  const c = new OffscreenCanvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d");
  ctx.drawImage(bmp, 0, 0);
  return ctx.getImageData(0, 0, bmp.width, bmp.height);
}

function crop(img, x0, y0, w, h) {
  const out = new ImageData(w, h);
  for (let y = 0; y < h; y++) {
    const s = ((y0 + y) * img.width + x0) * 4;
    out.data.set(img.data.subarray(s, s + w * 4), y * w * 4);
  }
  return out;
}

// ------------------------------------------------------------------ inference
function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const aa = (a.x2 - a.x1) * (a.y2 - a.y1), bb = (b.x2 - b.x1) * (b.y2 - b.y1);
  return { iou: inter / (aa + bb - inter + 1e-9), ios: inter / (Math.min(aa, bb) + 1e-9) };
}

// Per-class NMS, IoU 0.5 = Ultralytics 8.4 predict default (so counts match the Python app)
function nmsPerClass(dets, thr = 0.5) {
  const keep = [];
  const sorted = [...dets].sort((a, b) => b.conf - a.conf);
  const removed = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (removed[i]) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++)
      if (!removed[j] && sorted[j].cls === sorted[i].cls && iou(sorted[i], sorted[j]).iou > thr) removed[j] = 1;
  }
  return keep;
}

// Class-agnostic merge across tiles (same logic as local_app/detector.py)
function mergeTiles(dets, iouThr = 0.5, iosThr = 0.6) {
  const keep = [];
  const sorted = [...dets].sort((a, b) => b.conf - a.conf);
  const removed = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (removed[i]) continue;
    keep.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (removed[j]) continue;
      const o = iou(sorted[i], sorted[j]);
      if (o.iou > iouThr || o.ios > iosThr) removed[j] = 1;
    }
  }
  return keep;
}

async function predictYolo(model, img, conf) {
  const S = model.entry.imgsz;
  const r = Math.min(S / img.width, S / img.height);
  const nw = Math.round(img.width * r), nh = Math.round(img.height * r);
  const padX = (S - nw) / 2, padY = (S - nh) / 2;
  // letterbox with grey 114, like Ultralytics
  const src = new OffscreenCanvas(img.width, img.height);
  src.getContext("2d").putImageData(img, 0, 0);
  const c = new OffscreenCanvas(S, S);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, S, S);
  ctx.imageSmoothingQuality = "low";   // bilinear, closest to Ultralytics/OpenCV INTER_LINEAR
  ctx.drawImage(src, Math.round(padX - 0.1), Math.round(padY - 0.1), nw, nh);
  const px = ctx.getImageData(0, 0, S, S).data;
  const input = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++) {
    input[i] = px[i * 4] / 255;
    input[S * S + i] = px[i * 4 + 1] / 255;
    input[2 * S * S + i] = px[i * 4 + 2] / 255;
  }
  const feeds = { [model.session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, S, S]) };
  const out = (await model.session.run(feeds))[model.session.outputNames[0]];
  const [, ch, N] = out.dims;
  const d = out.data, nc = ch - 4;
  const dets = [];
  for (let j = 0; j < N; j++) {
    let best = -1, bs = 0;
    for (let k = 0; k < nc; k++) { const s = d[(4 + k) * N + j]; if (s > bs) { bs = s; best = k; } }
    if (bs < conf) continue;
    const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
    dets.push({
      x1: Math.max(0, (cx - w / 2 - padX) / r), y1: Math.max(0, (cy - h / 2 - padY) / r),
      x2: Math.min(img.width, (cx + w / 2 - padX) / r), y2: Math.min(img.height, (cy + h / 2 - padY) / r),
      conf: bs, cls: best,
    });
  }
  return nmsPerClass(dets);
}

// RF-DETR ONNX (outputs: boxes cxcywh normalised [1,Q,4], logits [1,Q,C]). UNTESTED until a real export exists.
async function predictRfdetr(model, img, conf) {
  const S = model.entry.imgsz;
  const src = new OffscreenCanvas(img.width, img.height);
  src.getContext("2d").putImageData(img, 0, 0);
  const c = new OffscreenCanvas(S, S);
  const ctx = c.getContext("2d");
  ctx.drawImage(src, 0, 0, S, S);
  const px = ctx.getImageData(0, 0, S, S).data;
  const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
  const input = new Float32Array(3 * S * S);
  for (let i = 0; i < S * S; i++)
    for (let k = 0; k < 3; k++) input[k * S * S + i] = (px[i * 4 + k] / 255 - mean[k]) / std[k];
  const outs = await model.session.run({ [model.session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, S, S]) });
  const [boxName, logitName] = model.session.outputNames;
  const boxes = outs[boxName], logits = outs[logitName];
  const Q = boxes.dims[1], C = logits.dims[2];
  const off = model.entry.class_offset || 0;   // set if the export has a background/placeholder class first
  const dets = [];
  for (let q = 0; q < Q; q++) {
    let best = -1, bs = 0;
    for (let k = off; k < C; k++) {
      const s = 1 / (1 + Math.exp(-logits.data[q * C + k]));
      if (s > bs) { bs = s; best = k - off; }
    }
    if (bs < conf || best >= model.entry.names.length) continue;
    const [cx, cy, w, h] = boxes.data.slice(q * 4, q * 4 + 4);
    dets.push({ x1: (cx - w / 2) * img.width, y1: (cy - h / 2) * img.height,
      x2: (cx + w / 2) * img.width, y2: (cy + h / 2) * img.height, conf: bs, cls: best });
  }
  return dets;
}

const predict = (model, img, conf) =>
  model.entry.format === "rfdetr" ? predictRfdetr(model, img, conf) : predictYolo(model, img, conf);

async function predictTiled(model, img, conf, tile, overlap = 0.2) {
  if (!tile || (img.width <= tile && img.height <= tile)) return predict(model, img, conf);
  const step = Math.max(1, Math.floor(tile * (1 - overlap)));
  const pos = (L) => { const p = []; for (let v = 0; v <= Math.max(L - tile, 0); v += step) p.push(v);
    if (p[p.length - 1] + tile < L) p.push(L - tile); return p; };
  const all = [];
  for (const y0 of pos(img.height)) for (const x0 of pos(img.width)) {
    const t = crop(img, x0, y0, Math.min(tile, img.width), Math.min(tile, img.height));
    for (const d of await predict(model, t, conf))
      all.push({ ...d, x1: d.x1 + x0, y1: d.y1 + y0, x2: d.x2 + x0, y2: d.y2 + y0 });
  }
  return mergeTiles(all);
}

// ------------------------------------------------------------------ drawing
function drawResult(img, dets, names, showLabels) {
  const c = document.createElement("canvas");
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext("2d");
  ctx.putImageData(img, 0, 0);
  const t = Math.max(1, Math.round(Math.min(img.width, img.height) / 500));
  ctx.lineWidth = t;
  ctx.font = `${10 * t}px sans-serif`;
  for (const d of dets) {
    ctx.strokeStyle = ctx.fillStyle = PALETTE[d.cls % PALETTE.length];
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    if (showLabels) ctx.fillText(`${names[d.cls]} ${d.conf.toFixed(2)}`, d.x1, Math.max(10 * t, d.y1 - 2));
  }
  return c;
}

// ------------------------------------------------------------------ main run
async function run() {
  const species = document.querySelector("input[name=species]:checked")?.value;
  const files = [...$("files").files];
  if (!species) return alert("Pick a species first.");
  if (!files.length) return alert("Add at least one image.");
  const conf = parseFloat($("conf").value);
  const tile = parseInt($("tile").value || "0", 10);
  const showLabels = $("labels").checked;

  $("run").disabled = true;
  $("gallery").innerHTML = "";
  $("resultsPanel").hidden = true;
  try {
    const model = await getModel(species);
    const names = model.entry.names;
    const legend = names.map((n, k) => `<span class="swatch" style="background:${PALETTE[k % PALETTE.length]}"></span>${n}`).join("");
    const summary = [], detRows = [], images = [], seen = new Map(), dupes = [];

    for (let i = 0; i < files.length; i++) {
      const f = files[i];
      $("progress").textContent = `Scoring ${i + 1} of ${files.length}: ${f.name}`;
      await yieldUI();
      const buf = await f.arrayBuffer();
      const digest = await sha1Hex(buf, 16);
      if (seen.has(digest)) { dupes.push(`${f.name} (same as ${seen.get(digest)})`); continue; }
      seen.set(digest, f.name);
      const img = await decodeImage(f, buf);
      const dets = await predictTiled(model, img, conf, tile);

      const counts = names.map(() => 0);
      dets.forEach((d) => counts[d.cls]++);
      const total = dets.length;
      const row = { image: f.name, species, model: model.label, conf_threshold: conf, tile, total };
      names.forEach((n, k) => (row[`n_${n}`] = counts[k]));
      names.forEach((n, k) => (row[`prop_${n}`] = total ? +(counts[k] / total).toFixed(4) : null));
      summary.push(row);
      dets.forEach((d) => detRows.push({ image: f.name, species, model: model.label, class: names[d.cls],
        conf: +d.conf.toFixed(4), x1: +d.x1.toFixed(1), y1: +d.y1.toFixed(1), x2: +d.x2.toFixed(1), y2: +d.y2.toFixed(1) }));

      const canvas = drawResult(img, dets, names, showLabels);
      const fig = document.createElement("figure");
      const cap = document.createElement("figcaption");
      cap.textContent = `${f.name}: ${total} detected`;
      fig.append(canvas, cap);
      $("gallery").append(fig);
      images.push({ index: images.length, name: f.name, w: img.width, h: img.height, dets, canvas });
    }

    if (summary.length > 1) {
      const pooled = { image: "ALL (pooled)", species, model: model.label, conf_threshold: conf, tile,
        total: summary.reduce((s, r) => s + r.total, 0) };
      names.forEach((n) => (pooled[`n_${n}`] = summary.reduce((s, r) => s + r[`n_${n}`], 0)));
      names.forEach((n) => (pooled[`prop_${n}`] = pooled.total ? +(pooled[`n_${n}`] / pooled.total).toFixed(4) : null));
      summary.push(pooled);
    }

    const cols = ["image", "species", "model", "conf_threshold", "tile", "total",
      ...names.map((n) => `n_${n}`), ...names.map((n) => `prop_${n}`)];
    LAST = { species, names, summary, detRows, images, cols, stamp: stamp() };

    $("info").innerHTML = `${model.note}<div class="small">Model: <code>${model.label}</code> · box colours: ${legend}</div>` +
      (dupes.length ? `<div class="warn">Skipped duplicate uploads: ${dupes.join("; ")}</div>` : "");
    renderTable(summary, names);
    $("resultsPanel").hidden = false;
    $("progress").textContent = `Done: ${images.length} image(s).`;
  } catch (e) {
    console.error(e);
    $("progress").textContent = "";
    $("info").innerHTML = `<div class="warn">Error: ${e.message}</div>`;
  } finally {
    $("run").disabled = false;
  }
}

function renderTable(rows, names) {
  const shown = ["image", "total", ...names.map((n) => `n_${n}`), ...names.map((n) => `prop_${n}`)];
  const head = `<tr>${shown.map((c) => `<th>${c}</th>`).join("")}</tr>`;
  const body = rows.map((r) => `<tr class="${r.image === "ALL (pooled)" ? "pooled" : ""}">${shown
    .map((c) => `<td>${r[c] ?? "–"}</td>`).join("")}</tr>`).join("");
  const t = $("summary");
  t.innerHTML = head + body;
}

// ------------------------------------------------------------------ exports
const DET_COLS = ["image", "species", "model", "class", "conf", "x1", "y1", "x2", "y2"];
function exportSummary() {
  download(`summary_${LAST.species}_${LAST.stamp}.csv`, new Blob([toCSV(LAST.cols, LAST.summary)], { type: "text/csv" }));
}
function exportDets() {
  download(`detections_${LAST.species}_${LAST.stamp}.csv`, new Blob([toCSV(DET_COLS, LAST.detRows)], { type: "text/csv" }));
}
async function exportZip() {
  const zip = new JSZip();
  zip.file("labels/classes.txt", LAST.names.join("\n") + "\n");
  zip.file(`summary_${LAST.species}_${LAST.stamp}.csv`, toCSV(LAST.cols, LAST.summary));
  zip.file(`detections_${LAST.species}_${LAST.stamp}.csv`, toCSV(DET_COLS, LAST.detRows));
  for (const im of LAST.images) {
    const stem = `${String(im.index).padStart(3, "0")}_${im.name.replace(/\.[^.]+$/, "")}`;
    const lines = im.dets.map((d) => [d.cls, ((d.x1 + d.x2) / 2 / im.w).toFixed(6), ((d.y1 + d.y2) / 2 / im.h).toFixed(6),
      ((d.x2 - d.x1) / im.w).toFixed(6), ((d.y2 - d.y1) / im.h).toFixed(6)].join(" "));
    zip.file(`labels/${stem}.txt`, lines.join("\n") + (lines.length ? "\n" : ""));
    const png = await new Promise((r) => im.canvas.toBlob(r, "image/png"));
    zip.file(`annotated/${stem}.png`, png);
  }
  download(`review_package_${LAST.species}_${LAST.stamp}.zip`, await zip.generateAsync({ type: "blob" }));
}

// ------------------------------------------------------------------ init
async function init() {
  try {
    MANIFEST = await (await fetch("models/models.json", { cache: "no-cache" })).json();
  } catch (e) {
    $("info").innerHTML = `<div class="warn">Could not load models/models.json: ${e.message}</div>`;
    return;
  }
  $("species").innerHTML = Object.keys(MANIFEST.species)
    .map((s) => `<label><input type="radio" name="species" value="${s}"> ${s}</label>`).join("");
  $("conf").addEventListener("input", () => ($("confOut").textContent = (+$("conf").value).toFixed(2)));
  $("files").addEventListener("change", () => {
    const n = $("files").files.length;
    $("filelist").textContent = n ? `${n} image(s) selected` : "";
  });
  $("run").addEventListener("click", run);
  $("dlSummary").addEventListener("click", exportSummary);
  $("dlDets").addEventListener("click", exportDets);
  $("dlZip").addEventListener("click", exportZip);
  $("run").disabled = false;
  $("run").textContent = "Score";
}
init();
