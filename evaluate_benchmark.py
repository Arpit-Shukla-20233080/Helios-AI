# evaluate_benchmark.py
"""
HELIOS-AI Benchmark & Latency Suite.

Benchmarks the pre-trained thermal detector ThermalDetector/Th_G_v9.pt on
Results/INPUT.jpg - no training, no downloads.

Measured
--------
* Latency over N timed iterations (after warm-up), per stage:
    - HELIOS preprocessing   EXIF / bit depth / aspect-true resize / grayscale
    - YOLO preprocessing     letterbox + tensor conversion   (Ultralytics profiler)
    - Forward pass           neural network inference         (Ultralytics profiler)
    - NMS / postprocessing   box decoding + class-agnostic NMS (Ultralytics profiler)
  plus total latency, effective FPS, and the full cockpit pipeline end to end
  (IPCV layers, homography, IEC enrichment, crops, base64 encoding).
* Model statistics: parameters, GFLOPs (thop), layers, weights size, parameter
  memory, process RSS before/after load and at peak.
* Determinism: identical detections across all timed iterations.
* Label-free robustness: prediction consistency (IoU >= 0.5 matching against the
  clean prediction) under photometric and geometric perturbations. This measures
  stability, not accuracy - no ground-truth labels are used.

Outputs
-------
* Formatted console summary
* Results/BENCHMARK_REPORT.png   executive performance card
* Results/benchmark_metrics.json machine-readable metrics (served by the API)

Usage
-----
    python evaluate_benchmark.py
    python evaluate_benchmark.py --iterations 50 --device cuda:0
    python evaluate_benchmark.py --no-robustness --no-plot
"""

import argparse
import copy
import datetime as dt
import json
import os
import platform
import statistics
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import psutil
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "Results", "INPUT.jpg")
DEFAULT_PNG = os.path.join(PROJECT_ROOT, "Results", "BENCHMARK_REPORT.png")
DEFAULT_JSON = os.path.join(PROJECT_ROOT, "Results", "benchmark_metrics.json")

# Chart tokens (dark surface) - shared with the cockpit's look.
SURFACE, PANEL = "#111316", "#1a1a19"
INK, INK_2, MUTED = "#f0efec", "#c3c2b7", "#898781"
GRID, BASELINE = "#2c2c2a", "#383835"
SERIES_1 = "#3987e5"
IEC_COLORS = {"Class 3": "#ef4444", "Class 2": "#f59e0b", "Class 1": "#3b82f6"}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _rss_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def _peak_rss_mb() -> Optional[float]:
    """Peak resident set size of this process in MB (Windows peak_wset / POSIX ru_maxrss)."""
    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak:
        return peak / 1e6
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss / 1e6 if sys.platform == "darwin" else maxrss / 1e3
    except (ImportError, AttributeError):
        return None


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def _greedy_match(base: np.ndarray, pred: np.ndarray, iou_thr: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU."""
    iou = _box_iou(base, pred)
    matches, used_b, used_p = [], set(), set()
    for flat in np.argsort(-iou, axis=None):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < iou_thr:
            break
        if i in used_b or j in used_p:
            continue
        used_b.add(i)
        used_p.add(j)
        matches.append((int(i), int(j), float(iou[i, j])))
    return matches


def _detections(result) -> Dict[str, np.ndarray]:
    boxes = result.boxes
    return {
        "xyxyn": boxes.xyxyn.cpu().numpy() if boxes is not None else np.zeros((0, 4)),
        "cls": boxes.cls.cpu().numpy().astype(int) if boxes is not None else np.zeros(0, int),
        "conf": boxes.conf.cpu().numpy() if boxes is not None else np.zeros(0),
    }


def _count_gflops(torch_model: torch.nn.Module, height: int, width: int) -> Optional[float]:
    """GFLOPs (2 x MACs) via thop, falling back to torch.utils.flop_counter."""
    probe = copy.deepcopy(torch_model).float().cpu().eval()
    x = torch.zeros(1, 3, height, width)
    try:
        import thop
        macs, _ = thop.profile(probe, inputs=[x], verbose=False)
        return 2.0 * macs / 1e9
    except Exception:
        pass
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as counter:
            probe(x)
        return counter.get_total_flops() / 1e9
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Perturbations for the label-free robustness suite (input: uint8 grayscale)
# ──────────────────────────────────────────────────────────────────────────────
def _jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE) if ok else img


def _perturbations() -> List[Tuple[str, Callable[[np.ndarray], np.ndarray], bool]]:
    """(name, transform, horizontally flipped?)"""
    rng = np.random.default_rng(42)
    return [
        ("Gaussian noise σ=8", lambda g: np.clip(g + rng.normal(0, 8, g.shape), 0, 255).astype(np.uint8), False),
        ("Gaussian blur σ=1.2", lambda g: cv2.GaussianBlur(g, (0, 0), 1.2), False),
        ("Brightness +20 DL", lambda g: cv2.add(g, 20), False),
        ("Contrast ×0.8", lambda g: np.clip((g.astype(np.float32) - g.mean()) * 0.8 + g.mean(), 0, 255).astype(np.uint8), False),
        ("Gamma 0.8", lambda g: (255.0 * (g / 255.0) ** 0.8).astype(np.uint8), False),
        ("JPEG quality 40", lambda g: _jpeg(g, 40), False),
        ("Half-res resample", lambda g: cv2.resize(cv2.resize(g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                                                   (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR), False),
        ("Horizontal flip", lambda g: cv2.flip(g, 1), True),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────
def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Benchmark image not found: {args.image}")

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    rss_before = _rss_mb()
    t0 = time.perf_counter()
    from model_loader import WEIGHTS_PATH, model          # noqa: E402  (load is part of the benchmark)
    from predictor import inspect_thermal_image, preprocess_image
    from solar_iec_analyzer import FAULT_KNOWLEDGE_BASE, generate_flir_ironbow, generate_otsu_threshold, generate_sobel_gradient
    load_seconds = time.perf_counter() - t0
    rss_after_load = _rss_mb()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    source = Image.open(args.image)
    source.load()
    predict_kwargs = dict(conf=args.conf, iou=0.7, agnostic_nms=True, verbose=False, device=device, imgsz=args.imgsz)

    def helios_preprocess() -> np.ndarray:
        return (preprocess_image(source, args.imgsz) * 255.0 + 0.5).astype(np.uint8)

    # Warm-up (graph/kernels/allocator), excluded from statistics.
    gray = helios_preprocess()
    for _ in range(args.warmup):
        model.predict(Image.fromarray(gray), **predict_kwargs)

    stage = {"helios_preprocess": [], "yolo_preprocess": [], "forward": [], "nms_postprocess": [],
             "ipcv_layers": [], "total": []}
    signatures, last = [], None
    peak_rss = rss_after_load
    for _ in range(args.iterations):
        t = time.perf_counter()
        gray = helios_preprocess()
        pil_in = Image.fromarray(gray)
        t_helios = (time.perf_counter() - t) * 1000.0
        result = model.predict(pil_in, **predict_kwargs)[0]
        speed = result.speed
        t = time.perf_counter()
        generate_flir_ironbow(gray)
        generate_sobel_gradient(gray)
        generate_otsu_threshold(gray)
        t_ipcv = (time.perf_counter() - t) * 1000.0

        stage["helios_preprocess"].append(t_helios)
        stage["yolo_preprocess"].append(float(speed["preprocess"]))
        stage["forward"].append(float(speed["inference"]))
        stage["nms_postprocess"].append(float(speed["postprocess"]))
        stage["ipcv_layers"].append(t_ipcv)
        stage["total"].append(t_helios + float(speed["preprocess"]) + float(speed["inference"]) + float(speed["postprocess"]))

        det = _detections(result)
        signatures.append(tuple((int(c), *np.round(b, 4)) for c, b in zip(det["cls"], det["xyxyn"])))
        last = result
        peak_rss = max(peak_rss, _rss_mb())

    # End-to-end cockpit pipeline (IPCV + geometry + enrichment + crops + encoding).
    e2e = []
    for _ in range(args.e2e_iterations):
        t = time.perf_counter()
        inspect_thermal_image(source, conf_threshold=args.conf, detection_floor=0.20)
        e2e.append((time.perf_counter() - t) * 1000.0)
        peak_rss = max(peak_rss, _rss_mb())

    latency = {k: _stats(v) for k, v in stage.items()}
    latency["end_to_end_pipeline"] = _stats(e2e) if e2e else None
    model_only = [a + b + c for a, b, c in zip(stage["yolo_preprocess"], stage["forward"], stage["nms_postprocess"])]
    fps = 1000.0 / latency["total"]["mean"]

    # Detection summary for the benchmark frame.
    base = _detections(last)
    names = model.names
    per_class: Dict[str, int] = {}
    severity: Dict[str, str] = {}
    for c in base["cls"]:
        per_class[names[int(c)]] = per_class.get(names[int(c)], 0) + 1
        severity[names[int(c)]] = FAULT_KNOWLEDGE_BASE.get(int(c), {}).get("iec_class", "Class 1")

    # Robustness (label-free consistency against the clean prediction).
    robustness = []
    if not args.no_robustness:
        clean_gray = helios_preprocess()
        for name, transform, flipped in _perturbations():
            perturbed = transform(clean_gray.copy())
            pred = _detections(model.predict(Image.fromarray(perturbed), **predict_kwargs)[0])
            boxes = pred["xyxyn"].copy()
            if flipped and len(boxes):
                boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]
            matches = _greedy_match(base["xyxyn"], boxes)
            n_base, n_pred = len(base["xyxyn"]), len(boxes)
            recall = len(matches) / n_base if n_base else 1.0
            precision = len(matches) / n_pred if n_pred else (1.0 if n_base == 0 else 0.0)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
            same_class = sum(1 for i, j, _ in matches if base["cls"][i] == pred["cls"][j])
            robustness.append({
                "perturbation": name,
                "detections": n_pred,
                "recall": round(recall, 3),
                "precision": round(precision, 3),
                "f1": round(f1, 3),
                "class_agreement": round(same_class / len(matches), 3) if matches else None,
                "mean_iou": round(float(np.mean([m[2] for m in matches])), 3) if matches else None,
                "mean_abs_conf_shift": round(float(np.mean([abs(base["conf"][i] - pred["conf"][j])
                                                            for i, j, _ in matches])), 3) if matches else None,
            })

    # Peak memory is read before FLOP profiling, which deep-copies the network.
    peak = _peak_rss_mb()

    # Model statistics (after timing so profiling cannot disturb the measurements).
    torch_model = model.model
    n_params = sum(p.numel() for p in torch_model.parameters())
    param_bytes = sum(p.numel() * p.element_size() for p in torch_model.parameters())
    input_h, input_w = (int(v) for v in last.orig_shape)
    gflops_640 = _count_gflops(torch_model, 640, 640)
    pad_h = int(np.ceil(input_h / 32) * 32)
    gflops_frame = _count_gflops(torch_model, pad_h, int(np.ceil(input_w / 32) * 32))

    return {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "image": os.path.relpath(args.image, PROJECT_ROOT),
        "weights": os.path.relpath(WEIGHTS_PATH, PROJECT_ROOT),
        "config": {"iterations": args.iterations, "warmup": args.warmup, "e2e_iterations": args.e2e_iterations,
                   "conf": args.conf, "iou": 0.7, "agnostic_nms": True, "imgsz": args.imgsz, "device": device},
        "system": {
            "platform": f"{platform.system()} {platform.release()}",
            "processor": platform.processor() or platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": __import__("ultralytics").__version__,
            "opencv": cv2.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        },
        "model": {
            "architecture": type(torch_model).__name__,
            "family": "YOLOv9 (GELAN) thermal detector",
            "layers": len(getattr(torch_model, "model", [])),
            "parameters": int(n_params),
            "parameters_m": round(n_params / 1e6, 3),
            "gflops_640": round(gflops_640, 2) if gflops_640 is not None else None,
            "gflops_frame": round(gflops_frame, 2) if gflops_frame is not None else None,
            "frame_shape": [input_h, input_w],
            "weights_size_mb": round(os.path.getsize(WEIGHTS_PATH) / 1e6, 2),
            "parameter_memory_mb": round(param_bytes / 1e6, 2),
            "classes": {str(k): v for k, v in names.items()},
        },
        "memory": {
            "rss_before_load_mb": round(rss_before, 1),
            "rss_after_load_mb": round(rss_after_load, 1),
            "rss_peak_sampled_mb": round(peak_rss, 1),
            "rss_peak_os_mb": round(peak, 1) if peak else None,
            "cuda_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device.startswith("cuda") else None,
        },
        "load_seconds": round(load_seconds, 3),
        "latency_ms": {k: ({kk: round(vv, 3) for kk, vv in v.items()} if v else None) for k, v in latency.items()},
        "latency_series_ms": {k: [round(x, 3) for x in v] for k, v in stage.items()},
        "model_only_ms": {kk: round(vv, 3) for kk, vv in _stats(model_only).items()},
        "fps": {"effective": round(fps, 2), "model_only": round(1000.0 / float(np.mean(model_only)), 2),
                "end_to_end": round(1000.0 / latency["end_to_end_pipeline"]["mean"], 2) if e2e else None},
        "deterministic": len(set(signatures)) == 1,
        "detections": {
            "total": int(len(base["cls"])),
            "mean_confidence": round(float(base["conf"].mean()), 3) if len(base["conf"]) else None,
            "per_class": per_class,
            "severity": severity,
        },
        "robustness": robustness,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(m: Dict[str, Any]) -> None:
    W = 78
    rule, thin = "═" * W, "─" * W

    def row(label: str, value: str) -> None:
        print(f"  {label:<34}{value}")

    lat = m["latency_ms"]
    print(rule)
    print("  HELIOS-AI  ·  MODEL BENCHMARK & LATENCY SUITE".ljust(W))
    print(f"  {m['weights']}  on  {m['image']}  ·  {m['generated_at']}")
    print(rule)
    s = m["system"]
    row("Device", f"{m['config']['device']}  ({s['cuda_device'] or s['processor']})")
    row("Runtime", f"Python {s['python']} · torch {s['torch']} · ultralytics {s['ultralytics']} · OpenCV {s['opencv']}")
    row("CPU threads (torch / logical)", f"{s['torch_threads']} / {s['cpu_count']}")
    print(thin)
    md = m["model"]
    row("Architecture", f"{md['family']} ({md['architecture']}, {md['layers']} blocks)")
    row("Parameters", f"{md['parameters_m']:.2f} M  ({md['parameters']:,})")
    row("GFLOPs @ 640×640", f"{md['gflops_640']}" if md["gflops_640"] is not None else "n/a")
    row(f"GFLOPs @ {md['frame_shape'][1]}×{md['frame_shape'][0]} frame",
        f"{md['gflops_frame']}" if md["gflops_frame"] is not None else "n/a")
    row("Weights file / parameter memory", f"{md['weights_size_mb']} MB / {md['parameter_memory_mb']} MB (fp32)")
    mem = m["memory"]
    row("Process RSS before → after load", f"{mem['rss_before_load_mb']} → {mem['rss_after_load_mb']} MB")
    row("Peak RSS (OS reported)", f"{mem['rss_peak_os_mb']} MB" if mem["rss_peak_os_mb"] else f"{mem['rss_peak_sampled_mb']} MB")
    if mem["cuda_peak_allocated_mb"] is not None:
        row("CUDA peak allocated", f"{mem['cuda_peak_allocated_mb']} MB")
    row("Model load time", f"{m['load_seconds']:.2f} s")
    print(thin)
    print(f"  {'Stage (' + str(m['config']['iterations']) + ' iterations, ms)':<34}{'mean':>9}{'std':>9}{'p50':>9}{'p95':>9}{'max':>9}")
    labels = [("helios_preprocess", "HELIOS preprocessing"), ("yolo_preprocess", "YOLO letterbox / tensor"),
              ("forward", "Neural network forward pass"), ("nms_postprocess", "NMS / postprocessing"),
              ("total", "TOTAL detection latency"), ("ipcv_layers", "IPCV layers (CLAHE/Sobel/Otsu)"),
              ("end_to_end_pipeline", "Full cockpit pipeline (E2E)")]
    for key, label in labels:
        v = lat.get(key)
        if not v:
            continue
        if key == "total":
            print("  " + "·" * (W - 2))
        print(f"  {label:<34}{v['mean']:>9.2f}{v['std']:>9.2f}{v['p50']:>9.2f}{v['p95']:>9.2f}{v['max']:>9.2f}")
    print(thin)
    fps = m["fps"]
    row("Effective FPS (detection)", f"{fps['effective']:.2f}")
    row("Model-only FPS (letterbox→NMS)", f"{fps['model_only']:.2f}")
    if fps["end_to_end"]:
        row("End-to-end FPS (cockpit pipeline)", f"{fps['end_to_end']:.2f}")
    row("Deterministic across iterations", "YES" if m["deterministic"] else "NO")
    print(thin)
    d = m["detections"]
    row(f"Detections @ conf ≥ {m['config']['conf']:.2f}", f"{d['total']} (mean confidence {d['mean_confidence']})")
    for name, count in sorted(d["per_class"].items(), key=lambda kv: -kv[1]):
        row(f"   {name}", f"{count}   [{d['severity'][name]}]")
    row("Classes", ", ".join(f"{k}:{v}" for k, v in md["classes"].items()))
    if m["robustness"]:
        print(thin)
        print(f"  {'Robustness (label-free consistency)':<34}{'dets':>6}{'recall':>8}{'prec.':>8}{'F1':>7}{'cls=':>7}{'IoU':>7}")
        for r in m["robustness"]:
            ca = f"{r['class_agreement']:.2f}" if r["class_agreement"] is not None else "  -"
            iou = f"{r['mean_iou']:.2f}" if r["mean_iou"] is not None else "  -"
            print(f"  {r['perturbation']:<34}{r['detections']:>6}{r['recall']:>8.2f}{r['precision']:>8.2f}"
                  f"{r['f1']:>7.2f}{ca:>7}{iou:>7}")
        mean_f1 = statistics.mean(r["f1"] for r in m["robustness"])
        row("Mean consistency F1", f"{mean_f1:.3f}")
    print(rule)


def save_report_card(m: Dict[str, Any], path: str) -> None:
    """Renders the executive performance card with Matplotlib (Agg backend)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
        "axes.titlecolor": INK, "text.color": INK, "axes.facecolor": PANEL,
    })
    fig = plt.figure(figsize=(16, 9), dpi=130, facecolor=SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.62, 1.25, 1.25], hspace=0.62, wspace=0.42,
                          left=0.125, right=0.975, top=0.86, bottom=0.07)

    md, lat, fps = m["model"], m["latency_ms"], m["fps"]
    fig.text(0.045, 0.945, "HELIOS-AI · Model Benchmark & Latency Card", fontsize=20, fontweight="bold", color=INK)
    s = m["system"]
    fig.text(0.045, 0.905,
             f"{m['weights']} · {m['image']} · {m['config']['device']} ({s['cuda_device'] or s['processor']}) · "
             f"torch {s['torch']} · ultralytics {s['ultralytics']} · {m['config']['iterations']} timed iterations · "
             f"{m['generated_at']}", fontsize=9.5, color=MUTED)

    # KPI tiles
    peak = m["memory"]["rss_peak_os_mb"] or m["memory"]["rss_peak_sampled_mb"]
    tiles = [
        ("Effective FPS", f"{fps['effective']:.2f}", "detection path"),
        ("Mean latency", f"{lat['total']['mean']:.0f} ms", f"p95 {lat['total']['p95']:.0f} ms"),
        ("Forward pass", f"{lat['forward']['mean']:.0f} ms",
         f"{lat['forward']['mean'] / lat['total']['mean'] * 100:.0f}% of total"),
        ("Parameters", f"{md['parameters_m']:.1f} M", f"{md['weights_size_mb']:.0f} MB weights"),
        ("Compute", f"{md['gflops_640']:.1f} GFLOPs" if md["gflops_640"] else "n/a", "at 640×640"),
        ("Peak memory", f"{peak:.0f} MB", "process RSS"),
    ]
    tile_box = gs[0, :].get_position(fig)
    tile_ax = fig.add_axes([0.045, tile_box.y0, 0.975 - 0.045, tile_box.height])  # align with the title, not the y-labels
    tile_ax.axis("off")
    n = len(tiles)
    for i, (label, value, sub) in enumerate(tiles):
        x0 = i / n + 0.004
        tile_ax.add_patch(plt.Rectangle((x0, 0.0), 1 / n - 0.012, 1.0, transform=tile_ax.transAxes,
                                        facecolor=PANEL, edgecolor=GRID, linewidth=1))
        tile_ax.text(x0 + 0.012, 0.74, label.upper(), transform=tile_ax.transAxes, fontsize=8.5, color=MUTED)
        tile_ax.text(x0 + 0.012, 0.34, value, transform=tile_ax.transAxes, fontsize=19, fontweight="bold", color=INK)
        tile_ax.text(x0 + 0.012, 0.08, sub, transform=tile_ax.transAxes, fontsize=8.5, color=INK_2)

    def style(ax, title):
        ax.set_title(title, loc="left", fontsize=11.5, pad=8, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(BASELINE)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(length=0, labelsize=9)

    # Stage latency breakdown
    ax = fig.add_subplot(gs[1, 0])
    keys = [("helios_preprocess", "HELIOS preprocess"), ("yolo_preprocess", "YOLO letterbox"),
            ("forward", "Forward pass"), ("nms_postprocess", "NMS / post"), ("ipcv_layers", "IPCV layers")]
    vals = [lat[k]["mean"] for k, _ in keys]
    errs = [lat[k]["std"] for k, _ in keys]
    ypos = np.arange(len(keys))[::-1]
    ax.barh(ypos, vals, height=0.56, color=SERIES_1, xerr=errs, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(ypos, [lbl for _, lbl in keys])
    for y, v in zip(ypos, vals):
        ax.text(v + max(vals) * 0.03, y, f"{v:.1f} ms", va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.set_xlabel("mean latency (ms), ±1 std", fontsize=9)
    style(ax, "Latency by pipeline stage")

    # Per-iteration latency
    ax = fig.add_subplot(gs[1, 1:])
    series = m["latency_series_ms"]["total"]
    it = np.arange(1, len(series) + 1)
    ax.plot(it, series, color=SERIES_1, linewidth=2, marker="o", markersize=4, label="total detection latency")
    ax.axhline(lat["total"]["mean"], color=MUTED, linestyle="--", linewidth=1)
    ax.text(it[-1], lat["total"]["mean"], f"  mean {lat['total']['mean']:.1f} ms", va="bottom", ha="right",
            fontsize=9, color=INK_2)
    ax.set_xlabel("iteration", fontsize=9)
    ax.set_ylabel("ms", fontsize=9)
    lo, hi = min(series), max(series)
    ax.set_ylim(max(0, lo - (hi - lo) * 0.6 - 1), hi + (hi - lo) * 0.6 + 1)
    ax.set_xticks(it if len(it) <= 25 else None)
    style(ax, f"Per-iteration latency (std {lat['total']['std']:.1f} ms · deterministic outputs: "
              f"{'yes' if m['deterministic'] else 'no'})")
    ax.grid(axis="y", color=GRID, linewidth=0.8)

    # Detections per class (IEC severity colour + label)
    ax = fig.add_subplot(gs[2, 0])
    det = m["detections"]
    items = sorted(det["per_class"].items(), key=lambda kv: kv[1])
    if items:
        y = np.arange(len(items))
        colors = [IEC_COLORS.get(det["severity"][k], SERIES_1) for k, _ in items]
        ax.barh(y, [v for _, v in items], height=0.56, color=colors)
        ax.set_yticks(y, [f"{k} · {det['severity'][k]}" for k, _ in items])
        for yy, (_, v) in zip(y, items):
            ax.text(v + 0.08, yy, str(v), va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, max(v for _, v in items) * 1.25 + 0.5)
    ax.set_xlabel("detections", fontsize=9)
    style(ax, f"Detections @ conf ≥ {m['config']['conf']:.2f} (n = {det['total']})")

    # Robustness
    ax = fig.add_subplot(gs[2, 1])
    rob = m["robustness"]
    if rob:
        y = np.arange(len(rob))[::-1]
        f1 = [r["f1"] for r in rob]
        ax.barh(y, f1, height=0.56, color=SERIES_1)
        ax.set_yticks(y, [r["perturbation"] for r in rob])
        for yy, v in zip(y, f1):
            ax.text(min(v + 0.02, 1.02), yy, f"{v:.2f}", va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("consistency F1 vs. clean prediction (IoU ≥ 0.5)", fontsize=9)
        style(ax, f"Label-free robustness (mean F1 {statistics.mean(f1):.2f})")
    else:
        ax.axis("off")

    # System / model facts
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    facts = [
        ("Architecture", f"{md['architecture']} · {md['layers']} blocks"),
        ("Frame", f"{md['frame_shape'][1]}×{md['frame_shape'][0]} (aspect-true)"),
        ("GFLOPs @ frame", f"{md['gflops_frame']}" if md["gflops_frame"] else "n/a"),
        ("Param memory", f"{md['parameter_memory_mb']} MB fp32"),
        ("Model load", f"{m['load_seconds']:.2f} s"),
        ("Model-only FPS", f"{fps['model_only']:.2f}"),
        ("End-to-end FPS", f"{fps['end_to_end']:.2f}" if fps["end_to_end"] else "n/a"),
        ("Torch threads", f"{s['torch_threads']} of {s['cpu_count']} logical CPUs"),
        ("NMS", "class-agnostic, IoU 0.70"),
    ]
    ax.set_title("System & model", loc="left", fontsize=11.5, pad=8, color=INK)
    for i, (k, v) in enumerate(facts):
        yy = 0.95 - i * 0.108
        ax.text(0.0, yy, k, fontsize=9.5, color=MUTED, transform=ax.transAxes, va="top")
        ax.text(0.42, yy, v, fontsize=9.5, color=INK, transform=ax.transAxes, va="top")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the HELIOS-AI YOLOv9 thermal detector.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Benchmark image (default: Results/INPUT.jpg)")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations (default: 20)")
    parser.add_argument("--warmup", type=int, default=3, help="Untimed warm-up iterations (default: 3)")
    parser.add_argument("--e2e-iterations", type=int, default=5,
                        help="Iterations of the full cockpit pipeline (default: 5, 0 to skip)")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference long side (default: 640)")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda:0")
    parser.add_argument("--output", default=DEFAULT_PNG, help="Performance card PNG path")
    parser.add_argument("--json", default=DEFAULT_JSON, help="Metrics JSON path")
    parser.add_argument("--no-robustness", action="store_true", help="Skip the perturbation consistency suite")
    parser.add_argument("--no-plot", action="store_true", help="Skip the PNG performance card")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")

    try:
        metrics = run_benchmark(args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[evaluate_benchmark] {exc}", file=sys.stderr)
        return 1

    print_summary(metrics)
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    print(f"  Metrics JSON          → {os.path.relpath(args.json, PROJECT_ROOT)}")
    if not args.no_plot:
        save_report_card(metrics, args.output)
        print(f"  Performance card PNG  → {os.path.relpath(args.output, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
