# predictor.py
"""
HELIOS-AI inspection pipeline.

    decode (EXIF / 16-bit radiometric aware) -> aspect-true grayscale normalisation
    -> YOLOv9 thermal detector (class-agnostic NMS)
    -> classical IPCV layers (CLAHE + Ironbow, Sobel |grad I|, Otsu) + image statistics
    -> homography geometry of module tables
    -> IEC TS 62446-3 enrichment, measured thermal signature, macro crops
    -> plant KPIs and per-stage latency telemetry

Everything returned by :func:`inspect_thermal_image` is JSON-serialisable; images are
base64 data URIs so the cockpit can render them without further requests.
"""

import base64
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from model_loader import model, model_lock
from solar_iec_analyzer import (
    DEFAULT_MODULE_WATTS,
    DEFAULT_MODULES_PER_STRING,
    DEFAULT_PEAK_SUN_HOURS,
    DEFAULT_TARIFF_USD_PER_KWH,
    RADIOMETRIC_SPAN_C,
    calculate_site_kpis,
    compute_ipcv_statistics,
    enrich_detection,
    generate_flir_ironbow,
    generate_otsu_threshold,
    generate_sobel_gradient,
    measure_thermal_signature,
    render_annotated_detections,
)
from ipcv_geometry import analyze_panel_geometry

ANALYSIS_LONG_SIDE = 640      # model input scale; the frame keeps its aspect ratio
DETECTION_FLOOR = 0.20        # lowest confidence returned to the cockpit slider
NMS_IOU = 0.7                 # Ultralytics default IoU for non-maximum suppression
CROP_PADDING = 0.25           # context margin around each macro crop


def _to_rgb8(img: Image.Image) -> np.ndarray:
    """
    Decodes any PIL image to an RGB uint8 array.

    * Applies the EXIF orientation tag (all 8 orientations).
    * 16-bit / 32-bit radiometric frames (mode I;16, I, F) are stretched to 8 bits with
      a robust 0.5-99.5 percentile window instead of being clipped.
    """
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    if img.mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            hi = lo + 1.0
        gray8 = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return np.stack([gray8] * 3, axis=-1)
    return np.asarray(img.convert("RGB"))


def preprocess_image(img: Image.Image, target_long_side: int = ANALYSIS_LONG_SIDE) -> np.ndarray:
    """
    Preprocesses an image: EXIF orientation, bit-depth normalisation, aspect-true resize
    (long side = ``target_long_side``) and conversion to normalised [0, 1] grayscale.

    The aspect ratio is preserved: YOLO letterboxes internally, so stretching a 640x512
    radiometric frame to a square would only distort module geometry and pin positions.
    """
    rgb = _to_rgb8(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]
    scale = target_long_side / float(max(h, w))
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if new_size != (w, h):
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, new_size, interpolation=interpolation)
    return gray.astype(np.float32) / 255.0


def _encode_data_uri(arr_or_img, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encodes a numpy array or PIL image as a base64 data URI (JPEG or PNG)."""
    pil_img = Image.fromarray(arr_or_img) if isinstance(arr_or_img, np.ndarray) else arr_or_img
    if fmt.upper() == "JPEG" and pil_img.mode not in ("RGB", "L"):
        pil_img = pil_img.convert("RGB")
    buf = BytesIO()
    if fmt.upper() == "JPEG":
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        pil_img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _pil_to_base64_data_uri(pil_img: Image.Image, format: str = "JPEG", quality: int = 85) -> str:
    """Converts a PIL image to a base64 data URI string (kept for backward compatibility)."""
    return _encode_data_uri(pil_img, fmt=format, quality=quality)


def _crop_box(bbox: Dict[str, int], width: int, height: int, min_side: int = 48) -> Optional[Tuple[int, int, int, int]]:
    """
    Square macro-crop window around a detection.

    The side is the longer box edge plus ``CROP_PADDING`` context on each side (at least
    ``min_side`` px), so thin hotspot boxes still show the neighbouring healthy cells
    used as the visual reference. The window is shifted, not truncated, at frame edges.
    """
    bw, bh = bbox["x2"] - bbox["x1"], bbox["y2"] - bbox["y1"]
    if bw <= 0 or bh <= 0:
        return None
    side = int(round(max(bw, bh) * (1.0 + 2.0 * CROP_PADDING)))
    side = min(max(side, min_side), width, height)
    cx, cy = (bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0
    x1 = int(round(min(max(cx - side / 2.0, 0), width - side)))
    y1 = int(round(min(max(cy - side / 2.0, 0), height - side)))
    return x1, y1, x1 + side, y1 + side


def warmup(iterations: int = 1) -> None:
    """Runs dummy inferences so the first real request does not pay predictor set-up cost."""
    dummy = Image.fromarray(np.zeros((ANALYSIS_LONG_SIDE * 4 // 5, ANALYSIS_LONG_SIDE), dtype=np.uint8))
    for _ in range(max(1, iterations)):
        run_detector(dummy, conf=DETECTION_FLOOR)


def run_detector(pil_input: Image.Image, conf: float, agnostic_nms: bool = True, iou: float = NMS_IOU):
    """Runs the YOLOv9 detector under the process-wide model lock."""
    with model_lock:
        return model.predict(pil_input, conf=conf, iou=iou, agnostic_nms=agnostic_nms, verbose=False)


def _device_name() -> str:
    try:
        return str(next(model.model.parameters()).device)
    except Exception:
        return "cpu"


def inspect_thermal_image(
    img: Image.Image,
    conf_threshold: float = 0.45,
    include_crops: bool = True,
    module_watts: float = DEFAULT_MODULE_WATTS,
    tariff_usd: float = DEFAULT_TARIFF_USD_PER_KWH,
    sun_hours: float = DEFAULT_PEAK_SUN_HOURS,
    modules_per_string: int = DEFAULT_MODULES_PER_STRING,
    detection_floor: Optional[float] = None,
    agnostic_nms: bool = True,
    include_geometry: bool = True,
) -> Dict[str, Any]:
    """
    Industry-grade analysis pipeline.

    1. Preprocesses the image (EXIF, bit depth, aspect-true resize, grayscale)
    2. Runs YOLOv9 thermal defect inference (class-agnostic NMS so one physical
       defect is never counted twice under two labels)
    3. Generates the classical IPCV layers: CLAHE + FLIR Ironbow, Sobel gradient
       magnitude, Otsu binarisation, plus histogram / Otsu / gradient statistics
    4. Detects module tables and rectifies the largest with a homography
    5. Extracts pin positions, macro crops and the measured thermal signature
    6. Calculates IEC TS 62446-3 severity, ΔT, power and revenue losses

    Args:
        conf_threshold: confidence at which a detection counts towards the KPIs.
        detection_floor: lowest confidence returned (lets the UI slider go below the
            threshold). Defaults to ``conf_threshold``.
        module_watts / tariff_usd / sun_hours / modules_per_string: economic model.
    """
    timings: Dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    rgb_original = _to_rgb8(img)
    orig_h, orig_w = rgb_original.shape[:2]
    pre_img = preprocess_image(Image.fromarray(rgb_original))
    uint_img = (pre_img * 255.0 + 0.5).astype(np.uint8)
    height, width = uint_img.shape[:2]
    pil_input = Image.fromarray(uint_img)
    timings["preprocess"] = (time.perf_counter() - t0) * 1000.0

    floor = conf_threshold if detection_floor is None else min(detection_floor, conf_threshold)
    floor = float(np.clip(floor, 0.01, 0.99))

    t0 = time.perf_counter()
    results = run_detector(pil_input, conf=floor, agnostic_nms=agnostic_nms)
    timings["inference_wall"] = (time.perf_counter() - t0) * 1000.0
    yolo_speed = dict(results[0].speed) if results else {}

    t0 = time.perf_counter()
    ironbow_np = generate_flir_ironbow(uint_img)
    sobel_np = generate_sobel_gradient(uint_img)
    otsu_np = generate_otsu_threshold(uint_img)
    ipcv_stats = compute_ipcv_statistics(uint_img)
    timings["ipcv_layers"] = (time.perf_counter() - t0) * 1000.0

    geometry_payload: Optional[Dict[str, Any]] = None
    if include_geometry:
        t0 = time.perf_counter()
        try:
            geo = analyze_panel_geometry(uint_img)
            geometry_payload = {
                "modules_detected": geo["modules_detected"],
                "quads": geo["quads"],
                "homography": geo["homography"],
                "diagnostics": geo["diagnostics"],
                "output_size": geo["output_size"],
                "target_method": geo.get("target_method"),
                "overlay_base64": _encode_data_uri(geo["overlay_rgb"]),
                "rectified_base64": _encode_data_uri(geo["rectified_rgb"]) if geo["rectified_rgb"] is not None else None,
                "rectified_aspect_base64": (_encode_data_uri(geo["rectified_aspect_rgb"])
                                            if geo["rectified_aspect_rgb"] is not None else None),
            }
        except Exception as exc:  # geometry is auxiliary; never fail the inspection for it
            geometry_payload = {"modules_detected": 0, "error": str(exc)}
        timings["geometry"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    raw_rgb = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2RGB)
    layer_sources = {"crop": raw_rgb, "ironbow_crop": ironbow_np, "sobel_crop": sobel_np, "otsu_crop": otsu_np}

    candidates: List[Tuple[int, float, List[float]]] = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        for idx in range(len(boxes)):
            conf = float(boxes.conf[idx])
            if conf < floor:
                continue
            candidates.append((int(boxes.cls[idx]), conf, boxes.xyxyn[idx].cpu().numpy().tolist()))

    enriched = [
        enrich_detection(0, class_id, conf, xyxyn, width, height,
                         module_watts=module_watts, modules_per_string=modules_per_string)
        for class_id, conf, xyxyn in candidates
    ]
    # Stable IDs: most critical first, then by confidence.
    enriched.sort(key=lambda d: (d["severity_rank"], -d["confidence_raw"]))

    detections: List[Dict[str, Any]] = []
    for number, det_obj in enumerate(enriched, start=1):
        det_obj["id"] = f"DEF-{number:03d}"
        det_obj["index"] = number
        det_obj["above_threshold"] = det_obj["confidence_raw"] >= conf_threshold
        det_obj["thermal_signature"] = measure_thermal_signature(uint_img, det_obj["bbox_pixels"])
        if include_crops:
            rect = _crop_box(det_obj["bbox_pixels"], width, height)
            for key, source in layer_sources.items():
                field = f"{key}_base64"
                if rect is None:
                    det_obj[field] = None
                    continue
                x1, y1, x2, y2 = rect
                region = source[y1:y2, x1:x2]
                det_obj[field] = _encode_data_uri(region, fmt="PNG" if key == "otsu_crop" else "JPEG", quality=90)
        detections.append(det_obj)

    counted = [d for d in detections if d["above_threshold"]]
    annotated_np = render_annotated_detections(raw_rgb, counted)
    timings["postprocess"] = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    payload_images = {
        "raw_image_base64": _encode_data_uri(raw_rgb),
        "annotated_image_base64": _encode_data_uri(annotated_np),
        "ironbow_image_base64": _encode_data_uri(ironbow_np),
        "sobel_image_base64": _encode_data_uri(sobel_np),
        "otsu_image_base64": _encode_data_uri(otsu_np, fmt="PNG"),
    }
    timings["encoding"] = (time.perf_counter() - t0) * 1000.0

    kpis = calculate_site_kpis(counted, module_watts=module_watts, tariff_usd=tariff_usd, sun_hours=sun_hours)
    timings["total"] = (time.perf_counter() - t_total) * 1000.0

    return {
        "status": "success",
        "image_dimensions": {"width": width, "height": height},
        "original_dimensions": {"width": orig_w, "height": orig_h},
        **payload_images,
        "kpis": kpis,
        "detections": detections,
        "ipcv": ipcv_stats,
        "geometry": geometry_payload,
        "analysis": {
            "conf_threshold": conf_threshold,
            "detection_floor": floor,
            "agnostic_nms": agnostic_nms,
            "nms_iou": NMS_IOU,
            "module_watts": module_watts,
            "tariff_usd_per_kwh": tariff_usd,
            "peak_sun_hours": sun_hours,
            "modules_per_string": modules_per_string,
            "radiometric_span_celsius": RADIOMETRIC_SPAN_C,
        },
        "timing_ms": {k: round(v, 2) for k, v in timings.items()},
        "yolo_speed_ms": {k: round(float(v), 2) for k, v in yolo_speed.items()},
        "model_info": {
            "name": "YOLOv9-Thermal",
            "weights": "ThermalDetector/Th_G_v9.pt",
            "standard": "IEC TS 62446-3 Aligned",
            "classes": {str(k): v for k, v in model.names.items()},
            "device": _device_name(),
        },
    }


# Backward compatibility with original API
def run_inference(img: Image.Image) -> Tuple[List[Dict[str, Any]], str, Optional[Image.Image]]:
    """
    Legacy helper maintained for backward compatibility with ``POST /predict/``.

    Returns (detections as {class_id, confidence, xyxyn}, status message, annotated PIL image)
    from a single inference pass.
    """
    analysis = inspect_thermal_image(img, conf_threshold=0.5, include_crops=False, include_geometry=False)
    legacy_dets = [
        {"class_id": d["class_id"], "confidence": d["confidence_raw"], "xyxyn": d["bbox_normalized"]}
        for d in analysis["detections"]
    ]
    encoded = analysis["annotated_image_base64"].split(",", 1)[1]
    annotated = Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    message = "Success" if legacy_dets else "No anomalies"
    return legacy_dets, message, annotated
