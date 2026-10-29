# solar_iec_analyzer.py
"""
Industrial Domain Intelligence & IEC TS 62446-3 Analytics for Solar Thermography.

Translates raw computer vision detections into quantitative engineering metrics,
severity classifications, and financial yield loss estimations. Also implements
the classical IPCV layer of the pipeline:

* CLAHE contrast enhancement + FLIR Ironbow / Inferno false colour
* Sobel thermal gradient magnitude  |grad I| = sqrt(Gx^2 + Gy^2)
* Otsu's optimal global threshold (OpenCV, cross-checked by a NumPy re-derivation)
* Per-defect thermal signature measured from pixel statistics (hotspot vs. healthy
  module surface in a surrounding annulus), giving an image-derived contrast and
  a calibrated ΔT under an explicit radiometric-span assumption.
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Defect taxonomy aligned with IEC TS 62446-3 and PV failure mode physics.
# ``loss_scope`` states whether the nominal loss applies to one module or the
# whole series string (string-level faults take every module in the string offline).
FAULT_KNOWLEDGE_BASE = {
    0: {
        "code": "MULTI_BYPASSED",
        "name": "Multiple Substrings Bypassed",
        "iec_class": "Class 2",
        "iec_label": "Class 2 (Medium Risk)",
        "color_hex": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.5)",
        "delta_t_range": (18.0, 28.0),
        "nominal_power_loss_pct": 66.6,
        "loss_scope": "module",
        "description": "Two or more substrings in the module have bypass diodes forward-biased, indicating significant internal shading, cracked cells, or severe cell mismatch.",
        "field_action": "Inspect for heavy localized soiling or physical cell fracture. Test individual diode forward voltages at module junction box."
    },
    1: {
        "code": "MULTI_DIODE",
        "name": "Multiple Bypass Diode Failure",
        "iec_class": "Class 3",
        "iec_label": "Class 3 (Critical Risk)",
        "color_hex": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.5)",
        "delta_t_range": (22.0, 36.0),
        "nominal_power_loss_pct": 85.0,
        "loss_scope": "module",
        "description": "Multiple junction box bypass diodes have suffered thermal breakdown or short circuit, causing severe localized junction heating and fire risk.",
        "field_action": "High thermal hazard. Isolate string, inspect junction box for melting/charring, and replace defective Schottky diode assembly."
    },
    2: {
        "code": "MULTI_HOTSPOT",
        "name": "Multiple Cell Hotspots",
        "iec_class": "Class 3",
        "iec_label": "Class 3 (Critical Risk)",
        "color_hex": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.5)",
        "delta_t_range": (25.0, 45.0),
        "nominal_power_loss_pct": 45.0,
        "loss_scope": "module",
        "description": "Multiple localized high-resistance solder joints, micro-cracks, or severe localized cell shunts dissipating power as intense localized heat.",
        "field_action": "Accelerated EVA browning/delamination hazard. Schedule immediate thermographic re-verification and module replacement."
    },
    3: {
        "code": "OPEN_PV",
        "name": "Module Open Circuit",
        "iec_class": "Class 2",
        "iec_label": "Class 2 (Medium Risk)",
        "color_hex": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.5)",
        "delta_t_range": (12.0, 20.0),
        "nominal_power_loss_pct": 100.0,
        "loss_scope": "module",
        "description": "Module completely disconnected or internal busbar ribbon severed; module floats at Voc and operates warmer than working neighbor modules.",
        "field_action": "Check MC4 connector seating and ribbon continuity with digital multimeter."
    },
    4: {
        "code": "SINGLE_BYPASSED",
        "name": "Single Substring Bypassed",
        "iec_class": "Class 1",
        "iec_label": "Class 1 (Low Risk / Monitor)",
        "color_hex": "#3b82f6",
        "badge_bg": "rgba(59, 130, 246, 0.15)",
        "badge_border": "rgba(59, 130, 246, 0.5)",
        "delta_t_range": (8.0, 15.0),
        "nominal_power_loss_pct": 33.3,
        "loss_scope": "module",
        "description": "One of three substrings has active diode conduction due to light partial shading, bird dropping, or minor cell defect.",
        "field_action": "Clean module surface; re-scan after 24 hours to ensure diode returns to non-conducting state."
    },
    5: {
        "code": "SINGLE_DIODE",
        "name": "Single Bypass Diode Fault",
        "iec_class": "Class 2",
        "iec_label": "Class 2 (Medium Risk)",
        "color_hex": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.5)",
        "delta_t_range": (15.0, 25.0),
        "nominal_power_loss_pct": 33.3,
        "loss_scope": "module",
        "description": "Overheated single bypass diode causing one module section to drop offline under load.",
        "field_action": "Verify bypass diode reverse leakage current and replace diode if breakdown occurs."
    },
    6: {
        "code": "SINGLE_HOTSPOT",
        "name": "Single Cell Hotspot",
        "iec_class": "Class 2",
        "iec_label": "Class 2 (Medium Risk)",
        "color_hex": "#f59e0b",
        "badge_bg": "rgba(245, 158, 11, 0.15)",
        "badge_border": "rgba(245, 158, 11, 0.5)",
        "delta_t_range": (14.0, 30.0),
        "nominal_power_loss_pct": 20.0,
        "loss_scope": "module",
        "description": "Single solar cell operating under reverse bias due to micro-fissure or point-shading, dissipating power from neighboring cells.",
        "field_action": "Perform I-V curve trace test to check fill factor reduction."
    },
    7: {
        "code": "STRING_OPEN_CIRCUIT",
        "name": "String Open Circuit Disconnection",
        "iec_class": "Class 3",
        "iec_label": "Class 3 (Critical Risk)",
        "color_hex": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.5)",
        "delta_t_range": (10.0, 18.0),
        "nominal_power_loss_pct": 100.0,  # of entire string
        "loss_scope": "string",
        "description": "Entire series string disconnected from combiner box / inverter MPPT; all modules running at Voc with zero power export.",
        "field_action": "Critical generation outage. Check string fuse, DC isolator switch, and inverter input channel immediately."
    },
    8: {
        "code": "STRING_REVERSED_POLARITY",
        "name": "String Reversed Polarity",
        "iec_class": "Class 3",
        "iec_label": "Class 3 (Critical Emergency)",
        "color_hex": "#ef4444",
        "badge_bg": "rgba(239, 68, 68, 0.15)",
        "badge_border": "rgba(239, 68, 68, 0.5)",
        "delta_t_range": (30.0, 55.0),
        "nominal_power_loss_pct": 100.0,  # of entire string
        "loss_scope": "string",
        "description": "Dangerous wiring fault where a string is cross-connected in reverse polarity, forcing reverse current through bypass diodes.",
        "field_action": "EMERGENCY. Immediately open DC disconnect breaker before catastrophic junction box thermal fire occurs."
    }
}

_UNKNOWN_FAULT = {
    "code": "UNKNOWN",
    "name": "Unclassified Thermal Anomaly",
    "iec_class": "Class 1",
    "iec_label": "Class 1 (Low Risk)",
    "color_hex": "#3b82f6",
    "badge_bg": "rgba(59, 130, 246, 0.15)",
    "badge_border": "rgba(59, 130, 246, 0.5)",
    "delta_t_range": (5.0, 12.0),
    "nominal_power_loss_pct": 15.0,
    "loss_scope": "module",
    "description": "Detected thermal anomaly requiring manual site inspection.",
    "field_action": "Inspect visually and verify with handheld multimeter."
}

DEFAULT_MODULE_WATTS = 400.0        # 400 W standard commercial panel
DEFAULT_PEAK_SUN_HOURS = 4.5        # 4.5 equivalent peak sun hours/day
DEFAULT_TARIFF_USD_PER_KWH = 0.12   # $0.12 per kWh
DEFAULT_MODULES_PER_STRING = 20     # typical 1500 V utility string length

# Assumed linear temperature span represented by the 8-bit thermal frame.
# Non-radiometric JPEG exports from LWIR payloads are AGC-scaled; a 40 °C scene span
# (e.g. 25-65 °C on an operating array) is a conservative mid-day assumption.
# Image-derived ΔT values scale linearly with this constant.
RADIOMETRIC_SPAN_C = 40.0

# Severity ordering used for sorting (critical first) and health penalties.
SEVERITY_RANK = {"Class 3": 0, "Class 2": 1, "Class 1": 2}
SEVERITY_PENALTY = {"Class 3": 15.0, "Class 2": 6.0, "Class 1": 2.0}

# CLAHE parameters shared by the Ironbow layer and reported in IPCV telemetry.
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID = (8, 8)


def _to_gray(np_img: np.ndarray) -> np.ndarray:
    """Returns a single-channel uint8 copy of an RGB or grayscale array."""
    if np_img.ndim == 3:
        return cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return np_img.copy()


def _hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    """Converts '#rrggbb' to an (r, g, b) tuple."""
    value = color_hex.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def generate_flir_ironbow(np_gray: np.ndarray) -> np.ndarray:
    """Applies CLAHE enhancement and FLIR Ironbow / Inferno false-color colormap."""
    gray = _to_gray(np_gray)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    enhanced = clahe.apply(gray)
    colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def generate_sobel_gradient(np_gray: np.ndarray) -> np.ndarray:
    """Computes Sobel thermal gradient magnitude to highlight cell busbars and thermal transitions."""
    gray = _to_gray(np_gray)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    grad_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # Apply colormap to make gradients stand out clearly
    colored = cv2.applyColorMap(grad_norm, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def generate_otsu_threshold(np_gray: np.ndarray) -> np.ndarray:
    """Calculates Otsu's optimal global adaptive threshold for hotspot binarization."""
    gray = _to_gray(np_gray)
    # Apply Gaussian blur to suppress camera sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu_mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Return as 3-channel RGB for display
    return cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2RGB)


def otsu_threshold_numpy(hist: np.ndarray) -> Tuple[int, float]:
    """
    Re-derives Otsu's threshold directly from a 256-bin histogram.

    Maximises the between-class variance
        sigma_B^2(t) = [mu_T * omega(t) - mu(t)]^2 / (omega(t) * (1 - omega(t)))
    where omega(t) is the cumulative class probability and mu(t) the cumulative mean.

    Returns:
        (t_star, sigma_b2_max) - the optimal threshold and the maximal between-class variance.
    """
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return 0, 0.0
    p /= total
    levels = np.arange(p.size, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    t_star = int(np.argmax(sigma_b2))
    return t_star, float(sigma_b2[t_star])


def compute_ipcv_statistics(np_gray: np.ndarray) -> Dict[str, Any]:
    """
    Computes classical image-statistics telemetry for the frame.

    Includes the 256-bin intensity histogram, Otsu's threshold (OpenCV and an
    independent NumPy derivation), Otsu separability eta = sigma_B^2 / sigma_T^2,
    hot-pixel fraction, Sobel gradient energy, edge density, Shannon entropy and
    RMS contrast. All values are JSON-serialisable.
    """
    gray = _to_gray(np_gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    t_cv, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hist_blur = cv2.calcHist([blurred], [0], None, [256], [0, 256]).ravel()
    t_np, sigma_b2 = otsu_threshold_numpy(hist_blur)
    sigma_t2 = float(blurred.astype(np.float64).var())
    separability = sigma_b2 / sigma_t2 if sigma_t2 > 0 else 0.0
    hot_fraction = float(np.mean(blurred > t_cv))

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    edge_threshold = float(np.percentile(magnitude, 90))
    edge_density = float(np.mean(magnitude > max(edge_threshold, 1e-6)))

    hist_raw = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    prob = hist_raw / max(hist_raw.sum(), 1.0)
    nz = prob[prob > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())

    return {
        "histogram": [int(v) for v in hist_raw],
        "otsu_threshold": int(t_cv),
        "otsu_threshold_numpy": int(t_np),
        "otsu_separability_eta": round(float(separability), 4),
        "hot_pixel_fraction": round(hot_fraction, 4),
        "mean_intensity": round(float(gray.mean()), 2),
        "rms_contrast": round(float(gray.std()), 2),
        "entropy_bits": round(entropy_bits, 3),
        "sobel_mean_magnitude": round(float(magnitude.mean()), 2),
        "sobel_p95_magnitude": round(float(np.percentile(magnitude, 95)), 2),
        "edge_density": round(edge_density, 4),
        "clahe": {"clip_limit": CLAHE_CLIP_LIMIT, "tile_grid": list(CLAHE_TILE_GRID)},
        "gaussian_kernel": [5, 5],
        "sobel_kernel": 3,
    }


def measure_thermal_signature(
    np_gray: np.ndarray,
    bbox_pixels: Dict[str, int],
    span_celsius: float = RADIOMETRIC_SPAN_C
) -> Optional[Dict[str, Any]]:
    """
    Measures the thermal signature of a detection directly from pixel intensities.

    The healthy reference is the median of the *module surface* in an annulus around
    the box: the annulus is segmented with a local Otsu threshold so that cold ground,
    inter-row gaps and frames do not bias the reference (IEC TS 62446-3 defines ΔT
    against a healthy reference area on comparable modules).

    Returns None when the box is degenerate.
    """
    gray = _to_gray(np_gray)
    h, w = gray.shape[:2]
    x1, y1 = max(0, bbox_pixels["x1"]), max(0, bbox_pixels["y1"])
    x2, y2 = min(w, bbox_pixels["x2"]), min(h, bbox_pixels["y2"])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    roi = gray[y1:y2, x1:x2].astype(np.float32)
    pad = max(6, int(0.75 * max(x2 - x1, y2 - y1)))
    ex1, ey1 = max(0, x1 - pad), max(0, y1 - pad)
    ex2, ey2 = min(w, x2 + pad), min(h, y2 + pad)
    window = gray[ey1:ey2, ex1:ex2]

    ring_mask = np.ones(window.shape, dtype=bool)
    ring_mask[y1 - ey1:y2 - ey1, x1 - ex1:x2 - ex1] = False
    ring = window[ring_mask].astype(np.float32)
    if ring.size < 10:
        ring = window.astype(np.float32).ravel()

    local_t, _ = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ring_surface = ring[ring > local_t]
    reference = ring_surface if ring_surface.size >= 20 else ring

    background_dl = float(np.median(reference))
    background_std = float(reference.std())
    roi_peak_dl = float(np.percentile(roi, 98))
    roi_mean_dl = float(roi.mean())
    contrast_dl = roi_peak_dl - background_dl
    hot_cut = background_dl + max(2.0 * background_std, 4.0)
    hot_area_pct = float(np.mean(roi > hot_cut) * 100.0)
    cnr = contrast_dl / background_std if background_std > 1e-6 else 0.0
    delta_t_image = max(0.0, contrast_dl) * span_celsius / 255.0

    return {
        "roi_peak_dl": round(roi_peak_dl, 1),
        "roi_mean_dl": round(roi_mean_dl, 1),
        "reference_dl": round(background_dl, 1),
        "contrast_dl": round(contrast_dl, 1),
        "contrast_to_noise": round(float(cnr), 2),
        "hot_area_pct": round(hot_area_pct, 1),
        "delta_t_image_celsius": round(delta_t_image, 1),
        "assumed_span_celsius": span_celsius,
        "local_otsu_threshold": int(local_t),
    }


def enrich_detection(
    det_id: int,
    class_id: int,
    confidence: float,
    xyxyn: List[float],
    img_width: int,
    img_height: int,
    module_watts: float = DEFAULT_MODULE_WATTS,
    modules_per_string: int = DEFAULT_MODULES_PER_STRING
) -> Dict[str, Any]:
    """Calculates normalized center points, pixel bounding boxes, and IEC TS 62446-3 metrics."""
    knowledge = FAULT_KNOWLEDGE_BASE.get(class_id)
    if knowledge is None:
        knowledge = dict(_UNKNOWN_FAULT, code=f"UNKNOWN_{class_id}", name=f"Defect Class {class_id}")

    xmin_n, ymin_n, xmax_n, ymax_n = [min(1.0, max(0.0, float(v))) for v in xyxyn]

    x_pct = round(((xmin_n + xmax_n) / 2.0) * 100, 2)
    y_pct = round(((ymin_n + ymax_n) / 2.0) * 100, 2)
    width_pct = round((xmax_n - xmin_n) * 100, 2)
    height_pct = round((ymax_n - ymin_n) * 100, 2)

    x1_px = max(0, int(round(xmin_n * img_width)))
    y1_px = max(0, int(round(ymin_n * img_height)))
    x2_px = min(img_width, int(round(xmax_n * img_width)))
    y2_px = min(img_height, int(round(ymax_n * img_height)))

    # Class-conditional ΔT prior, interpolated inside the physical range by confidence.
    dt_low, dt_high = knowledge["delta_t_range"]
    estimated_delta_t = round(dt_low + (dt_high - dt_low) * (confidence * 0.8 + 0.1), 1)

    loss_fraction = knowledge["nominal_power_loss_pct"] / 100.0
    affected_modules = modules_per_string if knowledge.get("loss_scope") == "string" else 1
    estimated_watt_loss = round(module_watts * loss_fraction * affected_modules, 1)

    return {
        "id": f"DEF-{det_id:03d}",
        "index": det_id,
        "class_id": class_id,
        "class_name": knowledge["name"],
        "code": knowledge["code"],
        "confidence": round(confidence * 100, 1),
        "confidence_raw": float(confidence),
        "iec_class": knowledge["iec_class"],
        "iec_label": knowledge["iec_label"],
        "severity_rank": SEVERITY_RANK.get(knowledge["iec_class"], 3),
        "color_hex": knowledge["color_hex"],
        "badge_bg": knowledge["badge_bg"],
        "badge_border": knowledge["badge_border"],
        "pin_position": {
            "x_pct": x_pct,
            "y_pct": y_pct,
            "width_pct": width_pct,
            "height_pct": height_pct
        },
        "bbox_pixels": {
            "x1": x1_px,
            "y1": y1_px,
            "x2": x2_px,
            "y2": y2_px
        },
        "bbox_normalized": [round(xmin_n, 5), round(ymin_n, 5), round(xmax_n, 5), round(ymax_n, 5)],
        "delta_t_celsius": estimated_delta_t,
        "delta_t_range": list(knowledge["delta_t_range"]),
        "power_loss_watts": estimated_watt_loss,
        "power_loss_pct": knowledge["nominal_power_loss_pct"],
        "loss_scope": knowledge.get("loss_scope", "module"),
        "affected_modules": affected_modules,
        "description": knowledge["description"],
        "field_action": knowledge["field_action"]
    }


def calculate_site_kpis(
    detections: List[Dict[str, Any]],
    module_watts: float = DEFAULT_MODULE_WATTS,
    tariff_usd: float = DEFAULT_TARIFF_USD_PER_KWH,
    sun_hours: float = DEFAULT_PEAK_SUN_HOURS
) -> Dict[str, Any]:
    """Aggregates detections into plant-level executive health and financial metrics."""
    assumptions = {
        "module_watts": module_watts,
        "tariff_usd_per_kwh": tariff_usd,
        "peak_sun_hours": sun_hours,
    }
    total_defects = len(detections)
    if total_defects == 0:
        return {
            "total_defects": 0,
            "critical_count": 0,
            "medium_count": 0,
            "minor_count": 0,
            "health_score": 100.0,
            "health_status": "Optimal",
            "total_watt_loss": 0.0,
            "total_kw_loss": 0.0,
            "daily_kwh_loss": 0.0,
            "annual_kwh_loss": 0.0,
            "annual_financial_loss_usd": 0.0,
            "primary_concern": "None",
            "assumptions": assumptions,
        }

    critical_count = sum(1 for d in detections if d["iec_class"] == "Class 3")
    medium_count = sum(1 for d in detections if d["iec_class"] == "Class 2")
    minor_count = sum(1 for d in detections if d["iec_class"] == "Class 1")

    total_watt_loss = sum(d["power_loss_watts"] for d in detections)
    daily_kwh_loss = (total_watt_loss / 1000.0) * sun_hours
    annual_kwh_loss = daily_kwh_loss * 365.0
    annual_financial_loss = round(annual_kwh_loss * tariff_usd, 2)

    penalty = sum(SEVERITY_PENALTY.get(d["iec_class"], 2.0) for d in detections)
    health_score = max(20.0, round(100.0 - penalty, 1))

    if health_score >= 90:
        health_status = "Optimal"
    elif health_score >= 70:
        health_status = "Attention Required"
    elif health_score >= 50:
        health_status = "Action Needed"
    else:
        health_status = "Critical Intervention"

    fault_counts: Dict[str, int] = {}
    for d in detections:
        fault_counts[d["class_name"]] = fault_counts.get(d["class_name"], 0) + 1
    primary_concern = max(fault_counts, key=fault_counts.get) if fault_counts else "None"

    return {
        "total_defects": total_defects,
        "critical_count": critical_count,
        "medium_count": medium_count,
        "minor_count": minor_count,
        "health_score": health_score,
        "health_status": health_status,
        "total_watt_loss": round(total_watt_loss, 1),
        "total_kw_loss": round(total_watt_loss / 1000.0, 3),
        "daily_kwh_loss": round(daily_kwh_loss, 2),
        "annual_kwh_loss": round(annual_kwh_loss, 1),
        "annual_financial_loss_usd": annual_financial_loss,
        "primary_concern": primary_concern,
        "fault_breakdown": fault_counts,
        "assumptions": assumptions,
    }


def render_annotated_detections(np_rgb: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    Draws IEC-colour-coded bounding boxes and '#n Class conf%' labels onto an RGB frame.

    Used for the API's annotated image and the legacy /predict/ endpoint so that the
    rendered evidence matches the severity scheme shown in the cockpit.
    """
    canvas = np_rgb.copy() if np_rgb.ndim == 3 else cv2.cvtColor(np_rgb, cv2.COLOR_GRAY2RGB)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.35, min(canvas.shape[0], canvas.shape[1]) / 1600.0)
    thickness = 1
    for det in detections:
        box = det["bbox_pixels"]
        color = _hex_to_rgb(det["color_hex"])
        cv2.rectangle(canvas, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2, cv2.LINE_AA)
        label = f"#{det.get('index', '?')} {det['class_name'].split(' ')[0]} {det['confidence']:.0f}%"
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
        ty = box["y1"] - 4 if box["y1"] - th - 6 > 0 else box["y2"] + th + 6
        cv2.rectangle(canvas, (box["x1"], ty - th - 3), (box["x1"] + tw + 6, ty + baseline - 1), color, -1)
        cv2.putText(canvas, label, (box["x1"] + 3, ty - 1), font, scale, (11, 11, 11), thickness, cv2.LINE_AA)
    return canvas


def knowledge_base_as_json() -> Dict[str, Dict[str, Any]]:
    """Returns the fault taxonomy with JSON-safe values (tuples converted to lists)."""
    out: Dict[str, Dict[str, Any]] = {}
    for class_id, entry in FAULT_KNOWLEDGE_BASE.items():
        item = dict(entry)
        item["delta_t_range"] = list(entry["delta_t_range"])
        out[str(class_id)] = item
    return out
