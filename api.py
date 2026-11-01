# api.py
"""
HELIOS-AI FastAPI service.

Endpoints
---------
GET  /                         Interactive web cockpit (static/index.html)
GET  /api/v1/health            Model, device and runtime telemetry
GET  /api/v1/taxonomy          IEC TS 62446-3 fault knowledge base
GET  /api/v1/sample            1-click inspection of the bundled flight frame
POST /api/v1/inspect           Full inspection of an uploaded thermal image
POST /api/v1/rectify           Homography rectification of module tables
GET  /api/v1/benchmark         Latest evaluate_benchmark.py metrics (JSON)
GET  /api/v1/benchmark/report  Latest benchmark performance card (PNG)
POST /predict/                 Legacy endpoint (annotated JPEG + count headers)
"""

import json
import os
import platform
import time
from contextlib import asynccontextmanager
from io import BytesIO

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

import ultralytics
from ipcv_geometry import analyze_panel_geometry
from model_loader import WEIGHTS_PATH, model
from predictor import (
    DETECTION_FLOOR,
    _encode_data_uri,
    inspect_thermal_image,
    preprocess_image,
    run_inference,
    warmup,
)
from solar_iec_analyzer import (
    DEFAULT_MODULE_WATTS,
    DEFAULT_MODULES_PER_STRING,
    DEFAULT_PEAK_SUN_HOURS,
    DEFAULT_TARIFF_USD_PER_KWH,
    knowledge_base_as_json,
)

API_VERSION = "2.1.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(BASE_DIR, "Results")
SAMPLE_PATH = os.path.join(RESULTS_DIR, "INPUT.jpg")
BENCHMARK_JSON = os.path.join(RESULTS_DIR, "benchmark_metrics.json")
BENCHMARK_PNG = os.path.join(RESULTS_DIR, "BENCHMARK_REPORT.png")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
STARTED_AT = time.time()

Image.MAX_IMAGE_PIXELS = 80_000_000  # guards against decompression bombs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warms the detector up at start-up so the first inspection is not penalised."""
    try:
        await run_in_threadpool(warmup, 1)
    except Exception as exc:  # a failed warm-up must not prevent the service from starting
        print(f"[helios] model warm-up skipped: {exc}")
    yield


app = FastAPI(
    title="HELIOS-AI: Solar Panel Defect Detection & IEC 62446-3 Diagnostics",
    description="Automated Aerial Thermography AI pipeline for photovoltaic defect detection, "
                "classical IPCV analysis, severity grading, and yield loss estimation.",
    version=API_VERSION,
    lifespan=lifespan,
)

# Enable CORS for flexible integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory if it exists
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _validate_parameters(conf_threshold: float, module_watts: float, tariff_usd: float,
                         sun_hours: float, modules_per_string: int) -> None:
    """Raises HTTP 422 for out-of-range analysis parameters."""
    problems = []
    if not 0.05 <= conf_threshold <= 0.95:
        problems.append("conf_threshold must be between 0.05 and 0.95")
    if not 10.0 <= module_watts <= 1000.0:
        problems.append("module_watts must be between 10 and 1000 W")
    if not 0.0 <= tariff_usd <= 5.0:
        problems.append("tariff_usd must be between 0 and 5 $/kWh")
    if not 0.5 <= sun_hours <= 12.0:
        problems.append("sun_hours must be between 0.5 and 12 h/day")
    if not 1 <= modules_per_string <= 60:
        problems.append("modules_per_string must be between 1 and 60")
    if problems:
        raise HTTPException(status_code=422, detail="; ".join(problems))


async def _read_upload_image(file: UploadFile) -> Image.Image:
    """Reads and decodes an uploaded image with size and format validation."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    try:
        img = Image.open(BytesIO(contents))
        img.load()
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=415, detail="Unsupported or corrupt image file: format not recognised.") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail=f"Image exceeds the {Image.MAX_IMAGE_PIXELS:,} pixel limit.") from exc
    except Exception as exc:  # truncated files, missing optional decoders (e.g. HEIF), ...
        raise HTTPException(status_code=415, detail=f"Image could not be decoded ({type(exc).__name__}).") from exc
    return img


@app.get("/", summary="Interactive Web Cockpit", include_in_schema=False)
def serve_ui():
    """Serves the interactive web cockpit for high-level evaluation and presentation."""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": "HELIOS-AI Solar Panel Defect Detection API is active.",
        "docs_url": "/docs"
    }


@app.get("/api/v1/sample", summary="Sample Drone Flight Analysis")
def inspect_sample_flight(
    conf_threshold: float = Query(0.45, description="Confidence counted towards KPIs"),
    module_watts: float = Query(DEFAULT_MODULE_WATTS, description="Nominal panel peak wattage (W)"),
    tariff_usd: float = Query(DEFAULT_TARIFF_USD_PER_KWH, description="Feed-in tariff ($/kWh)"),
    sun_hours: float = Query(DEFAULT_PEAK_SUN_HOURS, description="Peak sun hours per day"),
    modules_per_string: int = Query(DEFAULT_MODULES_PER_STRING, description="Modules per series string"),
):
    """
    Executes real-time inference on the bundled sample thermal flight image (Results/INPUT.jpg).
    Allows zero-friction 1-click demonstration during evaluation and presentations.
    """
    _validate_parameters(conf_threshold, module_watts, tariff_usd, sun_hours, modules_per_string)
    if not os.path.exists(SAMPLE_PATH):
        raise HTTPException(status_code=404, detail="Sample image not found on disk.")

    try:
        with Image.open(SAMPLE_PATH) as img:
            img.load()
            result = inspect_thermal_image(
                img, conf_threshold=conf_threshold, include_crops=True, module_watts=module_watts,
                tariff_usd=tariff_usd, sun_hours=sun_hours, modules_per_string=modules_per_string,
                detection_floor=DETECTION_FLOOR)
        result["source"] = {"name": "INPUT.jpg", "kind": "sample"}
        return JSONResponse(content=result, status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")


@app.post("/api/v1/inspect", summary="Quantitative Inspection & Diagnostic Endpoint")
async def inspect_uploaded_image(
    file: UploadFile = File(..., description="Infrared / Thermal image file (JPEG, PNG, TIFF)"),
    conf_threshold: float = Form(0.45, description="YOLO detection confidence threshold (0.05 to 0.95)"),
    module_watts: float = Form(DEFAULT_MODULE_WATTS, description="Nominal panel peak wattage (W)"),
    tariff_usd: float = Form(DEFAULT_TARIFF_USD_PER_KWH, description="Electricity feed-in tariff ($/kWh)"),
    sun_hours: float = Form(DEFAULT_PEAK_SUN_HOURS, description="Peak sun hours per day"),
    modules_per_string: int = Form(DEFAULT_MODULES_PER_STRING, description="Modules per series string"),
):
    """
    Full Industry-Standard Inspection:
    - Runs YOLOv9 defect detector (class-agnostic NMS)
    - Generates classical IPCV layers (CLAHE/Ironbow, Sobel, Otsu) and statistics
    - Detects module tables and computes a rectifying homography
    - Translates faults into IEC TS 62446-3 Class 1-3 severity tiers
    - Calculates estimated thermal delta (ΔT), measured thermal contrast and power yield loss
    - Returns base64 crops for interactive region zooming and telemetry display
    """
    _validate_parameters(conf_threshold, module_watts, tariff_usd, sun_hours, modules_per_string)
    img = await _read_upload_image(file)
    try:
        result = await run_in_threadpool(
            inspect_thermal_image, img, conf_threshold=conf_threshold, include_crops=True,
            module_watts=module_watts, tariff_usd=tariff_usd, sun_hours=sun_hours,
            modules_per_string=modules_per_string, detection_floor=DETECTION_FLOOR)
        result["source"] = {"name": file.filename or "upload", "kind": "upload"}
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inspection failed: {str(e)}")


@app.post("/api/v1/rectify", summary="Homography Rectification of Module Tables")
async def rectify_uploaded_image(
    file: UploadFile = File(..., description="Thermal image containing PV module tables"),
    output_size: int = Form(640, description="Normalised square output size in pixels (128-1280)"),
):
    """
    Classical geometric CV: Otsu segmentation -> findContours -> approxPolyDP with
    least-squares edge refinement -> getPerspectiveTransform -> warpPerspective.
    """
    if not 128 <= output_size <= 1280:
        raise HTTPException(status_code=422, detail="output_size must be between 128 and 1280")
    img = await _read_upload_image(file)

    def _run():
        gray = (preprocess_image(img) * 255.0 + 0.5).astype(np.uint8)
        geo = analyze_panel_geometry(gray, output_size=(output_size, output_size))
        return {
            "status": "success",
            "image_dimensions": {"width": int(gray.shape[1]), "height": int(gray.shape[0])},
            "modules_detected": geo["modules_detected"],
            "quads": geo["quads"],
            "homography": geo["homography"],
            "diagnostics": geo["diagnostics"],
            "target_method": geo.get("target_method"),
            "overlay_base64": _encode_data_uri(geo["overlay_rgb"]),
            "mask_base64": _encode_data_uri(cv2.cvtColor(geo["mask"], cv2.COLOR_GRAY2RGB), fmt="PNG"),
            "rectified_base64": _encode_data_uri(geo["rectified_rgb"]) if geo["rectified_rgb"] is not None else None,
            "rectified_aspect_base64": (_encode_data_uri(geo["rectified_aspect_rgb"])
                                        if geo["rectified_aspect_rgb"] is not None else None),
        }

    try:
        return JSONResponse(content=await run_in_threadpool(_run), status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rectification failed: {str(e)}")


@app.get("/api/v1/health", summary="Model & System Health")
def health_check():
    """Returns runtime telemetry, GPU acceleration status, and class mappings."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU (Standard)"
    try:
        n_params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        n_params = None

    return {
        "status": "healthy",
        "version": API_VERSION,
        "engine": "YOLOv9-Thermal",
        "device": device_name,
        "cuda_active": cuda_available,
        "classes_count": len(model.names),
        "classes": model.names,
        "parameters": n_params,
        "weights": os.path.relpath(WEIGHTS_PATH, BASE_DIR).replace(os.sep, "/"),
        "weights_size_mb": round(os.path.getsize(WEIGHTS_PATH) / 1e6, 2),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "python_version": platform.python_version(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "benchmark_available": os.path.exists(BENCHMARK_JSON),
        "standard_compliance": "IEC TS 62446-3"
    }


@app.get("/api/v1/taxonomy", summary="IEC TS 62446-3 Fault Knowledge Base")
def fault_taxonomy():
    """Returns the defect taxonomy: IEC class, ΔT range, nominal loss, root cause and field action."""
    return {"standard": "IEC TS 62446-3:2017", "classes": knowledge_base_as_json()}


@app.get("/api/v1/benchmark", summary="Latest Benchmark Metrics")
def benchmark_metrics():
    """Returns the metrics written by evaluate_benchmark.py (run it first)."""
    if not os.path.exists(BENCHMARK_JSON):
        raise HTTPException(status_code=404, detail="No benchmark yet. Run: python evaluate_benchmark.py")
    try:
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Benchmark metrics unreadable: {e}")


@app.get("/api/v1/benchmark/report", summary="Latest Benchmark Performance Card (PNG)")
def benchmark_report():
    """Serves Results/BENCHMARK_REPORT.png produced by evaluate_benchmark.py."""
    if not os.path.exists(BENCHMARK_PNG):
        raise HTTPException(status_code=404, detail="No benchmark card yet. Run: python evaluate_benchmark.py")
    return FileResponse(BENCHMARK_PNG, media_type="image/png", headers={"Cache-Control": "no-cache"})


# Backward-Compatible Legacy Endpoint
@app.post("/predict/", response_class=StreamingResponse, summary="Legacy Predict Endpoint")
async def predict_image(file: UploadFile = File(...)):
    """Legacy endpoint returning JPEG bytes with anomaly count in HTTP headers."""
    try:
        img = await _read_upload_image(file)
        detections, msg, output_image = await run_in_threadpool(run_inference, img)

        if output_image is None:
            return JSONResponse(content={"status": msg, "detections": detections}, status_code=200)

        img_io = BytesIO()
        output_image.save(img_io, format='JPEG')
        img_io.seek(0)

        return StreamingResponse(
            img_io,
            media_type="image/jpeg",
            headers={
                "X-Anomaly-Count": str(len(detections)),
                "X-Detection-Result": msg
            }
        )
    except HTTPException as e:
        return JSONResponse(content={"error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
