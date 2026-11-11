# HELIOS-AI — Presentation Speaker Notes

## Slide 1: Project Introduction

### What to Say
"Good morning/afternoon. Today we're presenting HELIOS-AI — an autonomous aerial thermography and photovoltaic anomaly intelligence system. Our system takes a single thermal image captured by a drone flying over a solar farm and transforms it into an auditable, quantified maintenance decision — detecting defects, grading their severity to international standards, and estimating the financial impact."

### Key Points to Emphasize
- The project name HELIOS-AI refers to the Greek sun god — fitting for solar energy
- This is NOT just a YOLO wrapper — the detector is one of six pipeline stages
- We use both deep learning AND classical image processing (IPCV)
- The system follows IEC TS 62446-3 standards for PV thermography
- Full pipeline runs in ~0.42 seconds on CPU

### Potential Questions
**Q: What does HELIOS-AI stand for?**
A: It's named after Helios, the Greek god of the sun. The "AI" reflects the combination of deep learning detection with classical computer vision intelligence.

**Q: How is this different from just running YOLO on thermal images?**
A: YOLO detection is only Stage 2 of our 6-stage pipeline. We add classical IPCV (CLAHE, Sobel, Otsu), module geometry with homography rectification, IEC TS 62446-3 severity grading, thermal physics analysis, and yield-loss economics.

---

## Slide 2: Problem Statement & Motivation

### What to Say
"Solar farms lose energy and revenue from faults that are invisible in normal photographs. Failed bypass diodes, cracked cells, reversed polarity — these show up only in infrared thermography. A single drone flight can produce thousands of thermal frames, and today these are interpreted manually by expert thermographers. This doesn't scale.

Our sample frame alone shows 9 defects that are dissipating 2.16 kilowatts — that's $425 per year in lost revenue from just one image. In a utility-scale plant, this multiplies enormously."

### How to Explain the Impact Cards
Point to each statistic: "$425/yr from 9 defects on one frame", "2.16 kW dissipated", "9 distinct fault classes", "under 0.5 seconds per frame". These are all real numbers from our system.

### Potential Questions
**Q: Why not just use visible-spectrum cameras?**
A: Most PV faults — bypass diode failures, cell-level hotspots, reversed polarity — have no visible signature. They only show up as temperature anomalies in longwave infrared (LWIR) thermography.

**Q: What's IEC TS 62446-3?**
A: It's the international standard (published 2017) that defines how outdoor infrared thermography should be conducted on PV plants — including how to classify defects by severity.

---

## Slide 3: Our Proposed Solution

### What to Say
"Our solution follows a clear Input → Processing → Output flow. The input is a single thermal image — JPEG, PNG, or even 16-bit radiometric TIFF. The processing involves classical IPCV analysis, deep learning detection, geometric rectification, IEC grading, and economic analysis. The output includes 6 visualization layers, an enriched defect register, plant-level KPIs, an interactive cockpit, and a printable audit certificate."

### Key Point About the Differentiator
Read the bottom banner: "HELIOS-AI is not a model wrapper. The detector is one of six pipeline stages." This is critical because evaluators often assume projects like this just wrap a pre-trained model.

### Potential Questions
**Q: Why do you call it 'not a model wrapper'?**
A: The YOLOv9 detector provides bounding boxes and class labels. But our system adds: (1) classical IPCV layers that explain what the model sees, (2) module geometry with homography, (3) IEC severity grading from the international standard, (4) thermal contrast measurement from the actual pixels, (5) power and revenue loss estimation, and (6) interactive visualization and reporting.

**Q: What formats do you support for input?**
A: JPEG, PNG, BMP, WebP, HEIC, and 16-bit/32-bit TIFF. For radiometric TIFFs, we use a robust 0.5-99.5 percentile window to stretch to 8-bit without clipping.

---

## Slide 4: System Architecture

### What to Say
"Here's our complete system architecture. At the top is the browser-based cockpit — it includes the main viewer, a guided tour engine, an IPCV lab for interactive analysis, and audit certificate generation.

The middle layer is our FastAPI service in api.py — it handles routing, input validation, and thread pool management.

The core layer has four Python modules: predictor.py orchestrates the pipeline, solar_iec_analyzer.py handles IPCV filters and IEC grading, ipcv_geometry.py does contour detection and homography, and model_loader.py manages the YOLOv9 model with thread-safe locking.

At the bottom, the 7-stage pipeline shows the data flow: from ingestion through classical IPCV, YOLOv9 detection, geometry, IEC enrichment, economics, to evidence packaging."

### How to Explain the Diagram
- Point to each layer: Browser → FastAPI → Core → Model
- Follow the data flow arrows downward
- The bottom pipeline row shows all 7 stages in sequence
- Note that Stages 1, 2, and 3 can run in parallel after Stage 0

### Potential Questions
**Q: Why do you use class-agnostic NMS?**
A: With per-class NMS, the model sometimes emits overlapping boxes with different labels on the same module — for example, SingleDiode at 0.54 and MultiHotSpot at 0.42 at identical coordinates. This double-counts power loss. Class-agnostic NMS keeps only the highest-confidence box at each location.

**Q: Why a thread pool?**
A: FastAPI uses async/await. The YOLOv9 inference is CPU-intensive and blocking. We run it in a thread pool so the event loop stays responsive. A process-wide lock prevents concurrent model access since Ultralytics predictors maintain internal state.

---

## Slide 5: Core Detection Methodology

### What to Say
"The left side shows our detection pipeline as four stages. First, we preprocess: aspect-true resize to 640px on the long side, EXIF transpose, and 16-bit stretch. Then YOLOv9 runs the forward pass — 25.5 million parameters, trained on aerial PV thermal data. Third, class-agnostic NMS at IoU 0.7 prevents double-counting. Finally, each detection is enriched with IEC severity, ΔT estimation, and power loss.

The right side shows all 9 fault classes our model detects, organized by IEC severity. Four are Class 3 (Critical — must address within 24 hours), four are Class 2 (Medium — scheduled within 30 days), and one is Class 1 (Low — monitor at next survey)."

### How to Explain the Fault Table
- String-level faults (StringReversedPolarity, StringOpenCircuit) cause 100% loss of the entire string
- Module-level faults affect individual modules
- The nominal loss percentage comes from the IEC standard and PV engineering literature

### Potential Questions
**Q: Did you train the YOLOv9 model?**
A: No. We use a pre-trained YOLOv9 checkpoint (Th_G_v9.pt, 51.6 MB) trained on aerial PV thermal data from the Kaggle Solar Panel dataset. We use it exactly as delivered — no re-training, no fine-tuning. Our contribution is everything around the detector.

**Q: What's the confidence threshold?**
A: The detection floor is 0.20 — we keep all candidates above this so the cockpit slider can adjust the display threshold without re-running inference. The default reporting threshold is 0.45, which determines what counts toward KPIs and the certificate.

**Q: Why 640px?**
A: YOLOv9 was trained at 640×640. We scale the long side to 640 and preserve the aspect ratio — YOLO letterboxes internally. Stretching a 640×512 frame to a square would distort module geometry and detection pin positions.

---

## Slide 6: Classical IPCV & Thermal Analysis

### What to Say
"This slide shows the foundational computer vision that makes our project more than a model wrapper. We implement three classical IPCV techniques:

1. CLAHE with Ironbow: Adaptive histogram equalization that reveals subtle temperature gradients, mapped to the standard Ironbow palette that thermographers use.

2. Sobel gradient magnitude: Highlights busbars, cell boundaries, and thermal transitions. The formula is |∇I| = √(Gx² + Gy²) using 3×3 Sobel kernels.

3. Otsu binarisation: We not only use OpenCV's Otsu but independently re-derive the threshold from first principles in NumPy — both give t* = 103 on our sample frame, with separability η = 0.82.

Below that, thermal signature measurement uses two independent ΔT figures. And module geometry uses least-squares edge refinement to achieve 0.48 pixel corner accuracy."

### Statistics to Know
- Otsu threshold: t* = 103 (verified by both OpenCV and NumPy)
- Separability η = 0.82 (high bimodality — modules and background separate well)
- Shannon entropy: 7.53 bits
- Hot pixel fraction: 63.7% (above Otsu threshold)
- Mean corner error: 0.48 px with LSQ refinement (vs 2.82 px with approxPolyDP alone)

### Potential Questions
**Q: Why implement Otsu twice (OpenCV and NumPy)?**
A: To demonstrate mathematical understanding. The OpenCV result is efficient but opaque. Our NumPy re-derivation computes the between-class variance σ²B(t) for every threshold and finds the argmax — showing we understand the algorithm, not just the API call. Both give identical results.

**Q: What is separability η?**
A: η = σ²B(t*) / σ²T — the ratio of between-class variance at the optimal threshold to total variance. It ranges from 0 to 1. η = 0.82 means the Otsu threshold cleanly separates the two populations (hot module surface vs. cool background).

**Q: How does the thermal signature measurement work?**
A: For each detection box, we create an analysis window padded by 75% of the box size. An annulus (window minus box) provides the healthy reference. We apply local Otsu to segment only module-surface pixels in the annulus. Then contrast = P98(box pixels) - median(reference). This follows IEC TS 62446-3 methodology.

---

## Slide 7: Implementation & Practical Demonstration

### What to Say
"Here you can see our actual running system. The cockpit [point to screenshot] shows the Ironbow thermal layer with severity-coded pins — red for Class 3 critical, amber for Class 2 medium. The KPI ribbon at the top shows health score, defect count, dissipated watts, and annual revenue loss.

Below is the IPCV Lab — an interactive histogram showing the Otsu threshold, the homography matrix, pipeline telemetry, and economics calculator.

On the right are our four main features: the interactive cockpit with 6 layers, the guided tour mode for hands-free presentations, the printable audit certificate, and data export capabilities."

### Live Demo Tips
If doing a live demo:
1. Open `http://localhost:8000/?sample=1` — instant analysis
2. Press keys 1-6 to switch layers
3. Click a red pin to show the telemetry drawer
4. Press T for guided tour
5. Try `/?sample=1&tour=1` for auto-start

### Potential Questions
**Q: Is the cockpit a separate application?**
A: No, it's a single HTML file (static/index.html) served by the FastAPI server. It uses Tailwind CSS, Lucide icons, vanilla JavaScript, and HTML5 Canvas — no heavy frameworks. All data comes from the API response as JSON with base64-encoded images.

**Q: How does the guided tour work?**
A: It's a requestAnimationFrame-driven engine. It walks through the IPCV layers first (Ironbow, Sobel, Otsu, Contours), then through each defect sorted by severity. For each defect, it smoothly zooms in, locks the reticle, and shows telemetry. The presenter can pause, resume, skip forward/back, and adjust speed.

---

## Slide 8: Evaluation & Results

### What to Say
"Our benchmark suite runs 20 timed iterations after 3 warm-ups. The forward pass dominates at 336 ms — that's 98% of detection latency. The classical IPCV layers add only 7 ms. Full end-to-end is about 420 ms or 2.38 FPS on CPU.

The robustness analysis tests prediction consistency under 8 perturbations. The model is invariant to brightness, contrast, gamma, and horizontal flip — all get F1 = 1.00. But it's sensitive to noise and blur: Gaussian noise σ=8 drops recall to 0.22. Importantly, precision stays 1.00 in all cases — no false alarms.

On our sample frame: 9 defects detected, 8 critical, 1 medium. Mean confidence 78.8%. The results are deterministic — identical boxes across all 20 iterations."

### Key Numbers
- Forward pass: 336.4 ms mean, 369.6 ms p95
- Full pipeline: 419.9 ms (including geometry, crops, encoding)
- 2.93 FPS detection, 2.38 FPS end-to-end
- 25.5M parameters, 103.7 GFLOPs at 640²
- Mean robustness F1: 0.85
- Sample frame: 9 defects, 2,160 W dissipated, $425.74/yr loss

### Potential Questions
**Q: Why is robustness measured without ground-truth labels?**
A: We don't have pixel-level ground truth for our sample frame. Instead, we measure prediction consistency: detections on perturbed images are matched to clean predictions by IoU ≥ 0.5. This measures stability, not accuracy. If the model gives the same detections under photometric changes, it's robust.

**Q: Why does noise hurt so much?**
A: Thermal sensors in small drones have limited SNR. Gaussian noise at σ=8 DL fills in the subtle temperature gradients the model relies on. Our robustness finding suggests that a denoising or focus-quality gate before inference would improve field reliability.

**Q: Is the model deterministic?**
A: Yes — verified across all 20 benchmark iterations. Same input → same boxes, classes, and confidences.

---

## Slide 9: Key Contributions, Limitations & Future Scope

### What to Say
"Let me summarize our contributions. First, we built a multi-layered system — not a YOLO wrapper. Six pipeline stages, each producing verifiable outputs. Second, we demonstrated classical IPCV rigor with mathematical verification. Third, we aligned with IEC TS 62446-3 standards. Fourth, we implemented production engineering: thread-safe inference, Docker deployment, input validation.

Our limitations are honest: temperatures are relative (8-bit AGC images give approximate ΔT), analysis is per-frame (no multi-image de-duplication), and detection quality is bounded by the pre-trained weights.

For future work: radiometric TIFF decoding for absolute temperatures, GPS integration for plant-wide mapping, ONNX/TensorRT export for GPU-accelerated inference."

### Potential Questions
**Q: If you didn't train the model, what is your contribution?**
A: The model gives 9 bounding boxes. Our contribution is everything else: (1) classical IPCV analysis that makes the system explainable, (2) IEC-standard severity grading, (3) thermal physics with two independent ΔT measurements, (4) sub-pixel homography for geometric accuracy, (5) economic yield-loss estimation, (6) a production-grade API, and (7) an interactive cockpit with guided tour and audit reporting.

**Q: Why are temperatures relative?**
A: Consumer thermal cameras (and most drone payloads) export AGC-scaled 8-bit images where pixel values represent relative intensity, not calibrated temperature. Our "image ΔT" assumes a linear 40°C span — this is stated on the certificate. For compliance-grade temperatures, you need radiometric data with calibration coefficients.

---

## Slide 10: Conclusion & Q&A

### What to Say
"To summarize: HELIOS-AI takes a single drone-captured thermal image and delivers an auditable maintenance decision. It detects 9 types of PV faults, enriches each with IEC severity and thermal physics, estimates financial impact, and presents everything through an interactive cockpit with a guided tour mode and printable audit certificate.

The full pipeline runs in under half a second on a standard CPU. Every result is deterministic and benchmarked.

Thank you for your time. We're happy to take any questions."

---

# Comprehensive Q&A Preparation

## System Architecture Questions

**Q1: Why did you choose FastAPI over Flask/Django?**
A: FastAPI provides async support (critical for non-blocking upload handling), automatic OpenAPI documentation at /docs, built-in request validation with Pydantic, and a thread pool for offloading CPU-intensive inference.

**Q2: Why is inference run in a thread pool instead of async?**
A: YOLOv9 inference is CPU-bound — it uses PyTorch's CPU threads internally. Running it directly in the async event loop would block all other requests. run_in_threadpool moves it to a background thread while the event loop remains responsive.

**Q3: What is the model_lock for?**
A: Ultralytics YOLO predictors maintain internal state between calls (pre-allocated buffers, previous results). If two threads call predict() simultaneously, they corrupt each other's state. The threading.Lock ensures mutual exclusion.

**Q4: How is the cockpit architecture organized?**
A: It's a single-page application in static/index.html. The server sends JSON with base64-encoded images; the client renders them on HTML5 Canvas with JavaScript. All filtering (confidence threshold, severity) happens client-side — no additional API calls needed.

## Model Selection Questions

**Q5: Why YOLOv9 specifically?**
A: YOLOv9 (GELAN architecture) provides strong detection performance at moderate compute cost. The pre-trained checkpoint was trained on aerial PV thermal data. At 25.5M parameters and 103.7 GFLOPs, it's practical for CPU inference.

**Q6: What are the 9 fault classes?**
A: MultiByPassed, MultiDiode, MultiHotSpot, OpenPV, SingleByPassed, SingleDiode, SingleHotSpot, StringOpenCircuit, StringReversedPolarity. They cover module-level and string-level faults at three severity tiers per IEC TS 62446-3.

**Q7: What dataset was the model trained on?**
A: The model was trained on the Kaggle Solar Panel thermal dataset (Kaggle: pkdarabi/solarpanel). We use the checkpoint as-is — no re-training.

## Implementation Questions

**Q8: How do you handle 16-bit radiometric TIFFs?**
A: PIL opens them in mode I;16 or I. We convert to float64, compute the 0.5th and 99.5th percentiles, and linearly stretch to 0-255 uint8. This robust windowing prevents outlier pixels from washing out the image.

**Q9: What does class-agnostic NMS mean?**
A: Standard NMS applies separately per class. Class-agnostic NMS treats all classes equally when suppressing overlapping boxes. If a MultiDiode box (conf 0.54) and a MultiHotSpot box (conf 0.42) overlap at IoU ≥ 0.7, only the higher-confidence one survives. This prevents double-counting power loss for the same physical defect.

**Q10: How does the warm-up work?**
A: At server startup (FastAPI lifespan), we run one dummy inference on a synthetic 512×640 blank image. This initializes all PyTorch buffers and JIT compilation, cutting the first real request from ~2.5s to ~0.45s.

## Thermal Analysis Questions

**Q11: What's the difference between estimated ΔT and image-measured ΔT?**
A: Estimated ΔT is a class-conditional prior from PV engineering literature, interpolated by detection confidence: ΔT_est = ΔT_lo + (ΔT_hi - ΔT_lo) × (0.8c + 0.1). Image-measured ΔT comes from actual pixel analysis: P98(box pixels) minus median(Otsu-segmented reference annulus), scaled by the assumed radiometric span.

**Q12: Why use an annulus as the healthy reference?**
A: The IEC standard defines ΔT against a healthy reference area. If we used the entire image, cold ground and sky would inflate the contrast artificially. The annulus (analysis window minus detection box) samples only nearby module surface, which we further filter by local Otsu to exclude non-module pixels.

**Q13: What is the radiometric span assumption?**
A: We assume the 0-255 pixel range maps linearly to a 40°C span. This is printed on the certificate as a stated assumption. For absolute temperatures, you'd need the camera's radiometric calibration data.

## Evaluation Metrics Questions

**Q14: How is robustness measured without labels?**
A: We compare perturbed predictions to the clean prediction. Detections are matched by IoU ≥ 0.5. Recall = matched/clean_total. Precision = matched/perturbed_total. This measures prediction stability, not absolute accuracy.

**Q15: Why is precision always 1.00?**
A: All perturbation-induced detections match a clean detection. The model never hallucinates new detections under perturbation — it only misses existing ones. This means no false alarms, which is reassuring for field deployment.

**Q16: What's the mean F1 of 0.85?**
A: Averaged across 8 perturbation types. The model is perfectly stable under photometric transforms (brightness, contrast, gamma, flip — all F1 = 1.00) but degrades under spatial corruption (blur F1 = 0.61, noise F1 = 0.36). The practical implication: sensor quality matters.

## Limitations Questions

**Q17: Can you use this for plant-wide monitoring?**
A: Currently, each frame is analyzed independently. For plant-wide monitoring, you'd need GPS tagging, frame overlap detection, and defect de-duplication across images — that's in our roadmap.

**Q18: Why not fine-tune the model?**
A: Our project scope focuses on the complete inspection pipeline, not model training. The pre-trained model detects 9 relevant fault classes with good confidence. Our contribution is the engineering around it — IPCV analysis, IEC grading, thermal physics, cockpit, and reporting.

**Q19: Is this production-ready?**
A: For single-frame analysis, yes — it has input validation, size limits, decompression bomb guards, thread-safe inference, Docker deployment, and a warm-up cycle. For production PV plant monitoring, you'd add authentication, geo-referencing, and batch processing.

## Possible Improvements Questions

**Q20: How would GPU inference help?**
A: The forward pass (336 ms) is 98% of detection latency. On a mid-range GPU, this drops to ~15-30 ms, enabling real-time analysis of flight video at 10-20 FPS. ONNX or TensorRT export would further optimize.

