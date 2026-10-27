# model_loader.py
"""
Single-instance loader for the pre-trained YOLOv9 thermal defect detector.

The weights path is resolved relative to this file (not the current working
directory), so the API, the benchmark suite and ad-hoc scripts all load the
same checkpoint no matter where they are launched from. Set the environment
variable ``HELIOS_WEIGHTS`` to point at an alternative checkpoint.

A process-wide lock is exported because Ultralytics predictors keep internal
state between calls; concurrent API requests must not interleave inside it.
"""

import os
import threading

# A production service must never pip-install packages at request time. Ultralytics
# otherwise auto-installs missing extras (e.g. pi-heif when PIL meets an unknown file).
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from ultralytics import YOLO  # noqa: E402  (must follow the environment setup above)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "ThermalDetector", "Th_G_v9.pt")
WEIGHTS_PATH = os.environ.get("HELIOS_WEIGHTS", DEFAULT_WEIGHTS_PATH)

if not os.path.isfile(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"YOLOv9 thermal weights not found at '{WEIGHTS_PATH}'. "
        "Expected the pre-trained checkpoint ThermalDetector/Th_G_v9.pt."
    )

model = YOLO(WEIGHTS_PATH)
model_lock = threading.Lock()
