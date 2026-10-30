# ipcv_geometry.py
"""
Classical geometric computer vision for aerial PV thermography.

Oblique drone captures foreshorten module tables into trapezoids, which skews pixel
areas, defect sizes and cell pitch. This module recovers a metric top-down view of a
module table with a planar homography - no learning involved:

    1. Segmentation   Gaussian blur -> Otsu threshold (valid pixels only) ->
                      morphological opening (cuts ground-clutter bridges between
                      tables) and closing (refills cell/busbar grid lines) give a
                      binary mask of module surfaces.
    2. Contours       cv2.findContours extracts each table outline; the convex hull is
                      simplified with cv2.approxPolyDP (epsilon swept until exactly four
                      vertices remain, minAreaRect as a fallback).
    3. Homography     The four ordered corners (TL, TR, BR, BL) and the target rectangle
                      define H via cv2.getPerspectiveTransform (direct linear transform
                      with 4 correspondences, 8 DoF).
    4. Rectification  cv2.warpPerspective resamples the table into a normalised
                      640 x 640 orthographic frame (and an aspect-true variant).

Run directly for an evaluator demonstration:

    python ipcv_geometry.py                       # sample image, synthetic 28 deg pitch
    python ipcv_geometry.py --image my_oblique.jpg --no-simulate

The demo synthesises an oblique capture from the nadir sample using a physically
modelled camera rotation H = K R K^-1, then recovers the top-down view and reports the
corner re-projection error against ground truth and the normalised cross-correlation
between the recovered and the true top-down module table.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "Results", "INPUT.jpg")
DEFAULT_FIGURE = os.path.join(PROJECT_ROOT, "Results", "HOMOGRAPHY_RECTIFICATION.png")
DEFAULT_RECTIFIED = os.path.join(PROJECT_ROOT, "Results", "RECTIFIED_PANEL.png")

CORNER_LABELS = ("TL", "TR", "BR", "BL")


@dataclass
class PanelQuad:
    """A module table outline approximated by an ordered quadrilateral."""

    corners: np.ndarray            # (4, 2) float32, ordered TL, TR, BR, BL
    contour_area: float            # pixel area enclosed by the raw contour
    quad_area: float               # pixel area of the fitted quadrilateral
    method: str                    # "approxPolyDP" or "minAreaRect"
    epsilon_frac: float            # approxPolyDP epsilon as a fraction of the perimeter
    contour: np.ndarray = field(repr=False, default=None)

    @property
    def rectangularity(self) -> float:
        """Contour area / quad area; 1.0 means the outline is a perfect quadrilateral."""
        return float(self.contour_area / self.quad_area) if self.quad_area > 0 else 0.0

    @property
    def centroid(self) -> np.ndarray:
        return self.corners.mean(axis=0)

    def edge_lengths(self) -> Tuple[float, float]:
        """Mean horizontal (top/bottom) and vertical (left/right) edge lengths in pixels."""
        tl, tr, br, bl = self.corners
        width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
        height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
        return float(width), float(height)

    def to_dict(self, img_w: int, img_h: int) -> Dict[str, object]:
        width, height = self.edge_lengths()
        # "+ 0.0" normalises IEEE negative zero (-0.0) produced by the line intersections.
        return {
            "corners_px": [[round(float(x), 1) + 0.0, round(float(y), 1) + 0.0] for x, y in self.corners],
            "corners_pct": [[round(float(x) / img_w * 100, 2) + 0.0, round(float(y) / img_h * 100, 2) + 0.0]
                            for x, y in self.corners],
            "area_px": round(self.contour_area, 1),
            "rectangularity": round(self.rectangularity, 3),
            "aspect_ratio": round(width / height, 3) if height > 0 else None,
            "method": self.method,
        }


@dataclass
class RectificationResult:
    """Output of a homography rectification."""

    warped: np.ndarray             # rectified image
    homography: np.ndarray         # 3x3 matrix mapping source -> rectified
    source_corners: np.ndarray     # (4, 2) source quad
    output_size: Tuple[int, int]   # (width, height)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Converts BGR/RGB/gray uint8 input to a single-channel uint8 image."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def segment_panel_mask(gray: np.ndarray, merge_rows: bool = False) -> np.ndarray:
    """
    Produces a binary mask of PV module surfaces.

    Module glass is warmer (brighter) than vegetation and gravel in operating
    conditions, so Otsu's threshold separates the two populations. The threshold is
    computed only over valid pixels, so zero padding from a previous warp cannot
    skew the histogram. Opening first (kernel wider than ground speckle bridges)
    disconnects neighbouring tables; a small closing then refills the dark
    inter-cell grid lines. With ``merge_rows`` a tall vertical closing also bridges
    the inter-row aisles so that a whole array block becomes one region.
    """
    gray = to_gray(gray)
    h, w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    valid = gray > 0
    if np.count_nonzero(valid) < 64:
        return np.zeros_like(gray)
    t_otsu, _ = cv2.threshold(blurred[valid].reshape(-1, 1), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = ((blurred > t_otsu) & valid).astype(np.uint8) * 255

    unit = max(3, int(round(min(h, w) / 128)))  # 4 px at 512-640 px resolution
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * unit - 1, 2 * unit - 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (unit + 1, unit + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    if merge_rows:
        tall = cv2.getStructuringElement(cv2.MORPH_RECT, (unit, 12 * unit))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, tall, iterations=1)

    # Fill interior holes (dark busbars / cold cells) so contours stay simple.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def order_corners(points: np.ndarray) -> np.ndarray:
    """
    Orders four points as TL, TR, BR, BL.

    Sorts by angle around the centroid (robust to rotations where the classic
    sum/difference heuristic fails), then rotates the cycle so the corner with the
    smallest x + y comes first.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]           # clockwise in image coordinates (y down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0).astype(np.float32)


def approximate_quadrilateral(contour: np.ndarray) -> Tuple[np.ndarray, str, float]:
    """
    Fits a 4-vertex polygon to a contour.

    The convex hull is simplified with cv2.approxPolyDP while epsilon is swept from 1%
    to 12% of the perimeter; the first 4-vertex result is accepted. If no epsilon
    yields exactly four vertices, the minimum-area rotated rectangle is used.
    """
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for eps_frac in np.linspace(0.01, 0.12, 23):
        approx = cv2.approxPolyDP(hull, float(eps_frac) * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2)), "approxPolyDP", float(eps_frac)
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(box), "minAreaRect", 0.0


def refine_corners_line_fit(contour: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Sub-pixel corner refinement by least-squares edge fitting.

    approxPolyDP can only return vertices that lie on the (pixel-quantised) contour.
    Here every contour point is assigned to its nearest quad side; points in the
    central 80% of each side are fitted with a robust (Huber) line via cv2.fitLine,
    and each corner is re-computed as the intersection of its two adjacent lines.
    This averages hundreds of boundary pixels per edge, typically cutting the corner
    error from a few pixels to well under one pixel.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    quad = order_corners(corners).astype(np.float64)
    if len(pts) < 16:
        return quad.astype(np.float32)

    sides = [(quad[i], quad[(i + 1) % 4]) for i in range(4)]
    distances, params = [], []
    for a, b in sides:
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
        distances.append(np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
        params.append(t)
    distances = np.stack(distances, axis=1)
    params = np.stack(params, axis=1)
    nearest = distances.argmin(axis=1)

    lines = []
    for i, (a, b) in enumerate(sides):
        selected = pts[(nearest == i) & (params[:, i] > 0.1) & (params[:, i] < 0.9)]
        if len(selected) < 4:
            selected = np.linspace(a, b, 10)
        vx, vy, x0, y0 = cv2.fitLine(selected.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)))

    refined = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        system = np.array([d1, -d2]).T
        if abs(np.linalg.det(system)) < 1e-9:
            refined.append(quad[i])
            continue
        s = np.linalg.solve(system, p2 - p1)
        refined.append(p1 + s[0] * d1)
    refined = np.array(refined, dtype=np.float32)

    # Guard against degenerate intersections (nearly parallel edges): keep the original.
    if np.max(np.linalg.norm(refined - quad.astype(np.float32), axis=1)) > 0.25 * np.ptp(quad, axis=0).max():
        return quad.astype(np.float32)
    return order_corners(refined)


def detect_panel_quads(
    image: np.ndarray,
    min_area_frac: float = 0.008,
    max_quads: int = 40,
    merge_rows: bool = False,
    refine: bool = True,
) -> Tuple[List[PanelQuad], np.ndarray]:
    """
    Detects module-table quadrilaterals.

    Contours smaller than ``min_area_frac`` of the frame are ignored. With ``refine``
    the approxPolyDP corners are refined to sub-pixel accuracy by edge line fitting.

    Returns:
        (quads sorted by area descending, binary segmentation mask)
    """
    gray = to_gray(image)
    h, w = gray.shape[:2]
    mask = segment_panel_mask(gray, merge_rows=merge_rows)
    # CHAIN_APPROX_NONE keeps every boundary pixel for the least-squares edge refinement.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    min_area = min_area_frac * h * w
    quads: List[PanelQuad] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        corners, method, eps = approximate_quadrilateral(contour)
        if refine:
            corners = refine_corners_line_fit(contour, corners)
            method = f"{method} + LSQ edge refinement"
        quad_area = float(cv2.contourArea(corners.reshape(-1, 1, 2)))
        if quad_area <= 0:
            continue
        quads.append(PanelQuad(corners=corners, contour_area=area, quad_area=quad_area,
                               method=method, epsilon_frac=eps, contour=contour))
    quads.sort(key=lambda q: q.contour_area, reverse=True)
    return quads[:max_quads], mask


def compute_homography(source_corners: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Solves the 3x3 homography mapping ordered source corners onto the output rectangle."""
    out_w, out_h = output_size
    destination = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                           dtype=np.float32)
    return cv2.getPerspectiveTransform(order_corners(source_corners), destination)


def rectify_quad(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: Tuple[int, int] = (640, 640),
) -> RectificationResult:
    """Warps the quadrilateral region into a fronto-parallel rectangle of ``output_size``."""
    H = compute_homography(corners, output_size)
    warped = cv2.warpPerspective(image, H, output_size, flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return RectificationResult(warped=warped, homography=H,
                               source_corners=order_corners(corners), output_size=output_size)


def aspect_true_size(quad: PanelQuad, width: int = 640) -> Tuple[int, int]:
    """Output size that preserves the table's measured aspect ratio at a fixed width."""
    w_px, h_px = quad.edge_lengths()
    ratio = h_px / w_px if w_px > 0 else 1.0
    return width, max(16, int(round(width * ratio)))


def _hartley_normalizer(points: np.ndarray) -> np.ndarray:
    """Similarity transform moving points to zero mean and mean distance sqrt(2)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    center = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - center, axis=1))
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array([[s, 0, -s * center[0]], [0, s, -s * center[1]], [0, 0, 1]], dtype=np.float64)


def homography_diagnostics(
    H: np.ndarray,
    source_corners: Optional[np.ndarray] = None,
    output_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, float]:
    """
    Numerical health of a homography.

    The raw condition number of H is dominated by its translation terms and is not
    meaningful, so when the correspondences are supplied the condition number is
    reported for the Hartley-normalised homography T_dst · H · T_src^-1 (the quantity
    that governs DLT stability). Also reports the 4-point re-projection residual and
    the projective (keystone) terms h31, h32.
    """
    Hn = H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H
    diag = {
        "determinant": float(np.linalg.det(Hn)),
        "perspective_h31": float(Hn[2, 0]),
        "perspective_h32": float(Hn[2, 1]),
    }
    if source_corners is not None and output_size is not None:
        out_w, out_h = output_size
        src = order_corners(source_corners).astype(np.float64)
        dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float64)
        H_norm = _hartley_normalizer(dst) @ Hn @ np.linalg.inv(_hartley_normalizer(src))
        H_norm /= H_norm[2, 2]
        projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hn).reshape(4, 2)
        diag["normalized_condition_number"] = float(np.linalg.cond(H_norm))
        diag["reprojection_residual_px"] = float(np.linalg.norm(projected - dst, axis=1).mean())
    else:
        diag["normalized_condition_number"] = float(np.linalg.cond(Hn))
    return diag


def simulate_oblique_view(
    image: np.ndarray,
    pitch_deg: float = 28.0,
    roll_deg: float = 7.0,
    yaw_deg: float = 0.0,
    focal_scale: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesises an oblique capture of a planar scene from a nadir frame.

    A pure camera rotation induces the homography H = K R K^-1 (K: pinhole intrinsics
    with f = focal_scale * width). The result is re-centred and scaled to fit the
    original canvas.

    Returns:
        (oblique image, 3x3 ground-truth homography nadir -> oblique)
    """
    h, w = image.shape[:2]
    f = focal_scale * w
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
    p, r, y = np.deg2rad([pitch_deg, roll_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    H_rot = K @ (Rz @ Ry @ Rx) @ np.linalg.inv(K)

    frame = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.float64)
    projected = cv2.perspectiveTransform(frame, H_rot).reshape(4, 2)
    min_xy, max_xy = projected.min(axis=0), projected.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    s = 0.96 * min(w / span[0], h / span[1])
    offset = np.array([w, h]) / 2.0 - s * (min_xy + max_xy) / 2.0
    S = np.array([[s, 0, offset[0]], [0, s, offset[1]], [0, 0, 1]], dtype=np.float64)
    H_true = S @ H_rot
    oblique = cv2.warpPerspective(image, H_true, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return oblique, H_true


def corner_reprojection_error(detected: np.ndarray, reference: np.ndarray) -> float:
    """Mean Euclidean distance (px) between two ordered corner sets."""
    return float(np.linalg.norm(order_corners(detected) - order_corners(reference), axis=1).mean())


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two equally sized grayscale images."""
    a = to_gray(a).astype(np.float64).ravel()
    b = to_gray(b).astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def draw_quads(
    image: np.ndarray,
    quads: Sequence[PanelQuad],
    target_index: Optional[int] = 0,
) -> np.ndarray:
    """Draws all detected quads (green) and the rectification target (amber, with corner labels)."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    scale = max(0.35, min(canvas.shape[:2]) / 1400.0)
    for i, quad in enumerate(quads):
        pts = quad.corners.astype(np.int32).reshape(-1, 1, 2)
        if quad.contour is not None:
            cv2.drawContours(canvas, [quad.contour], -1, (90, 160, 60), 1, cv2.LINE_AA)
        color = (11, 158, 245) if i == target_index else (80, 220, 120)
        cv2.polylines(canvas, [pts], True, color, 2 if i == target_index else 1, cv2.LINE_AA)
    if target_index is not None and 0 <= target_index < len(quads):
        for label, (x, y) in zip(CORNER_LABELS, quads[target_index].corners):
            center = (int(round(x)), int(round(y)))
            cv2.circle(canvas, center, 5, (11, 158, 245), -1, cv2.LINE_AA)
            cv2.circle(canvas, center, 5, (15, 15, 15), 1, cv2.LINE_AA)
            tx = min(max(center[0] + 6, 2), canvas.shape[1] - 30)
            ty = min(max(center[1] - 6, 14), canvas.shape[0] - 4)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (11, 158, 245), 1, cv2.LINE_AA)
    return canvas


def analyze_panel_geometry(image: np.ndarray, output_size: Tuple[int, int] = (640, 640)) -> Dict[str, object]:
    """
    One-call geometry analysis used by the inspection API.

    Returns numpy images (overlay in RGB, rectified tables in RGB) plus JSON-ready
    metadata. Images are left un-encoded so the caller decides on the transport format.
    """
    gray = to_gray(image)
    h, w = gray.shape[:2]
    quads, mask = detect_panel_quads(gray)
    overlay_bgr = draw_quads(gray, quads, target_index=0 if quads else None)
    result: Dict[str, object] = {
        "modules_detected": len(quads),
        "quads": [q.to_dict(w, h) for q in quads],
        "overlay_rgb": cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB),
        "mask": mask,
        "rectified_rgb": None,
        "rectified_aspect_rgb": None,
        "homography": None,
        "diagnostics": None,
        "output_size": list(output_size),
    }
    if quads:
        target = quads[0]
        rect = rectify_quad(gray, target.corners, output_size)
        rect_aspect = rectify_quad(gray, target.corners, aspect_true_size(target, output_size[0]))
        result["rectified_rgb"] = cv2.cvtColor(rect.warped, cv2.COLOR_GRAY2RGB)
        result["rectified_aspect_rgb"] = cv2.cvtColor(rect_aspect.warped, cv2.COLOR_GRAY2RGB)
        result["homography"] = [[round(float(v), 6) for v in row] for row in rect.homography]
        diag = homography_diagnostics(rect.homography, target.corners, output_size)
        result["diagnostics"] = {k: round(v, 6) for k, v in diag.items()}
        result["target_method"] = target.method
    return result


def _match_quad(quads: Sequence[PanelQuad], expected_centroid: np.ndarray) -> Optional[int]:
    """Index of the quad whose centroid is nearest ``expected_centroid``."""
    if not quads:
        return None
    distances = [float(np.linalg.norm(q.centroid - expected_centroid)) for q in quads]
    return int(np.argmin(distances))


def _format_matrix(H: np.ndarray) -> str:
    return "\n".join("      [" + "  ".join(f"{v:+11.5f}" for v in row) + " ]" for row in H)


def run_demo(
    image_path: str,
    figure_path: str,
    rectified_path: str,
    simulate: bool = True,
    pitch_deg: float = 28.0,
    roll_deg: float = 7.0,
    size: int = 640,
) -> Dict[str, object]:
    """
    End-to-end evaluator demonstration. Writes a figure and the rectified table and
    returns the numeric results (also printed to the console).
    """
    nadir = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if nadir is None:
        raise FileNotFoundError(f"Could not read image '{image_path}'.")

    if simulate:
        scene, H_true = simulate_oblique_view(nadir, pitch_deg=pitch_deg, roll_deg=roll_deg)
    else:
        scene, H_true = nadir, None

    quads, mask = detect_panel_quads(scene)
    if not quads:
        raise RuntimeError("No module-table contours found; check that the frame contains PV arrays.")

    report: Dict[str, object] = {"modules_detected": len(quads)}
    target_idx = 0
    reference_rectified = None

    if H_true is not None:
        # Pick the largest table in the nadir frame and locate it in the oblique frame.
        nadir_quads, _ = detect_panel_quads(nadir)
        if nadir_quads:
            ref = nadir_quads[0]
            expected = cv2.perspectiveTransform(ref.corners.reshape(-1, 1, 2).astype(np.float64),
                                                H_true).reshape(4, 2)
            target_idx = _match_quad(quads, expected.mean(axis=0)) or 0
            report["corner_reprojection_error_px"] = corner_reprojection_error(
                quads[target_idx].corners, expected)
            reference_rectified = rectify_quad(nadir, ref.corners, aspect_true_size(ref, size)).warped

    target = quads[target_idx]
    normalized = rectify_quad(scene, target.corners, (size, size))
    aspect_size = aspect_true_size(target, size)
    aspect = rectify_quad(scene, target.corners, aspect_size)

    if reference_rectified is not None:
        ref_resized = cv2.resize(reference_rectified, aspect_size, interpolation=cv2.INTER_AREA)
        report["ncc_vs_true_topdown"] = normalized_cross_correlation(aspect.warped, ref_resized)

    diag = homography_diagnostics(normalized.homography, target.corners, (size, size))
    report.update({
        "target_corners": target.corners.tolist(),
        "approximation": target.method,
        "epsilon_frac": target.epsilon_frac,
        "rectangularity": target.rectangularity,
        "homography": normalized.homography.tolist(),
        **diag,
    })

    os.makedirs(os.path.dirname(rectified_path) or ".", exist_ok=True)
    cv2.imwrite(rectified_path, normalized.warped)
    _save_demo_figure(figure_path, nadir, scene, mask, quads, target_idx, normalized.warped,
                      aspect.warped, report, simulate, pitch_deg, roll_deg)
    _print_report(report, image_path, figure_path, rectified_path, simulate, pitch_deg, roll_deg, size)
    return report


def _save_demo_figure(path, nadir, scene, mask, quads, target_idx, square, aspect, report,
                      simulate, pitch_deg, roll_deg) -> None:
    """Composes the 2x3 evaluator figure with Matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, surface = "#f0efec", "#a9a8a0", "#111316"
    overlay = cv2.cvtColor(draw_quads(scene, quads, target_idx), cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(16, 9.6), dpi=120, facecolor=surface)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.28, wspace=0.08,
                            left=0.03, right=0.97, top=0.88, bottom=0.05)
    panels = [
        (grid[0, 0], nadir, "gray", "(a) Nadir reference capture"),
        (grid[0, 1], scene, "gray",
         f"(b) Synthetic oblique view (pitch {pitch_deg:.0f}°, roll {roll_deg:.0f}°)" if simulate
         else "(b) Input capture"),
        (grid[0, 2], mask, "gray", "(c) Otsu → opening → closing mask"),
        (grid[1, 0], overlay, None, f"(d) findContours + approxPolyDP + LSQ edges: {len(quads)} tables"),
        (grid[1, 1], aspect, "gray", "(e) Rectified table, aspect-true"),
        (grid[1, 2], square, "gray", f"(f) Normalised {square.shape[1]}×{square.shape[0]} warp"),
    ]
    for spec, img, cmap, title in panels:
        ax = fig.add_subplot(spec)
        ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None,
                  interpolation="bilinear", aspect="equal")
        ax.set_title(title, color=ink, fontsize=11, loc="left", pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2c2c2a")

    fig.text(0.03, 0.955, "HELIOS-AI · Homography Rectification of PV Module Tables",
             color=ink, fontsize=17, fontweight="bold")
    metrics = [f"tables detected {report['modules_detected']}",
               f"normalised cond(H) {report['normalized_condition_number']:.1f}",
               f"rectangularity {report['rectangularity']:.3f}"]
    if "corner_reprojection_error_px" in report:
        metrics.append(f"corner error {report['corner_reprojection_error_px']:.2f} px")
    if "ncc_vs_true_topdown" in report:
        metrics.append(f"NCC vs. true top-down {report['ncc_vs_true_topdown']:.3f}")
    fig.text(0.03, 0.918, "   ·   ".join(metrics), color=muted, fontsize=11)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def _print_report(report, image_path, figure_path, rectified_path, simulate, pitch_deg, roll_deg, size) -> None:
    line = "─" * 72
    print(line)
    print("  HELIOS-AI  |  IPCV Homography Rectification")
    print(line)
    print(f"  Input image            : {os.path.relpath(image_path, PROJECT_ROOT)}")
    if simulate:
        print(f"  Synthetic oblique view : pitch {pitch_deg:.1f}°, roll {roll_deg:.1f}°  (H = K·R·K⁻¹)")
    print(f"  Module tables detected : {report['modules_detected']}")
    print(f"  Quad approximation     : {report['approximation']}")
    print(f"                           (approxPolyDP ε = {report['epsilon_frac'] * 100:.1f}% of perimeter)")
    print(f"  Rectangularity         : {report['rectangularity']:.3f}")
    corners = "  ".join(f"{lbl}({x:.0f},{y:.0f})" for lbl, (x, y) in zip(CORNER_LABELS, report["target_corners"]))
    print(f"  Target corners         : {corners}")
    print(f"  Homography H ({size}×{size} target):")
    print(_format_matrix(np.array(report["homography"])))
    print(f"  det(H) = {report['determinant']:.4e}   Hartley-normalised cond(H) = "
          f"{report['normalized_condition_number']:.2f}")
    print(f"  4-point re-projection residual              : {report['reprojection_residual_px']:.2e} px")
    if "corner_reprojection_error_px" in report:
        print(f"  Corner re-projection error vs. ground truth : {report['corner_reprojection_error_px']:.2f} px")
    if "ncc_vs_true_topdown" in report:
        print(f"  NCC(rectified, true top-down)               : {report['ncc_vs_true_topdown']:.4f}")
    print(f"  Figure written         : {os.path.relpath(figure_path, PROJECT_ROOT)}")
    print(f"  Rectified table        : {os.path.relpath(rectified_path, PROJECT_ROOT)}")
    print(line)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Homography rectification of PV module tables.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input thermal image (default: Results/INPUT.jpg)")
    parser.add_argument("--output", default=DEFAULT_FIGURE, help="Evaluator figure path")
    parser.add_argument("--rectified", default=DEFAULT_RECTIFIED, help="Rectified 640x640 output path")
    parser.add_argument("--no-simulate", action="store_true",
                        help="Rectify the input as-is instead of synthesising an oblique view")
    parser.add_argument("--pitch", type=float, default=28.0, help="Synthetic camera pitch in degrees")
    parser.add_argument("--roll", type=float, default=7.0, help="Synthetic camera roll in degrees")
    parser.add_argument("--size", type=int, default=640, help="Normalised output size in pixels")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    try:
        run_demo(args.image, args.output, args.rectified, simulate=not args.no_simulate,
                 pitch_deg=args.pitch, roll_deg=args.roll, size=args.size)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"[ipcv_geometry] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
