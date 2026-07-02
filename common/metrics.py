from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


def connected_overlap_components(mask: np.ndarray, min_area: int = 2000) -> List[Dict[str, int]]:
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps: List[Dict[str, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i].tolist()
        if area >= min_area:
            comps.append({
                'label': i,
                'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h),
                'area': int(area),
                'cx': float(centroids[i][0]), 'cy': float(centroids[i][1])
            })
    comps.sort(key=lambda c: c['x'])
    return comps


def edge_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, 60, 140).astype(np.float32) / 255.0


def gradient_mag(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def seam_energy_map(left: np.ndarray, right: np.ndarray, valid: np.ndarray, color_w: float = 0.35, edge_w: float = 0.55, grad_w: float = 0.10) -> np.ndarray:
    lf = left.astype(np.float32) / 255.0
    rf = right.astype(np.float32) / 255.0
    color = np.mean(np.abs(lf - rf), axis=2)
    edge = np.abs(edge_map(left) - edge_map(right))
    grad = np.abs(gradient_mag(left) - gradient_mag(right))
    grad = grad / (float(np.percentile(grad[valid > 0], 95)) + 1e-6) if np.count_nonzero(valid) > 100 else grad
    e = color_w * color + edge_w * edge + grad_w * grad
    e[valid == 0] = 1.0
    return np.clip(e, 0, 1).astype(np.float32)


def heatmap_u8(x: np.ndarray) -> np.ndarray:
    v = np.clip(x, 0, 1)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)


def summarize_values(vals: np.ndarray, mask: np.ndarray | None = None) -> Dict[str, float]:
    if mask is not None:
        vals = vals[mask > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {'count': 0, 'mean': None, 'rms': None, 'p95': None, 'max': None}
    return {
        'count': int(vals.size),
        'mean': float(np.mean(vals)),
        'rms': float(np.sqrt(np.mean(vals * vals))),
        'p95': float(np.percentile(vals, 95)),
        'max': float(np.max(vals)),
    }


def line_bending_from_points(points: np.ndarray) -> Dict[str, float]:
    if len(points) < 3:
        return {'count': int(len(points)), 'rms_px': None, 'max_px': None}
    pts = points.astype(np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    p = points.astype(np.float32)
    # distance from point to line
    d = np.abs((p[:, 0] - x0) * vy - (p[:, 1] - y0) * vx)
    return {'count': int(len(points)), 'rms_px': float(np.sqrt(np.mean(d * d))), 'max_px': float(np.max(d))}
