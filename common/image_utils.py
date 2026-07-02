from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    return np.clip(img.astype(np.float32), 0.0, 1.0)


def to_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def non_black_mask(img_bgr: np.ndarray, threshold: int = 12) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def gray_world_white_balance(img_bgr: np.ndarray, mask: np.ndarray | None = None, eps: float = 1e-6) -> Tuple[np.ndarray, Dict[str, float]]:
    img = to_float01(img_bgr)
    if mask is not None:
        m = mask > 0
        if np.count_nonzero(m) < 100:
            m = np.ones(img.shape[:2], dtype=bool)
    else:
        m = np.ones(img.shape[:2], dtype=bool)
    means = img[m].reshape(-1, 3).mean(axis=0)  # B,G,R
    target = float(means.mean())
    gains = target / np.maximum(means, eps)
    out = np.clip(img * gains.reshape(1, 1, 3), 0, 1)
    stats = {
        "mean_b_before": float(means[0]),
        "mean_g_before": float(means[1]),
        "mean_r_before": float(means[2]),
        "gain_b": float(gains[0]),
        "gain_g": float(gains[1]),
        "gain_r": float(gains[2]),
    }
    return to_u8(out), stats


def neutral_gray_white_balance(
    img_bgr: np.ndarray,
    mask: np.ndarray | None = None,
    sat_percentile: float = 35.0,
    value_low_percentile: float = 8.0,
    value_high_percentile: float = 96.0,
    strength: float = 1.0,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """White balance using low-saturation, mid/high-value pixels.

    Full-image gray-world can miss fluorescent green casts because colored scene
    content cancels the channel averages. This estimator focuses on likely
    neutral surfaces such as walls, ceiling, cabinets, and checkerboards.
    """
    img = to_float01(img_bgr)
    hsv = cv2.cvtColor(to_u8(img), cv2.COLOR_BGR2HSV)

    if mask is not None and np.count_nonzero(mask) > 100:
        base = mask > 0
    else:
        base = np.ones(img.shape[:2], dtype=bool)

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    sat_vals = sat[base]
    val_vals = val[base]

    if sat_vals.size < 100 or val_vals.size < 100:
        return img_bgr.copy(), {"method": "neutral_gray_white_balance", "skipped": True, "reason": "not_enough_mask_pixels"}

    sat_thr = float(np.percentile(sat_vals, sat_percentile))
    v_lo = float(np.percentile(val_vals, value_low_percentile))
    v_hi = float(np.percentile(val_vals, value_high_percentile))
    neutral = base & (sat <= sat_thr) & (val >= v_lo) & (val <= v_hi)

    if np.count_nonzero(neutral) < 1000:
        # Relax saturation before giving up; fisheye office scenes can have
        # few truly low-saturation pixels after clipping.
        sat_thr = float(np.percentile(sat_vals, min(65.0, sat_percentile + 25.0)))
        neutral = base & (sat <= sat_thr) & (val >= v_lo) & (val <= v_hi)

    if np.count_nonzero(neutral) < 100:
        return img_bgr.copy(), {
            "method": "neutral_gray_white_balance",
            "skipped": True,
            "reason": "not_enough_neutral_pixels",
            "sat_threshold": sat_thr,
            "value_low": v_lo,
            "value_high": v_hi,
        }

    means = img[neutral].reshape(-1, 3).mean(axis=0)
    target = float(means.mean())
    raw_gains = target / np.maximum(means, eps)
    s = float(np.clip(strength, 0.0, 1.0))
    gains = 1.0 + (raw_gains - 1.0) * s
    out = np.clip(img * gains.reshape(1, 1, 3), 0, 1)

    means_after = out[neutral].reshape(-1, 3).mean(axis=0)
    stats = {
        "method": "neutral_gray_white_balance",
        "skipped": False,
        "neutral_pixel_count": int(np.count_nonzero(neutral)),
        "sat_threshold": sat_thr,
        "value_low": v_lo,
        "value_high": v_hi,
        "strength": s,
        "mean_b_before": float(means[0]),
        "mean_g_before": float(means[1]),
        "mean_r_before": float(means[2]),
        "gain_b": float(gains[0]),
        "gain_g": float(gains[1]),
        "gain_r": float(gains[2]),
        "mean_b_after": float(means_after[0]),
        "mean_g_after": float(means_after[1]),
        "mean_r_after": float(means_after[2]),
    }
    return to_u8(out), stats


def percentile_contrast_stretch(img_bgr: np.ndarray, mask: np.ndarray | None, low: float, high: float) -> Tuple[np.ndarray, Dict[str, float]]:
    img = to_float01(img_bgr)
    if mask is not None and np.count_nonzero(mask) > 100:
        vals = img[mask > 0]
    else:
        vals = img.reshape(-1, 3)
    lo = np.percentile(vals, low, axis=0)
    hi = np.percentile(vals, high, axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    out = np.clip((img - lo.reshape(1, 1, 3)) / scale.reshape(1, 1, 3), 0, 1)
    stats = {f"p{low}_bgr_{i}": float(lo[i]) for i in range(3)}
    stats.update({f"p{high}_bgr_{i}": float(hi[i]) for i in range(3)})
    return to_u8(out), stats


def percentile_luma_contrast_stretch(img_bgr: np.ndarray, mask: np.ndarray | None, low: float, high: float) -> Tuple[np.ndarray, Dict[str, float]]:
    img = to_float01(img_bgr)
    luma = 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]
    if mask is not None and np.count_nonzero(mask) > 100:
        vals = luma[mask > 0]
    else:
        vals = luma.reshape(-1)

    lo = float(np.percentile(vals, low))
    hi = float(np.percentile(vals, high))
    scale = max(hi - lo, 1e-6)
    luma_stretched = np.clip((luma - lo) / scale, 0, 1)
    gain = luma_stretched / np.maximum(luma, 1e-6)
    out = np.clip(img * gain[:, :, None], 0, 1)
    stats = {
        "method": "luma_preserving",
        f"p{low}_luma": lo,
        f"p{high}_luma": hi,
    }
    return to_u8(out), stats


def draw_circle_overlay(img_bgr: np.ndarray, circle: Dict[str, float], color=(0, 255, 0)) -> np.ndarray:
    out = img_bgr.copy()
    c = (int(round(circle["cx"])), int(round(circle["cy"])))
    r = int(round(circle["radius"]))
    cv2.circle(out, c, r, color, 3, cv2.LINE_AA)
    cv2.drawMarker(out, c, (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    return out


def make_circle_mask(shape_hw: Tuple[int, int], circle: Dict[str, float], shrink_px: float = 0.0) -> np.ndarray:
    h, w = shape_hw
    y, x = np.ogrid[:h, :w]
    r = max(float(circle["radius"]) - shrink_px, 1.0)
    m = ((x - float(circle["cx"])) ** 2 + (y - float(circle["cy"])) ** 2 <= r * r).astype(np.uint8) * 255
    return m


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = max(ha, hb)
    if len(a.shape) == 2:
        a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
    if len(b.shape) == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    ca = np.zeros((h, wa, 3), dtype=np.uint8)
    cb = np.zeros((h, wb, 3), dtype=np.uint8)
    ca[:ha, :wa] = a
    cb[:hb, :wb] = b
    return np.hstack([ca, cb])
