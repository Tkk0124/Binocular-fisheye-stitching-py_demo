from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.image_utils import draw_circle_overlay, make_circle_mask, non_black_mask


def detect_circle(img_bgr, black_threshold: int, min_valid_area_ratio: float, fallback_radius_ratio: float, fit_mode: str = 'enclosing') -> Dict[str, float]:
    mask = non_black_mask(img_bgr, black_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = img_bgr.shape[:2]
    if not contours:
        return {'cx': w / 2.0, 'cy': h / 2.0, 'radius': min(h, w) * fallback_radius_ratio, 'method': 'fallback_no_contour'}
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < min_valid_area_ratio * h * w:
        return {'cx': w / 2.0, 'cy': h / 2.0, 'radius': min(h, w) * fallback_radius_ratio, 'method': 'fallback_small_contour'}
    (enc_cx, enc_cy), enc_r = cv2.minEnclosingCircle(cnt)
    cx, cy, r = float(enc_cx), float(enc_cy), float(enc_r)
    fit_info = {}
    # Least-squares is useful as a diagnostic, but with dark/partial fisheye
    # rims it tends to move inward and cuts valid edge pixels.
    pts = cnt.reshape(-1, 2).astype(np.float64)
    if len(pts) > 50:
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
        b = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
            cx2, cy2, c = sol
            r2 = np.sqrt(max(c + cx2 * cx2 + cy2 * cy2, 1.0))
            if np.isfinite(r2) and 0.2 * min(h, w) < r2 < 0.7 * max(h, w):
                fit_info = {'fit_cx': float(cx2), 'fit_cy': float(cy2), 'fit_radius': float(r2)}
                if fit_mode == 'least_squares':
                    cx, cy, r = float(cx2), float(cy2), float(r2)
        except Exception:
            pass
    result = {
        'cx': float(cx),
        'cy': float(cy),
        'radius': float(r),
        'method': 'contour_min_enclosing_circle' if fit_mode != 'least_squares' else 'contour_circle_fit',
        'contour_area': area,
        'enclosing_cx': float(enc_cx),
        'enclosing_cy': float(enc_cy),
        'enclosing_radius': float(enc_r),
    }
    result.update(fit_info)
    return result


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '02_circle_detect')
    c = cfg['circle_detect']
    left = read_image_bgr(ctx['left_preprocessed'])
    right = read_image_bgr(ctx['right_preprocessed'])
    fit_mode = c.get('fit_mode', 'enclosing')
    left_circle = detect_circle(left, c.get('black_threshold', 12), c.get('min_valid_area_ratio', 0.08), c.get('fallback_radius_ratio_to_short_side', 0.4687), fit_mode)
    right_circle = detect_circle(right, c.get('black_threshold', 12), c.get('min_valid_area_ratio', 0.08), c.get('fallback_radius_ratio_to_short_side', 0.4687), fit_mode)
    shrink = float(c.get('radius_shrink_px', 4))
    left_mask = make_circle_mask(left.shape[:2], left_circle, shrink)
    right_mask = make_circle_mask(right.shape[:2], right_circle, shrink)
    write_image(out_dir / 'left_circle_overlay.png', draw_circle_overlay(left, left_circle))
    write_image(out_dir / 'right_circle_overlay.png', draw_circle_overlay(right, right_circle))
    write_image(out_dir / 'left_valid_circle_mask.png', left_mask)
    write_image(out_dir / 'right_valid_circle_mask.png', right_mask)
    report = {'left_circle': left_circle, 'right_circle': right_circle, 'radius_shrink_px': shrink, 'fit_mode': fit_mode}
    write_json(out_dir / 'circle_report.json', report)
    ctx.update({
        'left_circle': left_circle,
        'right_circle': right_circle,
        'left_circle_mask': str(out_dir / 'left_valid_circle_mask.png'),
        'right_circle_mask': str(out_dir / 'right_valid_circle_mask.png'),
        'circle_report': str(out_dir / 'circle_report.json')
    })
    return ctx
