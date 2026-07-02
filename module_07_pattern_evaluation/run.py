from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.metrics import line_bending_from_points


def _try_chessboard(gray: np.ndarray, patterns: List[List[int]]):
    for p in patterns:
        size = (int(p[0]), int(p[1]))
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, size, flags)
        if ok and corners is not None:
            return size, corners.reshape(-1, 2)
    return None, None


def _pattern_metrics(corners: np.ndarray, size: Tuple[int, int]) -> Dict:
    cols, rows = size
    pts = corners.reshape(rows, cols, 2)
    row_bending = [line_bending_from_points(pts[r, :, :]) for r in range(rows)]
    col_bending = [line_bending_from_points(pts[:, c, :]) for c in range(cols)]
    h_spacing = np.linalg.norm(np.diff(pts, axis=1), axis=2).reshape(-1)
    v_spacing = np.linalg.norm(np.diff(pts, axis=0), axis=2).reshape(-1)
    return {
        'pattern_size': [cols, rows],
        'corner_count': int(corners.shape[0]),
        'horizontal_spacing_mean_px': float(np.mean(h_spacing)),
        'horizontal_spacing_cv': float(np.std(h_spacing) / (np.mean(h_spacing) + 1e-6)),
        'vertical_spacing_mean_px': float(np.mean(v_spacing)),
        'vertical_spacing_cv': float(np.std(v_spacing) / (np.mean(v_spacing) + 1e-6)),
        'row_line_bending_rms_mean_px': float(np.mean([m['rms_px'] for m in row_bending if m['rms_px'] is not None])) if row_bending else None,
        'col_line_bending_rms_mean_px': float(np.mean([m['rms_px'] for m in col_bending if m['rms_px'] is not None])) if col_bending else None,
    }


def _draw_corners(img: np.ndarray, corners: np.ndarray | None, size: Tuple[int, int] | None) -> np.ndarray:
    out = img.copy()
    if corners is not None and size is not None:
        cv2.drawChessboardCorners(out, size, corners.reshape(-1, 1, 2).astype(np.float32), True)
    return out


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '07_pattern_evaluation')
    pat = cfg['pattern']
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    patterns = pat.get('chessboard_patterns', [[11, 8], [10, 7], [9, 6], [8, 6], [7, 5]])
    report = {'enabled': bool(pat.get('enable_chessboard_detection', True)), 'components': []}
    if not report['enabled']:
        write_json(out_dir / 'pattern_evaluation_report.json', report)
        ctx.update({'pattern_evaluation_report': str(out_dir / 'pattern_evaluation_report.json')})
        return ctx

    for idx, c in enumerate(ctx.get('overlap_components', [])):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        cl = left[y:y+h, x:x+w]
        cr = right[y:y+h, x:x+w]
        gl = cv2.cvtColor(cl, cv2.COLOR_BGR2GRAY)
        gr = cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY)
        size_l, corners_l = _try_chessboard(gl, patterns)
        size_r, corners_r = _try_chessboard(gr, patterns)
        comp = {'component': idx, 'bbox': [x, y, w, h], 'left_found': corners_l is not None, 'right_found': corners_r is not None}
        if corners_l is not None:
            comp['left_metrics'] = _pattern_metrics(corners_l, size_l)
            write_image(out_dir / f'component_{idx:02d}_left_chessboard.png', _draw_corners(cl, corners_l, size_l))
        else:
            write_image(out_dir / f'component_{idx:02d}_left_chessboard_not_found.png', cl)
        if corners_r is not None:
            comp['right_metrics'] = _pattern_metrics(corners_r, size_r)
            write_image(out_dir / f'component_{idx:02d}_right_chessboard.png', _draw_corners(cr, corners_r, size_r))
        else:
            write_image(out_dir / f'component_{idx:02d}_right_chessboard_not_found.png', cr)
        # If same board detected in both crops, compare vertical offset distribution in crop coordinates.
        if corners_l is not None and corners_r is not None and size_l == size_r and len(corners_l) == len(corners_r):
            dy = corners_l[:, 1] - corners_r[:, 1]
            dx = corners_l[:, 0] - corners_r[:, 0]
            comp['left_right_corner_delta'] = {
                'mean_dx_px': float(np.mean(dx)), 'rms_dx_px': float(np.sqrt(np.mean(dx * dx))), 'max_abs_dx_px': float(np.max(np.abs(dx))),
                'mean_dy_px': float(np.mean(dy)), 'rms_dy_px': float(np.sqrt(np.mean(dy * dy))), 'max_abs_dy_px': float(np.max(np.abs(dy))),
            }
        report['components'].append(comp)
    write_json(out_dir / 'pattern_evaluation_report.json', report)
    ctx.update({'pattern_evaluation_report': str(out_dir / 'pattern_evaluation_report.json')})
    return ctx
