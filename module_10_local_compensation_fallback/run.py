from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json


def _estimate_shift(left_crop, right_crop, mask_crop):
    gl = cv2.cvtColor(left_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gr = cv2.cvtColor(right_crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    m = (mask_crop > 0).astype(np.float32)
    if np.count_nonzero(m) < 100:
        return None
    gl = gl * m
    gr = gr * m
    hann = cv2.createHanningWindow((gl.shape[1], gl.shape[0]), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(gl, gr, hann)
    return float(shift[0]), float(shift[1]), float(response)


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '10_local_compensation_fallback')
    fcfg = cfg['fallback_local_compensation']
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    overlap = cv2.imread(ctx['overlap_mask'], cv2.IMREAD_GRAYSCALE)
    report = {'enabled': bool(fcfg.get('enabled', False)), 'components': [], 'note': 'Fallback is diagnostic by default; it does not replace the final panorama unless enabled and manually wired.'}
    max_shift = float(fcfg.get('max_shift_px', 20))
    min_response = float(fcfg.get('min_response', 0.08))
    vis = cv2.addWeighted(left, 0.5, right, 0.5, 0)
    for idx, c in enumerate(ctx.get('overlap_components', [])):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        shift = _estimate_shift(left[y:y+h, x:x+w], right[y:y+h, x:x+w], overlap[y:y+h, x:x+w])
        item = {'component': idx, 'bbox': [x, y, w, h], 'phase_shift': None, 'accepted': False}
        if shift is not None:
            dx, dy, resp = shift
            accepted = abs(dx) <= max_shift and abs(dy) <= max_shift and resp >= min_response
            item['phase_shift'] = {'dx_px': dx, 'dy_px': dy, 'response': resp}
            item['accepted'] = bool(accepted)
            cv2.putText(vis, f"C{idx} dx={dx:.1f} dy={dy:.1f} r={resp:.2f}", (x+5, y+28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255) if accepted else (0,0,255), 2)
        report['components'].append(item)
    write_image(out_dir / 'local_shift_diagnostics.png', vis)
    write_json(out_dir / 'local_compensation_report.json', report)
    ctx.update({'local_compensation_report': str(out_dir / 'local_compensation_report.json')})
    return ctx
