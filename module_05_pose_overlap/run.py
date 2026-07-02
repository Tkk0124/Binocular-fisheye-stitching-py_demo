from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.metrics import connected_overlap_components


def _standard_dual_fisheye_seam_components(overlap: np.ndarray, cfg: Dict) -> List[Dict]:
    """Create two seam working strips around quarter and three-quarter longitudes.

    The raw overlap mask can be connected through polar regions when FOV > 180°.
    For stitching quality evaluation we need the two lateral seam zones, not the polar union.
    """
    H, W = overlap.shape
    width_ratio = float(cfg['overlap'].get('side_strip_width_ratio', 0.125))
    v_height_ratio = float(cfg['overlap'].get('strip_vertical_height_ratio', 0.5))
    sw = max(16, int(round(W * width_ratio)))
    y0 = int(round(H * (0.5 - v_height_ratio / 2.0)))
    y1 = int(round(H * (0.5 + v_height_ratio / 2.0)))
    centers = [W * 0.25, W * 0.75]
    comps = []
    for i, cx in enumerate(centers):
        x0 = max(0, int(round(cx - sw / 2)))
        x1 = min(W, int(round(cx + sw / 2)))
        crop = overlap[y0:y1, x0:x1]
        area = int(np.count_nonzero(crop))
        if area > 0:
            comps.append({
                'label': i + 1,
                'x': int(x0), 'y': int(y0), 'w': int(x1 - x0), 'h': int(y1 - y0),
                'area': area,
                'cx': float((x0 + x1) / 2.0), 'cy': float((y0 + y1) / 2.0),
                'source': 'standard_dual_fisheye_quarter_strip'
            })
    return comps


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '05_pose_overlap')
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    overlap = cv2.imread(ctx['overlap_mask'], cv2.IMREAD_GRAYSCALE)
    raw_comps = connected_overlap_components(overlap, int(cfg['overlap'].get('min_component_area', 2000)))
    comps = _standard_dual_fisheye_seam_components(overlap, cfg)
    vis = cv2.addWeighted(left, 0.5, right, 0.5, 0)
    raw_vis = vis.copy()
    for idx, c in enumerate(raw_comps):
        cv2.rectangle(raw_vis, (c['x'], c['y']), (c['x'] + c['w'], c['y'] + c['h']), (255, 128, 0), 2)
        cv2.putText(raw_vis, f"RAW{idx}:{c['area']}", (c['x'] + 5, c['y'] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    write_image(out_dir / 'raw_overlap_components_overlay.png', raw_vis)

    for idx, c in enumerate(comps):
        cv2.rectangle(vis, (c['x'], c['y']), (c['x'] + c['w'], c['y'] + c['h']), (0, 255, 0), 2)
        cv2.putText(vis, f"S{idx}:{c['area']}", (c['x'] + 5, c['y'] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        crop_l = left[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']]
        crop_r = right[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']]
        crop_m = overlap[c['y']:c['y']+c['h'], c['x']:c['x']+c['w']]
        write_image(out_dir / f'component_{idx:02d}_left.png', crop_l)
        write_image(out_dir / f'component_{idx:02d}_right.png', crop_r)
        write_image(out_dir / f'component_{idx:02d}_mask.png', crop_m)
    write_image(out_dir / 'overlap_components_overlay.png', vis)
    report = {'raw_component_count': len(raw_comps), 'raw_components': raw_comps, 'component_count': len(comps), 'components': comps, 'pose': cfg['pose']}
    write_json(out_dir / 'pose_overlap_report.json', report)
    ctx.update({'raw_overlap_components': raw_comps, 'overlap_components': comps, 'pose_overlap_report': str(out_dir / 'pose_overlap_report.json')})
    return ctx
