from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.metrics import seam_energy_map, heatmap_u8, summarize_values, edge_map, gradient_mag


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '06_overlap_evaluation')
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    overlap = cv2.imread(ctx['overlap_mask'], cv2.IMREAD_GRAYSCALE)
    ev = cfg['evaluation']
    energy = seam_energy_map(left, right, overlap, ev.get('color_weight', 0.35), ev.get('edge_weight', 0.55), ev.get('gradient_weight', 0.10))
    write_image(out_dir / 'seam_energy_heatmap.png', heatmap_u8(energy))
    write_image(out_dir / 'edge_left.png', (edge_map(left) * 255).astype(np.uint8))
    write_image(out_dir / 'edge_right.png', (edge_map(right) * 255).astype(np.uint8))
    diff = cv2.absdiff(left, right)
    diff[overlap == 0] = 0
    write_image(out_dir / 'color_absdiff.png', diff)
    report = {
        'energy': summarize_values(energy, overlap),
        'overlap_valid_pixels': int(np.count_nonzero(overlap)),
        'overlap_valid_ratio': float(np.count_nonzero(overlap) / overlap.size)
    }
    # Component-level stats
    component_stats = []
    for idx, c in enumerate(ctx.get('overlap_components', [])):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        e_crop = energy[y:y+h, x:x+w]
        m_crop = overlap[y:y+h, x:x+w]
        component_stats.append({'component': idx, 'bbox': [x, y, w, h], 'energy': summarize_values(e_crop, m_crop)})
        write_image(out_dir / f'component_{idx:02d}_energy_heatmap.png', heatmap_u8(e_crop))
    report['components'] = component_stats
    write_json(out_dir / 'overlap_evaluation_report.json', report)
    ctx.update({'seam_energy_map': str(out_dir / 'seam_energy_heatmap.png'), 'overlap_evaluation_report': str(out_dir / 'overlap_evaluation_report.json')})
    return ctx
