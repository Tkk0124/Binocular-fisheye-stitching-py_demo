from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.metrics import seam_energy_map, heatmap_u8, summarize_values, edge_map, gradient_mag


def _clean_old_debug_outputs(out_dir: Path) -> None:
    for pattern in (
        'component_*.png',
        'component_*_brightness_report.json',
        'seam_energy_heatmap.png',
        'edge_left.png',
        'edge_right.png',
        'color_absdiff.png',
        'seam_energy.npy',
        'overlap_evaluation_report.json',
    ):
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def _label_image(img: np.ndarray, label: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 360), 32), (0, 0, 0), -1)
    cv2.putText(out, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _side_by_side_labeled(items: Tuple[Tuple[str, np.ndarray], ...]) -> np.ndarray:
    spacer = None
    labeled = []
    for label, img in items:
        labeled_img = _label_image(img, label)
        labeled.append(labeled_img)
        if spacer is None:
            spacer = np.full((labeled_img.shape[0], 8, 3), 255, dtype=np.uint8)
    out = labeled[0]
    for img in labeled[1:]:
        out = np.hstack([out, spacer, img])
    return out


def _luma(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32) / 255.0
    return 0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]


def _make_luminance_cost_views(
    left_raw: np.ndarray,
    right_raw: np.ndarray,
    valid: np.ndarray,
    eval_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    left_cost = left_raw.copy()
    right_cost = right_raw.copy()
    y_left = _luma(left_raw)
    y_right = _luma(right_raw)
    valid_luma_min = float(eval_cfg.get('valid_luma_min', 0.03))
    valid_luma_max = float(eval_cfg.get('valid_luma_max', 0.95))
    gain_min = float(eval_cfg.get('gain_clip_min', 0.5))
    gain_max = float(eval_cfg.get('gain_clip_max', 2.0))
    stat_mask = (
        (valid > 0)
        & (y_left >= valid_luma_min)
        & (y_left <= valid_luma_max)
        & (y_right >= valid_luma_min)
        & (y_right <= valid_luma_max)
    )
    valid_count = int(np.count_nonzero(stat_mask))
    report = {
        'median_left_before': None,
        'median_right_before': None,
        'ratio_before': None,
        'gain_right_to_left': 1.0,
        'median_left_after': None,
        'median_right_after': None,
        'ratio_after': None,
        'valid_pixel_count': valid_count,
        'skipped': True,
    }
    if valid_count < 100:
        return left_cost, right_cost, report

    med_left = float(np.median(y_left[stat_mask]))
    med_right = float(np.median(y_right[stat_mask]))
    if med_left <= 1e-6 or med_right <= 1e-6:
        report.update({
            'median_left_before': med_left,
            'median_right_before': med_right,
            'ratio_before': None,
        })
        return left_cost, right_cost, report

    gain = float(np.clip(med_left / med_right, gain_min, gain_max))
    right_float = right_cost.astype(np.float32) * gain
    right_cost = np.clip(right_float, 0, 255).astype(np.uint8)
    y_right_after = _luma(right_cost)
    med_left_after = float(np.median(y_left[stat_mask]))
    med_right_after = float(np.median(y_right_after[stat_mask]))
    report.update({
        'median_left_before': med_left,
        'median_right_before': med_right,
        'ratio_before': float(med_right / med_left),
        'gain_right_to_left': gain,
        'median_left_after': med_left_after,
        'median_right_after': med_right_after,
        'ratio_after': float(med_right_after / med_left_after) if med_left_after > 1e-6 else None,
        'skipped': False,
    })
    return left_cost, right_cost, report


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '06_overlap_evaluation')
    _clean_old_debug_outputs(out_dir)
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    overlap = cv2.imread(ctx['overlap_mask'], cv2.IMREAD_GRAYSCALE)
    ev = cfg['evaluation']

    left_cost_view = left.copy()
    right_cost_view = right.copy()
    component_reports = []
    enable_luma_norm = bool(ev.get('enable_luminance_normalization', False))
    for idx, c in enumerate(ctx.get('overlap_components', [])):
        x, y, w, h = c['x'], c['y'], c['w'], c['h']
        raw_l = left[y:y+h, x:x+w]
        raw_r = right[y:y+h, x:x+w]
        m_crop = overlap[y:y+h, x:x+w]
        write_image(out_dir / f'component_{idx:02d}_raw_pair.png', _side_by_side_labeled((
            ('left raw', raw_l),
            ('right raw', raw_r),
        )))
        before_energy = seam_energy_map(raw_l, raw_r, m_crop, ev.get('color_weight', 0.35), ev.get('edge_weight', 0.55), ev.get('gradient_weight', 0.10))

        if enable_luma_norm:
            cost_l, cost_r, brightness_report = _make_luminance_cost_views(raw_l, raw_r, m_crop, ev)
        else:
            cost_l, cost_r = raw_l.copy(), raw_r.copy()
            brightness_report = {
                'median_left_before': None,
                'median_right_before': None,
                'ratio_before': None,
                'gain_right_to_left': 1.0,
                'median_left_after': None,
                'median_right_after': None,
                'ratio_after': None,
                'valid_pixel_count': int(np.count_nonzero(m_crop)),
                'skipped': True,
                'reason': 'enable_luminance_normalization=false',
            }

        left_cost_view[y:y+h, x:x+w] = cost_l
        right_cost_view[y:y+h, x:x+w] = cost_r
        after_energy = seam_energy_map(cost_l, cost_r, m_crop, ev.get('color_weight', 0.35), ev.get('edge_weight', 0.55), ev.get('gradient_weight', 0.10))
        right_diff = np.clip(cv2.absdiff(raw_r, cost_r).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
        write_image(out_dir / f'component_{idx:02d}_right_luma_norm_compare.png', _side_by_side_labeled((
            ('right raw', raw_r),
            ('right cost view', cost_r),
            ('absdiff x3', right_diff),
        )))
        before_heat = heatmap_u8(before_energy)
        after_heat = heatmap_u8(after_energy)
        heat_diff = np.clip(cv2.absdiff(before_heat, after_heat).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
        write_image(out_dir / f'component_{idx:02d}_cost_before_after_compare.png', _side_by_side_labeled((
            ('cost before', before_heat),
            ('cost after', after_heat),
            ('absdiff x3', heat_diff),
        )))
        write_json(out_dir / f'component_{idx:02d}_brightness_report.json', brightness_report)
        component_reports.append({'component': idx, 'bbox': [x, y, w, h], 'brightness': brightness_report})

    energy = seam_energy_map(left_cost_view, right_cost_view, overlap, ev.get('color_weight', 0.35), ev.get('edge_weight', 0.55), ev.get('gradient_weight', 0.10))
    energy_npy = out_dir / 'seam_energy.npy'
    np.save(str(energy_npy), energy)
    write_image(out_dir / 'seam_energy_heatmap.png', heatmap_u8(energy))
    write_image(out_dir / 'edge_left.png', (edge_map(left_cost_view) * 255).astype(np.uint8))
    write_image(out_dir / 'edge_right.png', (edge_map(right_cost_view) * 255).astype(np.uint8))
    diff = cv2.absdiff(left_cost_view, right_cost_view)
    diff[overlap == 0] = 0
    write_image(out_dir / 'color_absdiff.png', diff)
    report = {
        'energy': summarize_values(energy, overlap),
        'overlap_valid_pixels': int(np.count_nonzero(overlap)),
        'overlap_valid_ratio': float(np.count_nonzero(overlap) / overlap.size),
        'enable_luminance_normalization': enable_luma_norm,
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
    report['brightness_components'] = component_reports
    write_json(out_dir / 'overlap_evaluation_report.json', report)
    ctx.update({
        'seam_energy_map': str(out_dir / 'seam_energy_heatmap.png'),
        'seam_energy_npy': str(energy_npy),
        'overlap_evaluation_report': str(out_dir / 'overlap_evaluation_report.json'),
    })
    return ctx
