from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.metrics import seam_energy_map, heatmap_u8


def _dp_vertical_seam(cost: np.ndarray, valid: np.ndarray, smooth_penalty: float, invalid_cost: float) -> np.ndarray:
    h, w = cost.shape
    c = cost.astype(np.float32).copy()
    c[valid == 0] = invalid_cost
    dp = np.full_like(c, invalid_cost, dtype=np.float32)
    back = np.zeros((h, w), dtype=np.int16)
    dp[0] = c[0]
    for y in range(1, h):
        prev = dp[y - 1]
        for x in range(w):
            best_x = x
            best_v = prev[x]
            if x > 0 and prev[x - 1] + smooth_penalty < best_v:
                best_v = prev[x - 1] + smooth_penalty
                best_x = x - 1
            if x + 1 < w and prev[x + 1] + smooth_penalty < best_v:
                best_v = prev[x + 1] + smooth_penalty
                best_x = x + 1
            dp[y, x] = c[y, x] + best_v
            back[y, x] = best_x - x
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(dp[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = seam[y + 1] + int(back[y + 1, seam[y + 1]])
    return seam


def _draw_seam_overlay(base: np.ndarray, seam: np.ndarray, x0: int, y0: int) -> np.ndarray:
    out = base.copy()
    for yy, sx in enumerate(seam):
        y = y0 + yy
        x = x0 + int(sx)
        if 0 <= y < out.shape[0] and 0 <= x < out.shape[1]:
            out[y, x] = (0, 0, 255)
            if x + 1 < out.shape[1]:
                out[y, x + 1] = (0, 0, 255)
    return out


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '08_seam_graphcut_dp')
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    vl = cv2.imread(ctx['left_sphere_valid_mask'], cv2.IMREAD_GRAYSCALE)
    vr = cv2.imread(ctx['right_sphere_valid_mask'], cv2.IMREAD_GRAYSCALE)
    overlap = cv2.bitwise_and(vl, vr)
    se = cfg['seam']
    ev = cfg['evaluation']
    energy_path = ctx.get('seam_energy_npy')
    if energy_path:
        energy = np.load(energy_path).astype(np.float32)
    else:
        energy = seam_energy_map(left, right, overlap, ev.get('color_weight', 0.35), ev.get('edge_weight', 0.55), ev.get('gradient_weight', 0.10))

    H, W = overlap.shape
    selector_left = np.zeros((H, W), dtype=np.uint8)
    selector_left[(vl > 0) & (vr == 0)] = 255
    selector_left[(vl > 0) & (vr > 0)] = 255  # initial; overwritten by seam components

    report = {'components': [], 'method': 'dp_vertical', 'energy_source': energy_path or 'module_08_legacy_recomputed'}
    overlay = cv2.addWeighted(left, 0.5, right, 0.5, 0)
    pad = int(se.get('component_padding_px', 6))
    smooth = float(se.get('smooth_penalty', 0.12))
    invalid_cost = float(se.get('invalid_cost', 1000000.0))
    for idx, c in enumerate(ctx.get('overlap_components', [])):
        x = max(0, c['x'] - pad)
        y = max(0, c['y'] - pad)
        w = min(W - x, c['w'] + 2 * pad)
        h = min(H - y, c['h'] + 2 * pad)
        e_crop = energy[y:y+h, x:x+w]
        valid_crop = overlap[y:y+h, x:x+w]
        if h < 10 or w < 10 or np.count_nonzero(valid_crop) < 100:
            continue
        seam = _dp_vertical_seam(e_crop, valid_crop, smooth, invalid_cost)
        seam_mask = np.zeros((h, w), dtype=np.uint8)
        for yy, sx in enumerate(seam):
            seam_mask[yy, :max(0, int(sx))] = 255
        # Label convention: if component is left side of panorama, left part should come from right camera.
        # if component center is right side, left part should come from left camera.
        region_left = selector_left[y:y+h, x:x+w]
        if c['cx'] < W / 2.0:
            # left side of seam: right camera => selector_left=0; right side: left camera => 255
            region_left[valid_crop > 0] = 255
            region_left[(valid_crop > 0) & (seam_mask > 0)] = 0
        else:
            # left side: left camera; right side: right camera.
            region_left[valid_crop > 0] = 0
            region_left[(valid_crop > 0) & (seam_mask > 0)] = 255
        selector_left[y:y+h, x:x+w] = region_left
        local_overlay = _draw_seam_overlay(overlay[y:y+h, x:x+w], seam, 0, 0)
        write_image(out_dir / f'component_{idx:02d}_seam_overlay.png', local_overlay)
        write_image(out_dir / f'component_{idx:02d}_energy_heatmap.png', heatmap_u8(e_crop))
        write_image(out_dir / f'component_{idx:02d}_seam_side_mask.png', seam_mask)
        report['components'].append({'component': idx, 'bbox': [x, y, w, h], 'seam_x_mean': float(np.mean(seam)), 'seam_x_min': int(seam.min()), 'seam_x_max': int(seam.max())})
        overlay = _draw_seam_overlay(overlay, seam, x, y)

    hard = np.where(selector_left[:, :, None] > 0, left, right)
    write_image(out_dir / 'global_seam_overlay.png', overlay)
    write_image(out_dir / 'selector_left_mask.png', selector_left)
    write_image(out_dir / 'hard_composite.png', hard)
    write_json(out_dir / 'seam_report.json', report)
    ctx.update({'selector_left_mask': str(out_dir / 'selector_left_mask.png'), 'hard_composite': str(out_dir / 'hard_composite.png'), 'seam_report': str(out_dir / 'seam_report.json')})
    return ctx
