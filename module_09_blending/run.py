from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json


def _make_alpha(selector: np.ndarray, feather_radius: int) -> np.ndarray:
    alpha = (selector.astype(np.float32) / 255.0)
    if feather_radius > 0:
        k = max(3, int(feather_radius) * 2 + 1)
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return np.clip(alpha, 0, 1)


def _multiband_blend(left: np.ndarray, right: np.ndarray, alpha: np.ndarray, levels: int) -> np.ndarray:
    left_f = left.astype(np.float32) / 255.0
    right_f = right.astype(np.float32) / 255.0
    alpha_f = alpha.astype(np.float32)
    gp_a = [alpha_f]
    gp_l = [left_f]
    gp_r = [right_f]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_l.append(cv2.pyrDown(gp_l[-1]))
        gp_r.append(cv2.pyrDown(gp_r[-1]))
    lp_l = []
    lp_r = []
    for i in range(levels):
        size = (gp_l[i].shape[1], gp_l[i].shape[0])
        lp_l.append(gp_l[i] - cv2.pyrUp(gp_l[i + 1], dstsize=size))
        lp_r.append(gp_r[i] - cv2.pyrUp(gp_r[i + 1], dstsize=size))
    lp_l.append(gp_l[-1])
    lp_r.append(gp_r[-1])
    blended = []
    for ll, rr, aa in zip(lp_l, lp_r, gp_a):
        if aa.ndim == 2:
            aa = aa[:, :, None]
        blended.append(ll * aa + rr * (1 - aa))
    out = blended[-1]
    for i in range(levels - 1, -1, -1):
        size = (blended[i].shape[1], blended[i].shape[0])
        out = cv2.pyrUp(out, dstsize=size) + blended[i]
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '09_blending')
    left = read_image_bgr(ctx['left_sphere'])
    right = read_image_bgr(ctx['right_sphere'])
    selector = cv2.imread(ctx['selector_left_mask'], cv2.IMREAD_GRAYSCALE)
    bcfg = cfg['blend']
    alpha = _make_alpha(selector, int(bcfg.get('feather_radius_px', 24)))
    feather = np.clip(left.astype(np.float32) * alpha[:, :, None] + right.astype(np.float32) * (1 - alpha[:, :, None]), 0, 255).astype(np.uint8)
    write_image(out_dir / 'alpha_left_feathered.png', (alpha * 255).astype(np.uint8))
    write_image(out_dir / 'final_feather.png', feather)
    if bcfg.get('method', 'multiband') == 'multiband':
        final = _multiband_blend(left, right, alpha, int(bcfg.get('multiband_levels', 5)))
        write_image(out_dir / 'final_multiband.png', final)
        final_path = out_dir / 'final_multiband.png'
    else:
        final = feather
        final_path = out_dir / 'final_feather.png'
    report = {'method': bcfg.get('method', 'multiband'), 'final_image': str(final_path), 'feather_radius_px': int(bcfg.get('feather_radius_px', 24)), 'multiband_levels': int(bcfg.get('multiband_levels', 5))}
    write_json(out_dir / 'blend_report.json', report)
    ctx.update({'final_pano': str(final_path), 'blend_report': str(out_dir / 'blend_report.json')})
    return ctx
