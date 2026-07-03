from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.projection import make_fisheye_remap, remap_to_sphere, yaw_pitch_roll_matrix


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '04_sphere_projection')
    left = read_image_bgr(ctx['left_preprocessed'])
    right = read_image_bgr(ctx['right_preprocessed'])
    p = cfg['projection']
    pose = cfg['pose']
    W, H = int(p['pano_width']), int(p['pano_height'])
    R_left = yaw_pitch_roll_matrix(pose['left_yaw_deg'], pose['left_pitch_deg'], pose['left_roll_deg'])
    R_right = yaw_pitch_roll_matrix(pose['right_yaw_deg'], pose['right_pitch_deg'], pose['right_roll_deg'])
    mx_l, my_l, valid_l = make_fisheye_remap(W, H, ctx['left_circle'], ctx['radial_lut'], R_left, p.get('image_y_sign', -1))
    mx_r, my_r, valid_r = make_fisheye_remap(W, H, ctx['right_circle'], ctx['radial_lut'], R_right, p.get('image_y_sign', -1))
    sphere_l = remap_to_sphere(left, mx_l, my_l, valid_l, p.get('interpolation', 'linear'))
    sphere_r = remap_to_sphere(right, mx_r, my_r, valid_r, p.get('interpolation', 'linear'))
    write_image(out_dir / 'left_sphere.png', sphere_l)
    write_image(out_dir / 'right_sphere.png', sphere_r)
    write_image(out_dir / 'left_sphere_valid_mask.png', valid_l)
    write_image(out_dir / 'right_sphere_valid_mask.png', valid_r)
    both = cv2.bitwise_and(valid_l, valid_r)
    write_image(out_dir / 'overlap_mask.png', both)
    overlay = cv2.addWeighted(sphere_l, 0.5, sphere_r, 0.5, 0)
    overlay[both == 0] = (overlay[both == 0] * 0.35).astype(np.uint8)
    write_image(out_dir / 'sphere_overlap_overlay.png', overlay)
    report = {
        'pano_width': W,
        'pano_height': H,
        'left_circle_raw': ctx.get('left_circle_raw'),
        'right_circle_raw': ctx.get('right_circle_raw'),
        'left_circle_effective': ctx['left_circle'],
        'right_circle_effective': ctx['right_circle'],
        'left_projection_radius_px': float(ctx['left_circle']['radius']),
        'right_projection_radius_px': float(ctx['right_circle']['radius']),
        'left_valid_ratio': float(np.count_nonzero(valid_l) / valid_l.size),
        'right_valid_ratio': float(np.count_nonzero(valid_r) / valid_r.size),
        'overlap_ratio': float(np.count_nonzero(both) / both.size),
        'pose': pose
    }
    write_json(out_dir / 'projection_report.json', report)
    ctx.update({
        'left_sphere': str(out_dir / 'left_sphere.png'),
        'right_sphere': str(out_dir / 'right_sphere.png'),
        'left_sphere_valid_mask': str(out_dir / 'left_sphere_valid_mask.png'),
        'right_sphere_valid_mask': str(out_dir / 'right_sphere_valid_mask.png'),
        'overlap_mask': str(out_dir / 'overlap_mask.png'),
        'projection_report': str(out_dir / 'projection_report.json')
    })
    return ctx
