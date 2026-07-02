from __future__ import annotations

from pathlib import Path
from typing import Dict

from common.io_utils import ensure_dir, read_image_bgr, write_image, write_json
from common.image_utils import (
    gray_world_white_balance,
    neutral_gray_white_balance,
    non_black_mask,
    percentile_contrast_stretch,
    percentile_luma_contrast_stretch,
)


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '01_input_preprocess')
    pp = cfg['preprocess']
    cdet = cfg['circle_detect']
    left = read_image_bgr(cfg['input']['left_path'])
    right = read_image_bgr(cfg['input']['right_path'])
    left_mask0 = non_black_mask(left, cdet.get('black_threshold', 12))
    right_mask0 = non_black_mask(right, cdet.get('black_threshold', 12))

    stats = {'left_original_shape': list(left.shape), 'right_original_shape': list(right.shape)}
    if pp.get('gray_world_white_balance', True):
        left, s1 = gray_world_white_balance(left, left_mask0)
        right, s2 = gray_world_white_balance(right, right_mask0)
        stats['left_white_balance'] = s1
        stats['right_white_balance'] = s2
    if pp.get('neutral_gray_white_balance', False):
        left, s1 = neutral_gray_white_balance(
            left,
            left_mask0,
            pp.get('neutral_sat_percentile', 35.0),
            pp.get('neutral_value_low_percentile', 8.0),
            pp.get('neutral_value_high_percentile', 96.0),
            pp.get('neutral_awb_strength', 1.0),
        )
        right, s2 = neutral_gray_white_balance(
            right,
            right_mask0,
            pp.get('neutral_sat_percentile', 35.0),
            pp.get('neutral_value_low_percentile', 8.0),
            pp.get('neutral_value_high_percentile', 96.0),
            pp.get('neutral_awb_strength', 1.0),
        )
        stats['left_neutral_white_balance'] = s1
        stats['right_neutral_white_balance'] = s2
    if pp.get('green_cast_correction', True):
        # Gray-world already absorbs the main green cast. Kept as an explicit stage marker.
        stats['green_cast_correction_note'] = 'Applied through masked gray-world and optional neutral-gray channel gains.'
    if pp.get('percentile_clip_high', 100.0) < 100.0 or pp.get('percentile_clip_low', 0.0) > 0.0:
        stretch_fn = percentile_luma_contrast_stretch if pp.get('contrast_preserve_chroma', True) else percentile_contrast_stretch
        left, cs1 = stretch_fn(left, left_mask0, pp.get('percentile_clip_low', 0.2), pp.get('percentile_clip_high', 99.8))
        right, cs2 = stretch_fn(right, right_mask0, pp.get('percentile_clip_low', 0.2), pp.get('percentile_clip_high', 99.8))
        stats['left_contrast_stretch'] = cs1
        stats['right_contrast_stretch'] = cs2

    write_image(out_dir / 'left_preprocessed.png', left)
    write_image(out_dir / 'right_preprocessed.png', right)
    write_image(out_dir / 'left_nonblack_mask.png', left_mask0)
    write_image(out_dir / 'right_nonblack_mask.png', right_mask0)
    write_json(out_dir / 'preprocess_report.json', stats)
    ctx.update({
        'left_preprocessed': str(out_dir / 'left_preprocessed.png'),
        'right_preprocessed': str(out_dir / 'right_preprocessed.png'),
        'left_nonblack_mask': str(out_dir / 'left_nonblack_mask.png'),
        'right_nonblack_mask': str(out_dir / 'right_nonblack_mask.png'),
        'preprocess_report': str(out_dir / 'preprocess_report.json')
    })
    return ctx
