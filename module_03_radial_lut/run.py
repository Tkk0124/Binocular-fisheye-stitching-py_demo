from __future__ import annotations

from pathlib import Path
from typing import Dict

from common.io_utils import ensure_dir, write_json
from common.projection import build_radial_lut, radius_to_theta_deg
from common.xlsx_lens_parser import parse_lens_xlsx


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '03_radial_lut')
    xlsx_path = cfg.get('input', {}).get('lens_xlsx_path')
    parsed = parse_lens_xlsx(xlsx_path) if xlsx_path else {'found': False, 'notes': ['no xlsx path']}

    left_raw = ctx.get('left_circle_raw', ctx['left_circle'])
    right_raw = ctx.get('right_circle_raw', ctx['right_circle'])
    avg_raw_radius = 0.5 * (float(left_raw['radius']) + float(right_raw['radius']))
    avg_effective_radius = 0.5 * (float(ctx['left_circle']['radius']) + float(ctx['right_circle']['radius']))

    # Build the complete theta <-> radius LUT from the raw image circle.
    # The effective radius is then converted back to an effective theta/FOV through the LUT.
    lut = build_radial_lut(cfg['lens_model'], avg_raw_radius, parsed)
    full_theta_max_deg = float(lut['theta_max_deg'])
    effective_theta_max_deg = radius_to_theta_deg(avg_effective_radius, lut)
    effective_fov_deg = 2.0 * effective_theta_max_deg

    lut.update({
        'full_theta_max_deg': full_theta_max_deg,
        'full_fov_deg': 2.0 * full_theta_max_deg,
        'raw_radius_px': avg_raw_radius,
        'effective_radius_px': avg_effective_radius,
        'effective_theta_max_deg': effective_theta_max_deg,
        'effective_fov_deg': effective_fov_deg,
    })

    report = {
        'parsed_lens_xlsx': parsed,
        'average_raw_radius_px': avg_raw_radius,
        'average_effective_radius_px': avg_effective_radius,
        'left_circle_raw_radius_px': float(left_raw['radius']),
        'right_circle_raw_radius_px': float(right_raw['radius']),
        'left_circle_effective_radius_px': float(ctx['left_circle']['radius']),
        'right_circle_effective_radius_px': float(ctx['right_circle']['radius']),
        'radial_lut_summary': {
            'source': lut['source'],
            'full_theta_max_deg': lut['full_theta_max_deg'],
            'full_fov_deg': lut['full_fov_deg'],
            'effective_theta_max_deg': lut['effective_theta_max_deg'],
            'effective_fov_deg': lut['effective_fov_deg'],
            'first_radius_px': lut['radii_px'][0],
            'last_radius_px': lut['radii_px'][-1],
            'samples': len(lut['radii_px'])
        }
    }
    write_json(out_dir / 'radial_lut.json', lut)
    write_json(out_dir / 'radial_lut_report.json', report)
    ctx.update({'radial_lut': lut, 'radial_lut_json': str(out_dir / 'radial_lut.json'), 'radial_lut_report': str(out_dir / 'radial_lut_report.json')})
    return ctx
