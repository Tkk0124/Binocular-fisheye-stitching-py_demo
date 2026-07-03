from __future__ import annotations

from pathlib import Path
from typing import Dict

from common.io_utils import ensure_dir, write_json
from common.projection import build_radial_lut
from common.xlsx_lens_parser import parse_lens_xlsx


def run(ctx: Dict, cfg: Dict) -> Dict:
    out_dir = ensure_dir(Path(ctx['output_root']) / '03_radial_lut')
    xlsx_path = cfg.get('input', {}).get('lens_xlsx_path')
    parsed = parse_lens_xlsx(xlsx_path) if xlsx_path else {'found': False, 'notes': ['no xlsx path']}
    # Use average radius to keep left/right LUT scale consistent. Circle center remains per camera.
    avg_radius = 0.5 * (float(ctx['left_circle']['radius']) + float(ctx['right_circle']['radius']))
    lut = build_radial_lut(cfg['lens_model'], avg_radius, parsed)
    report = {
        'parsed_lens_xlsx': parsed,
        'average_effective_radius_px': avg_radius,
        'left_circle_raw_radius_px': float(ctx.get('left_circle_raw', ctx['left_circle'])['radius']),
        'right_circle_raw_radius_px': float(ctx.get('right_circle_raw', ctx['right_circle'])['radius']),
        'left_circle_effective_radius_px': float(ctx['left_circle']['radius']),
        'right_circle_effective_radius_px': float(ctx['right_circle']['radius']),
        'radial_lut_summary': {
            'source': lut['source'],
            'theta_max_deg': lut['theta_max_deg'],
            'first_radius_px': lut['radii_px'][0],
            'last_radius_px': lut['radii_px'][-1],
            'samples': len(lut['radii_px'])
        }
    }
    write_json(out_dir / 'radial_lut.json', lut)
    write_json(out_dir / 'radial_lut_report.json', report)
    ctx.update({'radial_lut': lut, 'radial_lut_json': str(out_dir / 'radial_lut.json'), 'radial_lut_report': str(out_dir / 'radial_lut_report.json')})
    return ctx
