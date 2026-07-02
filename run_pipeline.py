from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

# Allow running directly from the package root.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.io_utils import clean_dir, ensure_dir, load_config, write_json

from module_01_input_preprocess.run import run as run_01
from module_02_circle_detect.run import run as run_02
from module_03_radial_lut.run import run as run_03
from module_04_sphere_projection.run import run as run_04
from module_05_pose_overlap.run import run as run_05
from module_06_overlap_evaluation.run import run as run_06
from module_07_pattern_evaluation.run import run as run_07
from module_08_seam_graphcut_dp.run import run as run_08
from module_09_blending.run import run as run_09
from module_10_local_compensation_fallback.run import run as run_10

STAGES = [
    ("01_input_preprocess", run_01),
    ("02_circle_detect", run_02),
    ("03_radial_lut", run_03),
    ("04_sphere_projection", run_04),
    ("05_pose_overlap", run_05),
    ("06_overlap_evaluation", run_06),
    ("07_pattern_evaluation", run_07),
    ("08_seam_graphcut_dp", run_08),
    ("09_blending", run_09),
    ("10_local_compensation_fallback", run_10),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Dual-fisheye sphere projection + overlap seam pipeline")
    ap.add_argument("--config", default=str(ROOT / "config" / "default_config.json"), help="Base config JSON")
    ap.add_argument("--left", dest="left_path", default=None, help="Left fisheye BMP/PNG path")
    ap.add_argument("--right", dest="right_path", default=None, help="Right fisheye BMP/PNG path")
    ap.add_argument("--lens-xlsx", dest="lens_xlsx_path", default=None, help="Optional optical/lens xlsx path")
    ap.add_argument("--output", dest="output_root", default=None, help="Output root")
    ap.add_argument("--pano-width", type=int, default=None, help="Equirectangular output width")
    ap.add_argument("--pano-height", type=int, default=None, help="Equirectangular output height")
    ap.add_argument("--right-yaw", type=float, default=None, help="Right camera yaw in degrees; default 180")
    ap.add_argument("--left-yaw", type=float, default=None, help="Left camera yaw in degrees; default 0")
    ap.add_argument("--enable-local-fallback", action="store_true", help="Enable diagnostic local shift fallback flag")
    return ap.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict:
    o: Dict = {}
    if args.left_path:
        o.setdefault('input', {})['left_path'] = args.left_path
    if args.right_path:
        o.setdefault('input', {})['right_path'] = args.right_path
    if args.lens_xlsx_path:
        o.setdefault('input', {})['lens_xlsx_path'] = args.lens_xlsx_path
    if args.output_root:
        o.setdefault('runtime', {})['output_root'] = args.output_root
    if args.pano_width:
        o.setdefault('projection', {})['pano_width'] = args.pano_width
    if args.pano_height:
        o.setdefault('projection', {})['pano_height'] = args.pano_height
    if args.right_yaw is not None:
        o.setdefault('pose', {})['right_yaw_deg'] = args.right_yaw
    if args.left_yaw is not None:
        o.setdefault('pose', {})['left_yaw_deg'] = args.left_yaw
    if args.enable_local_fallback:
        o.setdefault('fallback_local_compensation', {})['enabled'] = True
    return o


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config, build_overrides(args))
    out_root = Path(cfg['runtime']['output_root'])
    if cfg['runtime'].get('overwrite', True):
        clean_dir(out_root)
    else:
        ensure_dir(out_root)
    # Save effective config first for traceability.
    write_json(out_root / 'effective_config.json', cfg)
    ctx: Dict = {'output_root': str(out_root), 'package_root': str(ROOT)}
    for name, fn in STAGES:
        print(f"[RUN] {name}")
        ctx = fn(ctx, cfg)
        write_json(out_root / 'context_latest.json', ctx)
    summary = {
        'status': 'ok',
        'final_pano': ctx.get('final_pano'),
        'output_root': str(out_root),
        'stage_reports': {
            'preprocess': ctx.get('preprocess_report'),
            'circle': ctx.get('circle_report'),
            'projection': ctx.get('projection_report'),
            'overlap': ctx.get('pose_overlap_report'),
            'overlap_evaluation': ctx.get('overlap_evaluation_report'),
            'pattern': ctx.get('pattern_evaluation_report'),
            'seam': ctx.get('seam_report'),
            'blend': ctx.get('blend_report'),
            'local_compensation': ctx.get('local_compensation_report')
        }
    }
    write_json(out_root / 'pipeline_summary.json', summary)
    print(f"[OK] final pano: {summary['final_pano']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
