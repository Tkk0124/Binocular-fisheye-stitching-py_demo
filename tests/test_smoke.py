from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_smoke_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    left = Path('/mnt/data/left.bmp')
    right = Path('/mnt/data/right.bmp')
    if not left.exists() or not right.exists():
        return
    out = tmp_path / 'smoke_output'
    cmd = [
        sys.executable, str(root / 'run_pipeline.py'),
        '--left', str(left),
        '--right', str(right),
        '--output', str(out),
        '--pano-width', '512',
        '--pano-height', '256',
    ]
    subprocess.check_call(cmd)
    summary = json.loads((out / 'pipeline_summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'ok'
    assert Path(summary['final_pano']).exists()
