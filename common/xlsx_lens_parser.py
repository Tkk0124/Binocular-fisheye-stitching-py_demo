from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _read_shared_strings(z: zipfile.ZipFile) -> List[str]:
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    root = ET.fromstring(z.read('xl/sharedStrings.xml'))
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    out = []
    for si in root.findall('a:si', ns):
        texts = []
        for t in si.findall('.//a:t', ns):
            texts.append(t.text or '')
        out.append(''.join(texts))
    return out


def _cell_value(c: ET.Element, shared: List[str]) -> str | float | None:
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    t = c.attrib.get('t')
    v = c.find('a:v', ns)
    if v is None or v.text is None:
        inline = c.find('.//a:t', ns)
        return inline.text if inline is not None else None
    txt = v.text
    if t == 's':
        idx = int(txt)
        return shared[idx] if 0 <= idx < len(shared) else None
    try:
        return float(txt)
    except Exception:
        return txt


def _col_index(ref: str) -> int:
    letters = ''.join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord('A') + 1)
    return n - 1


def _sheet_rows(z: zipfile.ZipFile, sheet_path: str, shared: List[str]) -> List[List[object]]:
    root = ET.fromstring(z.read(sheet_path))
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    rows = []
    for row in root.findall('.//a:sheetData/a:row', ns):
        vals: Dict[int, object] = {}
        max_col = -1
        for c in row.findall('a:c', ns):
            ref = c.attrib.get('r', '')
            col = _col_index(ref)
            max_col = max(max_col, col)
            vals[col] = _cell_value(c, shared)
        if max_col >= 0:
            rows.append([vals.get(i) for i in range(max_col + 1)])
    return rows


def parse_lens_xlsx(path: str | Path) -> Dict[str, object]:
    """Best-effort parser for optical xlsx.

    It extracts obvious numeric lens facts and tries to find an Angle ↔ Image Height table.
    It intentionally uses only stdlib so the pipeline has no hard spreadsheet dependency.
    """
    p = Path(path)
    result: Dict[str, object] = {
        'source': str(p),
        'found': False,
        'angle_height_table': [],
        'facts': {},
        'notes': []
    }
    if not p.exists():
        result['notes'].append('xlsx not found')
        return result
    try:
        with zipfile.ZipFile(p, 'r') as z:
            shared = _read_shared_strings(z)
            sheet_paths = sorted([n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml', n)])
            all_rows: List[List[object]] = []
            for sp in sheet_paths:
                all_rows.extend(_sheet_rows(z, sp, shared))
    except Exception as e:
        result['notes'].append(f'xlsx parse failed: {e}')
        return result

    result['found'] = True
    text_rows = []
    for row in all_rows:
        joined = ' '.join('' if v is None else str(v) for v in row)
        text_rows.append(joined)
        low = joined.lower()
        nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', joined)]
        if 'fov' in low and nums:
            result['facts'].setdefault('fov_candidates', []).extend(nums)
        if ('pixel' in low or '像元' in joined or '像素' in joined) and nums:
            result['facts'].setdefault('pixel_candidates', []).extend(nums)
        if ('image height' in low or '像高' in joined or 'height' in low) and nums:
            result['facts'].setdefault('height_candidates', []).extend(nums)
        if ('efl' in low or 'focal' in low or '焦距' in joined) and nums:
            result['facts'].setdefault('efl_candidates', []).extend(nums)

    # Best-effort table extraction: adjacent numeric columns with monotonic values.
    numeric_rows = []
    for row in all_rows:
        nums = []
        for v in row:
            if isinstance(v, (int, float)):
                nums.append(float(v))
            elif isinstance(v, str):
                try:
                    nums.append(float(v.strip()))
                except Exception:
                    pass
        if len(nums) >= 2:
            numeric_rows.append(nums)

    pairs: List[Tuple[float, float]] = []
    for nums in numeric_rows:
        # Common table can be angle, real height, f-theta distortion, relative illum, etc.
        for i in range(len(nums) - 1):
            a, h = nums[i], nums[i + 1]
            if 0.0 <= a <= 120.0 and 0.0 <= h <= 5.0:
                pairs.append((a, h))
    # Deduplicate and prefer increasing height by angle.
    uniq = {}
    for a, h in pairs:
        uniq[round(a, 4)] = h
    table = sorted([(float(a), float(h)) for a, h in uniq.items()], key=lambda x: x[0])
    filtered = []
    last_h = -1.0
    for a, h in table:
        if h + 1e-6 >= last_h:
            filtered.append((a, h))
            last_h = h
    if len(filtered) >= 5:
        result['angle_height_table'] = [{'angle_deg': a, 'height_mm': h} for a, h in filtered]
    else:
        result['notes'].append('no reliable angle-height table found; fallback radial model should be used')
    return result
