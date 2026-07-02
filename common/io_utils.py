from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_dir(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_image_bgr(path: str | Path) -> np.ndarray:
    path = str(path)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def write_image(path: str | Path, img: np.ndarray) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if img.dtype == np.float32 or img.dtype == np.float64:
        out = np.clip(img, 0.0, 1.0)
        out = (out * 255.0 + 0.5).astype(np.uint8)
    else:
        out = img
    ok = cv2.imwrite(str(p), out)
    if not ok:
        raise IOError(f"Failed to write image: {p}")


def relpath(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except Exception:
        return str(path)


def load_config(base_config: str | Path, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = read_json(base_config)
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst
