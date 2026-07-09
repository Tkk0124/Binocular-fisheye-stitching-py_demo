from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def yaw_pitch_roll_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    # yaw around global Y, pitch around X, roll around Z.
    return (rot_y(yaw) @ rot_x(pitch) @ rot_z(roll)).astype(np.float32)


def build_radial_lut(lens_cfg: Dict, circle_radius_px: float, parsed_xlsx: Dict | None = None) -> Dict[str, object]:
    fov = float(lens_cfg.get('diagonal_fov_deg', 210.0))
    theta_max = fov * 0.5
    samples = int(lens_cfg.get('lut_samples', 1024))
    prefer_xlsx = bool(lens_cfg.get('prefer_xlsx_lut', True))

    src = 'fallback_equidistant'
    angles = np.linspace(0, theta_max, samples, dtype=np.float32)
    radii = (angles / max(theta_max, 1e-6) * float(circle_radius_px)).astype(np.float32)

    if prefer_xlsx and parsed_xlsx:
        table = parsed_xlsx.get('angle_height_table') or []
        if len(table) >= 5:
            tab_a = np.array([float(r['angle_deg']) for r in table], dtype=np.float32)
            tab_h = np.array([float(r['height_mm']) for r in table], dtype=np.float32)
            # Reject accidental numeric pairs from non-optical tables. A usable fisheye angle-height
            # table must cover a large part of the declared half-FOV and remain physically plausible.
            if float(tab_a.max()) >= max(60.0, theta_max * 0.65) and 0.1 <= float(tab_h.max()) <= 5.5:
                pitch_um = float(lens_cfg.get('effective_pixel_pitch_um', 1.6))
                pitch_mm = pitch_um * 1e-3
                tab_r = tab_h / max(pitch_mm, 1e-9)
                # Normalize the complete lens table to the detected raw circle radius.
                # Effective-radius trimming is applied later by inverse-looking up theta from this LUT.
                if tab_r.max() > 0:
                    tab_r = tab_r / tab_r.max() * float(circle_radius_px)
                max_a = float(min(theta_max, tab_a.max()))
                angles = np.linspace(0, max_a, samples, dtype=np.float32)
                radii = np.interp(angles, tab_a, tab_r).astype(np.float32)
                theta_max = max_a
                src = 'xlsx_angle_height_normalized'

    return {
        'source': src,
        'theta_max_deg': float(theta_max),
        'angles_deg': angles.tolist(),
        'radii_px': radii.tolist(),
    }


def radius_to_theta_deg(radius_px: float, lut: Dict[str, object]) -> float:
    angles = np.asarray(lut['angles_deg'], dtype=np.float32)
    radii = np.asarray(lut['radii_px'], dtype=np.float32)
    if len(angles) == 0 or len(radii) == 0 or len(angles) != len(radii):
        raise ValueError('invalid radial LUT')
    return float(np.interp(float(radius_px), radii, angles))


def _interp_radius(theta_deg: np.ndarray, lut: Dict[str, object]) -> np.ndarray:
    a = np.asarray(lut['angles_deg'], dtype=np.float32)
    r = np.asarray(lut['radii_px'], dtype=np.float32)
    return np.interp(theta_deg, a, r).astype(np.float32)


def equirectangular_rays(width: int, height: int) -> np.ndarray:
    xs = (np.arange(width, dtype=np.float32) + 0.5) / width
    ys = (np.arange(height, dtype=np.float32) + 0.5) / height
    lon = (xs - 0.5) * (2.0 * np.pi)
    lat = (0.5 - ys) * np.pi
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    cos_lat = np.cos(lat_grid)
    ray = np.stack([
        cos_lat * np.sin(lon_grid),
        np.sin(lat_grid),
        cos_lat * np.cos(lon_grid),
    ], axis=-1).astype(np.float32)
    return ray


def make_fisheye_remap(width: int, height: int, circle: Dict[str, float], radial_lut: Dict[str, object], pose_cam_to_global: np.ndarray, image_y_sign: int = -1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ray_global = equirectangular_rays(width, height).reshape(-1, 3)
    ray_cam = (pose_cam_to_global.T @ ray_global.T).T
    z = np.clip(ray_cam[:, 2], -1.0, 1.0)
    theta = np.degrees(np.arccos(z)).astype(np.float32)
    theta_max = float(radial_lut.get('effective_theta_max_deg', radial_lut['theta_max_deg']))
    valid = theta <= theta_max
    r = _interp_radius(theta, radial_lut)
    xy_norm = np.sqrt(np.maximum(ray_cam[:, 0] ** 2 + ray_cam[:, 1] ** 2, 1e-12))
    cos_a = ray_cam[:, 0] / xy_norm
    sin_a = ray_cam[:, 1] / xy_norm
    map_x = float(circle['cx']) + r * cos_a
    map_y = float(circle['cy']) + (image_y_sign * r * sin_a)
    valid &= np.isfinite(map_x) & np.isfinite(map_y)
    map_x = map_x.reshape(height, width).astype(np.float32)
    map_y = map_y.reshape(height, width).astype(np.float32)
    valid = valid.reshape(height, width).astype(np.uint8) * 255
    return map_x, map_y, valid


def remap_to_sphere(img_bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray, valid: np.ndarray, interpolation: str = 'linear') -> np.ndarray:
    interp = cv2.INTER_LINEAR if interpolation == 'linear' else cv2.INTER_NEAREST
    out = cv2.remap(img_bgr, map_x, map_y, interp, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    out[valid == 0] = 0
    return out
