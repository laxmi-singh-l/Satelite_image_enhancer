"""Georeferenced satellite image I/O.

Read/write multi-band GeoTIFFs while preserving CRS and geotransform metadata.
On SR enhancement the pixel footprint shrinks by the scale factor, so the
output geotransform is scaled accordingly (geospatial consistency requirement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _cvt_bands(img: np.ndarray) -> np.ndarray:
    """Normalise cv2-style (H, W, BGR) to (H, W, channels)."""
    if img.ndim == 3:
        if img.shape[2] == 3:
            return img[..., ::-1].copy()  # BGR -> RGB
        if img.shape[2] == 4:
            return img[..., [2, 1, 0, 3]]  # BGRA -> RGBA
    return img


def _read_cv2(path: str) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Failed to load image: {path}')
    return _cvt_bands(img)


def read_geo_image(path: str) -> tuple[np.ndarray, dict | None]:
    """Read an image array ``(H, W, C)`` and optional rasterio geospatial meta."""
    p = Path(path)
    meta = None
    if p.suffix.lower() in ('.tif', '.tiff'):
        try:
            import rasterio
            with rasterio.open(p) as src:
                array = src.read()  # (C, H, W), band order preserved
                meta = {
                    'driver': src.driver,
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'nodata': src.nodata,
                }
            return np.moveaxis(array, 0, -1), meta
        except (ImportError, Exception):
            meta = None

    return _read_cv2(path), meta


def write_geo_image(path: str, array: np.ndarray, meta: dict | None,
                    scale: int = 1, dtype: str = 'uint8') -> None:
    """Write an array ``(H, W)`` or ``(H, W, C)`` to GeoTIFF (or PNG fallback).

    ``meta`` is the input rasterio metadata dict from ``read_geo_image``. The
    output geotransform is scaled by ``scale`` so geo-coordinates line up with
    the higher-resolution grid. If no meta is available a plain PNG/TIFF is
    written via OpenCV.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3:
        raise ValueError('array must be (H, W) or (H, W, C)')

    if meta is not None:
        try:
            import rasterio
            from rasterio.transform import Affine
            h, w, c = array.shape
            t = meta['transform']
            profile = {
                'driver': 'GTiff',
                'height': h,
                'width': w,
                'count': c,
                'dtype': dtype,
                'crs': meta.get('crs'),
                'nodata': meta.get('nodata'),
                'transform': Affine(
                    t.a / scale, t.b, t.c,
                    t.d, t.e / scale, t.f,
                ),
            }
            with rasterio.open(path, 'w', **profile) as dst:
                for i in range(c):
                    dst.write(array[..., i].astype(dtype), i + 1)
            return
        except (ImportError, Exception):
            pass

    import cv2
    c = array.shape[2]
    if c == 3:
        save_arr = array[..., ::-1].copy()  # RGB -> BGR
        cv2.imwrite(str(path), save_arr)
    elif c == 1:
        cv2.imwrite(str(path), array[..., 0])
    else:
        cv2.imwrite(str(path), array[..., :3][..., ::-1].copy())


def pixel_resolution(meta: dict | None, default_m: float = 10.0) -> float:
    """Ground-sample distance in meters from geotransform meta."""
    if meta and meta.get('transform') is not None:
        a = meta['transform'].a
        return float(abs(a)) if a else default_m
    return default_m