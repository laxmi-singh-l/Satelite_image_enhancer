"""Paired LR/HR satellite dataset for super-resolution training.

Supports two data layouts:

1. **Paired** — a manifest CSV with columns ``lr,hr`` referencing exact
   alignments where ``hr`` is a ``scale`` multiple of ``lr``. Aligned random
   patches are cropped jointly.
2. **HR-only + downsampling** — a manifest CSV with only an ``hr`` column (or
   ``--images-dir``), where the LR counterpart is synthesised on the fly with
   OpenCV ``INTER_AREA`` downsampling (the common SatSR degradation model).

Bands are preserved as-is: a 4-band RGBNIR GeoTIFF stays a 4-channel tensor.
Inputs are normalised to ``[0, 1]`` floats (``uint8`` values are scaled).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_image(path: str | None) -> np.ndarray | None:
    if path is None:
        return None

    suffix = Path(path).suffix.lower()
    if suffix in ('.tif', '.tiff'):
        try:
            import rasterio
            with rasterio.open(path) as src:
                img = src.read()  # (bands, H, W), band order preserved
            img = np.moveaxis(img, 0, -1)  # (H, W, bands)
            return img
        except (ImportError, Exception):
            pass

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Failed to load image: {path}')
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img


def _to_float(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.uint16):
        img = arr.astype(np.float32) / 65535.0
    elif np.issubdtype(arr.dtype, np.uint8):
        img = arr.astype(np.float32) / 255.0
    else:
        img = arr.astype(np.float32)
        if img.size and img.max() > 1.5:
            img = img / img.max()
    return np.clip(img, 0.0, 1.0)


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    if x.ndim == 2:
        x = x[..., np.newaxis]
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class PairedSuperResolutionDataset(Dataset):
    def __init__(self, samples, scale=4, hr_patch_size=256, generate_lr=False,
                 crop='random', augment=False, seed=None):
        """``samples``: list of ``{'lr': path|None, 'hr': path}``."""
        super().__init__()
        self.samples = samples
        self.scale = scale
        self.hr_patch_size = hr_patch_size
        self.generate_lr = generate_lr
        self.crop = crop
        self.augment = augment
        self.rng = random.Random(seed)
        first_hr = _load_image(samples[0]['hr'])
        self.n_channels = first_hr.shape[2] if first_hr.ndim == 3 else 1

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _same_augment(lr, hr, rng) -> tuple[np.ndarray, np.ndarray]:
        k = rng.choice([0, 1, 2, 3])
        if k:
            lr = np.rot90(lr, k)
            hr = np.rot90(hr, k)
        if rng.random() < 0.5:
            lr = lr[:, ::-1]
            hr = hr[:, ::-1]
        return lr, hr

    def _crop_and_derive_lr(self, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.scale
        h, w = hr.shape[:2]
        ph = min(self.hr_patch_size, (h // scale) * scale)
        pw = min(self.hr_patch_size, (w // scale) * scale)
        y = self.rng.randint(0, h - ph) if self.crop == 'random' else 0
        x = self.rng.randint(0, w - pw) if self.crop == 'random' else 0
        hr_crop = hr[y:y + ph, x:x + pw]
        lr_crop = cv2.resize(
            hr_crop,
            (pw // scale, ph // scale),
            interpolation=cv2.INTER_AREA,
        )
        return lr_crop, hr_crop

    def _aligned_pair(self, lr: np.ndarray, hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.scale
        lr_h, lr_w = lr.shape[:2]
        lph = min(self.hr_patch_size // scale, lr_h)
        lpw = min(self.hr_patch_size // scale, lr_w)
        ly = self.rng.randint(0, lr_h - lph) if self.crop == 'random' else 0
        lx = self.rng.randint(0, lr_w - lpw) if self.crop == 'random' else 0
        lr_crop = lr[ly:ly + lph, lx:lx + lpw]
        hr_crop = hr[ly * scale:(ly + lph) * scale, lx * scale:(lx + lpw) * scale]
        return lr_crop, hr_crop

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        lr_path, hr_path = sample.get('lr'), sample.get('hr')
        lr = _load_image(lr_path)
        hr = _load_image(hr_path)

        if lr is None or self.generate_lr:
            lr_crop, hr_crop = self._crop_and_derive_lr(hr)
        else:
            lr_crop, hr_crop = self._aligned_pair(lr, hr)

        if self.augment:
            lr_crop, hr_crop = self._same_augment(lr_crop, hr_crop, self.rng)

        return _to_tensor(_to_float(lr_crop)), _to_tensor(_to_float(hr_crop))


def _resolve_path(root: str | None, p: str) -> str:
    return str((Path(root) / p).resolve()) if root and not Path(p).is_absolute() else p


def read_manifest(manifest: str, root: str | None = None) -> list[dict]:
    """Read a CSV with ``lr,hr`` columns (either may be empty for HR-only)."""
    samples = []
    with open(manifest, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lr = (row.get('lr') or '').strip()
            hr = (row.get('hr') or '').strip()
            if not hr:
                continue
            samples.append({
                'lr': _resolve_path(root, lr) if lr else None,
                'hr': _resolve_path(root, hr),
            })
    if not samples:
        raise ValueError(f'No valid rows in manifest: {manifest}')
    return samples


def folder_manifest(img_dir: str, extensions=('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
    """Build HR-only samples from a directory of high-resolution images."""
    files = sorted(
        p for p in Path(img_dir).iterdir()
        if p.suffix.lower() in extensions and not p.name.startswith('.')
    )
    if not files:
        raise ValueError(f'No images found in: {img_dir}')
    return [{'lr': None, 'hr': str(p)} for p in files]


def split_sources(samples: list[dict], split=(0.7, 0.15, 0.15),
                  seed=13) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministic stratified-free random split into train/val/test lists."""
    shuffled = list(samples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_tr = round(n * split[0])
    n_va = round(n * split[1])
    return shuffled[:n_tr], shuffled[n_tr:n_tr + n_va], shuffled[n_tr + n_va:]


def build_sr_datasets(manifest: str | None, images_dir: str | None = None,
                      root: str | None = None, scale=4, hr_patch_size=256,
                      augment=True, split=(0.7, 0.15, 0.15),
                      shuffle_seed=13) -> tuple[Dataset, Dataset, Dataset, int]:
    """Build train/val/test datasets plus the detected band count."""
    if manifest:
        samples = read_manifest(manifest, root)
    elif images_dir:
        samples = folder_manifest(images_dir)
    else:
        raise ValueError('Provide either --manifest or --images-dir')

    train_src, val_src, test_src = split_sources(samples, split, shuffle_seed)
    train_ds = PairedSuperResolutionDataset(
        train_src, scale=scale, hr_patch_size=hr_patch_size,
        generate_lr=manifest is None, augment=augment, seed=shuffle_seed,
    )
    val_ds = PairedSuperResolutionDataset(
        val_src, scale=scale, hr_patch_size=hr_patch_size,
        generate_lr=manifest is None, augment=False,
    )
    test_ds = PairedSuperResolutionDataset(
        test_src, scale=scale, hr_patch_size=hr_patch_size,
        generate_lr=manifest is None, crop='full', augment=False,
    )
    return train_ds, val_ds, test_ds, train_ds.n_channels