import numpy as np
# from os import PathLike
import cv2
from PIL import Image
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Image not found: {path}')

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f'Failed to load image: {path}')

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    return img


def save_image(image: np.ndarray, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim == 3 and image.shape[2] == 3:
        save_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        save_img = image

    cv2.imwrite(str(path), save_img)


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
