# Wrote projects/satellite-enhancer/utils/visualization.py
import numpy as np
import cv2
from models.detector import LANDSAT_CLASSES


def colorize_segmentation_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, (name, color) in LANDSAT_CLASSES.items():
        rgb[mask == class_id] = color
    return rgb


def create_segmentation_overlay(image: np.ndarray, mask: np.ndarray,alpha: float = 0.5) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    seg_colored = colorize_segmentation_mask(mask)
    overlay = cv2.addWeighted(image, 1 - alpha, seg_colored, alpha, 0)
    return overlay


def create_comparison_view(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    if before.ndim == 2:
        before = cv2.cvtColor(before, cv2.COLOR_GRAY2RGB)
    if after.ndim == 2:
        after = cv2.cvtColor(after, cv2.COLOR_GRAY2RGB)

    h1, w1 = before.shape[:2]
    h2, w2 = after.shape[:2]
    target_h = max(h1, h2)
    target_w = max(w1, w2)

    before_pad = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    after_pad = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    before_pad[:h1, :w1] = before
    after_pad[:h2, :w2] = after

    divider = np.zeros((target_h, 4, 3), dtype=np.uint8)
    divider[:] = (255, 255, 255)

    comparison = np.hstack([before_pad, divider, after_pad])
    return comparison


def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    result = image.copy()
    if result.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['class']}: {det['confidence']:.2f}"
        color = (0, 255, 0)

        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return result
