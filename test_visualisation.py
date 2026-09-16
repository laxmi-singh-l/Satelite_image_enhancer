# test_visualization.py
import numpy as np
import cv2
from utils.visualization import (
    colorize_segmentation_mask,
    create_segmentation_overlay,
    create_comparison_view,
    draw_detections,
)

# Minimal fake LANDSAT_CLASSES if models.detector is not ready
from models.detector import LANDSAT_CLASSES  # make sure this file exists

def main():
    # Create a fake image (256x256 RGB)
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :] = (50, 100, 150)

    # Create a fake mask with two classes
    mask = np.zeros((256, 256), dtype=np.int32)
    mask[:, :128] = 0  # class 0
    mask[:, 128:] = 1  # class 1

    # Segmentation overlay
    overlay = create_segmentation_overlay(image, mask)

    # Fake "enhanced" image (brighter)
    enhanced = cv2.convertScaleAbs(image, alpha=1.5, beta=20)
    comparison = create_comparison_view(image, enhanced)

    # Fake detections
    detections = [
        {"bbox": [30, 30, 120, 120], "class": "object", "confidence": 0.93},
        {"bbox": [150, 100, 220, 200], "class": "object2", "confidence": 0.85},
    ]
    detected = draw_detections(image, detections)

    # Save results
    cv2.imwrite("overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("comparison.png", cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    cv2.imwrite("detections.png", cv2.cvtColor(detected, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
