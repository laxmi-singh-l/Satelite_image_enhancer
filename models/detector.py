import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict


LANDSAT_CLASSES = {
    0: ('water', (60, 119, 181)),
    1: ('forest', (34, 139, 34)),
    2: ('agriculture', (154, 205, 50)),
    3: ('urban', (178, 34, 34)),
    4: ('barren', (210, 180, 140)),
    5: ('wetland', (0, 206, 209)),
    6: ('grassland', (124, 252, 0)),
}

LANDSAT_CLASS_NAMES = [v[0] for v in LANDSAT_CLASSES.values()]


class DeepLabV3Segmenter:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            self.model = torch.hub.load(
                'pytorch/vision:v0.17.0',
                'deeplabv3_resnet101',
                pretrained=True,
            )
            self.model.to(self.device)
            self.model.eval()
        except (ImportError, RuntimeError):
            try:
                import torchvision
                self.model = torchvision.models.segmentation.deeplabv3_resnet101(
                    weights='DEFAULT'
                )
                self.model.to(self.device)
                self.model.eval()
            except (ImportError, RuntimeError):
                self.model = None

    @staticmethod
    def _prepare_input(image: np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image_rgb = cv2.resize(image, (520, 520))
        image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor, image.shape[:2]

    def segment(self, image: np.ndarray) -> np.ndarray:
        if self.model is not None:
            try:
                return self._segment_deeplab(image)
            except (RuntimeError, Exception):
                self.model = None
        return self._segment_fallback(image)

    def _segment_deeplab(self, image: np.ndarray) -> np.ndarray:
        tensor, (orig_h, orig_w) = self._prepare_input(image)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))['out'][0]
        mask = output.argmax(dim=0).cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask = self._remap_classes(mask)
        return mask

    @staticmethod
    def _remap_classes(mask: np.ndarray) -> np.ndarray:
        mapping = {
            0: 3, 1: 3, 2: 3,
            3: 6, 4: 6, 5: 6, 6: 6, 7: 6,
            8: 1, 9: 6, 10: 6,
            11: 0, 12: 0, 13: 0,
            14: 4, 15: 4,
            16: 6, 17: 6,
            18: 3, 19: 3, 20: 3,
            21: 0,
        }
        remapped = np.zeros_like(mask)
        for src, dst in mapping.items():
            remapped[mask == src] = dst
        return remapped

    @staticmethod
    def _segment_fallback(image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape
        result = np.zeros((h, w), dtype=np.uint8)

        water_thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
        bright_thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
        mid_thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]

        result[water_thresh == 0] = 0
        result[bright_thresh > 0] = 4
        result[mid_thresh > 0] = 1

        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((5, 5), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=2)
        result[edges_dilated > 0] = 3

        return result


class LandCoverSegmenter:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.segmenter = DeepLabV3Segmenter(device=self.device)
        self.yolo = None
        self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')
        except (ImportError, Exception):
            self.yolo = None

    def analyze(self, image: np.ndarray) -> dict:
        seg_mask = self.segmenter.segment(image)
        stats = self._compute_landcover_stats(seg_mask)
        objects = self._detect_objects(image)
        return {
            'segmentation_mask': seg_mask,
            'land_cover': stats,
            'objects': objects,
        }

    @staticmethod
    def _compute_landcover_stats(mask: np.ndarray) -> dict:
        total = mask.size
        stats = {}
        for class_id, (name, color) in LANDSAT_CLASSES.items():
            count = int(np.sum(mask == class_id))
            stats[name] = {
                'percentage': round(count / total * 100, 2),
                'pixel_count': count,
                'color': color,
            }
        return stats

    def _detect_objects(self, image: np.ndarray) -> list:
        if self.yolo is None:
            return self._detect_objects_fallback(image)

        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            results = self.yolo(img_rgb, verbose=False)
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    detections.append({
                        'bbox': box.tolist(),
                        'class': r.names[int(cls)],
                        'confidence': float(conf),
                    })
            return detections
        except (RuntimeError, Exception):
            self.yolo = None
            return self._detect_objects_fallback(image)

    @staticmethod
    def _detect_objects_fallback(image: np.ndarray) -> list:
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        detections = []
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 50 < area < 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append({
                    'bbox': [float(x), float(y), float(x + w), float(y + h)],
                    'class': 'bright_anomaly',
                    'confidence': min(1.0, area / 5000),
                })

        return detections

    def __call__(self, image: np.ndarray) -> dict:
        return self.analyze(image)


class ObjectDetector:
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = None
        self._load_yolo()

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            self.yolo = YOLO('yolov8n.pt')
        except (ImportError, Exception):
            self.yolo = None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.25) -> list:
        if self.yolo is not None:
            return self._detect_yolo(image, conf_threshold)
        return self._detect_fallback(image)

    def _detect_yolo(self, image: np.ndarray, conf_threshold: float) -> list:
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            results = self.yolo(img_rgb, verbose=False)
            detections = []
            for r in results:
                if r.boxes is None:
                    continue
                for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                    if float(conf) >= conf_threshold:
                        detections.append({
                            'bbox': box.tolist(),
                            'class': r.names[int(cls)],
                            'confidence': float(conf),
                        })
            return detections
        except (RuntimeError, Exception):
            self.yolo = None
            return self._detect_fallback(image)

    @staticmethod
    def _detect_fallback(image: np.ndarray) -> list:
        return []
