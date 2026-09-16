import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional
from collections import Counter


SCENE_TEMPLATES = {
    'aquatic': [
        'Scene is dominated by {water_percent:.0f}% water coverage.',
        'Large water body detected with {water_quality} conditions.',
    ],
    'urban': [
        'Urban area with {urban_percent:.0f}% built-up coverage.',
        'Dense infrastructure detected including {object_summary}.',
    ],
    'vegetated': [
        'Vegetation-rich landscape with {forest_percent:.0f}% forest and {grassland_percent:.0f}% grassland.',
        'Scene shows {vegetation_density} vegetation cover.',
    ],
    'agricultural': [
        'Agricultural zone with {agriculture_percent:.0f}% cropland.',
        'Cultivated fields pattern detected across {agriculture_percent:.0f}% of the scene.',
    ],
    'barren': [
        'Arid/barren terrain covering {barren_percent:.0f}% of the scene.',
        'Sparse vegetation with significant exposed surface.',
    ],
    'wetland': [
        'Wetland ecosystem with {wetland_percent:.0f}% wetland coverage.',
        'Transition zone between aquatic and terrestrial environments.',
    ],
    'mixed': [
        'Mixed landscape with diverse land-cover types.',
    ],
}


@dataclass
class SceneReport:
    scene_type: str
    land_cover: dict
    objects_detected: list
    object_counts: dict
    description: str
    confidence: float
    time_analysis: str
    summary_stats: dict

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class SceneAnalyzer:
    def __init__(self):
        self.captioner = None
        self._load_captioner()

    def _load_captioner(self):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.processor = BlipProcessor.from_pretrained(
                'Salesforce/blip-image-captioning-base'
            )
            self.model = BlipForConditionalGeneration.from_pretrained(
                'Salesforce/blip-image-captioning-base'
            )
            self.captioner = True
        except (ImportError, OSError, Exception):
            self.captioner = None

    def analyze(self, land_cover: dict, objects: list,
                rgb_image: Optional[np.ndarray] = None) -> SceneReport:
        scene_type = self._classify_scene(land_cover)
        object_counts = Counter(obj['class'] for obj in objects)
        object_summary = ', '.join(
            f"{count} {cls}" for cls, count in object_counts.most_common(5)
        )

        blip_caption = self._generate_caption(rgb_image) if rgb_image is not None else ''
        template_desc = self._build_template_description(
            scene_type, land_cover, object_summary
        )

        description = self._merge_descriptions(blip_caption, template_desc)
        summary_stats = self._compute_summary_stats(land_cover)

        total_pixels = sum(
            v['pixel_count'] for v in land_cover.values()
        )

        return SceneReport(
            scene_type=scene_type,
            land_cover=land_cover,
            objects_detected=objects,
            object_counts=dict(object_counts),
            description=description,
            confidence=self._estimate_confidence(land_cover),
            time_analysis=self._generate_time_context(),
            summary_stats=summary_stats,
        )

    @staticmethod
    def _classify_scene(land_cover: dict) -> str:
        if not land_cover:
            return 'unknown'

        dominant = max(land_cover.items(), key=lambda x: x[1]['percentage'])

        if dominant[1]['percentage'] > 50:
            for scene_type, keywords in [
                ('aquatic', ['water']),
                ('urban', ['urban']),
                ('agricultural', ['agriculture']),
                ('barren', ['barren']),
                ('wetland', ['wetland']),
                ('vegetated', ['forest', 'grassland']),
            ]:
                if dominant[0] in keywords:
                    return scene_type

        return 'mixed'

    @staticmethod
    def _build_template_description(scene_type: str, land_cover: dict,
                                     object_summary: str) -> str:
        templates = SCENE_TEMPLATES.get(scene_type, SCENE_TEMPLATES['mixed'])
        water = land_cover.get('water', {}).get('percentage', 0)
        urban = land_cover.get('urban', {}).get('percentage', 0)
        forest = land_cover.get('forest', {}).get('percentage', 0)
        grassland = land_cover.get('grassland', {}).get('percentage', 0)
        agriculture = land_cover.get('agriculture', {}).get('percentage', 0)
        barren = land_cover.get('barren', {}).get('percentage', 0)
        wetland = land_cover.get('wetland', {}).get('percentage', 0)

        water_quality = 'clear' if water > 30 else 'sparse'
        vegetation_density = 'dense' if (forest + grassland) > 40 else 'moderate'

        parts = []
        for t in templates:
            try:
                parts.append(t.format(
                    water_percent=water,
                    urban_percent=urban,
                    forest_percent=forest,
                    grassland_percent=grassland,
                    agriculture_percent=agriculture,
                    barren_percent=barren,
                    wetland_percent=wetland,
                    water_quality=water_quality,
                    vegetation_density=vegetation_density,
                    object_summary=object_summary or 'no significant objects',
                ))
            except KeyError:
                parts.append(t)

        return ' '.join(parts)

    def _generate_caption(self, rgb_image: np.ndarray) -> str:
        if self.captioner is None:
            return ''

        try:
            from PIL import Image
            pil_img = Image.fromarray(rgb_image)
            inputs = self.processor(pil_img, return_tensors='pt')
            out = self.model.generate(**inputs, max_length=50)
            return self.processor.decode(out[0], skip_special_tokens=True)
        except Exception:
            return ''

    @staticmethod
    def _merge_descriptions(blip_caption: str, template_desc: str) -> str:
        if blip_caption:
            blip_caption = blip_caption.strip().capitalize()
            if not blip_caption.endswith('.'):
                blip_caption += '.'
            return f'{blip_caption} {template_desc}'
        return template_desc.strip()

    @staticmethod
    def _compute_summary_stats(land_cover: dict) -> dict:
        if not land_cover:
            return {}

        percentages = {k: v['percentage'] for k, v in land_cover.items()}
        sorted_cover = sorted(percentages.items(), key=lambda x: -x[1])

        natural = sum(
            v['percentage'] for k, v in land_cover.items()
            if k in ('forest', 'grassland', 'water', 'wetland')
        )
        anthropogenic = sum(
            v['percentage'] for k, v in land_cover.items()
            if k in ('urban', 'agriculture')
        )

        return {
            'dominant_class': sorted_cover[0][0] if sorted_cover else 'unknown',
            'dominant_percentage': sorted_cover[0][1] if sorted_cover else 0,
            'natural_coverage': round(natural, 2),
            'anthropogenic_coverage': round(anthropogenic, 2),
            'class_diversity': len([v for v in percentages.values() if v > 5]),
        }

    @staticmethod
    def _estimate_confidence(land_cover: dict) -> float:
        if not land_cover:
            return 0.0
        confidence = 0.0
        dominant = max(v['percentage'] for v in land_cover.values())
        if dominant > 70:
            confidence = 0.95
        elif dominant > 50:
            confidence = 0.85
        elif dominant > 30:
            confidence = 0.70
        else:
            confidence = 0.60
        return round(confidence, 2)

    @staticmethod
    def _generate_time_context() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

    def __call__(self, land_cover: dict, objects: list,
                 rgb_image: Optional[np.ndarray] = None) -> SceneReport:
        return self.analyze(land_cover, objects, rgb_image)
