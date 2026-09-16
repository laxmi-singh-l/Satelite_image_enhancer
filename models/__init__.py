# models/__init__.py
"""
Convenience imports for the models package.

All model classes are imported from their canonical source modules.
Fallback mechanisms are built into each class (e.g. EDSR falls back to
OpenCV if no checkpoint is found).
"""

from .detector import LANDSAT_CLASSES, LandCoverSegmenter, ObjectDetector
from .enhance import (
    IRSuperResolution,
    SuperResolutionEDSR,
    SatelliteSuperResolution,
)
from .colorize import IR2RGB, UNetGenerator, PatchGANDiscriminator
from .gan import ESRGANDiscriminator
from .uncertainty import (
    estimate_uncertainty,
    summarize_uncertainty,
    aggregate_uncertainty,
)
