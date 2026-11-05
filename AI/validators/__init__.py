"""
Dedicated Modality Validator Package.
Provides clean two-stage architecture:
- Stage 1: verify_base_authenticity (Universal authenticity: blank, noise, photo, screenshot, DICOM)
- Stage 2: Modality-specific validators (XR, CT, MR, US)
"""

from validators.base import verify_base_authenticity, compute_shannon_entropy
from validators.xr import validate_chest_xray
from validators.ct import validate_ct_chest
from validators.mri import validate_brain_mri
from validators.us import validate_ultrasound

__all__ = [
    "verify_base_authenticity",
    "compute_shannon_entropy",
    "validate_chest_xray",
    "validate_ct_chest",
    "validate_brain_mri",
    "validate_ultrasound",
]
