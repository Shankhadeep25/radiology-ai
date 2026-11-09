"""
Medical image authenticity and radiographic validity validator.
Employs a decoupled Two-Stage Architecture:
- Stage 1: Universal Medical Scan Authenticity (rejects non-medical, blank, photo, document, noise).
- Stage 2: Dedicated Modality-Specific Validators (XR, CT, MR, US) without cross-contamination.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# Re-export dedicated validators and stage 1 guards
from validators.base import verify_base_authenticity, compute_shannon_entropy
from validators.xr import validate_chest_xray
from validators.ct import validate_ct_chest
from validators.mri import validate_brain_mri
from validators.us import validate_ultrasound

logger = logging.getLogger(__name__)

# Backwards compatibility aliases
_compute_shannon_entropy = compute_shannon_entropy
_verify_chest_xray_anatomy = validate_chest_xray


def verify_medical_image(
    image_input: str | Path | np.ndarray,
    expected_modality: str = "XR",
) -> tuple[bool, str, dict[str, Any]]:
    """
    Two-Stage Multi-Modality Medical Image Verification.
    
    Stage 1: Universal Base Authenticity Guard
             Screens out corrupted DICOMs, blank canvases, natural color photos,
             cartoons, synthetic noise, and text documents/screenshots.
    
    Stage 2: Dedicated Modality Validator
             Routes to the specific validator tailored to the acquisition physics
             of that modality (XR, CT, MR, US) to extract telemetry flags
             without cross-modality interference.

    Returns:
        (is_valid: bool, reason: str, telemetry: dict)
    """
    expected = (expected_modality or "XR").upper()
    if expected == "MRI":
        expected = "MR"

    # Stage 1: Universal Modality-Agnostic Authenticity Guard
    is_valid, reason, base_telemetry, gray = verify_base_authenticity(image_input, expected)
    if not is_valid or gray is None:
        return False, reason, base_telemetry

    # Stage 2: Dispatch to the dedicated modality validator
    modality_validators = {
        "XR": validate_chest_xray,
        "CT": validate_ct_chest,
        "MR": validate_brain_mri,
        "US": validate_ultrasound,
    }

    validator_fn = modality_validators.get(expected, validate_chest_xray)
    mod_valid, mod_reason, mod_metrics = validator_fn(gray)

    combined_telemetry = {**base_telemetry, **mod_metrics}
    return bool(mod_valid), mod_reason, combined_telemetry
