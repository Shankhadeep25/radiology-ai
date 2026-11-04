"""
Automated Multi-Slice Montage & Volumetric Contact Sheet Decomposition.

Detects when an uploaded scan is a composite multi-slice grid (e.g., 2x4, 2x3, 2x2, 3x3)
representing different volumetric slices or angles of an MRI/CT scan, cleanly splits
the sub-slices, and aggregates slice-by-slice inference into a unified 3D study report.
"""

from __future__ import annotations

from typing import Any, Tuple
import cv2
import numpy as np


def detect_montage_grid(gray: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """
    Detect whether a grayscale image is a multi-slice grid montage.
    Returns (is_montage, grid_metadata).
    """
    h, w = gray.shape[:2]
    if h < 80 or w < 80:
        return False, {}

    row_means = np.mean(gray, axis=1)
    col_means = np.mean(gray, axis=0)
    img_mean = float(np.mean(gray))

    # Candidate grids: (rows, cols)
    candidates = [
        (2, 4), (2, 3), (2, 2),
        (3, 4), (3, 3), (3, 2),
        (4, 4), (4, 3), (4, 2),
        (1, 2), (1, 3), (1, 4),
        (2, 1), (3, 1), (4, 1),
    ]

    best_grid = None
    max_score = 0

    for R, C in candidates:
        if R * C < 2:
            continue

        cell_h = h / R
        cell_w = w / C
        if cell_h < 40 or cell_w < 40:
            continue

        y_divs = [int(h * r / R) for r in range(1, R)]
        x_divs = [int(w * c / C) for c in range(1, C)]

        # Check for valley / border drops along dividers
        y_valid = True
        for y in y_divs:
            window = row_means[max(0, y - 5) : min(h, y + 6)]
            if len(window) > 0 and (np.min(window) < img_mean * 0.70 or np.min(window) < 45):
                continue
            y_valid = False
            break

        x_valid = True
        for x in x_divs:
            window = col_means[max(0, x - 5) : min(w, x + 6)]
            if len(window) > 0 and (np.min(window) < img_mean * 0.70 or np.min(window) < 45):
                continue
            x_valid = False
            break

        if y_valid and x_valid:
            score = R * C
            if score > max_score:
                max_score = score
                best_grid = {
                    "rows": R,
                    "cols": C,
                    "count": R * C,
                    "y_divs": y_divs,
                    "x_divs": x_divs,
                    "cell_h": cell_h,
                    "cell_w": cell_w,
                }

    if best_grid is not None:
        return True, best_grid

    return False, {}


def extract_montage_slices(
    gray: np.ndarray,
    grid_meta: dict[str, Any],
    target_size: int = 224,
) -> list[dict[str, Any]]:
    """
    Decompose a multi-slice montage into clean individual slices.
    Returns list of slice records with cropped image, index, and slice coordinates.
    """
    h, w = gray.shape[:2]
    R = grid_meta["rows"]
    C = grid_meta["cols"]

    y_bounds = [0] + grid_meta.get("y_divs", []) + [h]
    x_bounds = [0] + grid_meta.get("x_divs", []) + [w]

    slices = []
    idx = 0

    for r in range(R):
        for c in range(C):
            y1 = y_bounds[r]
            y2 = y_bounds[r + 1]
            x1 = x_bounds[c]
            x2 = x_bounds[c + 1]

            # Inset 2px to avoid divider line borders
            inset_y = 2 if (y2 - y1) > 20 else 0
            inset_x = 2 if (x2 - x1) > 20 else 0

            sub = gray[y1 + inset_y : y2 - inset_y, x1 + inset_x : x2 - inset_x]
            if sub.size == 0 or sub.shape[0] < 15 or sub.shape[1] < 15:
                continue

            # High-fidelity resize to target model resolution
            resized = cv2.resize(sub, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

            slices.append({
                "slice_index": idx + 1,
                "row": r,
                "col": c,
                "bbox": {"x": int(x1), "y": int(y1), "width": int(x2 - x1), "height": int(y2 - y1)},
                "image": resized,
                "raw_crop": sub,
            })
            idx += 1

    return slices

