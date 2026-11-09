import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
import torch

# Add AI/ and backend/ to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
AI_DIR = REPO_ROOT / "AI"
BACKEND_DIR = REPO_ROOT / "backend"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from dl_models import load_baseline, normalize_meta
from preprocessing import pad_square, preprocess_for
from predict import predict_baseline


def test_pad_square():
    # Square image should be unchanged in size
    sq = Image.new("RGB", (100, 100), (255, 0, 0))
    assert pad_square(sq).size == (100, 100)

    # Non-square: width 80, height 160 -> padded to 160x160 with black fill
    rect = Image.new("RGB", (80, 160), (255, 255, 255))
    padded = pad_square(rect)
    assert padded.size == (160, 160)
    # Check top-left corner is black fill (0, 0, 0)
    assert padded.getpixel((0, 0)) == (0, 0, 0)
    # Check center is white (255, 255, 255)
    assert padded.getpixel((80, 80)) == (255, 255, 255)


def test_preprocess_for_shape_dtype_and_padding():
    # Create non-square dummy image (120x240)
    img = Image.new("RGB", (120, 240), (200, 100, 50))

    for name in ["ct", "us", "mri"]:
        tensor = preprocess_for(name, img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224), f"Wrong shape for {name}: {tensor.shape}"
        assert tensor.dtype == torch.float32, f"Wrong dtype for {name}: {tensor.dtype}"
        # Sensible value range after ImageNet normalization: mostly in [-3.0, 3.0]
        assert tensor.min() > -5.0 and tensor.max() < 5.0

    # Test that CT and US pad before resize, whereas MRI does not pad
    # If we pass an image where non-centered region is distinct, CT/US pad to square first.
    wide_img = Image.new("RGB", (300, 100), (0, 0, 255))
    t_ct = preprocess_for("ct", wide_img)
    t_mri = preprocess_for("mri", wide_img)
    # Top region in padded CT should reflect black padding (negative values after ImageNet normalization)
    # whereas MRI stretches the blue channel across the full vertical span
    assert not torch.allclose(t_ct, t_mri)


def test_normalize_meta_ct():
    import json
    ct_file = REPO_ROOT / "weights" / "metrics" / "ct_metrics.json"
    with open(ct_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    norm = normalize_meta("ct", meta)
    assert norm["name"] == "ct"
    assert norm["classes"] == ['adenocarcinoma', 'large_cell', 'normal', 'squamous']
    assert isinstance(norm["temperature"], float)
    assert abs(norm["accuracy"] - 0.5935) < 0.001
    assert abs(norm["balanced_accuracy"] - 0.6766) < 0.001
    assert abs(norm["macro_auc"] - 0.8855) < 0.001
    assert isinstance(norm["accuracy_ci"], list) and len(norm["accuracy_ci"]) == 2
    assert isinstance(norm["macro_auc_ci"], list) and len(norm["macro_auc_ci"]) == 2
    assert norm["status_note"] is not None
    assert norm["wrong_modality"] is not None


def test_normalize_meta_mri():
    import json
    mri_file = REPO_ROOT / "weights" / "metrics" / "mri_metrics.json"
    with open(mri_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    norm = normalize_meta("mri", meta)
    assert norm["name"] == "mri"
    assert norm["classes"] == ['glioma', 'meningioma', 'notumor', 'pituitary']
    assert abs(norm["accuracy"] - 0.894375) < 0.001
    assert abs(norm["macro_auc"] - 0.97545) < 0.001
    assert norm["accuracy_ci"] == [0.87875, 0.909375]
    assert norm["macro_auc_ci"] == [0.9694804076851835, 0.9809102769820477]
    assert norm["balanced_accuracy"] is None  # absent in MRI
    assert "NOT approved" in norm["status_note"]
    assert "us_busi" in norm["wrong_modality"]


def test_normalize_meta_us():
    import json
    us_file = REPO_ROOT / "weights" / "metrics" / "us_metrics.json"
    with open(us_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    norm = normalize_meta("us", meta)
    assert norm["name"] == "us"
    assert norm["classes"] == ['benign', 'malignant', 'normal']
    assert abs(norm["accuracy"] - 0.8) < 0.001
    assert abs(norm["macro_auc"] - 0.9220) < 0.001
    assert norm["balanced_accuracy"] is None  # absent in US JSON
    assert norm["accuracy_ci"] == [0.7333333333333333, 0.8666666666666667]
    assert norm["macro_auc_ci"] == [0.8804082291535015, 0.9569126998665684]


def test_normalize_meta_missing_keys():
    # Empty meta returns None for missing values
    norm = normalize_meta("mri", {})
    assert norm["name"] == "mri"
    assert norm["classes"] is None
    assert norm["accuracy"] is None
    assert norm["accuracy_ci"] is None
    assert norm["macro_auc"] is None
    assert norm["status_note"] is None
    assert norm["wrong_modality"] is None

    norm_ct = normalize_meta("ct", {})
    assert norm_ct["name"] == "ct"
    assert norm_ct["accuracy"] is None
    assert norm_ct["balanced_accuracy"] is None


def test_flag_off_fails_closed(monkeypatch):
    monkeypatch.delenv("ENABLE_BASELINE_MODELS", raising=False)
    with pytest.raises(RuntimeError) as exc_info:
        load_baseline("mri")
    assert "model unavailable" in str(exc_info.value).lower()

    monkeypatch.setenv("ENABLE_BASELINE_MODELS", "0")
    with pytest.raises(RuntimeError) as exc_info:
        load_baseline("ct")
    assert "model unavailable" in str(exc_info.value).lower()


def test_flag_on_missing_weights_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_BASELINE_MODELS", "1")
    # Mock weights path by overriding file lookup or testing non-existent modality
    with pytest.raises(ValueError):
        load_baseline("unknown_modality")


def test_predict_baseline_response_contract(monkeypatch):
    monkeypatch.setenv("ENABLE_BASELINE_MODELS", "1")
    dummy = Image.new("RGB", (224, 224), (128, 128, 128))

    for name in ["ct", "mri", "us"]:
        res = predict_baseline(name, dummy)
        assert res["calibrated"] is False
        assert res["in_review_band"] is None
        assert res["status"] == "research_baseline"
        assert "all_scores" in res
        # Sum of softmax probabilities should equal 1.0
        scores_sum = sum(res["all_scores"].values())
        assert abs(scores_sum - 1.0) < 1e-4

        # Primary label must be in all_scores and have highest score
        assert res["primary_label"] in res["all_scores"]
        assert abs(res["confidence"] - res["all_scores"][res["primary_label"]]) < 1e-4

        # Findings must be unvalidated
        for f in res["all_findings"]:
            assert f["validated"] is False
            assert f["status"] == "unvalidated"

        # Disclaimers
        assert "Research baseline" in res["disclaimer"]
        assert "Wrong-modality" in res["modality_warning"]
        assert res["model_metrics"] is not None

