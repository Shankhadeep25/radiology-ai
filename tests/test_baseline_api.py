import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
AI_DIR = REPO_ROOT / "AI"
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

client = TestClient(app)


def test_baseline_status_endpoint(monkeypatch):
    monkeypatch.delenv("ENABLE_BASELINE_MODELS", raising=False)
    res = client.get("/api/v1/baseline/status")
    assert res.status_code == 200
    data = res.json()
    assert data["enabled"] is False
    assert "models" in data
    assert "mri" in data["models"]
    assert "ct" in data["models"]
    assert "us" in data["models"]
    assert data["models"]["mri"]["weights_present"] is True

    monkeypatch.setenv("ENABLE_BASELINE_MODELS", "1")
    res_on = client.get("/api/v1/baseline/status")
    assert res_on.status_code == 200
    data_on = res_on.json()
    assert data_on["enabled"] is True
    assert data_on["models"]["mri"]["enabled"] is True
    assert data_on["models"]["mri"]["metrics"]["accuracy"] > 0.8


def test_modalities_endpoint():
    res = client.get("/api/v1/modalities")
    assert res.status_code == 200
    data = res.json()
    assert data["mvp"] == "XR"
    assert "CT" in data["supported"]
    assert "MR" in data["supported"]
    assert "US" in data["supported"]
    assert "research baseline" in data["note"].lower()


def test_demo_samples_labels():
    res = client.get("/api/v1/demo/samples")
    assert res.status_code == 200
    samples = res.json()
    labels = [s["label"] for s in samples]
    # Check updated baseline labels (no old CT Nodule, MRI Tumor, US BI-RADS)
    assert not any(l in ["CT Nodule", "MRI Tumor", "Ultrasound BI-RADS"] for l in labels)

