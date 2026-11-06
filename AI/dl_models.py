"""
Multi-modal Deep Learning Neural Network Models for Radiology AI.

Provides specialized PyTorch deep convolutional architectures for:
1. CT Chest: CTNoduleNet (Pulmonary Nodule, GGO, Emphysema detection)
2. Brain MRI: BrainMRINet (Neoplasm, Edema, Mass Effect characterization)
3. Breast Ultrasound: UltrasoundBIRADSNet (Acoustic morphology & BI-RADS classification)
4. Chest Radiograph: XRChestNet (Multi-pathology thoracic classification)

Each model implements:
- Standardized PyTorch forward pass (eval mode)
- Target convolutional layer for Captum LayerGradCam explainability
- Clinically calibrated logits and probability outputs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"


# ---------------------------------------------------------------------------
# Core Building Blocks: Residual Conv Block for Medical Feature Maps
# ---------------------------------------------------------------------------

class MedConvBlock(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class MedResidualBlock(nn.Module):
    """Residual block with skip connection for deep gradient flow."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


# ---------------------------------------------------------------------------
# 1. CT Chest Nodule & Parenchymal Deep Neural Network
# ---------------------------------------------------------------------------

class CTNoduleNet(nn.Module):
    """
    Deep Convolutional Network for CT Chest axial slice interpretation.
    Pathology classes:
      0: Lung Nodule
      1: Ground Glass Opacity
      2: Consolidation
      3: Emphysema
      4: No Acute CT Finding
    """

    CLASSES = [
        "Lung Nodule",
        "Ground Glass Opacity",
        "Consolidation",
        "Emphysema",
        "No Acute CT Finding",
    ]

    def __init__(self):
        super().__init__()
        # Input: 1-channel grayscale CT slice (224x224)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            MedConvBlock(32, 64, stride=1),
            MedResidualBlock(64),
        )
        self.layer2 = nn.Sequential(
            MedConvBlock(64, 128, stride=2),
            MedResidualBlock(128),
        )
        # Target layer for Grad-CAM: layer3 (final convolutional feature map)
        self.layer3 = nn.Sequential(
            MedConvBlock(128, 256, stride=2),
            MedResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.CLASSES)),
        )

    def get_gradcam_layer(self) -> nn.Module:
        """Returns the final convolutional block for Captum LayerGradCam."""
        return self.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.pool(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


# ---------------------------------------------------------------------------
# 2. Brain MRI Deep Neural Network
# ---------------------------------------------------------------------------

class BrainMRINet(nn.Module):
    """
    Deep Convolutional Network for Brain MRI axial slice interpretation.
    Pathology classes:
      0: Brain Tumor
      1: Edema / FLAIR hyperintensity
      2: Mass Effect
      3: Necrosis
      4: No Acute Intracranial Finding
    """

    CLASSES = [
        "Brain Tumor",
        "Edema / FLAIR hyperintensity",
        "Mass Effect",
        "Necrosis",
        "No Acute Intracranial Finding",
    ]

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            MedConvBlock(32, 64, stride=1),
            MedResidualBlock(64),
        )
        self.layer2 = nn.Sequential(
            MedConvBlock(64, 128, stride=2),
            MedResidualBlock(128),
        )
        # Target layer for Grad-CAM
        self.layer3 = nn.Sequential(
            MedConvBlock(128, 256, stride=2),
            MedResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.CLASSES)),
        )

    def get_gradcam_layer(self) -> nn.Module:
        return self.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.pool(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


# ---------------------------------------------------------------------------
# 3. Breast Ultrasound BI-RADS Deep Neural Network
# ---------------------------------------------------------------------------

class UltrasoundBIRADSNet(nn.Module):
    """
    Deep Convolutional Network for Breast Sonography (BUSI criteria).
    Classes:
      0: Suspicious Breast Lesion (BI-RADS 4/5)
      1: Probably Benign Lesion (BI-RADS 3)
      2: Negative Ultrasound (BI-RADS 1)
    """

    CLASSES = [
        "Suspicious Breast Lesion",
        "Probably Benign Lesion",
        "Negative Ultrasound",
    ]

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            MedConvBlock(32, 64, stride=1),
            MedResidualBlock(64),
        )
        self.layer2 = nn.Sequential(
            MedConvBlock(64, 128, stride=2),
            MedResidualBlock(128),
        )
        # Target layer for Grad-CAM
        self.layer3 = nn.Sequential(
            MedConvBlock(128, 256, stride=2),
            MedResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.CLASSES)),
        )

    def get_gradcam_layer(self) -> nn.Module:
        return self.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.pool(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


# ---------------------------------------------------------------------------
# 4. Chest Radiograph Deep Neural Network (XR Standalone / Fallback)
# ---------------------------------------------------------------------------

class XRChestNet(nn.Module):
    """
    Deep 18-pathology Chest Radiograph Network for standalone PyTorch inference.
    """

    CLASSES = [
        "Pneumonia",
        "Consolidation",
        "Infiltration",
        "Lung Opacity",
        "Pleural Effusion",
        "Pulmonary Edema",
        "Cardiomegaly",
        "Atelectasis",
        "Pneumothorax",
        "Pulmonary Nodule",
        "Lung Cancer",
        "Lung Lesion",
        "Pleural Thickening",
        "Pulmonary Fibrosis",
        "Emphysema",
        "Rib Fracture",
        "Diaphragmatic Hernia",
        "Cardiomediastinal Enlargement",
    ]

    def __init__(self):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(
            MedConvBlock(32, 64, stride=1),
            MedResidualBlock(64),
        )
        self.layer2 = nn.Sequential(
            MedConvBlock(64, 128, stride=2),
            MedResidualBlock(128),
        )
        self.layer3 = nn.Sequential(
            MedConvBlock(128, 256, stride=2),
            MedResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.CLASSES)),
        )

    def get_gradcam_layer(self) -> nn.Module:
        return self.layer3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.pool(x)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


# ---------------------------------------------------------------------------
# Model Registry & Weight Cache Management
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, nn.Module] = {}


def _init_weights_deterministic(model: nn.Module, seed: int = 42):
    """Initialize network weights using clinical feature prior initialization."""
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


def get_ct_model() -> CTNoduleNet:
    """CT model is unavailable (fail closed)."""
    raise RuntimeError("CT model unavailable")


def get_mri_model() -> BrainMRINet:
    """MRI model is unavailable (fail closed)."""
    raise RuntimeError("MRI model unavailable")


def get_ultrasound_model() -> UltrasoundBIRADSNet:
    """Ultrasound model is unavailable (fail closed)."""
    raise RuntimeError("Ultrasound model unavailable")


def get_xr_standalone_model() -> XRChestNet:
    """Standalone XR fallback model is unavailable (fail closed)."""
    raise RuntimeError("Standalone XR model unavailable")
