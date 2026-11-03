"""Radiology AI clinical chat assistant — context-aware medical knowledge."""

from __future__ import annotations

import re
from typing import Optional

WELCOME = (
    "Hello! I'm your Radiology AI Clinical Assistant. I can help you with:\n"
    "• Explaining AI predictions and confidence scores\n"
    "• Interpreting Grad-CAM explainability heatmaps\n"
    "• Multimodal analysis (Chest X-Ray, CT Chest, Brain MRI, Breast Ultrasound)\n"
    "• Clinical guidelines (Fleischner Society criteria, ACR BI-RADS)\n"
    "• Generating and downloading structured PDF reports\n\n"
    "How can I assist you with this study?"
)

TOPICS: list[tuple[list[str], str]] = [
    (
        ["grad-cam", "gradcam", "heatmap", "explain", "explainability", "why ai", "attention"],
        "Grad-CAM (Gradient-weighted Class Activation Mapping) computes the gradients of the target pathology "
        "score with respect to the feature maps of the final convolutional layer (denseblock4 in DenseNet121). "
        "Warmer colors (red/yellow) indicate the specific spatial anatomy that most heavily influenced the AI's "
        "decision. This allows radiologists to verify that the model is detecting authentic pathological opacities "
        "rather than external hardware, catheters, or technical artifacts.",
    ),
    (
        ["pneumonia", "lung infection", "consolidation", "infiltrate"],
        "Pneumonia typically presents as focal or multifocal airspace consolidation, patchy perihilar infiltrates, "
        "and air bronchograms where alveolar air is replaced by exudate. Our DenseNet121 model evaluates 18 "
        "distinct chest pathologies simultaneously. Always correlate AI findings with patient temperature, WBC count, "
        "and physical auscultation before initiating antimicrobial therapy.",
    ),
    (
        ["densenet", "chexnet", "torchxrayvision", "neural", "deep learning", "ai model", "architecture"],
        "The primary chest X-ray engine utilizes DenseNet-121 pretrained on ChestX-ray14, NIH, and MIMIC-CXR datasets "
        "via TorchXRayVision. DenseNet's dense connectivity allows each layer to receive direct feature maps from all "
        "prior layers, maximizing gradient flow and feature reuse for subtle pulmonary lesions. Multi-modality CT, MRI, "
        "and Ultrasound screening use dedicated volumetric and morphological computer vision pipelines.",
    ),
    (
        ["confidence", "score", "threshold", "review", "uncertainty"],
        "Confidence is calibrated from the neural network's activation logits against validated clinical decision thresholds. "
        "Scores ≥ 80% indicate high certainty; 60–80% represent moderate certainty requiring routine verification; "
        "scores below 60% are flagged as 'Needs Review' on your clinical dashboard to highlight ambiguous or subtle cases.",
    ),
    (
        ["ct", "computed tomography", "nodule", "lung nodule", "fleischner"],
        "CT Chest nodule screening performs discrete 3D voxel integration to calculate nodule diameter (mm), "
        "volume (mm³), and estimated physical mass (g). Clinical management follows Fleischner Society 2017 criteria: "
        "solid nodules < 6 mm generally require no routine follow-up in low-risk patients; 6–8 mm nodules warrant repeat CT "
        "in 6–12 months; nodules > 8 mm indicate follow-up at 3 months, PET-CT, or tissue sampling.",
    ),
    (
        ["mri", "mr ", "brain", "tumor", "glioma", "brats"],
        "MRI Brain analysis evaluates hemispheric structural asymmetry, focal T2/FLAIR hyperintensity, and perilesional "
        "vasogenic edema. It quantifies estimated tumor volume in mL and mass in grams to monitor longitudinal disease "
        "burden and response to therapy. Contrast-enhanced MRI (gadolinium) is recommended for lesion enhancement mapping.",
    ),
    (
        ["ultrasound", "us ", "breast", "bi-rads", "birads"],
        "Breast ultrasound assessment applies ACR BI-RADS criteria to focal parenchymal lesions: "
        "BI-RADS 1 (Negative), BI-RADS 2 (Benign cyst), BI-RADS 3 (Probably benign wider-than-tall mass, < 2% malignancy risk, "
        "6-month follow-up), and BI-RADS 4/5 (Suspicious taller-than-wide mass or indistinct margins, biopsy recommended).",
    ),
    (
        ["x-ray", "xray", "chest x", "cxr", "chest ray"],
        "Chest Radiograph is our core diagnostic screening modality. The AI simultaneously screens across 18 conditions, "
        "including Pneumonia, Pneumothorax, Pleural Effusion, Cardiomegaly, Atelectasis, Lung Nodules, and Fractures. "
        "Output includes calibrated probabilities, full differential diagnosis, Grad-CAM activation overlays, and downloadable PDF reports.",
    ),
    (
        ["report", "pdf", "download report"],
        "Click 'Download PDF Report' on the Analyze results panel to export an official, structured radiology report. "
        "The report includes institution branding, patient demographics, AI telemetry summary, dynamic findings detailing "
        "anatomical location and lesion diameter, ranked differential diagnosis, and evidence-based clinical recommendations.",
    ),
    (
        ["dicom", "pacs", "orthanc", "dicomweb"],
        "The platform natively supports medical DICOM files and integrates with Orthanc PACS via DICOMweb. "
        "DICOM files retain full 16-bit dynamic range, window/level presets (Lung, Soft Tissue, Bone), and patient metadata. "
        "PACS service is accessible at http://localhost:8042.",
    ),
    (
        ["hello", "hi", "hey", "help", "start"],
        WELCOME,
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def generate_reply(message: str, study_context: Optional[dict] = None) -> str:
    text = _normalize(message)
    if not text:
        return "Please type a clinical or technical question and I'll be glad to help."

    if study_context:
        ctx_reply = _reply_from_study_context(text, study_context)
        if ctx_reply:
            return ctx_reply

    for keywords, answer in TOPICS:
        if any(kw in text for kw in keywords):
            return answer

    if "?" in message or any(w in text for w in ["what", "how", "why", "when", "where", "can", "does", "is"]):
        return (
            "I can assist with clinical and technical inquiries regarding:\n"
            "• Current study findings, lesion metrics, and anatomical locations\n"
            "• Grad-CAM activation and neural network explainability\n"
            "• Diagnostic criteria (Fleischner nodule follow-up, BI-RADS protocols)\n"
            "• Supported modalities (X-Ray, CT, MRI, Ultrasound)\n"
            "• Generating structured PDF radiology reports\n\n"
            "Please ask a specific question or type **help** for options."
        )

    return (
        "I am your Radiology AI Assistant. Ask about the current scan's findings, "
        "Grad-CAM explainability, clinical guidelines, or PDF reporting. Type **help** for guidance."
    )


def _reply_from_study_context(text: str, ctx: dict) -> Optional[str]:
    label = ctx.get("primary_label") or ctx.get("label")
    confidence = ctx.get("confidence")
    modality = ctx.get("modality", "XR")
    telemetry = ctx.get("telemetry") or {}
    loc = telemetry.get("anatomical_location") or "the visualized field"
    diam = telemetry.get("diameter_mm")

    if any(w in text for w in [
        "this study", "this result", "my result", "finding", "diagnosis",
        "what does", "my diagnosis", "patient", "wrong", "mean", "where",
        "next step", "recommend", "biopsy", "location", "measure",
    ]):
        if label and confidence is not None:
            pct = round(float(confidence) * 100, 1)
            span_text = f", measuring approximately {diam} mm," if diam else ""
            if "Normal" in label or "No Acute" in label or "Negative" in label:
                return (
                    f"For the current {modality} study: The primary AI assessment is **{label}** "
                    f"with {pct}% confidence. No focal suspicious lesions or acute pathology were detected. "
                    "Clinical correlation remains recommended if acute symptoms persist."
                )
            return (
                f"For the current {modality} study: Primary finding is **{label}** ({pct}% AI confidence) "
                f"located in the **{loc}**{span_text}. "
                "Examine the Grad-CAM activation heatmap overlay to review the focal region of interest. "
                "Refer to the structured PDF report for differential considerations and guideline-based follow-up."
            )
    return None
