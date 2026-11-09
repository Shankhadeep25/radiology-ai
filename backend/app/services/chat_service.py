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
        ["validator", "rejected", "reject", "authenticity", "fake scan", "non-medical", "invalid image", "selfie", "portrait"],
        "Our platform enforces strict multi-tier medical image authenticity guards (AI/validator.py). "
        "Every upload is analyzed for authentic radiological characteristics, including dynamic range, "
        "grayscale entropy, aspect ratios, and modality-specific anatomical profiles (e.g. bilateral lung field "
        "symmetry for X-rays, axial calvarium oval for brain MRI, thoracic wall contour for CT, or acoustic fan for US). "
        "Monochrome selfies, natural photos, and cross-modality mismatches are automatically rejected to prevent "
        "ungrounded AI hallucinations and safeguard clinical decision pipelines.",
    ),
    (
        ["montage", "slice grid", "series", "multi-slice", "sub-slices", "grid"],
        "Clinical CT and MRI scans frequently arrive as 2x2, 3x3, or 4x4 multi-slice montage grids. "
        "Our automated montage detector (AI/montage_detector.py) identifies uniform tile layouts, "
        "segments each anatomical axial section, and evaluates each slice individually through deep learning vision backbones. "
        "This allows discrete 3D voxel volume integration and spatial localization without requiring heavy desktop PACS installations.",
    ),
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
        ["pneumothorax", "collapsed lung"],
        "Pneumothorax is an acute clinical emergency indicated by separation of the visceral pleural line "
        "from the chest wall with complete absence of peripheral lung markings. Our system flags pneumothorax "
        "with Critical/High urgency, prompting immediate clinician assessment for tension physiology and tube thoracostomy.",
    ),
    (
        ["cardiomegaly", "enlarged heart", "ctr"],
        "Cardiomegaly indicates an enlarged cardiac silhouette where the transverse cardiothoracic ratio (CTR) "
        "exceeds 0.50 on standard PA chest radiographs. It frequently correlates with chronic hypertension, cardiomyopathy, "
        "or congestive heart failure and is often accompanied by upper-lobe venous prominence and pleural effusions.",
    ),
    (
        ["pleural effusion", "effusion", "fluid"],
        "Pleural effusion manifests as dependent homogeneous fluid density blunting the costophrenic angles. "
        "Larger collections cause compressive passive atelectasis at the adjacent lung base. Management focuses on "
        "evaluating fluid volume and determining whether the effusion is transudative or exudative.",
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
        "vasogenic edema. It quantifies estimated tumor volume in mL, mass in grams, and midline shift in mm "
        "to monitor longitudinal mass effect and response to therapy. Contrast-enhanced MRI (gadolinium) is recommended.",
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

    # Immediate greeting handling
    if text in ("hello", "hi", "hey", "start", "help", "welcome", "good morning", "good afternoon", "good evening"):
        return WELCOME

    # Conversational pleasantries & acknowledgments (must not dump patient/medical data)
    if re.search(r"^(thanks?|thank you|thx|much appreciated|appreciate it|thank you so much)[\s!.]*$", text):
        return "You're welcome! Let me know if you have any questions about this study or need help interpreting findings."

    if re.search(r"^(ok|okay|got it|understood|sure|alright|great|perfect|sounds good)[\s!.]*$", text):
        return "Understood. Feel free to ask whenever you need further details or guideline recommendations."

    if re.search(r"^(bye|goodbye|see you|take care)[\s!.]*$", text):
        return "Goodbye! Feel free to return whenever you need clinical imaging support."

    # Route through the Clinical RAG Service
    try:
        from app.services.rag_service import answer_clinical_query_sync
        return answer_clinical_query_sync(message, study_context)
    except Exception as exc:
        pass

    # Safety Fallback to rule engine if RAG unavailable
    if study_context:
        ctx_reply = _reply_from_study_context(text, study_context)
        if ctx_reply:
            return ctx_reply

    for keywords, answer in TOPICS:
        if any(kw in text for kw in keywords):
            return answer

    return (
        "I am your Radiology AI Clinical Assistant. You can ask about:\n"
        "• Specific findings, lesion diameter, volume, and anatomical localization\n"
        "• Clinical next steps per Fleischner Society 2017 criteria or ACR BI-RADS\n"
        "• Grad-CAM explainability and neural network attention\n"
        "• Medical image validation and montage scan processing\n\n"
        "Please ask your question and I will correlate it with current clinical evidence."
    )


def _reply_from_study_context(text: str, ctx: dict) -> Optional[str]:
    label = ctx.get("primary_label") or ctx.get("label")
    confidence = ctx.get("confidence")
    modality = ctx.get("modality", "XR")
    telemetry = ctx.get("telemetry") or {}
    loc = telemetry.get("anatomical_location") or "the visualized field"
    diam = telemetry.get("diameter_mm")
    vol_mm3 = telemetry.get("volume_mm3")
    vol_ml = telemetry.get("volume_ml")
    birads = telemetry.get("birads")
    fleischner = telemetry.get("fleischner_category")
    shift_mm = telemetry.get("midline_shift_mm")

    if not label or confidence is None:
        return None

    pct = round(float(confidence) * 100, 1)

    # 1. Plain language / patient consultation explanation
    if any(w in text for w in ["plain", "simple", "patient", "layman", "easy words", "simple terms", "consult"]):
        if any(norm in label for norm in ("Normal", "No Acute", "Negative")):
            return (
                f"**Patient Summary ({modality})**: Your scan is **normal** ({pct}% confidence) with no signs of acute infection or masses. "
                "Your doctor will discuss your symptoms and ongoing care."
            )
        return (
            f"**Patient Summary ({modality})**: The scan identified **{label}** in the **{loc}** ({pct}% confidence). "
            "This warrants clinical attention; your physician and radiologist will recommend the next steps."
        )

    # 2. Measurement and quantitative telemetry queries
    if any(w in text for w in ["measure", "diameter", "volume", "mass", "metrics", "size", "span", "telemetry"]):
        details = [f"**Measurements ({modality} - {label})**:", f"• **Location**: {loc}"]
        if diam:
            details.append(f"• **Diameter**: {diam} mm")
        if vol_mm3:
            details.append(f"• **3D Volume**: {vol_mm3} mm³")
        if vol_ml:
            details.append(f"• **Volume**: {vol_ml} mL")
        if shift_mm is not None and shift_mm > 0:
            details.append(f"• **Midline Shift**: {shift_mm} mm")
        if fleischner:
            details.append(f"• **Fleischner**: {fleischner}")
        if birads:
            details.append(f"• **BI-RADS**: {birads}")
        return "\n".join(details)

    # 3. Clinical next steps and management protocol
    if any(w in text for w in ["next step", "recommend", "protocol", "what should", "what to do", "follow up", "biopsy"]):
        if modality == "CT":
            return (
                f"**Next Steps (CT - {label})**:\n"
                f"1. Lesion in **{loc}** ({pct}% confidence).\n"
                f"2. Apply Fleischner 2017: {fleischner or 'Repeat CT at 3–12 months'}.\n"
                "3. Correlate with clinical risk factors and history."
            )
        if modality == "MR":
            return (
                f"**Next Steps (Brain MRI - {label})**:\n"
                f"1. Abnormal mass in **{loc}** ({pct}% confidence).\n"
                "2. Contrast-enhanced MRI (gadolinium) recommended.\n"
                "3. Neurosurgical evaluation for mass effect management."
            )
        if modality == "US":
            return (
                f"**Next Steps (Breast US - {label})**:\n"
                f"1. Lesion in **{loc}** (**{birads or 'BI-RADS 4'}**, {pct}% confidence).\n"
                "2. Ultrasound-guided core needle biopsy indicated.\n"
                "3. Bilateral diagnostic mammography correlation."
            )
        return (
            f"**Next Steps (CXR - {label})**:\n"
            f"1. Finding centered in **{loc}** ({pct}% confidence).\n"
            "2. Correlate with auscultation, temperature, and inflammatory labs (CBC, CRP).\n"
            "3. Repeat CXR in 4–6 weeks to verify radiographic resolution."
        )

    # 4. General study finding questions
    if any(w in text for w in [
        "this study", "this result", "my result", "finding", "diagnosis",
        "what does", "my diagnosis", "wrong", "mean", "where",
    ]):
        span_text = f", measuring approximately {diam} mm," if diam else ""
        if any(norm in label for norm in ("Normal", "No Acute", "Negative")):
            return f"**Study Finding ({modality})**: Primary assessment is **{label}** ({pct}% confidence). No acute focal lesions detected."
        return f"**Study Finding ({modality})**: Primary finding is **{label}** ({pct}% confidence) in the **{loc}**{span_text}. Review the Grad-CAM heatmap overlay to view the focal region."

    return None

    return None
