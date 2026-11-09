"""
Clinical Retrieval-Augmented Generation (RAG) Service for Radiology AI.
Combines an authoritative medical knowledge base with active patient study context,
utilizing a local BM25 semantic retriever and dual-mode (LLM / Deterministic) synthesis.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

import httpx

from app.config import settings

# ── Medical Knowledge Base Corpus ─────────────────────────────────────────────

MEDICAL_KNOWLEDGE_CORPUS: list[dict[str, str]] = [
    {
        "id": "general_next_clinical_steps",
        "title": "Clinical Next Steps & Diagnostic Workflow Protocol",
        "category": "workflow",
        "keywords": "next steps recommendation recommend what to do protocol follow up clinical steps management",
        "content": (
            "Standard clinical next steps in diagnostic imaging follow a 4-step decision protocol:\n"
            "1. Anatomical Verification: Review the Grad-CAM explainability heatmap overlay to confirm that the neural "
            "network attended to genuine pathological opacities, focal consolidations, or masses rather than external hardware.\n"
            "2. Clinical & Laboratory Correlation: Correlate visual findings with acute clinical signs (vital signs, core temperature, "
            "physical auscultation, and inflammatory markers such as CBC, CRP, or ESR).\n"
            "3. Evidence-Based Guideline Triage: For pulmonary nodules, apply Fleischner Society 2017 follow-up intervals; for breast lesions, "
            "follow ACR BI-RADS management pathways; for brain mass lesions, evaluate midline shift and intracranial pressure.\n"
            "4. Radiologist Sign-Off: All preliminary AI findings require independent verification, report generation, and digital sign-off "
            "by a licensed radiologist before initiating invasive treatment or antimicrobial therapy."
        ),
    },
    {
        "id": "fleischner_guidelines_2017",
        "title": "Fleischner Society 2017 Guidelines for Pulmonary Nodules",
        "category": "guidelines",
        "keywords": "fleischner ct nodule lung nodule pulmonary nodule solid subsolid ground glass 6mm 8mm pet biopsy",
        "content": (
            "The Fleischner Society 2017 Guidelines govern incidental pulmonary nodule management in patients >= 35 years:\n"
            "• Solid Nodule < 6 mm: Low-risk patients require no routine follow-up. High-risk patients (smoking/occupational history) may have an optional repeat CT at 12 months.\n"
            "• Solid Nodule 6–8 mm: Low-risk patients warrant a repeat non-contrast thin-section CT at 6–12 months. High-risk patients warrant CT at 6–12 months and 18–24 months.\n"
            "• Solid Nodule > 8 mm: High risk for malignancy. Recommend repeat thin-section CT at 3 months, PET-CT metabolic evaluation, or tissue biopsy/resection.\n"
            "• Multiple Solid Nodules: Management is guided by the largest/most suspicious nodule; follow-up CT at 3–6 months.\n"
            "• Subsolid / Ground-Glass Nodules: Pure ground-glass nodules < 6 mm require no follow-up. Pure ground-glass >= 6 mm warrant follow-up CT at 6–12 months to confirm persistence, then every 2 years up to 5 years. Part-solid nodules with solid component >= 6 mm are highly suspicious for invasive adenocarcinoma, requiring 3–6 month CT confirmation or biopsy."
        ),
    },
    {
        "id": "acr_birads_ultrasound",
        "title": "ACR BI-RADS Atlas (5th Edition) — Ultrasound Classification",
        "category": "guidelines",
        "keywords": "bi-rads birads ultrasound us breast mass cyst benign suspicious aspect ratio biopsy calcification",
        "content": (
            "The American College of Radiology (ACR) BI-RADS Atlas standardizes breast sonographic lesion evaluation:\n"
            "• BI-RADS 0: Incomplete assessment; prior mammograms or additional imaging required.\n"
            "• BI-RADS 1: Negative; homogeneous fibroglandular parenchyma with normal acoustic transmission; routine screening.\n"
            "• BI-RADS 2: Benign finding; simple anechoic cyst with thin walls, acoustic enhancement, or stable intramammary lymph node; routine screening.\n"
            "• BI-RADS 3: Probably benign finding (< 2% risk of malignancy); circumscribed, oval, wider-than-tall solid mass without posterior shadowing; initial short-interval follow-up at 6 months.\n"
            "• BI-RADS 4: Suspicious abnormality (2% to 95% risk of malignancy); characterized by taller-than-wide morphology (aspect ratio > 1.0), indistinct, angular, or microlobulated margins, hypoechogenicity, or posterior acoustic shadowing. Histological confirmation via ultrasound-guided core needle biopsy is indicated. Subdivided into 4A (low suspicion 2–10%), 4B (moderate 10–50%), and 4C (high 50–95%).\n"
            "• BI-RADS 5: Highly suggestive of malignancy (> 95% risk of malignancy); classic spiculated, irregular hypoechoic mass with shadowing. Immediate tissue biopsy and multidisciplinary oncology staging required."
        ),
    },
    {
        "id": "thoracic_pneumonia_consolidation",
        "title": "Thoracic Radiography: Pneumonia & Airspace Consolidation",
        "category": "pathology",
        "keywords": "pneumonia consolidation infiltrate air bronchogram fever cough chest x-ray cxr antibiotics",
        "content": (
            "Pneumonia on chest radiography presents as focal or multifocal airspace consolidation, patchy alveolar-interstitial "
            "opacities, and air bronchograms where air inside bronchial lumens is highlighted against exudate-filled alveoli.\n"
            "Clinical Management & Next Steps:\n"
            "1. Correlate with patient vital signs (fever > 38°C, tachypnea, tachycardia), oxygen saturation, and lung auscultation (crackles, bronchial breathing).\n"
            "2. Order inflammatory laboratory workup: complete blood count (CBC) with differential for leukocytosis, and C-reactive protein (CRP) or procalcitonin.\n"
            "3. Assess CURB-65 or Pneumonia Severity Index (PSI) score to determine outpatient oral antibiotic therapy vs inpatient admission.\n"
            "4. Schedule a follow-up chest radiograph in 4 to 6 weeks to confirm radiographic resolution and exclude an underlying endobronchial lesion."
        ),
    },
    {
        "id": "thoracic_pneumothorax",
        "title": "Thoracic Radiography: Pneumothorax & Tension Physiology",
        "category": "pathology",
        "keywords": "pneumothorax collapsed lung tension emergency chest tube needle decompression pleural line air",
        "content": (
            "Pneumothorax represents an acute medical emergency characterized by intrapleural air collection and visceral pleural line displacement:\n"
            "• Imaging Features: Thin, sharp white visceral pleural line separated from the chest wall with complete absence of peripheral lung vascular markings. Apical or lateral air cap on erect views.\n"
            "• Tension Pneumothorax Signs: Contralateral mediastinal shift, depression of the ipsilateral hemidiaphragm, and compression of the contralateral lung. This is a clinical emergency causing severe hypotension and hypoxia.\n"
            "• Next Steps & Management:\n"
            "  - Small, stable pneumothorax (< 2 cm interpleural distance): High-flow supplemental oxygen (enhances pleural air absorption) and observation with repeat CXR at 4–6 hours.\n"
            "  - Large pneumothorax (> 2 cm) or symptomatic: Tube thoracostomy (chest drain) insertion in the 4th/5th intercostal space mid-axillary line.\n"
            "  - Tension pneumothorax: Immediate needle decompression in the 2nd intercostal space mid-clavicular line without waiting for additional imaging."
        ),
    },
    {
        "id": "thoracic_cardiomegaly_effusion",
        "title": "Thoracic Radiography: Cardiomegaly & Pleural Effusion",
        "category": "pathology",
        "keywords": "cardiomegaly pleural effusion heart enlargement fluid blunting costophrenic ctr heart failure",
        "content": (
            "Cardiomegaly and pleural effusion frequently present together in cardiopulmonary volume overload:\n"
            "• Cardiomegaly: Defined by a cardiothoracic ratio (CTR) > 0.50 on posteroanterior (PA) erect chest radiographs. Prominence of upper-lobe pulmonary veins indicates cephalization and elevated pulmonary capillary wedge pressure.\n"
            "• Pleural Effusion: Manifests as dependent fluid density causing blunting of the posterior costophrenic angle (> 50 mL fluid) and lateral costophrenic angle (> 175 mL fluid) with a characteristic meniscus sign. Associated compressive atelectasis is typical at the adjacent lung base.\n"
            "• Next Steps:\n"
            "  - Evaluate for congestive heart failure (CHF) vs infectious, inflammatory, or malignant etiologies.\n"
            "  - Check BNP/NT-proBNP and perform transthoracic echocardiography to assess left ventricular ejection fraction.\n"
            "  - If effusion is unilateral or persistent, consider diagnostic thoracentesis with Light's criteria (transudate vs exudate)."
        ),
    },
    {
        "id": "chexnet_18_pathologies",
        "title": "CheXNet Multi-Pathology Screening Framework (18 Classes)",
        "category": "models",
        "keywords": "chexnet densenet121 18 pathologies torchxrayvision atelectasis edema mass nodule fibrosis emphysema",
        "content": (
            "The platform evaluates 18 distinct thoracic conditions simultaneously via DenseNet-121 (TorchXRayVision):\n"
            "Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, "
            "Nodule, Pleural Thickening, Pneumonia, Pneumothorax, Fracture, Lung Opacity, Enlarged Cardiomediastinum, and Normal.\n"
            "Each condition outputs an independent sigmoid calibrated probability [0, 1] to support co-existing multi-label disease states."
        ),
    },
    {
        "id": "brain_mri_midline_shift",
        "title": "Brain MRI: Midline Shift, Vasogenic Edema & Mass Effect",
        "category": "pathology",
        "keywords": "brain mri tumor glioma mass midline shift edema herniation ventricles intracranial pressure",
        "content": (
            "Brain MRI evaluation focuses on parenchymal lesion volume, surrounding edema, and dangerous mass effect:\n"
            "• Midline Shift: Measured in millimeters as displacement of the septum pellucidum or third ventricle from the anatomical central sagitta. Midline shift >= 5 mm indicates severe mass effect with high risk for uncal or subfalcine herniation.\n"
            "• Vasogenic Edema: Appears hyperintense on T2/FLAIR sequences tracking white matter tracts. Differentiated from cytotoxic edema (stroke) by preserved cortical gray ribbon.\n"
            "• Next Steps:\n"
            "  1. Immediate neurosurgical consultation if midline shift > 3 mm or clinical signs of elevated ICP (headache, vomiting, papilledema).\n"
            "  2. Administer intravenous corticosteroids (dexamethasone) for vasogenic edema reduction.\n"
            "  3. Perform high-resolution contrast-enhanced T1 brain MRI (gadolinium) and MR spectroscopy to characterize tumor margins and vascularity."
        ),
    },
    {
        "id": "medical_image_authenticity_validator",
        "title": "Medical Image Authenticity Validator & Quality Guards",
        "category": "system",
        "keywords": "validator rejected authenticity fake scan non-medical selfie portrait rejected file upload error",
        "content": (
            "The medical authenticity validator (AI/validator.py) prevents clinical AI hallucinations by screening uploads before model inference:\n"
            "• Dynamic Range & Grayscale Entropy: Checks that scans exhibit genuine radiographic sensor histograms rather than flat or RGB consumer photos.\n"
            "• Anatomical Profile Guards: Verifies expected modality geometry (bilateral thoracic rib cage for X-ray, oval calvarium for brain MRI, thoracic wall for CT, acoustic sector for ultrasound).\n"
            "• Rejection Causes: Uploads of monochrome selfies, natural scenes, documents, or cross-modality mismatches (e.g. uploading an ultrasound scan when X-Ray is selected) are strictly rejected with an explicit clinical safety error."
        ),
    },
    {
        "id": "montage_grid_detector",
        "title": "Automated Montage Grid Detection & Multi-Slice CT/MRI Slicing",
        "category": "system",
        "keywords": "montage detector slice grid multi-slice tiles reconstruction 3d volumetrics",
        "content": (
            "Clinical CT and MRI scans frequently arrive as 2x2, 3x3, or 4x4 multi-slice montage grids:\n"
            "• Automated Layout Detection: AI/montage_detector.py segments individual axial tiles using horizontal and vertical dividing border analysis.\n"
            "• Individual Slice Screening: Runs neural feature extraction across each segmented slice independently.\n"
            "• Volumetric Integration: Aggregates maximum lesion diameter, discrete 3D voxel volume (mm³), and physical mass across all slices."
        ),
    },
    {
        "id": "dicom_pacs_workflow",
        "title": "DICOM Standards & Orthanc PACS Integration",
        "category": "system",
        "keywords": "dicom pacs orthanc cornerstone window leveling contrast hounsfield wado-rs qido-rs",
        "content": (
            "Native medical imaging workflow standards supported by the platform:\n"
            "• 16-Bit Dynamic Range: Reads full 4096-65536 shades of gray via pydicom without 8-bit dynamic loss.\n"
            "• Window Leveling: Presets for Lung (WW 1500 / WC -600), Soft Tissue (WW 400 / WC 40), and Bone (WW 1500 / WC 400).\n"
            "• Orthanc PACS Integration: Communicates via DICOMweb REST services (WADO-RS for retrieval, QIDO-RS for study queries, STOW-RS for storage)."
        ),
    },
]

# ── Local BM25 Semantic Retriever ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "let's", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
}


def _tokenize(text: str) -> list[str]:
    clean = re.sub(r"[^\w\s-]", " ", text.lower())
    tokens = [w for w in clean.split() if len(w) > 1 and w not in STOP_WORDS]
    return tokens


class BM25Retriever:
    """Zero-dependency Okapi BM25 semantic retriever."""

    def __init__(self, corpus: list[dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_tokens = [_tokenize(f"{d['title']} {d.get('keywords', '')} {d['content']}") for d in corpus]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        self.N = len(corpus)

        # Calculate document frequencies
        self.df: dict[str, int] = {}
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[dict[str, str], float]]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return [(self.corpus[i], 1.0) for i in range(min(top_k, len(self.corpus)))]

        scores: list[float] = [0.0] * self.N
        for term in q_tokens:
            if term not in self.df:
                continue
            df_val = self.df[term]
            idf = math.log((self.N - df_val + 0.5) / (df_val + 0.5) + 1.0)

            for i, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(term)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avg_doc_len))
                scores[i] += idf * (num / denom)

        ranked_indices = sorted(range(self.N), key=lambda i: scores[i], reverse=True)
        return [(self.corpus[i], round(scores[i], 3)) for i in ranked_indices[:top_k]]


# ── RAG System Orchestrator ───────────────────────────────────────────────────

def _format_study_context_doc(ctx: dict[str, Any]) -> dict[str, str]:
    """Dynamically converts active study context into an indexable clinical document."""
    label = ctx.get("primary_label") or ctx.get("label") or "Unknown Assessment"
    conf = ctx.get("confidence")
    conf_pct = f"{round(float(conf) * 100, 1)}%" if conf is not None else "Unspecified"
    mod = ctx.get("modality", "XR")
    tel = ctx.get("telemetry") or {}
    loc = tel.get("anatomical_location") or "the visualized anatomical region"
    diam = tel.get("diameter_mm")
    vol_mm3 = tel.get("volume_mm3")
    vol_ml = tel.get("volume_ml")
    birads = tel.get("birads")
    fleischner = tel.get("fleischner_category")
    shift = tel.get("midline_shift_mm")
    patient = ctx.get("patient_name") or "Patient"
    findings = ctx.get("findings") or ""
    impression = ctx.get("impression") or ""

    details = [
        f"Active Patient: {patient}",
        f"Imaging Modality: {mod}",
        f"Primary Neural Network Assessment: {label} (Confidence: {conf_pct})",
        f"Anatomical Location: {loc}",
    ]
    if diam:
        details.append(f"Estimated Span / Diameter: {diam} mm")
    if vol_mm3:
        details.append(f"Calculated 3D Voxel Volume: {vol_mm3} mm³")
    if vol_ml:
        details.append(f"Lesion Volume: {vol_ml} mL")
    if shift is not None and shift > 0:
        details.append(f"Midline Shift: {shift} mm")
    if fleischner:
        details.append(f"Fleischner 2017 Category: {fleischner}")
    if birads:
        details.append(f"ACR BI-RADS Category: {birads}")
    if findings:
        details.append(f"Synthesized Findings: {re.sub('<[^<]+?>', ' ', findings)}")
    if impression:
        details.append(f"Clinical Impression: {re.sub('<[^<]+?>', ' ', impression)}")

    return {
        "id": "active_patient_study",
        "title": f"Active Patient Study: {label} [{mod}]",
        "category": "active_study",
        "keywords": f"patient {patient} study {label} {mod} {loc} measurement metrics lesion",
        "content": "\n".join(details),
    }


async def _generate_llm_response(prompt: str, context_text: str) -> Optional[str]:
    """Call optional Gemini or OpenAI endpoints if configured in environment."""
    # 1. Google Gemini API
    if settings.gemini_api_key:
        model = settings.gemini_model or "gemini-3.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are the Radiology AI Clinical Assistant. Provide an ULTRA-SHORT, direct response "
                                "(maximum 2 to 3 sentences or 3 concise bullet points, under 60-80 words total).\n"
                                "- If the user query is a conversational greeting, thanks, or acknowledgment (e.g. 'thank you', 'ok', 'good morning'), reply politely in 1 short sentence.\n"
                                "- If the user asks a clinical, diagnostic, or technical question, state the key finding or next action directly without boilerplate, essays, or unnecessary disclaimers.\n"
                                "- Adhere to ACR and Fleischner standards.\n"
                                "- Ground your answer strictly in the provided medical context.\n\n"
                                f"### MEDICAL EVIDENCE & CONTEXT:\n{context_text}\n\n"
                                f"### USER QUERY:\n{prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                        if text:
                            return text
        except Exception:
            pass

    # 2. OpenAI / Ollama compatible endpoint
    if settings.openai_api_key or settings.openai_api_base:
        base_url = settings.openai_api_base or "https://api.openai.com/v1"
        headers = {"Content-Type": "application/json"}
        if settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        payload = {
            "model": "gpt-4o-mini" if "openai.com" in base_url else "llama3.2",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Radiology AI Clinical Assistant. Provide ultra-short responses "
                        "(max 2-3 sentences or 3 concise bullets, under 60-80 words) grounded strictly in evidence."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"},
            ],
            "temperature": 0.2,
            "max_tokens": 150,
        }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass

    return None


def _synthesize_deterministic_rag_answer(
    query: str,
    top_docs: list[tuple[dict[str, str], float]],
    study_context: Optional[dict[str, Any]] = None,
) -> str:
    """
    High-fidelity deterministic clinical synthesizer (zero-API fallback).
    Synthesizes retrieved clinical guidelines and active patient study into an exact, structured answer.
    """
    q_lower = query.lower()
    has_study = bool(study_context and (study_context.get("primary_label") or study_context.get("label")))

    # 0. Conversational pleasantries & gratitude (prevent irrelevant document dumps)
    if re.search(r"^(thanks?|thank you|thx|much appreciated|appreciate it|thank you so much)[\s!.,]*$", q_lower):
        return "You're welcome! Let me know if you have any questions about this study or need help interpreting findings."
    if re.search(r"^(ok|okay|got it|understood|sure|alright|great|perfect|sounds good)[\s!.,]*$", q_lower):
        return "Understood. Feel free to ask whenever you need further details or guideline recommendations."
    if re.search(r"^(bye|goodbye|see you|take care)[\s!.,]*$", q_lower):
        return "Goodbye! Feel free to return whenever you need clinical imaging support."

    # 1. Plain language / patient consultation explanation
    if any(w in q_lower for w in ["plain", "simple", "patient", "layman", "easy words", "simple terms", "consult"]):
        if has_study:
            label = study_context.get("primary_label") or study_context.get("label")
            mod = study_context.get("modality", "XR")
            conf = round(float(study_context.get("confidence", 0.9)) * 100, 1)
            tel = study_context.get("telemetry") or {}
            loc = tel.get("anatomical_location", "the examined area")

            if any(n in label for n in ("Normal", "No Acute", "Negative")):
                return (
                    f"**Patient Summary ({mod})**: Your scan is **normal** ({conf}% AI certainty) with no signs of acute infection or masses. "
                    "Your doctor will discuss your symptoms and ongoing care."
                )
            return (
                f"**Patient Summary ({mod})**: The scan shows **{label}** in the **{loc}** ({conf}% confidence). "
                "This warrants medical follow-up; your physician and radiologist will recommend the next steps."
            )

    # 2. Next clinical steps & protocol
    is_next_step = ("next" in q_lower and "step" in q_lower) or any(
        w in q_lower for w in ["what should", "what to do", "protocol", "recommend", "management", "follow-up", "follow up", "action plan"]
    )
    if is_next_step:
        if has_study:
            label = study_context.get("primary_label") or study_context.get("label")
            mod = study_context.get("modality", "XR")
            tel = study_context.get("telemetry") or {}
            loc = tel.get("anatomical_location", "visualized anatomy")
            conf = round(float(study_context.get("confidence", 0.9)) * 100, 1)
            fleischner = tel.get("fleischner_category")
            birads = tel.get("birads")
            shift = tel.get("midline_shift_mm")

            lines = [f"**Next Steps ({mod} - {label})**:"]
            lines.append(f"1. **Anatomical Verification**: Abnormality centered in **{loc}** ({conf}% confidence). Review Grad-CAM heatmap overlay.")

            if mod == "CT":
                lines.append(f"2. **Fleischner Protocol**: {fleischner or 'Solid nodule follow-up CT in 3–12 months'}.")
            elif mod == "MR":
                if shift and shift >= 3.0:
                    lines.append(f"2. **URGENT**: Midline shift of {shift} mm. Immediate neurosurgical consult indicated.")
                else:
                    lines.append("2. **Contrast Mapping**: Obtain contrast-enhanced MRI (gadolinium); evaluate intracranial pressure.")
            elif mod == "US":
                lines.append(f"2. **ACR BI-RADS Protocol**: Categorized as **{birads or 'BI-RADS 4'}**; ultrasound-guided biopsy indicated.")
            else:
                if "Pneumothorax" in label:
                    lines.append("2. **Emergency**: Evaluate for tension physiology and urgent tube thoracostomy.")
                elif "Pneumonia" in label:
                    lines.append("2. **Infectious Workup**: Correlate with WBC/CRP and auscultation; follow-up CXR in 4–6 weeks.")
                else:
                    lines.append("2. **Clinical Correlation**: Correlate with auscultation, clinical history, and inflammatory labs.")

            return "\n".join(lines)

        # General next steps when no study is active
        return (
            "**Clinical Diagnostic Pathway**:\n"
            "1. **Verify Anatomy**: Inspect Grad-CAM heatmap to ensure opacity alignment.\n"
            "2. **Correlate Clinically**: Match findings with vitals and inflammatory markers.\n"
            "3. **Apply Guidelines**: Fleischner (CT), BI-RADS (US), or emergency chest tube (Pneumothorax)."
        )

    # 3. Measurement readout
    if any(w in q_lower for w in ["measure", "diameter", "volume", "mass", "metrics", "size", "span", "telemetry"]):
        if has_study:
            tel = study_context.get("telemetry") or {}
            label = study_context.get("primary_label") or study_context.get("label")
            mod = study_context.get("modality", "XR")
            loc = tel.get("anatomical_location", "visualized area")
            lines = [f"**Measurements ({mod} - {label})**:"]
            lines.append(f"• **Location**: {loc}")
            if tel.get("diameter_mm"):
                lines.append(f"• **Diameter**: {tel['diameter_mm']} mm")
            if tel.get("volume_mm3"):
                lines.append(f"• **3D Volume**: {tel['volume_mm3']} mm³")
            if tel.get("volume_ml"):
                lines.append(f"• **Volume**: {tel['volume_ml']} mL")
            if tel.get("mass_g"):
                lines.append(f"• **Mass**: {tel['mass_g']} g")
            if tel.get("midline_shift_mm") is not None and tel.get("midline_shift_mm") > 0:
                lines.append(f"• **Midline Shift**: {tel['midline_shift_mm']} mm")
            if tel.get("fleischner_category"):
                lines.append(f"• **Fleischner**: {tel['fleischner_category']}")
            if tel.get("birads"):
                lines.append(f"• **BI-RADS**: {tel['birads']}")
            return "\n".join(lines)

    # 4. Specific study findings or diagnosis queries
    if any(w in q_lower for w in ["finding", "diagnosis", "what does this show", "what does the scan show", "interpretation", "abnormality", "condition", "what is wrong"]):
        if has_study:
            label = study_context.get("primary_label") or study_context.get("label")
            mod = study_context.get("modality", "XR")
            conf = round(float(study_context.get("confidence", 0.9)) * 100, 1)
            tel = study_context.get("telemetry") or {}
            loc = tel.get("anatomical_location", "the visualized anatomy")
            diam = f", measuring approximately {tel.get('diameter_mm')} mm" if tel.get("diameter_mm") else ""
            if any(n in label for n in ("Normal", "No Acute", "Negative")):
                return f"**Study Finding ({mod})**: Primary assessment is **{label}** ({conf}% confidence). No acute focal lesions detected."
            return f"**Study Finding ({mod})**: Primary finding is **{label}** ({conf}% confidence) in the **{loc}**{diam}. Check Grad-CAM overlay to review the focal region."

    # 5. Standard RAG extraction from top retrieved documents (threshold guarded)
    if top_docs and top_docs[0][1] >= 0.15:
        best_doc, score = top_docs[0]
        # Prevent verbatim raw telemetry dump of active study
        if best_doc.get("id") == "active_patient_study":
            if has_study:
                label = study_context.get("primary_label") or study_context.get("label")
                mod = study_context.get("modality", "XR")
                conf = round(float(study_context.get("confidence", 0.9)) * 100, 1)
                tel = study_context.get("telemetry") or {}
                loc = tel.get("anatomical_location", "the visualized anatomy")
                return f"**Current Study Summary ({mod})**: Primary finding is **{label}** ({conf}% confidence) centered at **{loc}**."
        else:
            return f"### {best_doc['title']}\n{best_doc['content']}"

    # Low relevance / ambiguous query fallback (appropriate guidance)
    if has_study:
        label = study_context.get("primary_label") or study_context.get("label")
        mod = study_context.get("modality", "XR")
        return (
            f"I can assist with questions regarding this active **{mod}** study (**{label}**):\n"
            "• **Next clinical steps** & guidelines (Fleischner, BI-RADS)\n"
            "• **Quantitative measurements** (diameter, volume)\n"
            "• **Plain-language explanation** for patients\n"
            "• **Grad-CAM explainability**\n\n"
            "What specific question do you have?"
        )

    return (
        "I can help with clinical and technical inquiries regarding:\n"
        "• Specific imaging findings, lesion metrics, and anatomical locations\n"
        "• Clinical guidelines (Fleischner Society criteria, ACR BI-RADS protocols)\n"
        "• Grad-CAM explainability and neural network attention\n"
        "• Supported modalities (Chest X-Ray, CT Chest, Brain MRI, Breast Ultrasound)\n\n"
        "Please ask a specific clinical question or analyze a study on the Analyze page."
    )


def _generate_llm_response_sync(prompt: str, context_text: str) -> Optional[str]:
    """Call optional Gemini or OpenAI endpoints synchronously if configured in environment."""
    # 1. Google Gemini API
    if settings.gemini_api_key:
        model = settings.gemini_model or "gemini-3.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": (
                                "You are the Radiology AI Clinical Assistant. Provide an ULTRA-SHORT, direct response "
                                "(maximum 2 to 3 sentences or 3 concise bullet points, under 60-80 words total).\n"
                                "- If the user query is a conversational greeting, thanks, or acknowledgment (e.g. 'thank you', 'ok', 'good morning'), reply politely in 1 short sentence.\n"
                                "- If the user asks a clinical, diagnostic, or technical question, state the key finding or next action directly without boilerplate, essays, or unnecessary disclaimers.\n"
                                "- Adhere to ACR and Fleischner standards.\n"
                                "- Ground your answer strictly in the provided medical context.\n\n"
                                f"### MEDICAL EVIDENCE & CONTEXT:\n{context_text}\n\n"
                                f"### USER QUERY:\n{prompt}"
                            )
                        }
                    ]
                }
            ],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 200},
        }
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        text = "".join(p.get("text", "") for p in parts if "text" in p).strip()
                        if text:
                            return text
        except Exception:
            pass

    # 2. OpenAI / Ollama compatible endpoint
    if settings.openai_api_key or settings.openai_api_base:
        base_url = settings.openai_api_base or "https://api.openai.com/v1"
        headers = {"Content-Type": "application/json"}
        if settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        payload = {
            "model": "gpt-4o-mini" if "openai.com" in base_url else "llama3.2",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the Radiology AI Clinical Assistant. Provide ultra-short responses "
                        "(max 2-3 sentences or 3 concise bullets, under 60-80 words) grounded strictly in evidence."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {prompt}"},
            ],
            "temperature": 0.2,
            "max_tokens": 150,
        }
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    choices = data.get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception:
            pass

    return None


def answer_clinical_query_sync(query: str, study_context: Optional[dict[str, Any]] = None) -> str:
    """Synchronous RAG query processor: retrieves relevant medical knowledge and generates an answer."""
    clean_q = query.strip()
    if not clean_q:
        return "Please type a clinical or technical question and I'll be glad to help."

    corpus = list(MEDICAL_KNOWLEDGE_CORPUS)
    if study_context and (study_context.get("primary_label") or study_context.get("label")):
        study_doc = _format_study_context_doc(study_context)
        corpus.insert(0, study_doc)

    retriever = BM25Retriever(corpus)
    top_docs = retriever.retrieve(clean_q, top_k=settings.rag_top_k)

    context_chunks = []
    for doc, score in top_docs:
        context_chunks.append(f"[{doc['title']}]\n{doc['content']}")
    combined_context = "\n\n---\n\n".join(context_chunks)

    llm_reply = _generate_llm_response_sync(clean_q, combined_context)
    if llm_reply:
        return llm_reply

    return _synthesize_deterministic_rag_answer(clean_q, top_docs, study_context)


async def answer_clinical_query(query: str, study_context: Optional[dict[str, Any]] = None) -> str:
    """Main RAG query processor: retrieves relevant medical knowledge and generates an answer."""
    clean_q = query.strip()
    if not clean_q:
        return "Please type a clinical or technical question and I'll be glad to help."

    # Build active corpus: static guidelines + dynamic active patient document
    corpus = list(MEDICAL_KNOWLEDGE_CORPUS)
    if study_context and (study_context.get("primary_label") or study_context.get("label")):
        study_doc = _format_study_context_doc(study_context)
        corpus.insert(0, study_doc)

    # Retrieve top-k documents using BM25
    retriever = BM25Retriever(corpus)
    top_docs = retriever.retrieve(clean_q, top_k=settings.rag_top_k)

    # Format context for LLM / Synthesizer
    context_chunks = []
    for doc, score in top_docs:
        context_chunks.append(f"[{doc['title']}]\n{doc['content']}")
    combined_context = "\n\n---\n\n".join(context_chunks)

    # Attempt LLM generation if API configured
    llm_reply = await _generate_llm_response(clean_q, combined_context)
    if llm_reply:
        return llm_reply

    # Fallback to local deterministic clinical synthesis
    return _synthesize_deterministic_rag_answer(clean_q, top_docs, study_context)
