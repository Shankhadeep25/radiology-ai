# 12-Slide PPT Outline — Clinical Presentation Outline

**Title:** Radiology AI Assistant — Explainable Multi-Modality Clinical Decision Support

Copy each slide into PowerPoint. Demo X-ray live; show CT/MRI/US as architecture + working pipelines.

---

## Slide 1 — Title
**Radiology AI Assistant**  
Computer vision platform with PACS integration and explainable imaging analytics  

Team | Radiology AI 2026  
*Assists radiologists. Does not replace them.*

---

## Slide 2 — Problem
- Imaging volumes rising; radiologist shortage (India ~1 : 100,000)
- Fatigue → missed findings
- Existing AI: black box, no PACS, no reports
- **Need:** triage + detect + **explain** + hospital workflow

---

## Slide 3 — Solution
One platform:
- Upload / PACS DICOM (X-ray, CT, MRI, Ultrasound)
- Modality-specific AI
- Grad-CAM heatmaps + confidence + uncertainty
- Structured report + PDF
- Orthanc PACS, DICOMweb, HL7/FHIR

---

## Slide 4 — MVP vs Full Spec (honest)

| Now (internal round) | Next phases |
|----------------------|-------------|
| **Chest X-ray** DenseNet121 / CheXNet | CT nodules (nnU-Net, LIDC-IDRI) |
| 12+ chest findings + Grad-CAM | MRI tumors + volume (BraTS, MONAI) |
| Report + PDF + dashboard | US BI-RADS (BUSI, MedSAM) |
| Orthanc + DICOMweb + FHIR stubs | Full QIDO/WADO production |

**Judges want one polished workflow** — X-ray is that workflow. CT/MRI/US are **live pipelines** in the same product.

---

## Slide 5 — Live Demo (2 min)
1. Login doctor@radiology.ai  
2. Analyze → Pneumonia Case → Run  
3. Show: prediction, confidence, Grad-CAM, 12 findings, PDF  
4. Switch modality to CT / MRI / US and upload a scan  
5. Patients → progression graph  

---

## Slide 6 — Architecture
```
User / PACS → Ingestion (pydicom, CLAHE, windowing)
           → Inference engine (XR DenseNet | CT | MR | US)
           → Explainability (Captum Grad-CAM)
           → CDS (triage, report, FHIR DiagnosticReport)
           → Dashboard + Cornerstone viewer + Orthanc
```

---

## Slide 7 — Tech Stack
AI: PyTorch, TorchXRayVision, Captum, OpenCV  
Future: MONAI, nnU-Net, MedSAM  
Backend: FastAPI, Celery, SQLite, JWT  
Frontend: React, Tailwind, Chart.js, Cornerstone.js  
PACS: Orthanc, DICOMweb QIDO-RS / WADO-RS  
Interop: HL7 FHIR ImagingStudy + DiagnosticReport  
Deploy: Docker Compose

---

## Slide 8 — Datasets & Models
| Modality | Dataset | Model |
|----------|---------|--------|
| X-ray | ChestX-ray14, RSNA Pneumonia | DenseNet121 / CheXNet |
| CT | LIDC-IDRI, DeepLesion | nnU-Net (hooked) |
| MRI | BraTS | 3D U-Net / MONAI (hooked) |
| US | BUSI | MedSAM + BI-RADS (hooked) |

Metrics: Accuracy, Precision, Recall, F1, AUC

---

## Slide 9 — Explainability & Safety
- Grad-CAM overlay (why the AI decided)
- Confidence meter + “Needs human review”
- Severity + differential diagnosis
- Audit logs, RBAC, encrypted images
- Every report: *final interpretation by radiologist required*

---

## Slide 10 — PACS / Hospital
- Orthanc Docker simulation
- DICOM upload → PACS
- DICOMweb QIDO-RS, WADO-RS APIs
- FHIR ImagingStudy + DiagnosticReport
- Cornerstone.js viewer

---

## Slide 11 — Impact & Market
- 30 min report → ~30 sec AI draft
- Rural / tele-radiology
- India imaging market growth
- SaaS hospital + API for telemedicine

---

## Slide 12 — Ask & Close
Built in hackathon window: working X-ray CDS + multi-modality architecture.  
**Ask:** mentor radiologist, hospital pilot, validated datasets.  
**Line:** *We don’t replace radiologists. We give them superpowers.*

---

## Judge one-liner
> “MVP is explainable chest X-ray CDS. Same platform already routes CT nodule, MRI tumor volume, and ultrasound BI-RADS, with Orthanc, DICOMweb, and FHIR — so we match the full problem statement without shipping four unfinished models.”
