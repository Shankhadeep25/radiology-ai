/** Client-side radiology assistant — works even if API is unreachable. */

const TOPICS = [
  {
    keys: ['next clinical', 'next step', 'clinical step', 'what to do', 'protocol', 'management', 'recommendation', 'follow up', 'clinical steps'],
    answer:
      '**Standard Clinical Diagnostic Pathway**:\n' +
      '1. **Verify Anatomy**: Inspect Grad-CAM heatmap to ensure genuine pathological opacity.\n' +
      '2. **Correlate Clinically**: Check vitals and inflammatory markers (CBC, CRP).\n' +
      '3. **Apply Guidelines**: Fleischner criteria for CT nodules, BI-RADS for US, emergency chest tube for pneumothorax.\n\n' +
      '*Analyze a scan on the Analyze page to receive patient-specific next steps.*',
  },
  {
    keys: ['validator', 'rejected', 'reject', 'authenticity', 'fake scan', 'non-medical', 'invalid image', 'selfie', 'portrait'],
    answer:
      'Our medical authenticity validator screens dynamic range, grayscale entropy, and anatomical contours. Monochrome selfies, natural photos, or cross-modality mismatches are automatically rejected.',
  },
  {
    keys: ['montage', 'slice grid', 'series', 'multi-slice', 'sub-slices', 'grid'],
    answer:
      'The automated montage detector segments 2x2, 3x3, or 4x4 multi-slice CT/MRI grids, running deep learning on each slice to compute 3D volume (mm³) and maximum lesion diameter.',
  },
  {
    keys: ['pneumothorax', 'collapsed lung'],
    answer:
      '**Pneumothorax** is an acute emergency marked by visceral pleural line separation and absent lung markings. Flagged as High Urgency for immediate evaluation of tension physiology.',
  },
  {
    keys: ['cardiomegaly', 'enlarged heart', 'ctr'],
    answer:
      '**Cardiomegaly** is diagnosed when the cardiothoracic ratio (CTR) exceeds 0.50 on PA chest radiographs, frequently associated with heart failure or hypertension.',
  },
  {
    keys: ['pleural effusion', 'effusion', 'fluid'],
    answer:
      '**Pleural effusion** appears as dependent fluid blunting costophrenic angles with meniscus sign. Correlate with ultrasound/CT and consider thoracentesis if unilateral.',
  },
  {
    keys: ['grad-cam', 'gradcam', 'heatmap', 'explain', 'explainability', 'why ai', 'attention'],
    answer:
      '**Grad-CAM** highlights the convolutional regions that most influenced the AI prediction. Warmer colors (red/yellow) indicate focal abnormalities like consolidations or masses.',
  },
  {
    keys: ['pneumonia', 'lung infection', 'consolidation', 'infiltrate'],
    answer:
      '**Pneumonia** presents with focal airspace consolidation and air bronchograms. Correlate with fever, auscultation, and inflammatory markers before prescribing antibiotics.',
  },
  {
    keys: ['densenet', 'chexnet', 'torchxrayvision', 'neural', 'deep learning', 'ai model', 'torch'],
    answer:
      'The platform uses **DenseNet-121** pretrained on ChestX-ray14 and MIMIC-CXR for 18 chest conditions, alongside dedicated 3D volumetric pipelines for CT, MRI, and Ultrasound.',
  },
  {
    keys: ['confidence', 'score', 'threshold', 'review', 'uncertainty'],
    answer:
      'Scores ≥80% indicate high certainty; 60–80% indicate moderate certainty; scores <60% are flagged as **Needs Review** for clinician verification.',
  },
  {
    keys: ['ct', 'computed tomography', 'nodule', 'lung nodule', 'fleischner'],
    answer:
      '**CT Nodule Protocol (Fleischner 2017)**:\n• Solid <6 mm: No routine follow-up\n• Solid 6–8 mm: Repeat CT in 6–12 months\n• Solid >8 mm: CT at 3 months, PET-CT, or biopsy.',
  },
  {
    keys: ['mri', 'mr ', 'brain', 'tumor', 'brats'],
    answer:
      '**Brain MRI** evaluates parenchymal asymmetry, vasogenic edema, lesion volume (mL), and midline shift (mm). Shift ≥5 mm indicates urgent mass effect.',
  },
  {
    keys: ['ultrasound', 'us ', 'breast', 'bi-rads', 'birads'],
    answer:
      '**Breast Ultrasound (ACR BI-RADS)**: Evaluates aspect ratio and margin roughness. BI-RADS 3 has <2% risk (6-mo follow-up); BI-RADS 4/5 warrants core needle biopsy.',
  },
  {
    keys: ['x-ray', 'xray', 'chest x', 'cxr', 'chest ray'],
    answer:
      '**Chest X-Ray**: Screens 18 thoracic conditions with calibrated probabilities, Grad-CAM overlays, and structured downloadable PDF reports.',
  },
  {
    keys: ['report', 'pdf', 'download'],
    answer:
      'Click **Download PDF Report** on the Analyze panel to export a signed report with patient data, AI telemetry, dynamic findings, and recommendations.',
  },
  {
    keys: ['dicom', 'pacs', 'orthanc', 'dicomweb'],
    answer:
      'Integrated with Orthanc PACS via DICOMweb at http://localhost:8042 with 16-bit window/level presets (Lung, Soft Tissue, Bone).',
  },
  {
    keys: ['upload', 'analyze', 'how to', 'use app', 'how do i', 'steps', 'workflow', 'start'],
    answer:
      'Workflow: Analyze → select patient & modality → upload scan or pick demo case → click **Run AI Analysis** → inspect Grad-CAM & telemetry → download report.',
  },
  {
    keys: ['normal', 'negative', 'no finding', 'healthy'],
    answer:
      '**Normal**: No acute consolidation, pneumothorax, effusion, or suspicious mass detected. Clinical correlation is recommended if symptoms persist.',
  },
  {
    keys: ['hello', 'hi', 'hey', 'help', 'start'],
    answer:
      "Hello! I'm your Radiology AI Clinical Assistant. Ask about findings, Grad-CAM overlays, next clinical steps, or diagnostic guidelines.",
  },
]

function normalize(text) {
  return text.toLowerCase().replace(/\s+/g, ' ').trim()
}

function replyFromStudy(text, ctx) {
  if (!ctx?.primary_label) return null
  const t = normalize(text)
  const pct = ctx.confidence != null ? Math.round(Number(ctx.confidence) * 1000) / 10 : null
  const conf = pct != null ? `${pct}%` : 'N/A'
  const tel = ctx.telemetry || {}
  const loc = tel.anatomical_location || 'the visualized anatomy'
  const diam = tel.diameter_mm ? `, measuring approximately ${tel.diameter_mm} mm,` : ''
  const modality = ctx.modality || 'XR'
  const label = ctx.primary_label

  // 1. Plain language / patient consultation
  if (/(plain|simple|patient|layman|easy words|simple terms|consult)/.test(t)) {
    if (label.includes('Normal') || label.includes('No Acute') || label.includes('Negative')) {
      return `**Patient Summary (${modality})**:\nYour scan is **normal** (${conf} confidence) with no signs of acute infection or mass. Your physician will review this alongside any ongoing symptoms.`
    }
    return `**Patient Summary (${modality})**:\nThe scan identified **${label}** in the **${loc}** (${conf} confidence). This warrants clinical attention; your physician and radiologist will recommend appropriate next steps.`
  }

  // 2. Quantitative measurements
  if (/(measure|diameter|volume|mass|metrics|size|span|telemetry)/.test(t)) {
    const lines = [`**Measurements (${modality} - ${label})**:`]
    lines.push(`• **Location**: ${loc}`)
    if (tel.diameter_mm) lines.push(`• **Diameter**: ${tel.diameter_mm} mm`)
    if (tel.volume_mm3) lines.push(`• **3D Volume**: ${tel.volume_mm3} mm³`)
    if (tel.volume_ml) lines.push(`• **Volume**: ${tel.volume_ml} mL`)
    if (tel.midline_shift_mm != null && tel.midline_shift_mm > 0) lines.push(`• **Midline Shift**: ${tel.midline_shift_mm} mm`)
    if (tel.fleischner_category) lines.push(`• **Fleischner**: ${tel.fleischner_category}`)
    if (tel.birads) lines.push(`• **BI-RADS**: ${tel.birads}`)
    return lines.join('\n')
  }

  // 3. Clinical next steps
  if (/(next.*step|clinical.*step|recommend|protocol|what should|what to do|follow[- ]?up|biopsy|management)/.test(t)) {
    if (modality === 'CT') {
      return `**Next Steps (CT - ${label})**:\n1. Lesion identified in **${loc}** (${conf} confidence).\n2. Apply Fleischner 2017 criteria: ${tel.fleischner_category || 'CT follow-up at 3–12 months'}.\n3. Correlate with patient smoking and clinical risk profile.`
    }
    if (modality === 'MR') {
      return `**Next Steps (Brain MRI - ${label})**:\n1. Abnormal mass in **${loc}** (${conf} confidence).\n2. Contrast-enhanced MRI (gadolinium) recommended.\n3. Urgent neurosurgical evaluation if midline shift/mass effect is present.`
    }
    if (modality === 'US') {
      return `**Next Steps (Breast US - ${label})**:\n1. Lesion in **${loc}** (**${tel.birads || 'BI-RADS 4'}**, ${conf} confidence).\n2. Ultrasound-guided core needle biopsy indicated.\n3. Correlate with bilateral diagnostic mammography.`
    }
    return `**Next Steps (CXR - ${label})**:\n1. Finding centered in **${loc}** (${conf} confidence).\n2. Correlate with auscultation, temperature, and inflammatory labs (CBC, CRP).\n3. Repeat CXR in 4–6 weeks to verify radiographic resolution.`
  }

  // 4. General study findings
  if (/(this|my|result|finding|diagnosis|study|mean|wrong|disease|where|location)/.test(t)) {
    if (label.includes('Normal') || label.includes('No Acute') || label.includes('Negative')) {
      return `**Study Finding (${modality})**: Primary assessment is **${label}** (${conf} confidence). No acute focal lesions or consolidations were detected.`
    }
    return `**Study Finding (${modality})**: Primary finding is **${label}** (${conf} confidence) in the **${loc}**${diam}. Review the Grad-CAM heatmap overlay for focal localization.`
  }

  return null
}

export function getLocalChatReply(message, studyContext = null) {
  const text = normalize(message)
  if (!text) return 'Please type a question.'

  // Conversational pleasantries & gratitude
  if (/^(thanks?|thank you|thx|much appreciated|appreciate it|thank you so much)[\s!.,]*$/.test(text)) {
    return "You're welcome! Let me know if you have any questions about this study or need help interpreting findings."
  }
  if (/^(ok|okay|got it|understood|sure|alright|great|perfect|sounds good)[\s!.,]*$/.test(text)) {
    return "Understood. Feel free to ask whenever you need further details or guideline recommendations."
  }
  if (/^(bye|goodbye|see you|take care)[\s!.,]*$/.test(text)) {
    return "Goodbye! Feel free to return whenever you need clinical imaging support."
  }
  if (/^(hello|hi|hey|start|help|welcome|good morning|good evening|good afternoon)[\s!.,]*$/.test(text)) {
    return "Hello! I'm your Radiology AI Clinical Assistant. Ask about scan findings, clinical next steps, Grad-CAM explainability, or diagnostic guidelines."
  }

  const studyReply = replyFromStudy(text, studyContext)
  if (studyReply) return studyReply

  for (const { keys, answer } of TOPICS) {
    if (keys.some((k) => text.includes(k))) return answer
  }

  return (
    'I am your Radiology AI Clinical Assistant. You can ask about:\n' +
    '• Next clinical steps per Fleischner criteria or ACR BI-RADS\n' +
    '• Current scan findings, lesion diameter, volume, and location\n' +
    '• Grad-CAM explainability and neural network attention\n' +
    '• Medical image validation and montage scan processing'
  )
}
