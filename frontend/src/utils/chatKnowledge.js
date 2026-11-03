/** Client-side radiology assistant — works even if API is unreachable. */

const TOPICS = [
  {
    keys: ['grad-cam', 'gradcam', 'heatmap', 'explain', 'explainability', 'why ai', 'attention'],
    answer:
      'Grad-CAM highlights the exact image regions that most influenced the neural network prediction. Warmer colors (red/yellow) indicate focal features like consolidations, effusions, or masses that DenseNet121 focused on. You can view the overlay directly in the results panel.',
  },
  {
    keys: ['pneumonia', 'lung infection', 'consolidation', 'infiltrate'],
    answer:
      'Pneumonia typically appears as focal airspace consolidation, patchy perihilar opacities, and air bronchograms. Our DenseNet121 model evaluates 18 chest pathologies simultaneously. Always correlate with clinical history and inflammatory markers.',
  },
  {
    keys: ['densenet', 'chexnet', 'torchxrayvision', 'neural', 'deep learning', 'ai model', 'torch'],
    answer:
      'We use DenseNet121 pretrained on ChestX-ray14 and MIMIC-CXR via TorchXRayVision. CT, MRI, and Ultrasound use dedicated morphological and 3D volumetric lesion screening pipelines.',
  },
  {
    keys: ['confidence', 'score', 'threshold', 'review', 'uncertainty'],
    answer:
      'Confidence reflects the calibrated neural network activation (0–100%). Scores ≥80% indicate high certainty; 60–80% warrant standard verification; below 60% is flagged as Needs Review.',
  },
  {
    keys: ['ct', 'computed tomography', 'nodule', 'lung nodule', 'fleischner'],
    answer:
      'CT Chest screening evaluates pulmonary nodules with discrete 3D voxel volumetrics. Management follows Fleischner Society guidelines: nodules <6 mm need no routine follow-up; 6–8 mm warrant repeat CT at 6–12 months; >8 mm suggest CT at 3 months, PET-CT, or biopsy.',
  },
  {
    keys: ['mri', 'mr ', 'brain', 'tumor', 'brats'],
    answer:
      'MRI Brain analysis evaluates hemispheric structural asymmetry, focal hyperintensity, and vasogenic edema. It calculates estimated lesion volume in mL and mass in grams.',
  },
  {
    keys: ['ultrasound', 'us ', 'breast', 'bi-rads', 'birads'],
    answer:
      'Ultrasound breast analysis evaluates focal lesions using ACR BI-RADS criteria: aspect ratio (taller-than-wide indicates BI-RADS 4/5 suspicion), margin roughness, and acoustic transmission.',
  },
  {
    keys: ['x-ray', 'xray', 'chest x', 'cxr', 'chest ray'],
    answer:
      'Chest X-Ray is our core screening modality. The AI screens across 18 pathologies, providing calibrated probabilities, differential diagnosis, Grad-CAM overlays, and PDF reports.',
  },
  {
    keys: ['report', 'pdf', 'download'],
    answer:
      'Click Download PDF Report on the Analyze results panel for an official structured report with patient demographics, AI telemetry, dynamic findings, and clinical impressions.',
  },
  {
    keys: ['dicom', 'pacs', 'orthanc', 'dicomweb'],
    answer:
      'Orthanc PACS integrates via DICOMweb at http://localhost:8042. Full 16-bit dynamic range, window/level presets, and metadata are preserved.',
  },
  {
    keys: ['upload', 'analyze', 'how to', 'use app', 'how do i', 'steps', 'workflow', 'start'],
    answer:
      'Workflow: Analyze → enter patient name → select modality → upload scan or pick demo case → click Run AI Analysis → inspect Grad-CAM and telemetry → download PDF report.',
  },
  {
    keys: ['normal', 'negative', 'no finding', 'healthy'],
    answer:
      'Normal means no acute focal consolidation, pneumothorax, effusion, or mass was detected above threshold. Clinical correlation remains essential if symptoms persist.',
  },
  {
    keys: ['hello', 'hi', 'hey', 'help', 'start'],
    answer:
      "Hello! I'm your Radiology AI clinical assistant. Ask about the current scan's findings, Grad-CAM explainability, Fleischner criteria, BI-RADS classifications, or PDF reports.",
  },
]

function normalize(text) {
  return text.toLowerCase().replace(/\s+/g, ' ').trim()
}

function replyFromStudy(text, ctx) {
  if (!ctx?.primary_label) return null
  const t = normalize(text)
  if (!/(this|my|result|finding|diagnosis|study|patient|mean|wrong|disease|where|location|measure|biopsy|next|do)/.test(t)) return null
  const pct = ctx.confidence != null ? Math.round(Number(ctx.confidence) * 1000) / 10 : null
  const conf = pct != null ? `${pct}%` : 'N/A'
  const tel = ctx.telemetry || {}
  const loc = tel.anatomical_location || 'the visualized anatomy'
  const diam = tel.diameter_mm ? `, measuring approximately ${tel.diameter_mm} mm,` : ''

  if (ctx.primary_label.includes('Normal') || ctx.primary_label.includes('No Acute') || ctx.primary_label.includes('Negative')) {
    return `For the current ${ctx.modality || 'XR'} study: The primary AI assessment is ${ctx.primary_label} (${conf} confidence). No acute focal lesions or consolidations were identified. Clinical correlation is recommended if acute symptoms persist.`
  }

  return `For the current ${ctx.modality || 'XR'} study: Primary finding is ${ctx.primary_label} (${conf} AI confidence) located in the ${loc}${diam}. Review the Grad-CAM heatmap overlay and structured PDF report for differential considerations and guideline-based follow-up.`
}

export function getLocalChatReply(message, studyContext = null) {
  const text = normalize(message)
  if (!text) return 'Please type a question.'

  const studyReply = replyFromStudy(text, studyContext)
  if (studyReply) return studyReply

  for (const { keys, answer } of TOPICS) {
    if (keys.some((k) => text.includes(k))) return answer
  }

  if (/\?/.test(message) || /^(what|how|why|when|where|can|does|is|who|which)\b/.test(text)) {
    return (
      'I can assist with clinical and technical inquiries regarding:\n' +
      '• Current study findings, lesion metrics, and anatomical locations\n' +
      '• Grad-CAM activation and neural network explainability\n' +
      '• Diagnostic guidelines (Fleischner nodule follow-up, BI-RADS protocols)\n' +
      '• Supported modalities (X-Ray, CT, MRI, Ultrasound)\n' +
      '• Generating structured PDF radiology reports\n\n' +
      'Please ask a specific question or type "help" for options.'
    )
  }

  return (
    'I\'m your Radiology AI assistant. Ask about findings, Grad-CAM, supported modalities, uploading scans, or generating reports. Type "help" for examples.'
  )
}
