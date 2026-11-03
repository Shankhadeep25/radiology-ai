"""Supported medical diagnoses and clinical metadata across modalities."""

SUPPORTED_DIAGNOSES = [
    "Pneumonia",
    "Tuberculosis",
    "COVID-19",
    "Pulmonary Edema",
    "Lung Cancer",
    "Pulmonary Nodule",
    "Pleural Effusion",
    "Cardiomegaly",
    "Atelectasis",
    "Consolidation",
    "Pneumothorax",
    "Lung Opacity",
    "Infiltration",
    "Pulmonary Fibrosis",
    "Emphysema",
    "Pleural Thickening",
    "Lung Lesion",
    "Rib Fracture",
    "Normal",
    # Modality specific (CT, MRI, Ultrasound)
    "Lung Nodule",
    "No Acute CT Finding",
    "Brain Tumor",
    "No Acute Intracranial Finding",
    "Suspicious Breast Lesion",
    "Probably Benign Lesion",
    "Negative Ultrasound",
]

DETECTION_THRESHOLD = 0.25  # Show as "possible"
POSITIVE_THRESHOLD = 0.45   # Show as "detected"

CONDITION_INFO = {
    "Pneumonia": {"category": "Infectious", "urgency": "high", "icon": "🫁"},
    "Tuberculosis": {"category": "Infectious", "urgency": "high", "icon": "🔬"},
    "COVID-19": {"category": "Infectious", "urgency": "high", "icon": "🦠"},
    "Pulmonary Edema": {"category": "Cardiovascular", "urgency": "high", "icon": "💧"},
    "Lung Cancer": {"category": "Oncology", "urgency": "critical", "icon": "⚠️"},
    "Pulmonary Nodule": {"category": "Oncology", "urgency": "medium", "icon": "🔴"},
    "Pleural Effusion": {"category": "Pleural", "urgency": "medium", "icon": "💦"},
    "Cardiomegaly": {"category": "Cardiovascular", "urgency": "medium", "icon": "❤️"},
    "Atelectasis": {"category": "Airway", "urgency": "medium", "icon": "📉"},
    "Consolidation": {"category": "Airspace", "urgency": "high", "icon": "🌫️"},
    "Pneumothorax": {"category": "Pleural", "urgency": "critical", "icon": "🚨"},
    "Lung Opacity": {"category": "Airspace", "urgency": "medium", "icon": "🌁"},
    "Infiltration": {"category": "Airspace", "urgency": "medium", "icon": "☁️"},
    "Pulmonary Fibrosis": {"category": "Interstitial", "urgency": "medium", "icon": "🕸️"},
    "Emphysema": {"category": "Airway", "urgency": "medium", "icon": "💨"},
    "Pleural Thickening": {"category": "Pleural", "urgency": "low", "icon": "🛡️"},
    "Lung Lesion": {"category": "Oncology", "urgency": "high", "icon": "🎯"},
    "Rib Fracture": {"category": "Musculoskeletal", "urgency": "high", "icon": "🦴"},
    "Normal": {"category": "Normal", "urgency": "none", "icon": "✅"},
    # CT
    "Lung Nodule": {"category": "Oncology", "urgency": "high", "icon": "🔴"},
    "No Acute CT Finding": {"category": "Normal", "urgency": "none", "icon": "✅"},
    # MRI
    "Brain Tumor": {"category": "Neuro-Oncology", "urgency": "critical", "icon": "🧠"},
    "No Acute Intracranial Finding": {"category": "Normal", "urgency": "none", "icon": "✅"},
    # Ultrasound
    "Suspicious Breast Lesion": {"category": "Breast Imaging", "urgency": "high", "icon": "⚡"},
    "Probably Benign Lesion": {"category": "Breast Imaging", "urgency": "medium", "icon": "🔍"},
    "Negative Ultrasound": {"category": "Normal", "urgency": "none", "icon": "✅"},
}
