# Baseline Models — Integration Reference

> **RESEARCH BASELINES. NOT CLINICALLY VALIDATED.**
> These models are DenseNet121 classifiers trained on single public datasets as part of a 7th-semester minor project.
> They are disabled by default and must be explicitly enabled only for demonstration purposes.

---

## Enabling Baseline Models

Add the following to your `.env` file (repo root):

```
ENABLE_BASELINE_MODELS=1
```

When the flag is `0`, absent, or unset the backend returns **HTTP 503** for every CT / MRI / US request.  
The X-ray path is unaffected by this flag.

---

## Weight Files

Place the trained weights in the repo-root `weights/` directory:

| File | Modality | Architecture |
|---|---|---|
| `weights/mri_densenet121.pt` | Brain MRI | DenseNet121, classifier → `nn.Linear(1024, 4)` |
| `weights/us_densenet121.pt` | Breast Ultrasound | DenseNet121, classifier → `nn.Linear(1024, 3)` |
| `weights/ct_densenet121.pt` | Chest CT | DenseNet121, classifier → `nn.Linear(1024, 4)` |

These files are listed in `.gitignore` (`weights/*.pt`) and are **never committed to the repository**.

Metrics JSONs (tracked):

| File | Contents |
|---|---|
| `weights/metrics/mri_metrics.json` | Accuracy CI, AUC CI, per-class sensitivity/specificity, wrong-modality test |
| `weights/metrics/us_metrics.json` | Accuracy CI, AUC CI, per-class sensitivity/specificity, wrong-modality test |
| `weights/metrics/ct_metrics.json` | Accuracy CI, balanced accuracy CI, AUC CI, per-class metrics, wrong-modality test |

---

## Model Details

### MRI — Brain Tumour Classification

**Dataset**: [Brain Tumour MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)  
**Classes** (order from metrics JSON): `glioma`, `meningioma`, `notumor`, `pituitary`  
**Backbone**: `torchvision.models.densenet121`, ImageNet pre-trained, classifier replaced  
**Preprocessing**: `Image.open().convert("RGB")` → `Resize(224, 224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`  
**No padding** (confirmed from training notebook save cell).

#### Test-set Metrics (hold-out, 1 600 images)

| Metric | Value | 95 % CI |
|---|---|---|
| Accuracy | 0.894 | 0.879 – 0.909 |
| Macro AUC | 0.975 | 0.969 – 0.981 |
| Balanced accuracy | — | — |

#### Per-class Sensitivity (test set)

| Class | Sensitivity | Specificity |
|---|---|---|
| glioma (0) | 0.785 | 0.996 |
| meningioma (1) | 0.818 | 0.977 |
| notumor (2) | 0.995 | 0.898 |
| pituitary (3) | 0.980 | 0.988 |

#### Wrong-Modality Behaviour (100 images each)

| Input modality | Mean top confidence | Fraction above 0.9 review threshold |
|---|---|---|
| Breast ultrasound | 0.683 | 97 % |
| Chest CT | 0.858 | 100 % |
| Chest X-ray | 0.771 | 99 % |

#### Known Limitations

- ~15 % of tumour images are predicted as `notumor` (meningioma and glioma sensitivity < 82 %).
- Calibration did not hold on the validation split (validation set likely shares near-copies with training).
- Review threshold from the JSON `review` block **did not hold** — it is ignored in the integration.
- Wrong-modality inputs receive high-confidence answers (68–86 % mean confidence).
- Trained on public data only; not evaluated on real clinical images.

---

### US — Breast Ultrasound Classification

**Dataset**: [BUSI — Breast Ultrasound Images Dataset (Kaggle)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)  
**Classes** (order from metrics JSON): `benign`, `malignant`, `normal`  
**Backbone**: `torchvision.models.densenet121`, ImageNet pre-trained, classifier replaced  
**Preprocessing**: `Image.open().convert("RGB")` → `pad_square` (black fill, centre crop to max(w,h)) → `Resize(224, 224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`

#### Test-set Metrics (hold-out, 150 images)

| Metric | Value | 95 % CI |
|---|---|---|
| Accuracy | 0.800 | 0.733 – 0.867 |
| Macro AUC | 0.922 | 0.880 – 0.957 |
| Balanced accuracy | — | — |

#### Per-class Sensitivity (test set)

| Class | Sensitivity | Specificity | AUC |
|---|---|---|---|
| benign | 0.851 | 0.849 | 0.920 |
| malignant | 0.735 | 0.849 | 0.881 |
| normal | 0.737 | 0.970 | 0.965 |

#### Wrong-Modality Behaviour (100 images each)

| Input modality | Mean top confidence | Fraction above 0.8 confidence |
|---|---|---|
| Brain MRI | 0.889 | 78 % |
| Chest CT | 0.892 | 78 % |

#### Known Limitations

- Malignant sensitivity is 0.735 — 26 % of malignant cases misclassified.
- `malignant` called `normal`: 95 % CI upper bound 14.7 % (not zero).
- Calibration did not transfer to the held-out split.
- Wrong-modality inputs (brain MRI, chest CT) receive 78 %+ of predictions above 0.8 confidence.
- Trained on public data only; not evaluated on real clinical images.

---

### CT — Chest CT Cancer Classification

**Dataset**: [Chest CT-Scan Images Dataset (Kaggle)](https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images)  
**Classes** (order from metrics JSON): `adenocarcinoma`, `large_cell`, `normal`, `squamous`  
**Backbone**: `torchvision.models.densenet121`, ImageNet pre-trained, classifier replaced  
**Preprocessing**: `Image.open().convert("RGB")` → `pad_square` (black fill, centre crop to max(w,h)) → `Resize(224, 224)` → `ToTensor` → `Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])`

#### Test-set Metrics (hold-out, 139 images)

| Metric | Value | 95 % CI |
|---|---|---|
| Accuracy | 0.594 | 0.532 – 0.651 |
| Balanced accuracy | 0.677 | 0.617 – 0.730 |
| Macro AUC | 0.885 | 0.854 – 0.911 |

#### Per-class Sensitivity (test set)

| Class | Sensitivity | Specificity | AUC |
|---|---|---|---|
| adenocarcinoma | 0.633 | 0.747 | 0.778 |
| large_cell | 0.961 | 0.705 | 0.918 |
| normal | 0.824 | 0.989 | 0.965 |
| squamous | 0.289 | 0.984 | 0.882 |

#### Wrong-Modality Behaviour (100 images each)

| Input modality | Mean top confidence | Fraction above 0.8 confidence | Most common prediction |
|---|---|---|---|
| Brain MRI | 0.742 | 42 % | normal (61 %) |
| Breast ultrasound | 0.677 | 33 % | normal (71 %) |

#### Known Limitations

- **Squamous sensitivity is 0.289** — 71 % of squamous carcinoma cases misclassified.
- Overall accuracy 0.594 is below the balanced-accuracy figure because class distributions are uneven.
- Confidence is overstated (calibration did not hold).
- Wrong-modality inputs predominantly predicted as `normal` with moderate-to-high confidence.
- Trained on public data only; not evaluated on real clinical images.

---

## Common Limitations (All Three Models)

- Trained on a **single public dataset** each. No multi-site, multi-scanner, or clinical validation.
- **Wrong-modality inputs receive confident answers** — the models have no out-of-distribution detector.
- Calibration did not hold for any model on the held-out validation split.
- No patient-level de-identification was verified by the project team.
- **Not intended and not suitable for diagnosis or patient care.**

---

## Running the Check Script

After placing sample images in `demo_samples/baseline/<model>/<true_class>/`, run:

```bash
python scripts/check_baselines.py
```

Expected folder structure:

```
demo_samples/baseline/
  mri/
    glioma/      ← .jpg or .png files
    meningioma/
    notumor/
    pituitary/
  us/
    benign/
    malignant/
    normal/
  ct/
    adenocarcinoma/
    large_cell/
    normal/
    squamous/
```

If the folder does not exist, the script prints these instructions and exits cleanly.

The script requires `ENABLE_BASELINE_MODELS=1` in the environment (or `.env`).

---

## API Endpoints

| Endpoint | Behaviour when flag is ON | Behaviour when flag is OFF |
|---|---|---|
| `POST /api/v1/predict` with CT/MR/US | Returns baseline prediction JSON | HTTP 503 |
| `POST /api/v1/gradcam` with CT/MR/US | HTTP 503 (Grad-CAM not implemented for baselines) | HTTP 503 |
| `POST /api/v1/report` with CT/MR/US | Returns report with disclaimer header | HTTP 503 |
| `GET /api/v1/baseline/status` | Returns `enabled`, `weights_present`, `loaded`, `metrics` per model | Same (flag state reported accurately) |

All baseline predictions include:

```json
{
  "calibrated": false,
  "in_review_band": null,
  "status": "research_baseline",
  "disclaimer": "Research baseline trained on one public dataset. Not clinically validated. Probabilities are not calibrated.",
  "modality_warning": "Wrong-modality images can still receive confident answers. Use only images of the matching modality.",
  "model_metrics": { "accuracy": ..., "macro_auc": ..., "accuracy_ci": [...], "macro_auc_ci": [...] }
}
```

