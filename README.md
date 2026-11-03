# Radiology AI Assistant

**Explainable AI Clinical Decision Support for Chest X-Ray Analysis**

An advanced clinical decision support radiology assistant that detects pneumonia from chest X-rays using pretrained DenseNet121, provides Grad-CAM explainability, generates structured radiology reports, and presents everything in a polished doctor dashboard.

## Features

- **Chest X-Ray Analysis** — Upload PNG, JPG, or DICOM scans
- **AI Prediction** — DenseNet121 via TorchXRayVision (18 pathology classes)
- **Grad-CAM Explainability** — Visual heatmap highlighting areas of concern
- **Auto Report Generation** — Clinical findings + impression with PDF export
- **Differential Diagnosis** — Ranked probability list (Pneumonia, TB, Edema, etc.)
- **Confidence Meter** — AI uncertainty with "Needs Human Review" flagging
- **Severity Scoring** — Mild / Moderate / Severe classification
- **Doctor Dashboard** — Metrics, charts, study history
- **JWT Authentication** — Role-based access (doctor, radiologist, admin)
- **Docker Ready** — One-command deployment with optional Orthanc PACS

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Docker & Docker Compose

### Backend

```bash
cd radiology-ai/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

API docs: http://localhost:8000/docs

### Frontend

```bash
cd radiology-ai/frontend
npm install
npm run dev
```

App: http://localhost:5173

**Demo login:** `doctor@radiology.ai` / `demo1234`

### Docker (Full Stack)

```bash
cd radiology-ai/docker
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Orthanc PACS (optional): `docker compose --profile pacs up`

### Cloud Deploy

See [docs/DEPLOY.md](docs/DEPLOY.md) for Render + Vercel (public URL) or VPS instructions.

## Project Structure

```
radiology-ai/
├── backend/           # FastAPI REST API
│   ├── app.py         # Entry point
│   ├── app/
│   │   ├── api/       # Route handlers
│   │   ├── models/    # SQLAlchemy + Pydantic schemas
│   │   └── services/  # Business logic
│   └── requirements.txt
├── frontend/          # React + Tailwind dashboard
│   └── src/
│       ├── pages/     # Dashboard, Analyze, History
│       └── components/
├── AI/                # ML pipeline
│   ├── preprocessing.py
│   ├── predict.py     # TorchXRayVision inference
│   ├── gradcam.py     # Explainability
│   ├── report_generator.py
│   └── train.py       # Fine-tuning script
├── docker/            # Docker Compose + Nginx
├── docs/              # Architecture + roadmap
├── weights/           # Custom model weights
└── datasets/          # Training data
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/upload` | Upload chest X-ray |
| POST | `/api/v1/predict/{id}` | Run AI analysis |
| GET | `/api/v1/history` | Study history |
| GET | `/api/v1/metrics` | Dashboard metrics |
| GET | `/api/v1/patients` | List patients |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI | PyTorch, TorchXRayVision, OpenCV, Captum, pydicom |
| Backend | FastAPI, SQLAlchemy, SQLite, JWT |
| Frontend | React, Tailwind CSS, Chart.js |
| Deployment | Docker, Nginx, Redis, Orthanc |

## Training (Optional)

```bash
# Place images in datasets/rsna/NORMAL and datasets/rsna/PNEUMONIA
cd AI
python train.py --epochs 10 --batch-size 32
```

## License

MIT License
