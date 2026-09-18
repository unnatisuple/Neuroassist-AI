# 🧠 NeuroAssist AI v2

**Doctor-Only Explainable Clinical Decision Support System for Alzheimer's Disease**

> ⚠️ **INVESTIGATIONAL SOFTWARE** — Not FDA/CE cleared. Not a substitute for clinical judgment.

---

## Overview

NeuroAssist AI v2 is a secure, clinician-only platform for Alzheimer's disease analysis:

- **4-class MRI classification**: Non-Demented, Very Mild, Mild, Moderate
  - ⚠️ Severe Dementia is NOT supported by current training data
- **Class Probability Distribution**: Real-time softmax probability bars across all 4 stages with winner highlighting
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Macro convolutional attention localization (`layer4[1].conv2`)
  - **Grad-CAM++**: Focal, higher-order gradient-weighted feature attribution
  - **Integrated Gradients**: Captum path-integrated pixel-level attribution with black baseline
  - **Brain-Region Analysis**: Quantitative 2D image-space spatial attribution across 8 canonical anatomical regions (Hippocampus, Entorhinal Cortex, Temporal, Parietal, Frontal, Occipital, Cingulate, Precuneus) with clinical notes and disclaimers
- **Clinical risk assessment**: Transparent, rule-based scoring (NOT ML-based)
- **RAG-grounded reports**: Groq-generated reports citing clinical guidelines
- **Multilingual PDF export**: English, Hindi, Marathi with embedded live XAI visual overlays and regional attribution tables
- **Medical AI assistant**: Scope-limited Groq chatbot (Alzheimer's/dementia only)
- **Patient record management**: Longitudinal tracking and trend analysis
- **Audit trail**: Every prediction and report access is logged

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS + Material UI |
| Backend | FastAPI + Uvicorn + Pydantic + Motor (async MongoDB) |
| AI/ML | PyTorch + torchvision + timm + pytorch-grad-cam + Captum |
| LLM | **Groq API** (Configurable via GROQ_API_KEY & GROQ_MODEL) |
| RAG | FAISS + Sentence-Transformers |
| Database | MongoDB (Atlas or local) |
| PDF | ReportLab + Jinja2 |

---

## Project Structure

```
neuroassist-ai-v2/
├── frontend/               React + TypeScript + Vite
│   └── src/
│       ├── pages/          Login, Dashboard, MRI Analysis, Risk, Reports, Chatbot, Admin
│       ├── components/     Layout, shared components
│       ├── context/        AuthContext (JWT)
│       └── services/       Axios API client
├── backend/                FastAPI application
│   ├── routers/            auth, mri, risk, report, chatbot, patients, admin, health
│   ├── services/           model, explainability, RAG, PDF, recommendations
│   ├── middleware/         Global error handler, audit logging
│   ├── schemas/            Pydantic request/response models
│   ├── auth/               JWT + bcrypt
│   └── db/                 Motor async MongoDB
├── ml/                     Training pipeline (separate from serving)
│   ├── data/               Dataset download + preprocessing
│   ├── training/           Unified training script (7 architectures)
│   └── reports/            Evaluation artifacts + model cards
├── infra/                  Docker Compose (MongoDB + Redis)
└── .env.example            All required secrets
```

---

## Quick Start

### 1. Clone & Configure
```bash
git clone <repo-url>
cd neuroassist-ai-v2
cp .env.example .env
# Edit .env or .env.local: set JWT_SECRET_KEY, GROQ_API_KEY, MONGODB_URI
```

### 2. Start Infrastructure
```bash
cd infra
docker-compose up -d  # MongoDB + Redis
```

### 3. Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
# Build the RAG vector index (seeds sample guidelines out-of-the-box):
python backend/services/build_rag_index.py
# Start the API server:
uvicorn backend.main:app --reload --port 8000
# Default admin: admin@neuroassist.ai / admin123!CHANGE_ME
```

### 4. Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### 5. Train & Evaluate Models (4-Class Alzheimer MRI Dataset)
The ML pipeline uses the 6,400 MRI scans dataset organized into 4 clinical stages:
- **NonDemented**: 3,200 scans (50.0%)
- **VeryMildDemented**: 2,240 scans (35.0%)
- **MildDemented**: 896 scans (14.0%)
- **ModerateDemented**: 64 scans (1.0%)

All scripts use repository-relative paths and can be executed from anywhere in the repository:
```bash
# Generate healthy brain template (used by explainability and difference mapping):
python ml/training/generate_healthy_template.py

# Quick train a model (ResNet-18 or CNN):
python ml/train.py --arch resnet18 --epochs 10 --batch_size 32

# Evaluate best checkpoint on test set:
python ml/evaluate.py --checkpoint ml/checkpoints/best_model.pt

# Run experiment levers (A through E):
python ml/run_experiments.py --lever b
```

---

## Key Design Decisions

1. **Doctor-only**: No patient-facing routes. Login says "Clinician Portal."
2. **No fake data**: If the model isn't trained, the UI shows "Model not loaded" — never mock predictions.
3. **Groq LLM**: Official Groq SDK provider abstraction powering the medical assistant and clinical reports.
4. **Rule-based risk**: Clinical risk assessment is explicitly labeled as rule-based, not ML.
5. **Audit trail**: Every prediction and report access is logged with trace IDs.
6. **Regulatory disclaimers**: Visible on every clinical output screen and report.
7. **File Encryption at Rest**: MRI scans uploaded to the system are symmetrically encrypted (AES-256) on disk using cryptography Fernet, ensuring HIPAA/GDPR compliance for local patient files.

---

## Regulatory Notice

This system is a **decision-support tool**, not an autonomous diagnostic device. Regulatory clearance (FDA SaMD, CE under MDR, CDSCO) is a separate legal/business process. This software includes:
- Disclaimers on every clinical output
- Audit logging of every prediction and who reviewed it
- `DECISION_SUPPORT_MODE_ONLY=true` config flag
- Model version + checksum tracking for auditability

---

## License

MIT — for academic and research use only.
