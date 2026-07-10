# 🧠 NeuroAssist AI v2

**Doctor-Only Explainable Clinical Decision Support System for Alzheimer's Disease**

> ⚠️ **INVESTIGATIONAL SOFTWARE** — Not FDA/CE cleared. Not a substitute for clinical judgment.

---

## Overview

NeuroAssist AI v2 is a secure, clinician-only platform for Alzheimer's disease analysis:

- **4-class MRI classification**: Non-Demented, Very Mild, Mild, Moderate
  - ⚠️ Severe Dementia is NOT supported by current training data
- **Explainable AI**: Grad-CAM, Grad-CAM++, HiResCAM, Integrated Gradients, Guided Backprop
- **Clinical risk assessment**: Transparent, rule-based scoring (NOT ML-based)
- **RAG-grounded reports**: Gemini-generated reports citing clinical guidelines
- **Multilingual PDF export**: English, Hindi, Marathi
- **Medical AI assistant**: Scope-limited Gemini chatbot (Alzheimer's/dementia only)
- **Patient record management**: Longitudinal tracking and trend analysis
- **Audit trail**: Every prediction and report access is logged

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19 + TypeScript + Vite + Tailwind CSS + Material UI |
| Backend | FastAPI + Uvicorn + Pydantic + Motor (async MongoDB) |
| AI/ML | PyTorch + torchvision + timm + pytorch-grad-cam + Captum |
| LLM | **Google Gemini API ONLY** (no Groq/OpenAI/Anthropic) |
| RAG | FAISS + Sentence-Transformers + LangChain |
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
# Edit .env: set JWT_SECRET_KEY, GEMINI_API_KEY, MONGODB_URI
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

### 5. Train Models (requires GPU + Kaggle dataset)
```bash
pip install kaggle
python ml/data/download_dataset.py
# Run the preprocessing pipeline (applies skull-stripping approximation, intensity normalization, and subject splits):
python ml/data/preprocess.py --input-dir ml/data/raw --output-dir ml/data/preprocessed
# Train models:
python ml/training/train.py --data-dir ml/data/preprocessed --save-dir ml/checkpoints
```

---

## Key Design Decisions

1. **Doctor-only**: No patient-facing routes. Login says "Clinician Portal."
2. **No fake data**: If the model isn't trained, the UI shows "Model not loaded" — never mock predictions.
3. **Gemini only**: Zero imports of Groq, OpenAI, or Anthropic anywhere.
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
