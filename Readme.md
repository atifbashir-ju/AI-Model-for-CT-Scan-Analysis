# AI Agent-Driven CT Scan Analysis System

An agentic, human-in-the-loop medical imaging system for automated lung nodule detection in CT scans, built around a controlled continuous-learning pipeline: **DICOM → AI Prediction → Radiologist Feedback → Disagreement Detection → Candidate Model Retraining → Evaluation → Controlled Deployment.**

> **Status: In Development** — core pipeline (detection, feedback, model registry) is functional and tested end-to-end. Automated retraining triggers and the full evaluation-gated promotion cycle are implemented but pending large-scale validation. See [Roadmap](#roadmap) below.

---

## Overview

This project explores how a cloud/PACS-inspired architecture can bring adaptive AI into radiology workflows **without** removing the radiologist from the loop. Instead of a single black-box model, the system is built as a set of cooperating components orchestrated by an **AI Agent**, closely following the disagreement-driven, feedback-gated model-update pattern described in recent PACS research literature.

The core design principle: **the model never updates itself blindly.** Every prediction is reviewable, every correction is logged, and a new model version is only promoted to production after it demonstrably outperforms the current one on held-out data — with human approval as the final gate.

---

## Key Features

- **3D CNN Vision Model** — a custom PyTorch Convolutional Neural Network trained to detect lung nodules in CT volumes
- **Full-Volume Sliding-Window Inference** — scans the entire 3D CT volume (not just fixed patches) and merges overlapping detections
- **Multi-Format DICOM Processing** — supports DICOM series, MetaImage (`.mhd`/`.raw`), NIfTI, and NRRD formats
- **Mixup Detection** — automatically validates that all uploaded files belong to the same patient/study, rejecting mismatched uploads
- **Radiologist Review Interface** — an interactive Streamlit UI where clinicians can view AI findings slice-by-slice and mark corrections directly on the image
- **Structured Feedback Store** — every Accept/Reject/Correct decision is logged with full context (AI confidence, model version, radiologist notes, timestamps)
- **Patient History Tracking** — recognizes returning patients and surfaces prior findings and feedback
- **Model Registry & Versioning** — trained models are never overwritten; each version is stored with full metadata (accuracy, training data, date)
- **Safety-Gated Retraining** — retraining is only triggered once enough validated feedback accumulates (not after a single correction), and candidate models must pass accuracy **and** recall checks against the current production model before promotion
- **AI Agent Orchestration** — a central orchestrator coordinates all of the above as discrete tools, with full activity logging for auditability

---

## Architecture

```
                         USER
                           │
                           ▼
                    DICOM UPLOAD
                           │
                           ▼
                    ┌─────────────┐
                    │  AI AGENT   │  (orchestrator)
                    └──────┬──────┘
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
       DICOM Tool     Vision Model    Feedback Store
    (load/validate)   (3D CNN CNN     (structured
                       inference)      corrections)
             │             │              │
             └──────┬──────┘              │
                     ▼                     │
              Prediction + Confidence      │
                     │                     │
                     ▼                     │
              Radiologist Review ──────────┘
                     │
             ┌───────┴───────┐
           Agree           Disagree
             │                │
             ▼                ▼
           Done          Store correction
                                │
                                ▼
                     Training Queue Check
                     (threshold-gated)
                                │
                                ▼
                     Candidate Model Training
                                │
                                ▼
                  Evaluation vs. Production Model
                     (accuracy + recall checks)
                          │           │
                        Pass        Fail
                          │           │
                          ▼           ▼
                   Human Approval   Reject
                          │
                          ▼
                  Promote to Production
                  (Model Registry v2, v3...)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | PyTorch (3D CNN) |
| Medical Imaging I/O | pydicom, SimpleITK |
| Image Processing | OpenCV, NumPy |
| Web Interface | Streamlit, streamlit-image-coordinates |
| Data Handling | Pandas, scikit-learn |
| Storage | JSON-based feedback log & model registry (file-based, DB-ready design) |

---

## Dataset

Trained on **[LUNA16](https://luna16.grand-challenge.org/)** (LUng Nodule Analysis 2016) — a public, radiologist-annotated dataset of 888 CT scans derived from LIDC-IDRI, with nodule annotations validated by 4 experienced radiologists.

Current model (v1) trained on:
- **391 CT scans** (subsets 0–4)
- **1,064 labeled 3D patches** (522 positive / 542 negative)

---

## Results (Model v1)

| Metric | Value |
|---|---|
| Test Accuracy | **94.84%** |
| Test Samples | 213 (unseen) |
| Correctly Identified Nodules | 115 |
| Correctly Identified Normal | 87 |
| False Positives | 7 |
| False Negatives | 4 |

> These results reflect a research prototype trained on a subset of the full dataset. They demonstrate feasibility of the architecture, not clinical-grade performance.

---

## Project Structure

```
├── src/
│   ├── agent.py                 # Orchestrator - coordinates all tools
│   ├── dicom_processor.py       # DICOM/MHD/NIfTI/NRRD loading + mixup detection
│   ├── vision_model.py          # 3D CNN architecture + sliding-window inference
│   ├── model_registry.py        # Model versioning and promotion logic
│   ├── feedback_store.py        # Radiologist feedback storage & patient history
│   ├── training_pipeline.py     # Candidate model training from feedback
│   ├── evaluator.py             # Candidate vs. production model comparison
│   ├── app.py                   # Streamlit UI
│   └── test_agent_pipeline.py   # End-to-end pipeline test
├── models_registry/              # Versioned trained models + metadata
├── feedback/                     # Structured radiologist feedback log
├── logs/                         # Agent activity logs
└── train_model_v1.ipynb          # Original model training notebook
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- LUNA16 dataset (or your own DICOM/MHD data) for training

### Installation

```bash
git clone https://github.com/atifbashir-ju/AI-Model-of-CT-SCAN-Analysis.git
cd AI-Model-of-CT-SCAN-Analysis

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install torch pydicom SimpleITK opencv-python numpy pandas scikit-learn \
            streamlit streamlit-image-coordinates pyarrow
```

### Running the App

```bash
cd src
streamlit run app.py
```

### Running the Pipeline Test

```bash
cd src
python test_agent_pipeline.py
```

---

## Roadmap

- [x] 3D CNN model training on LUNA16
- [x] Full-volume sliding-window inference
- [x] Multi-format DICOM/MHD/NIfTI/NRRD loading with mixup detection
- [x] Structured feedback store with patient history
- [x] Model registry with versioning
- [x] Agent orchestration layer with activity logging
- [x] Safety-gated retraining trigger (threshold-based)
- [x] Candidate vs. production model evaluation (accuracy + recall)
- [ ] Full end-to-end retraining → evaluation → promotion cycle tested at scale
- [ ] Training on full LUNA16 dataset (888 scans)
- [ ] Cloud deployment (AWS/Streamlit Cloud)
- [ ] Database-backed storage (replacing JSON files)

---

## Disclaimer

This is a **research and educational prototype**. It is **not a certified medical device** and must not be used for actual clinical diagnosis. All predictions are intended to support, not replace, review by a qualified radiologist.

---

## Acknowledgments

- [LUNA16 Challenge](https://luna16.grand-challenge.org/) for the annotated dataset
- Architecture inspired by cloud-based PACS research on disagreement-driven adaptive radiographic AI

---

## Author

**Atif Bashir**
[GitHub](https://github.com/atifbashir-ju)