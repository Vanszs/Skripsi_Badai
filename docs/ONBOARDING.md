# Developer Onboarding Guide — RA-Diffusion Gede–Pangrango

Welcome to the **RA-Diffusion Gede–Pangrango** codebase! This document provides a complete onboarding walkthrough for new developers and researchers joining the project.

---

## 1. Project Overview

- **Title:** Nowcasting Probabilistik Cuaca Multi-Variabel untuk Mitigasi Risiko Pendakian di Gunung Gede–Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning
- **Author:** Bevantyo Satria Pinandhita (NPM 22081010153)
- **Tech Stack:**
  - **Language:** Python 3.11.9, TypeScript, React 18, Shell
  - **Deep Learning:** PyTorch 2.6.0+cu124, PyTorch Geometric 2.7.0 (`torch-geometric`), HuggingFace Diffusers 0.36.0 (`diffusers`)
  - **Vector Indexing:** FAISS (`faiss-cpu`)
  - **API Backend:** FastAPI, Uvicorn
  - **Data Ingestion:** Open-Meteo ERA5 Archive API

---

## 2. Architecture Layers

The codebase is organized into **8 distinct architectural layers**:

```
 ┌──────────────────────────────────────────────────────────┐
 │ 1. Frontend Web Application (React / Vite Visualizer)    │
 └────────────────────────────┬─────────────────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. API Services & Dashboard (FastAPI Routers)            │
 └────────────────────────────┬─────────────────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. Probabilistic Evaluation & Metrics (CRPS / Brier / CSI)│
 └────────────────────────────┬─────────────────────────────┘
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 4. Core Deep Learning Models (RA-Diffusion + GNN)        │
 └─────────────┬──────────────────────────────┬─────────────┘
               ▼                              ▼
 ┌───────────────────────────┐  ┌───────────────────────────┐
 │ 5. FAISS Retrieval Engine │  │ 6. Data Pipeline Loaders  │
 └───────────────────────────┘  └───────────────────────────┘
```

1. **`layer:frontend-ui` (Frontend Web Application):** Interactive dashboard, topology overlay, and weather visualizer built in React/Vite.
2. **`layer:api-service` (API Services & Dashboard):** FastAPI REST backend (`api/routers/nowcast.py`, `api/routers/dashboard.py`).
3. **`layer:evaluation` (Probabilistic Metrics):** Evaluation suite (`core/src/evaluation/probabilistic_metrics.py`).
4. **`layer:core-model` (Core Deep Learning Models):** PyTorch models (`core/src/models/diffusion.py`, `core/src/models/gnn.py`, `core/src/models/mlp_baseline.py`).
5. **`layer:retrieval` (Retrieval Engine):** FAISS index query database (`core/src/retrieval/base.py`).
6. **`layer:data-pipeline` (Data Pipeline & Loaders):** Open-Meteo fetcher (`ingest.py`), PyTorch DataLoader (`temporal_loader.py`), Graph Topology (`builder.py`).
7. **`layer:documentation` (Documentation & Specifications):** Research documentation and thesis notes (`docs/`, `.kimchi/docs/`).
8. **`layer:infrastructure` (Infrastructure & Config):** Skill scripts, hooks, and project settings.

---

## 3. Guided Walkthrough (5 Key Steps)

1. **Data Ingestion & Graph Construction (`core/src/data/ingest.py`, `temporal_loader.py`):**
   Downloads 20-year hourly ERA5 data for 3 stations (Puncak, Cibodas, Cianjur), applies `log1p` for rain, Z-score normalization, and builds sliding temporal windows (`seq_len=6`).
2. **FAISS Retrieval Query (`core/src/retrieval/base.py`):**
   Encodes the current weather window and searches the historical index (`IndexFlatL2`) for top $k=3$ nearest analogs.
3. **GNN & Diffusion Conditioning (`core/src/models/gnn.py`, `diffusion.py`):**
   `SpatioTemporalGNN` extracts spatial message passing across stations + multihead temporal attention. `ConditionalDiffusionModel` performs 20-step DDIM reverse sampling over 30 ensemble members.
4. **Probabilistic Evaluation (`core/src/evaluation/probabilistic_metrics.py`):**
   Computes CRPS, Brier score, POD, FAR, CSI, RMSE, MAE, and Pearson correlation.
5. **FastAPI & Web UI (`api/routers/nowcast.py`, `web/src/`):**
   Streams probabilistic nowcasting results to the web visualizer for hiking risk mitigation.

---

## 4. Key Complexity Hotspots (Handle With Care)

- **`core/src/models/diffusion.py` (Complex):** Conditional U-Net MLP with sinusoidal time embeddings and additive context/retrieval/graph embeddings.
- **`core/src/models/gnn.py` (Moderate):** SpatioTemporalGNN with 2-layer GATConv and PyTorch MultiheadAttention.
- **`core/src/evaluation/probabilistic_metrics.py` (Moderate):** Vectorized CRPS calculations and threshold contingency tables.

---

## 5. Quick Start Commands

```bash
# 1. Activate virtual environment
source .venv/bin/activate  # on Linux

# 2. Run model training
python -m src.train

# 3. Run model evaluation
python -m notebooks.evaluasi_model

# 4. Start API server
uvicorn api.main:app --reload --port 8000
```
