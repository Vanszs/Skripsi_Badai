# Migration Checklist Status (Verified)

Verification date: `2026-04-21`

Scope: canonical migration to 5-node star graph with main-node-only target.

## A. CONFIG

- [O] node definitions correct (`src/config.py`)
- [O] `node_roles` exist (`src/config.py`)
- [O] node order correct `[MAIN,UP,DOWN,LEFT,RIGHT]` (`src/config.py`)
- [O] main node identity consistent `(-6.75, 107.00)` (`src/config.py`, checkpoint config)
- [O] no active 3-node logic in runtime paths (`src/`, `run_eval_final.py`)

## B. DATA

- [O] 5-node ingestion (`src/data/ingest.py`, generated parquet)
- [O] ERA5 enforced (`models=era5` in ingestion params and persisted `model_mode`)
- [O] grid centers verified (`data/raw/pangrango_era5_5node_grid_validation.json`)
- [O] no reuse of 3-node data in active pipeline (3-node parquet moved to `_archive/legacy_3node_data/`)

## C. GRAPH

- [O] star topology (`src/config.py`, `build_star_edge_index`)
- [O] edge count = 8 (`STAR_EDGE_COUNT=8`, runtime checks in train/inference)
- [O] training graph = inference graph (verified by edge equality check from checkpoint metadata)

## D. LOADER

- [O] shape correct (`[seq_len, 5, features]`, runtime dataset check)
- [O] fail-fast checks working (wrong order, missing node, wrong name)
- [O] no silent drop (strict row-count and uniqueness checks per timestamp)

## E. TRAINING

- [O] main node target only (`TemporalGraphDataset.__getitem__`)
- [O] no node averaging for target/context
- [O] batch runs OK (train smoke run completed)
- [O] checkpoint complete metadata (`models/diffusion_chkpt.pth`)

## F. INFERENCE

- [O] no hardcode node count
- [O] graph reconstructed from checkpoint metadata
- [O] output targets main node only (`main_node_name`, `target_node_policy`)

## G. EVALUATION

- [O] evaluation main-node only (`run_eval_final.py`)
- [O] baseline aligned to same target policy (`src/train_baseline.py`)
- [O] reports regenerated (`result_test/comparison/*.csv/json`, `result_test/EVALUATION_REPORT.md`)

## H. DOCS

- [O] active docs updated to canonical 5-node (`docs/ACTIVE_5NODE_STAR_MAIN.md`)
- [O] notebook updated (`notebooks/evaluasi_model.ipynb`)
- [O] archive labeled legacy (`_archive/README.md`)

## I. VALIDITY

- [O] grid sharing handled (collision detector + fail-fast in ingestion)
- [O] no false spatial independence in active node set (5 canonical nodes map to distinct centers)
- [O] interpretation policy documented (grid-center strict + non-independence policy in grid report)

## J. QUALITY

- [O] no runtime error on end-to-end smoke
- [O] no shape mismatch on dataset/train/inference/eval paths
- [O] pipeline works end-to-end (`ingest -> train -> baseline -> inference -> eval`)
- [O] automated tests pass (`python -m unittest discover -s tests -v`)

## Runtime Evidence Artifacts

- Dataset:
  - `data/raw/pangrango_era5_5node_2005_2025.parquet`
  - `data/raw/pangrango_era5_5node_2024_2024.parquet`
- Grid validation:
  - `data/raw/pangrango_era5_5node_grid_validation.json`
- Checkpoints:
  - `models/diffusion_chkpt.pth`
  - `models/mlp_baseline_chkpt.pth`
- Evaluation:
  - `result_test/comparison/comparison_summary.csv`
  - `result_test/comparison/comparison_summary.json`
  - `result_test/EVALUATION_REPORT.md`

## K. DATASET + TRAIN/VAL/TEST CROSSCHECK (Split, Normalization, Leakage)

- [O] temporal split is strictly chronological (`train <= train_end < val <= val_end < test`) in `src/train.py`
- [O] split happens before model fitting/stat computation (`temporal_split` then `compute_stats_from_training`) in `src/train.py`
- [O] normalization stats are fit from training scope only (main-node training rows) in `src/train.py`
- [O] val/test/eval reuse training-derived stats (no refit on test) in `src/train.py`, `run_eval_final.py`
- [O] retrieval context/index is built from training window only (`date <= train_end`, `node == MAIN`) in `src/train.py`, `src/inference.py`
- [O] loader fail-fast prevents structural leakage/silent corruption (missing node, unknown node, wrong order, duplicate, wrong row-count) in `src/data/temporal_loader.py`
- [O] lag feature uses backward shift only (`precipitation_lag1 = shift(1)`), no future target reference in `src/data/ingest.py`
- [O] evaluation follows checkpoint split boundaries (`train_end`, `val_end`) and targets MAIN only in `run_eval_final.py`
- [O] automated regression checks for contract/split/inference pass (`python -m unittest discover -s tests -v` => 7/7 OK)

### Internet Best-Practice References (checked `2026-04-20`)

- scikit-learn: Common pitfalls (data leakage, preprocessing must be learned on train only): `https://sklearn.org/stable/common_pitfalls.html`
- scikit-learn: TimeSeriesSplit (time-ordered CV for temporal data): `https://sklearn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html`
- TensorFlow time-series tutorial (chronological split and train-stat normalization pattern): `https://www.tensorflow.org/tutorials/structured_data/time_series`

## L. ALGORITHM FORMULA CROSSCHECK (3 COMPONENTS)

### L1. Spatio-Temporal Graph (GAT + Temporal Attention)

- [O] spatial graph operator implemented with GAT neighborhood attention (2-layer `GATConv`) in `src/models/gnn.py`
- [O] temporal aggregation implemented via multi-head self-attention across sequence in `src/models/gnn.py`
- [O] graph conditioning is injected into diffusion as dedicated embedding branch (`graph_mlp`) in `src/models/diffusion.py`
- [O] temporal attention now includes learnable positional encoding and causal mask (`pos_embedding`, `attn_mask`) in `src/models/gnn.py`

### L2. Retrieval Module (kNN + External Memory)

- [O] retrieval uses FAISS kNN L2 search (`IndexFlatL2` / `index.search`) in `src/retrieval/base.py`
- [O] retrieval pool for validation/test/inference is restricted to training-period main-node rows (`date <= train_end`) in `src/train.py`, `src/inference.py`
- [O] training-time retrieval precompute excludes self and enforces strict-past neighbors (`exclude_self=True`, `strict_past=True`) in `src/train.py`
- [O] retrieval backend is aligned between train and inference (both use FAISS `IndexFlatL2`) in `src/train.py`, `src/retrieval/base.py`, `src/inference.py`

### L3. Retrieval-Augmented Diffusion Model

- [O] DDPM-style objective path exists in training loop (`noise -> add_noise -> predict_noise -> MSE`) in `src/train.py`, `src/models/diffusion.py`
- [O] reverse denoising for inference exists (`DDPM` and accelerated `DDIM`) in `src/models/diffusion.py`
- [O] retrieval and graph conditions are both fused into diffusion conditioning stream in `src/models/diffusion.py`
- [O] diffusion comments/contracts are consistent with active 3-target output (`FINAL_TARGET_COLS`) in `src/models/diffusion.py`, `src/config.py`
- [O] main training path uses weighted noise loss (`RainForecaster.weighted_noise_loss`) in `src/train.py`

### L4. Correlation and Train/Val/Test Relation

- [O] temporal split remains non-overlapping and chronological in `src/train.py`
- [O] test correlation is computed and exported per scenario/variable in `run_eval_final.py`, `result_test/comparison/comparison_summary.csv`
- [O] training/validation correlation metrics are logged explicitly per epoch and persisted in training summary/checkpoint config in `src/train.py`

### Formula References From Primary Sources (checked `2026-04-20`)

- Graph Attention Networks (official paper + operator form):  
  - `https://arxiv.org/abs/1710.10903`  
  - `https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html`
- Retrieval-augmented paradigm for diffusion models (semi-parametric retrieval conditioning):  
  - `https://arxiv.org/abs/2204.11824`
- FAISS kNN-L2 retrieval semantics (`IndexFlatL2`, k-nearest search):  
  - `https://github.com/facebookresearch/faiss/wiki/Getting-started`
- DDPM forward/reverse process and simplified noise-prediction objective:  
  - `https://arxiv.org/abs/2006.11239`

## M. RAIN-SPECIALIZED ROBUST CROSSCHECK (checked `2026-04-21`)

- [O] rain-specialized training path implemented (auxiliary wet/dry head + weighted BCE + validation threshold calibration) in `src/models/diffusion.py`, `src/train.py`
- [O] inference/evaluation uses checkpoint rain metadata (`rain_specialization`) and applies calibrated rain gate in `src/inference.py`, `run_eval_final.py`
- [O] robust weekly crosscheck artifacts regenerated in `result_test/nowcasting_hourly_week/weekly_crosscheck_rain_specialized.csv`
- [O] training is numerically stable on GPU **without** AMP (no skipped non-finite batch; early-stop run completed, best epoch 13)
- [X] training with AMP is stable for this configuration (observed non-finite train batches when AMP enabled; mitigation currently: run GPU without AMP)
- [X] rain-specialized model consistently beats persistence on weekly robust crosscheck (current artifact still shows positive RMSE delta on driest/median/wettest weeks)

### Rain-Method Research References (primary)

- Multi-satellite nowcasting paper showing classification objective better for extreme-rain CSI than pure regression:
  - `https://arxiv.org/abs/2307.10843`
- Multi-task precipitation benchmark with weighted loss for heavy-rain performance:
  - `https://arxiv.org/abs/2310.02676`
- Zero-inflated precipitation modeling motivation:
  - `https://arxiv.org/abs/2504.11058`
