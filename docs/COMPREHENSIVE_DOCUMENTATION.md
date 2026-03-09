# DOKUMENTASI LENGKAP PROJECT
## Nowcasting Probabilistik Cuaca Multi-Variabel untuk Mitigasi Risiko Pendakian

**Versi:** 3.0  
**Tanggal:** Maret 2026  
**Status:** Phase 1+2 selesai (~80-85% sidang-ready). Model terbukti belajar.

---

# BAGIAN 1: KONTEKS PROJECT

## 1.1 Judul Lengkap Skripsi

> **"Nowcasting Probabilistik Cuaca Multi-Variabel untuk Mitigasi Risiko Pendakian di Gunung Gede-Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**

## 1.2 Breakdown Judul

| Komponen | Penjelasan |
|----------|------------|
| **Nowcasting** | Prediksi cuaca jangka pendek (1 jam ke depan) |
| **Probabilistik** | Output berupa distribusi probabilitas (N=30 sampel ensemble) |
| **Cuaca Multi-Variabel** | Prediksi simultan 3 variabel: presipitasi, kecepatan angin, kelembapan |
| **Mitigasi Risiko Pendakian** | Fokus pada kombinasi faktor risiko bagi pendaki |
| **Gunung Gede-Pangrango** | Lokasi studi kasus di Jawa Barat — 3 node: Puncak, Cibodas, Cianjur |
| **Retrieval-Augmented** | FAISS IndexFlatL2 mencari k=3 analog historis untuk conditioning |
| **Diffusion Model** | Conditional DDPM (1000 timesteps, DDIM 20 steps saat inference) |
| **Spatio-Temporal Graph Conditioning** | GAT 2-layer (4 heads) + MultiheadAttention temporal -> graph embedding 64-dim |

## 1.3 Tujuan Project

### Rumusan Masalah (4 RQ):
1. **RQ1**: Bagaimana kinerja model probabilistik (diffusion) dalam memprediksi presipitasi, kecepatan angin, dan kelembapan?
2. **RQ2**: Sejauh mana komponen retrieval dan spatio-temporal graph meningkatkan akurasi model?
3. **RQ3**: Bagaimana perbandingan model probabilistik (diffusion) vs deterministik (MLP baseline)?
4. **RQ4**: Bagaimana trade-off antara akurasi rata-rata dan sensitivitas terhadap kejadian ekstrem?

### Skenario Eksperimen (6 Skenario):
| # | Skenario | Menjawab RQ |
|---|----------|-------------|
| 1 | Persistence (naive baseline) | RQ3 |
| 2 | MLP Baseline (deterministik) | RQ3 |
| 3 | Diff Only (tanpa retrieval, tanpa GNN) | RQ2 |
| 4 | Diff+Retrieval (tanpa GNN) | RQ2 |
| 5 | Diff+GNN (tanpa retrieval) | RQ2 |
| 6 | Full Model (Diffusion + Retrieval + GNN) | RQ1, RQ2 |

---

# BAGIAN 2: DATA

## 2.1 Sumber Data

| Aspek | Detail |
|-------|--------|
| **Provider** | Open-Meteo API (wrapper untuk ERA5) |
| **Dataset** | ERA5 Reanalysis |
| **Temporal Range** | 2005-01-01 hingga 2025-01-01 |
| **Temporal Resolution** | Hourly (per jam) |
| **Spatial Resolution** | ~25 km grid |
| **Total Records** | ~526,032 titik data (3 nodes x 20 tahun x ~8760 jam) |

### Koordinat Lokasi (3 Nodes)

| Node | Nama | Latitude | Longitude | Elevasi |
|------|------|----------|-----------|---------|
| **Puncak** | Puncak Gede-Pangrango | -6.7698 | 106.9636 | ~3,019 m |
| **Lereng** | Cibodas | -6.7308 | 107.0026 | ~1,800 m |
| **Hilir** | Cianjur | -6.8160 | 107.1330 | ~500 m |

## 2.2 Variabel

### Variabel Target (3 Output):
| Variabel | Unit | Transform |
|----------|------|-----------|
| `precipitation` | mm/jam | log1p + z-score |
| `wind_speed_10m` | m/s | z-score |
| `relative_humidity_2m` | % | z-score |

### Variabel Fitur Input (9 features):
| # | Variabel | Unit |
|---|----------|------|
| 1 | `temperature_2m` | C |
| 2 | `relative_humidity_2m` | % |
| 3 | `dewpoint_2m` | C |
| 4 | `surface_pressure` | hPa |
| 5 | `wind_speed_10m` | m/s |
| 6 | `wind_direction_10m` | deg |
| 7 | `cloud_cover` | % |
| 8 | `precipitation_lag1` | mm |
| 9 | `elevation` | m (statis) |

## 2.3 Normalisasi

### Target Normalization Stats (dari training set):
```
t_mean: [0.170, 4.003, 83.869]   (precip_log, wind, humidity)
t_std:  [0.361, 2.285, 14.669]   (precip_log, wind, humidity)
```

### Transform Pipeline:
```python
# Forward (training):
y_log = np.log1p(y_raw)               # Presipitasi saja
y_norm = (y_log - t_mean) / t_std     # Z-score standar

# Inverse (inference):
y_log_hat = y_norm * t_std + t_mean
y_raw_hat = np.expm1(y_log_hat)       # Presipitasi: inverse log1p
y_raw_hat = clamp(min=0)              # Presipitasi >= 0
```

---

# BAGIAN 3: METODE

## 3.1 Arsitektur Model

```
Input Sequence (6 timesteps x 9 features x 3 nodes)
                    |
    +---------------+---------------+
    |                               |
    v                               v
  ST-GNN                      FAISS Retrieval
  (GAT 2-layer, 4 heads)      (k=3 neighbors)
  + TemporalAttention          -> Retrieval MLP
  -> graph_emb [64-dim]        -> ret_emb [hidden_dim]
    |                               |
    +---------------+---------------+
                    |
                    v
        Conditional Diffusion Model
        (DDPM 1000 steps / DDIM 20 steps)
        Conditioning: t_emb + c_emb + r_emb + g_emb
        Backbone: U-Net-like MLP (3->128->256->128->3)
                    |
                    v
        30 probabilistic samples [30, 3]
                    |
                    v
        Denormalization -> Final output
        (precip mm, wind m/s, humidity %)
```

## 3.2 Komponen Detail

### A. Spatio-Temporal GNN (`src/models/gnn.py`)

| Parameter | Value |
|-----------|-------|
| Input dim | 9 (features) |
| Hidden dim | 64 |
| Output dim | 64 (graph embedding) |
| GAT heads (layer 1) | 4 (concat) |
| GAT heads (layer 2) | 1 |
| Temporal Attention | MultiheadAttention (embed_dim=64, num_heads=4) |
| Dropout | 0.1 |
| Pooling | global_mean_pool |

### B. Retrieval Database (`src/retrieval/base.py`)

| Parameter | Value |
|-----------|-------|
| Index type | FAISS IndexFlatL2 |
| Vector dim | 9 (features) |
| k neighbors | 3 |

### C. Diffusion Model (`src/models/diffusion.py`)

| Parameter | Value |
|-----------|-------|
| Input dim | 3 (target variables) |
| Timesteps | 1000 (DDPM training) |
| Inference | DDIM 20 steps, clip_sample=False |
| Hidden dim | 128 |
| Context dim | 9 |
| Retrieval dim | 27 (k=3 x 9) |
| Graph dim | 64 |
| Conditioning | Additive: t_emb + c_emb + r_emb + g_emb |
| Num ensemble | 30 |

## 3.3 Training Configuration

| Parameter | Diffusion | MLP Baseline |
|-----------|-----------|--------------|
| Epochs | 20 | 51 (early stopped) |
| Batch size | 512 | 512 |
| Optimizer | AdamW (lr=1e-3, wd=1e-4) | AdamW (lr=1e-3, wd=1e-4) |
| Scheduler | Tidak ada | CosineAnnealingLR |
| Loss | F.mse_loss (standar MSE) | F.mse_loss |
| Early Stopping | Best val checkpoint | patience=10 |
| Mixed Precision | AMP (fp16) | AMP (fp16) |
| Best val_loss | 0.1210 | - |

## 3.4 Temporal Split

| Set | Period | Years | Purpose |
|-----|--------|-------|---------|
| **Train** | 2005.01.01 - 2018.12.31 | 14 | Model learning |
| **Validation** | 2019.01.01 - 2021.12.31 | 3 | Best model selection |
| **Test** | 2022.01.01 - 2025.01.01 | 3+ | Final evaluation |

Normalisasi stats dan FAISS index dibangun HANYA dari training data.

---

# BAGIAN 4: HASIL EVALUASI

## 4.1 Evaluasi Harian (EVAL_STEP=24, 1098 titik)

| Skenario | Precip RMSE | Wind RMSE | Hum RMSE | Precip Corr | Wind Corr | Hum Corr |
|----------|-------------|-----------|----------|-------------|-----------|----------|
| Persistence | 1.631 | 1.471 | 4.573 | 0.581 | 0.855 | 0.956 |
| MLP Baseline | 1.591 | 1.334 | 3.893 | 0.543 | 0.879 | 0.954 |
| Diff Only | 1.923 | 1.483 | 5.938 | 0.451 | 0.848 | 0.946 |
| Diff+Retrieval | 1.929 | 1.502 | 6.750 | 0.360 | 0.845 | 0.945 |
| Diff+GNN | 1.884 | 1.447 | 4.132 | 0.467 | 0.858 | 0.950 |
| **Full Model** | **1.856** | **1.433** | **4.006** | **0.418** | **0.858** | **0.952** |

### Temuan:
- Full model beats persistence pada wind RMSE dan humidity RMSE
- GNN kontribusi terbesar: humidity RMSE 5.94 -> 4.13 (turun 30%)
- Retrieval saja kurang efektif; kombinasi retrieval+GNN optimal

## 4.2 Evaluasi Hourly Nowcasting (EVAL_STEP=1, 336 titik)

| Metrik | Precipitation | Wind Speed | Humidity |
|--------|---------------|------------|----------|
| RMSE Diff | 1.301 | 1.539 | **2.917** |
| RMSE Pers | 1.233 | 1.478 | 3.481 |
| **Skill Score** | -5.5% | -4.1% | **+16.2%** |
| **CRPS Diff** | **0.544** | **0.886** | **1.450** |
| MAE Pers | 0.645 | 1.146 | 2.175 |
| CRPS < MAE? | Ya (-16%) | Ya (-23%) | Ya (-33%) |

**CRPS diffusion mengalahkan MAE persistence di semua 3 variabel.**

## 4.3 Precipitation Threshold (EVAL_STEP=24)

| Threshold | Pers POD | Full POD | Pers CSI | Full CSI |
|-----------|----------|----------|----------|----------|
| 2 mm | 56.6% | 8.6% | 36.3% | 8.0% |
| 5 mm | 50.0% | 0.0% | 27.8% | 0.0% |
| 10 mm | 0.0% | 0.0% | 0.0% | 0.0% |

Deteksi hujan ekstrem masih lemah — konsisten dengan literatur ERA5.

---

# BAGIAN 5: BUG FIXES (Semua Selesai)

| # | Bug | Fix | Dampak |
|---|-----|-----|--------|
| 1 | T_STD_MULTIPLIER=5.0 | Dihapus | Target presipitasi kembali ke range N(0,1) |
| 2 | Weighted loss 5x/10x | Ganti F.mse_loss standar | Training stabil |
| 3 | GNN .repeat() mismatch | Per-node features saat inference | GNN berfungsi benar |
| 4 | Val loss < train loss | Keduanya pakai MSE standar | Metrik sebanding |
| 5 | MLP eval inkonsisten | Target rata-rata | Evaluasi konsisten |
| 6 | DDIM clip_sample=True | clip_sample=False | Wind corr: -0.24 -> +0.85 |

Detail lengkap: lihat `docs/FIX_PLAN.md`

---

# BAGIAN 6: FILE STRUCTURE

```
d:\SKRIPSI\Skripsi_Bevan\
+-- data/
|   +-- raw/
|       +-- pangrango_era5_2005_2025.parquet
+-- docs/
|   +-- COMPREHENSIVE_DOCUMENTATION.md   <- Dokumentasi ini
|   +-- DATASET_STRUCTURE.md             <- Struktur data
|   +-- FIX_PLAN.md                      <- Rencana perbaikan (DONE)
|   +-- LAPORAN_STATUS_SKRIPSI.md        <- Status progress
|   +-- diagrams_and_tables.md           <- Diagram & tabel untuk thesis
|   +-- PANDUAN_SEMINAR_PROPOSAL.md      <- Panduan sempro
|   +-- PANDUAN_SEMPRO.md                <- Panduan sempro
+-- models/
|   +-- diffusion_chkpt.pth             <- Checkpoint diffusion+GNN (retrained)
|   +-- mlp_baseline_chkpt.pth          <- Checkpoint MLP (retrained)
+-- notebooks/
|   +-- evaluasi_model.ipynb
+-- result_test/
|   +-- EVALUATION_REPORT.md            <- Laporan 6 skenario
|   +-- comparison/                     <- CSV/JSON metrik perbandingan
|   +-- persistence/                    <- Metrik persistence
|   +-- mlp_baseline/                   <- Metrik MLP
|   +-- diff_only/                      <- Metrik diff tanpa ret/gnn
|   +-- diff_retrieval/                 <- Metrik diff+retrieval
|   +-- diff_gnn/                       <- Metrik diff+gnn
|   +-- full_model/                     <- Metrik full model
|   +-- diffusion_pure/                 <- Metrik diffusion pure (legacy)
|   +-- plots/                          <- Visualisasi (23 PNG)
+-- src/
|   +-- data/
|   |   +-- ingest.py                   <- Fetch ERA5 dari Open-Meteo
|   |   +-- temporal_loader.py          <- PyTorch Dataset (sliding window)
|   +-- graph/
|   |   +-- builder.py                  <- Graph construction (3 nodes)
|   +-- models/
|   |   +-- diffusion.py                <- ConditionalDiffusionModel + RainForecaster
|   |   +-- gnn.py                      <- SpatioTemporalGNN (GAT + Temporal)
|   |   +-- mlp_baseline.py             <- MLP Baseline
|   +-- retrieval/
|   |   +-- base.py                     <- FAISS retrieval database
|   +-- evaluation/
|   |   +-- __init__.py
|   |   +-- probabilistic_metrics.py    <- CRPS, Brier Score, dll.
|   +-- train.py                        <- Training diffusion + GNN
|   +-- train_baseline.py               <- Training MLP baseline
|   +-- inference.py                    <- Inference (run_inference_real)
+-- run_eval_final.py                   <- Script evaluasi 6 skenario
+-- requirements.txt
+-- _archive/                           <- File-file lama (tidak dipakai)
```

---

# BAGIAN 7: REFERENSI LITERATUR

1. **Hourly rainfall nowcasting sulit** — DL "still challenged by prediction of heavy precipitation" (NHESS 2025)
2. **RMSE ~1-2 mm/jam normal** — CNN/DL estimasi hujan satelit RMSE ~1.6 mm/jam; ensemble Indonesia RMSE 1.85-3.08 mm/jam
3. **ERA5 lemah di ekstrem** — "poorly representation of daily extreme rainfall" di Indonesia (Adv. Meteorology 2022)

---

**END OF DOCUMENTATION**
