---
trigger: always_on
---

setiap saya memberikan prompt, selalu balas dengan md artifact untuk saya review untuk didiskusikan, kecuali saya bilang "langsung eksekusi, langsung jalankan" atau sejenisnya.

jika anda membuat test file, setelah test langsung hapus saja.

---

# Context Bridge — Skripsi RA-Diffusion Gede–Pangrango

> **Dokumen ini adalah sumber kebenaran tunggal (single source of truth)** untuk setiap sesi AI baru.
> Semua angka, arsitektur, dan path di bawah ini **mencerminkan implementasi aktual** di workspace, bukan rencana awal.
> **Terakhir diverifikasi:** 4 Maret 2026.

---

## 0. Identitas Proyek

| Field | Nilai |
|-------|-------|
| **Judul Skripsi** | Nowcasting Probabilistik Cuaca Multi-Variabel untuk Mitigasi Risiko Pendakian di Gunung Gede–Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning |
| **Mahasiswa** | Bevantyo Satria Pinandhita (NPM 22081010153) |
| **Pembimbing** | Faisal Muttaqin, S.Kom, M.T. & Andreas Nugroho Sihananto, S.Kom., M.Kom. |
| **Universitas** | UPN Veteran Jawa Timur — Program Studi Informatika |
| **Status** | Training selesai, Evaluasi selesai, Visualisasi selesai, Notebook evaluasi dibuat |

---

## 1. Workspace Structure (Aktual)

```
D:\SKRIPSI\Skripsi_Bevan\
├── .agent/rules/respon.md          ← FILE INI (context bridge)
├── .gitignore                      ← Exclude .venv, data/raw, models/*.pth
├── requirements.txt                ← 16 dependencies
│
├── data/raw/
│   └── pangrango_era5_2005_2025.parquet   ← Dataset utama (~526K rows)
│
├── docs/
│   ├── COMPREHENSIVE_DOCUMENTATION.md
│   ├── DATASET_STRUCTURE.md
│   └── FINAL_AUDIT_REPORT.md
│
├── models/
│   ├── diffusion_chkpt.pth         ← Checkpoint terbaik (val loss 0.0966)
│   └── mlp_baseline_chkpt.pth      ← MLP baseline checkpoint
│
├── notebooks/
│   └── evaluasi_model.ipynb        ← Notebook evaluasi lengkap (Indonesian)
│
├── src/
│   ├── train.py                    ← Training diffusion + GNN
│   ├── train_baseline.py           ← Training MLP baseline
│   ├── inference.py                ← Inference engine (pure diffusion)
│   ├── data/
│   │   ├── ingest.py               ← Download ERA5 via Open-Meteo
│   │   └── temporal_loader.py      ← DataLoader + normalisasi + graph
│   ├── models/
│   │   ├── diffusion.py            ← ConditionalDiffusionModel (294,019 params)
│   │   ├── gnn.py                  ← SpatioTemporalGNN (GAT + TemporalAttention)
│   │   └── mlp_baseline.py         ← MLP baseline (3-layer)
│   ├── graph/
│   │   └── builder.py              ← PangrangoGraphBuilder
│   ├── retrieval/
│   │   └── base.py                 ← FAISS retrieval (IndexFlatL2)
│   └── evaluation/
│       ├── __init__.py
│       └── probabilistic_metrics.py ← CRPS, Brier, POD, FAR, CSI, RMSE, MAE, Corr
│
├── results/
│   ├── plots/                      ← 8 visualisasi final (PNG)
│   │   ├── timeseries_1week_puncak.png
│   │   ├── timeseries_1week_lereng_cibodas.png
│   │   ├── timeseries_1week_hilir_cianjur.png
│   │   ├── scatter_actual_vs_predicted.png
│   │   ├── ensemble_spread.png
│   │   ├── confusion_matrices.png
│   │   ├── reliability_diagram.png
│   │   └── training_curves.png
│   ├── tables/
│   │   ├── metrics_summary.csv
│   │   └── metrics_summary.json
│   ├── training_logs/              ← Loss curves (source PNG)
│   ├── diffusion_results/diffusion_metrics.json   ← Per-node + aggregated
│   ├── baseline_results/baseline_metrics.json     ← MLP baseline
│   ├── probabilistic_metrics.json
│   ├── probabilistic_results.csv   ← Reliability diagram data
│   ├── threshold_sensitivity.json  ← 2/5/10 mm threshold analysis
│   ├── evaluation_report.md
│   └── model_analysis_report.md    ← 351-line comprehensive report
│
├── _archive/                       ← 9 file arsip (debug, old notebooks, drafts)
└── .venv/                          ← Python 3.11.9 virtual environment
```

---

## 2. Environment Teknis

| Komponen | Nilai |
|----------|-------|
| Python | 3.11.9 (venv di `.venv/`) |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA RTX 3050 4GB |
| torch-geometric | 2.7.0 |
| diffusers | 0.36.0 |
| faiss-cpu | 1.13.2 |
| OS | Windows |

---

## 3. Data Pipeline (Implementasi Aktual)

### 3.1 Sumber Data
- **API:** Open-Meteo ERA5 Archive (`archive-api.open-meteo.com/v1/archive`)
- **Periode:** 2005-01-01 s/d 2025-12-31 (20 tahun hourly)
- **Output:** `data/raw/pangrango_era5_2005_2025.parquet` (~526K rows)
- **Timezone:** `Asia/Singapore`

### 3.2 Node Observasi (3 Node)

| Node | Lat | Lon | Ketinggian | ERA5 Grid Cell |
|------|-----|-----|------------|---------------|
| **Puncak** | -6.769797 | 106.963583 | ~2,958 mdpl | (-6.75, 107.00) |
| **Lereng_Cibodas** | -6.751722 | 106.987160 | ~1,275 mdpl | (-6.75, 107.00) — **SAMA** |
| **Hilir_Cianjur** | -6.816000 | 107.133000 | ~450 mdpl | (-6.75, 107.25) |

> Puncak & Lereng_Cibodas jatuh di grid cell ERA5 yang sama karena resolusi ~25 km.

### 3.3 Variabel yang Didownload (8 dynamic + 2 static + 2 lag)

**Dynamic (hourly):** `precipitation`, `temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `cloudcover`

**Static:** `elevation` (dari Elevation API), `land_sea_mask` (derived)

**Lag features:** `precipitation_lag1` (shift 1h), `precipitation_lag3` (shift 3h)

### 3.4 Target Variables — 3 variabel (BUKAN 4)

| # | Target | Satuan | Keterangan |
|---|--------|--------|------------|
| 0 | `precipitation` | mm/jam | Log1p transform sebelum normalisasi |
| 1 | `wind_speed_10m` | m/s | Z-score langsung |
| 2 | `relative_humidity_2m` | % | Z-score langsung |

> `temperature_2m` TIDAK termasuk target — hanya digunakan sebagai input feature.

### 3.5 Feature Columns (Input ke Model)

`temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover`, `precipitation_lag1`, `elevation` — (9 fitur, filtered by availability)

### 3.6 Temporal Split (Strict, No Leakage)

| Split | Periode | ~Persentase |
|-------|---------|-------------|
| **Training** | 2005-01-01 → 2018-12-31 | ~67% |
| **Validation** | 2019-01-01 → 2021-12-31 | ~14% |
| **Test** | 2022-01-01 → 2025-12-31 | ~19% |

### 3.7 Normalisasi

- **Precipitation:** `log1p(x)` lalu z-score `(x - mean) / (std + 1e-5)`
- **Fitur & target lain:** z-score langsung
- **Stats sumber:** Training set only
- **T_STD_MULTIPLIER:** 5.0 (hanya untuk precipitation std, memperlebar distribusi prediksi)

---

## 4. Arsitektur Model (Implementasi Aktual)

### 4.1 Pipeline End-to-End

```
ERA5 Data → Sliding Window (seq_len=6) → Feature Normalization (z-score)
  │
  ├─► FAISS Retrieval (k=3 nearest historical analogs)
  │     Index: IndexIVFFlat (256 cells, nprobe=16) saat training
  │     Index: IndexFlatL2 saat inference
  │
  ├─► SpatioTemporalGNN (GAT 2-layer + TemporalAttention)
  │     Per-timestep: GATConv(in, 64, heads=4) → GATConv(256, 64, heads=1)
  │     Cross-time: MultiheadAttention(64, heads=4) → mean pool
  │     Output: graph_embedding [batch, 64]
  │
  ├─► Context Features (mean over nodes, last timestep)
  │
  └─► ConditionalDiffusionModel
        Conditioning = time_emb + context_emb + retrieval_emb + graph_emb (additive)
        Architecture: U-Net-like MLP (down1→down2→mid→skip→up1→out)
        Training scheduler: DDPMScheduler (1000 timesteps)
        Inference scheduler: DDIMScheduler (20 steps)
        30 ensemble samples → median = point prediction
            │
            ▼
        Denormalisasi + Clamping
            │
            ▼
        Output: 3 variabel × distribusi probabilistik
```

### 4.2 ConditionalDiffusionModel (294,019 params)

| Layer | Dimensi |
|-------|---------|
| `input_dim` | 3 (= NUM_TARGETS) |
| `context_dim` | 64 |
| `retrieval_dim` | 32 |
| `graph_dim` | 64 |
| `hidden_dim` | 64 |
| `time_mlp` | SinusoidalEmb(64) → Linear(64,128) → GELU → Linear(128,64) |
| `cond_mlp` | Linear(64,64) → SiLU → Linear(64,64) |
| `retrieval_mlp` | Linear(32,64) → SiLU → Linear(64,64) |
| `graph_mlp` | Linear(64,64) → SiLU → Linear(64,64) |
| `down1` | Linear(3,64) → SiLU |
| `down2` | Linear(64,128) → SiLU |
| `mid` | Linear(128,128) → SiLU |
| `up1` | Linear(256,64) → SiLU (skip: cat mid+down2) |
| `out` | Linear(64,3) |

### 4.3 SpatioTemporalGNN

| Komponen | Detail |
|----------|--------|
| `SpatialGNN` | 2-layer GAT: GATConv(in, 64, heads=4, concat=True) → ReLU → GATConv(256, 64, heads=1) → global_mean_pool |
| `TemporalAttention` | MultiheadAttention(64, heads=4, dropout=0.1) + LayerNorm + residual → mean(dim=1) |
| `output_proj` | Linear(64, 64) |

### 4.4 Graph Construction

- **Topologi:** Fully-connected static (3 nodes → 6 directed edges, no self-loops)
- **Edge attributes:** Euclidean distance antara (lat, lon) pairs
- **Dynamic edges** (di `builder.py`): Wind-direction-based, tapi **TIDAK DIPAKAI** di pipeline aktual — selalu static fully-connected

### 4.5 MLP Baseline

Linear(input, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 3)

---

## 5. Training Config (Implementasi Aktual)

| Parameter | Nilai |
|-----------|-------|
| SEQ_LEN | 6 |
| BATCH_SIZE | 512 |
| EPOCHS | 20 |
| HIDDEN_DIM | 128 |
| GRAPH_DIM | 64 |
| K_NEIGHBORS (FAISS) | 3 |
| NUM_TARGETS | 3 |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) — gabungan params GNN + Diffusion |
| AMP | Enabled on CUDA (GradScaler) |
| Loss | Weighted MSE: base=1.0, ×5 if \|target\|>1.0, ×10 if \|target\|>3.0 |
| Best Val Loss | **0.0966** (saved to `models/diffusion_chkpt.pth`) |

---

## 6. Inference — Pure Diffusion (TANPA Hybrid)

> **CATATAN:** Hybrid persistence telah DIHAPUS dari thesis. Fokus sesuai judul:
> Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning.

### 6.1 Pipeline Inferensi

1. Input sequence → Conditioning (context + retrieval + GNN graph)
2. DDIM reverse sampling (20 steps) × 30 ensemble members
3. Denormalisasi + clamping
4. Output ensemble → metrik deterministik (median) + probabilistik (distribusi)

### 6.2 Denormalisasi

1. `samples_denorm = samples × t_std + t_mean`
2. Precipitation (idx 0): `torch.expm1()` (inverse log1p)
3. Clamping: precipitation ≥ 0, humidity ∈ [0, 100], wind_speed unclamped

### 6.3 Inference Config (Evaluasi Final)

| Parameter | Nilai |
|-----------|-------|
| Ensemble size | **30** |
| DDIM steps | **20** |
| EVAL_STEP | **24** (daily subsampling) |
| Test period | 2022–2025 |
| Sampel per node | 1,097 |
| Total valid samples | **3,291** |
| NaN skipped | **0** |

---

## 7. Hasil Evaluasi (Final, Verified)

### 7.1 Metrik Deterministik (Agregasi Semua Node)

| Variabel | RMSE | MAE | Correlation |
|----------|------|-----|-------------|
| Curah Hujan | 2.007 mm/jam | 0.849 | 0.455 |
| Kec. Angin | 2.009 m/s | 1.514 | 0.833 |
| Kelembapan | 6.712 % | 5.169 | 0.923 |

### 7.2 Metrik Probabilistik (Default Threshold)

| Variabel | Threshold | CRPS | Brier | POD | FAR | CSI |
|----------|-----------|------|-------|-----|-----|-----|
| Precipitation | 10 mm/jam | 0.830 | 0.020 | 0.059 | 0.875 | 0.042 |
| Wind Speed | 10 m/s | 1.488 | 0.099 | 0.505 | 0.166 | 0.459 |
| Humidity | 90% | 5.002 | 0.081 | 0.351 | 0.256 | 0.313 |

### 7.3 Threshold Sensitivity — Curah Hujan

| Threshold | N Events | POD | FAR | CSI | Brier |
|-----------|----------|-----|-----|-----|-------|
| **2 mm/jam** | 424 | **0.415** | 0.389 | **0.328** | 0.106 |
| 5 mm/jam | 153 | 0.248 | 0.616 | 0.178 | 0.052 |
| 10 mm/jam | 51 | 0.059 | 0.870 | 0.042 | 0.020 |

> CSI meningkat **7.8×** dari 10mm→2mm. Threshold 10 mm/jam terlalu tinggi karena hanya 0.151% data melebihi nilai ini.

### 7.4 Perbandingan vs Persistence Baseline (24h lag)

| Variabel | Metrik | Persistence | Model | Improvement |
|----------|--------|-------------|-------|-------------|
| Precipitation | RMSE | 1.385 | 2.007 | -45.0% |
| Precipitation | MAE | 0.432 | 0.849 | -96.7% |
| Precipitation | Corr | 0.259 | **0.455** | +0.196 |
| Wind Speed | RMSE | 2.878 | **2.009** | +30.2% |
| Wind Speed | MAE | 2.179 | **1.514** | +30.5% |
| Wind Speed | Corr | 0.551 | **0.833** | +0.282 |
| Humidity | RMSE | 7.631 | **6.712** | +12.0% |
| Humidity | MAE | 5.262 | **5.169** | +1.8% |
| Humidity | Corr | 0.809 | **0.923** | +0.115 |

**Skor: Model mengungguli persistence pada 7/9 metrik.** Precipitation RMSE/MAE lebih tinggi karena persistence mendapat "keuntungan gratis" dari 64% data bernilai 0 mm — tapi model memiliki korelasi jauh lebih tinggi (0.455 vs 0.259).

### 7.5 Precipitation Baseline Comparison

| Metode | RMSE | MAE | Correlation |
|--------|------|-----|-------------|
| Climatology Mean | 2.198 | 0.926 | 0.000 |
| Persistence (t-24) | 2.621 | 1.242 | 0.206 |
| Persist + Bias Corr | 2.621 | 1.242 | 0.206 |
| **RA-Diffusion (Ours)** | **2.009** | **0.849** | **0.455** |

### 7.6 Distribusi Curah Hujan (Fakta Penting)

- 64.2% data bernilai **0 mm** (tidak hujan)
- Hanya 4.08% > 2 mm/jam, 0.93% > 5 mm/jam, **0.151%** > 10 mm/jam
- P50 = 0.000, P90 = 0.800, P95 = 1.700, P99 = 4.900, Max = 30.4 mm/jam

---

## 8. Temuan & Interpretasi Kunci

### Strengths
1. **Pipeline end-to-end lengkap** — Retrieval-Augmented + Diffusion + Spatio-Temporal Graph, semua berfungsi terintegrasi
2. **Wind speed & humidity excellent** — Corr 0.83 dan 0.92, mengungguli persistence pada semua metrik
3. **Output probabilistik** — 30 ensemble members memungkinkan uncertainty quantification
4. **Numerically stable** — 0 NaN dari 3,291 sampel evaluasi
5. **Skill score positif** — Model mengungguli persistence pada 7/9 metrik

### Limitations
1. **Precipitation extreme** — POD≈0 pada 10 mm/jam (class imbalance 0.151%), POD=0.42 pada 2 mm/jam (meaningful)
2. **ERA5 resolusi** (~25 km) — Puncak & Lereng_Cibodas di grid cell yang sama
3. ~~**Hybrid weight tinggi**~~ — DIHAPUS dari thesis, fokus pure diffusion
4. **Precipitation RMSE > persistence** — Konsekuensi dari distribusi 64% zero; persistence "menang" dengan prediksi "tetap 0"

### Rekomendasi Perbaikan (untuk Future Work)
1. Data observasi BMKG AWS (resolusi tinggi)
2. Cost-sensitive / focal loss untuk heavy rain
3. Downscaling ERA5 + koreksi topografi
4. Fine-tuning diffusion model (epoch, loss, T_STD_MULTIPLIER)

---

## 9. Metrik yang Diimplementasi

| Metrik | File | Default Threshold |
|--------|------|-------------------|
| RMSE | `probabilistic_metrics.py` | — |
| MAE | `probabilistic_metrics.py` | — |
| Correlation (Pearson) | `probabilistic_metrics.py` | — |
| CRPS | `probabilistic_metrics.py` | — |
| Brier Score | `probabilistic_metrics.py` | 10.0 mm/jam |
| POD | `probabilistic_metrics.py` | 10.0 mm/jam, prob_threshold=0.5 |
| FAR | `probabilistic_metrics.py` | 10.0 mm/jam, prob_threshold=0.5 |
| CSI | `probabilistic_metrics.py` | 10.0 mm/jam, prob_threshold=0.5 |

> ROC-AUC dan F1 **TIDAK** diimplementasi — jangan menyebut dalam konteks evaluasi.

---

## 10. File Kunci & Fungsinya

| File | Fungsi | Kapan Dijalankan |
|------|--------|-----------------|
| `src/data/ingest.py` | Download ERA5 → parquet | Sekali (data sudah ada) |
| `src/data/temporal_loader.py` | DataLoader + normalisasi + graph | Dipanggil oleh train.py |
| `src/models/diffusion.py` | ConditionalDiffusionModel + RainForecaster | Dipanggil oleh train.py |
| `src/models/gnn.py` | SpatioTemporalGNN | Dipanggil oleh train.py |
| `src/models/mlp_baseline.py` | MLP baseline | Dipanggil oleh train_baseline.py |
| `src/graph/builder.py` | PangrangoGraphBuilder | Dipanggil oleh temporal_loader.py |
| `src/retrieval/base.py` | RetrievalDatabase (FAISS) | Dipanggil oleh train.py & inference.py |
| `src/train.py` | Training loop utama | `python -m src.train` |
| `src/train_baseline.py` | Training MLP baseline | `python -m src.train_baseline` |
| `src/inference.py` | Inference engine (pure diffusion) | Dipanggil oleh evaluasi scripts |
| `src/evaluation/probabilistic_metrics.py` | Semua metrik evaluasi | Dipanggil oleh evaluasi scripts |

---

## 11. Catatan untuk AI Session Berikutnya

1. **Jangan ubah arsitektur tanpa konfirmasi** — model sudah trained dan evaluated, checkpoint sudah final
2. **Jika user minta evaluasi ulang** — gunakan pola di `_archive/generate_plots.py` sebagai referensi pipeline inference
3. **Jika user minta re-training** — config ada di `src/train.py`, jalankan `python -m src.train`
4. **Precipitation RMSE tinggi bukan bug** — ini konsekuensi distribusi data (64% zero), sudah dianalisis dan didokumentasi
5. **Puncak & Lereng_Cibodas identik** — karena grid ERA5 sama, bukan bug
6. **Notebook evaluasi** sudah ada di `notebooks/evaluasi_model.ipynb` — berisi semua plot, metrik, dan interpretasi dalam Bahasa Indonesia
7. **`WeatherStateEncoder`** di `retrieval/base.py` ada tapi **tidak dipakai** — FAISS langsung pakai raw normalized features
8. **Dynamic edges** di `builder.py` ada tapi **tidak dipakai** — pipeline selalu pakai static fully-connected graph
