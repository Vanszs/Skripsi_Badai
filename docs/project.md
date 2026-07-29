# PROJECT OVERVIEW: SKRIPSI NOWCASTING PRESIPITASI OROGRAFIS GUNUNG GEDE-PANGRANGO

Dokumen ini menyajikan penjelas komprehensif mengenai latar belakang, tujuan, arsitektur sistem, dan formulasi ilmiah dari riset skripsi berjudul:

> **"Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning untuk Nowcasting Presipitasi di Gunung Gede-Pangrango"**

---

## 1. Tentang Proyek Ini (Apa dan Bagaimana)

Proyek ini adalah sistem **nowcasting presipitasi probabilistik jam-an (1-step-ahead rolling forecast)** yang dirancang khusus untuk kawasan pegunungan tropis dengan dinamika topografi kompleks, yaitu **Gunung Gede-Pangrango, Jawa Barat**.

### Masalah Utama yang Dihadapi:
1. **Dinamika Orografis & Kebasahan Dingin Ekstrem**: Wilayah puncak pegunungan sering mengalami pengangkatan massa udara paksa (*orographic lifting*), yang memicu hujan deras mendadak dan angin kencang. Fenomena ini menjadi pemicu utama kejadian hipotermia basah-dingin bagi pendaki.
2. **Keterbatasan Model NWP & Deterministik**: Model prakiraan cuaca fisik murni (NWP) atau jaringan saraf deterministik (seperti MLP/U-Net standar) cenderung menghasilkan prediksi yang terlalu halus (*over-smoothing*) dan sering meremehkan (*under-prediction*) intensitas hujan lebat pada skala mikro jam-an.
3. **Data Langka (Rare Events)**: Kejadian hujan lebat ekstrem tergolong langka dalam histori pencatatan, sehingga model generatif biasa sulit menangkap Trajektori cuaca ekstrem tanpa bimbingan konteks historis.

### Cara Kerja Sistem (Bagaimana Model Bekerja):
Sistem menggabungkan tiga pilar arsitektur utama:
1. **Spatio-Temporal Graph Neural Network (ST-GNN)**: Mengodekan ketergantungan spasial antar 5 stasiun pengamatan lereng-puncak dan evolusi temporalnya.
2. **Retrieval-Augmented Mechanism (FAISS)**: Mencari $k=3$ analog cuaca ekstrem historis yang paling mirip dengan kondisi atmosfer saat ini.
3. **Conditional Denoising Diffusion Model (DDPM/DDIM)**: Menggenerasi distribusi prediksi ensemble Monte Carlo (50 sampel) untuk menangkap ketidakpastian presipitasi secara probabilistik.

---

## 2. Tujuan Proyek

### Tujuan Ilmiah & Teknis:
1. **Meningkatkan Akurasi Prediksi Kejadian Ekstrem**: Mengatasi kecenderungan *over-smoothing* dengan menghasilkan sampel prediksi yang tajam dan peka terhadap presipitasi lebat ($>10 \text{ mm/jam}$).
2. **Memodelkan Ketidakpastian (Probabilistic Forecasting)**: Menyajikan distribusi probabilitas eksedansi hujan (persentil P10, P50/median, P90) alih-alih hanya satu nilai titik deterministik tunggal.
3. **Mengevaluasi Sinergi Komponen (Ablation Study)**: Mengukur kontribusi relatif integrasi ST-GNN dan Retrieval Analogs terhadap performa diffusion model melalui 6 skenario eksperimen.

### Tujuan Praktis & Keselamatan Pendaki:
1. **Sistem Peringatan Dini Hipotermia (EWS)**: Memberikan estimasi risiko hipotermia pendaki di titik Puncak Pangrango (`MAIN`) berdasarkan kombinasi suhu *wind chill*, kecepatan angin, dan intensitas hujan.
2. **Mitigasi Operasional Pendakian**: Menyediakan panduan bagi Balai Besar Taman Nasional Gunung Gede Pangrango (TNGGP) dan Tim SAR untuk penutupan jalur atau instruksi evakuasi dini pendaki.

---

## 3. Arsitektur Sistem & Komponen Pembentuk

Arsitektur model dibangun atas **5 layer utama** yang saling terintegrasi:

```
[ Data Ingestion ERA5 (5 Node) ] ──> [ Preprocessing: Lag-1, log1p, Z-score ]
                                                    │
                                                    ▼
                 ┌──────────────────────────────────┴──────────────────────────────────┐
                 │                                                                     │
                 ▼                                                                     ▼
   [ ST-GNN Module (5-Node Star) ]                                   [ FAISS L2 Retrieval Module ]
   ├─ SpatialGNN (GATConv 2-layer)                                   ├─ IndexFlatL2 (144-dim embedding)
   └─ TemporalAttention (Causal Mask)                                 └─ k=3 Historical Analogs (R_t)
                 │                                                                     │
                 └──────────────────────────────────┬──────────────────────────────────┘
                                                    │ Joint Conditioning (c_t)
                                                    ▼
                                 [ Conditional Denoising Network (f_theta) ]
                                 ├─ Timestep Embedding (Sinusoidal)
                                 ├─ ResNet Block + Multi-head Attention
                                 └─ DDPMScheduler / DDIMScheduler (1000 timesteps)
                                                    │
                                                    ▼
                                 [ Monte Carlo Ensemble Sampling (50 Samples) ]
                                                    │
                                                    ▼
                                 [ Evaluation & Hypothermia Risk Assessment ]
```

---

## 4. Rincian Arsitektur & Persamaan Matematika

### A. Topologi Graf Star 5 Node
Model menggunakan topologi graf star terarah dengan 5 simpul observasi:
* **`MAIN` (-6.70°, 106.98°, 3008 mdpl)**: Puncak Pangrango / Mandalawangi (**Target Utama Prediksi**).
* **`UP` (-6.45°, 106.98°, 1450 mdpl)**: Konteks Lereng Utara.
* **`DOWN` (-6.95°, 106.98°, 980 mdpl)**: Konteks Lereng Selatan.
* **`LEFT` (-6.70°, 106.73°, 1120 mdpl)**: Konteks Lereng Barat.
* **`RIGHT` (-6.70°, 107.23°, 1250 mdpl)**: Konteks Lereng Timur.

Topologi ini terdiri dari **8 edge terarah**: `MAIN ↔ UP`, `MAIN ↔ DOWN`, `MAIN ↔ LEFT`, dan `MAIN ↔ RIGHT`.

### B. Preprocessing Data (Eq 3.1 - 3.3)
1. **Fitur Lag 1 Jam** (Eq 3.1):
   $$\text{PrecipitationLag1}_t = \text{Precipitation}_{t-1}$$
2. **Transformasi Logaritmik** (Eq 3.2):
   $$y' = \log(1 + y) \quad (\text{native } \texttt{torch.log1p})$$
3. **Normalisasi Z-Score** (Eq 3.3):
   $$z = \frac{y' - \mu}{\sigma + \varepsilon} \quad (\text{dengan } \varepsilon = 10^{-5})$$

### C. Spatio-Temporal Graph Neural Network (ST-GNN) (Eq 3.4 - 3.7)
1. **Graph Sequence Construction** (Eq 3.4 & 3.5):
   $$X_t = \{x_{t-L+1}, \dots, x_t\}, \quad g = \{G_{t-L+1}, \dots, G_t\}$$
2. **Spatial Graph Attention Network (GAT)** (Eq 3.6):
   $$h_\tau = \text{GAT}(G_\tau)$$
   Menggunakan 2 layer `GATConv` dengan 4 attention heads dan `edge_dim=1`.
3. **Temporal Attention Aggregation** (Eq 3.7):
   $$h_t^G = \text{TemporalAttention}(h_{t-L+1}, \dots, h_t)$$
   Self-attention berpenanda posisi (*positional embedding*) dengan *causal mask* untuk mencegah bocornya informasi masa depan.

### D. Retrieval-Augmented Historical Analogs (FAISS L2) (Eq 2.4 & 3.8)
1. **Euclidean Distance L2 Index** (Eq 2.4):
   $$d(x_q, x_i) = \sqrt{\sum_{j=1}^m (x_{q,j} - x_{i,j})^2}$$
   Pencarian $k=3$ analog cuaca ekstrem historis menggunakan `faiss.IndexFlatL2`.
2. **Retrieval Feature Vector** (Eq 3.8):
   $$R_t = k\text{NN}(c_t, \mathcal{D}_{\text{train}}, k)$$

### E. Conditional Denoising Diffusion Model (Eq 2.2, 2.3, 3.9 - 3.11)
1. **Forward Diffusion Process** (Eq 2.2 & 3.9):
   $$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)$$
   $$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
2. **Conditional Noise Predictor** (Eq 3.10):
   $$\hat{\epsilon}_\theta = f_\theta(x_t, t, h_t^G, R_t)$$
3. **Denoising Loss Function** (Eq 2.3):
   $$\mathcal{L}_{\text{diff}} = \mathbb{E}_{x_0, \epsilon, t} \left[ \|\epsilon - \hat{\epsilon}_\theta(x_t, t, c)\|_2^2 \right]$$
   Diperkaya dengan *Weighted MSE* untuk bobot presipitasi ekstrem.
4. **Probabilistic Prediction Distribution** (Eq 3.11):
   $$p(y_{t+1}^{\text{MAIN}} \mid X_t)$$
   Dihasilkan dari 50 sampel ensemble Monte Carlo via `RainForecaster.sample_fast`.

---

## 5. Ringkasan Skenario Eksperimen Ablation Study (6 Skenario)

Untuk membuktikan kontribusi masing-masing komponen, dilakukan evaluasi 6 skenario:

| Skenario | Deskripsi Model | Komponen Aktif |
| :--- | :--- | :--- |
| **1. Persistence** | Naif baseline ($t+1 = t$) | Tanpa ML |
| **2. MLP Baseline** | Deterministik Neural Network | Single Node MAIN |
| **3. Diffusion Only** | Conditional Diffusion tanpa GNN & Retrieval | Diffusion Denoising |
| **4. Diffusion + Retrieval** | Diffusion + FAISS Historical Analogs | Diffusion + FAISS L2 |
| **5. Diffusion + ST-GNN** | Diffusion + 5-Node Spatial GAT & Temporal Attention | Diffusion + ST-GNN |
| **6. Full Proposed Model** | Model Utuh (Conditional Diffusion + ST-GNN + Retrieval) | **Full Component Synergy** |

---

## 6. Struktur Direktori Repository

* `src/config.py`: Kontrak sistem 5-node star graph, parameter fisik, & metadata.
* `src/data/ingest.py`: Modul pengambilan data meteorologi Open-Meteo ERA5 & pembuat lag-1.
* `src/data/temporal_loader.py`: Preprocessing `log1p`, normalisasi `z-score`, & PyG DataLoader.
* `src/models/gnn.py`: Modul Spatial GAT & Temporal Attention (`SpatioTemporalGNN`).
* `src/models/diffusion.py`: Arsitektur `ConditionalDiffusionModel`, DDPMScheduler, & sampler.
* `src/retrieval/base.py`: Wrapper `faiss.IndexFlatL2` pencarian $k$-NN analog historis.
* `src/evaluation/probabilistic_metrics.py`: Evaluasi MAE, RMSE, Pearson $r$, CRPS, Brier Score, POD, FAR, CSI, & Reliability Diagram.
* `src/train.py`: Pipeline pelatihan model utama.
* `src/inference.py`: Pipeline inferensi one-step-ahead probabilistik rolling forecast.
* `run_eval_final.py`: Ekskusi evaluasi komprehensif 6 skenario eksperimen.

---
*Dokumen ini dibuat otomatis sebagai rangkuman resmi riset skripsi.*
