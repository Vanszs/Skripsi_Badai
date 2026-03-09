# Diagram dan Tabel Skripsi
## Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning
### Nowcasting Probabilistik Cuaca Multi-Variabel Gunung Gede-Pangrango

---

## DAFTAR DIAGRAM

---

### Diagram 1 — Alur Penelitian

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam roundCorner 10
skinparam defaultFontName Arial
skinparam ActivityBackgroundColor #E8F4FD
skinparam ActivityBorderColor #2E86C1
skinparam ArrowColor #2E86C1

title Diagram Alur Penelitian

start

:=== **Pengambilan Data** ===
ERA5 Reanalysis via Open-Meteo API
Periode: 2005–2025 (hourly)
3 Node: Puncak, Lereng Cibodas, Hilir Cianjur;

:=== **Pra-Pemrosesan Data** ===
- Transformasi log1p (presipitasi)
- Z-normalisasi (stats dari training saja)
- Pembuatan fitur lag (lag-1, lag-3)
- Pengambilan elevasi & land-sea mask;

:=== **Pembagian Dataset Temporal** ===
- Training: 2005–2018 (14 tahun)
- Validasi: 2019–2021 (3 tahun)
- Pengujian: 2022–2025 (4 tahun);

:=== **Pembentukan Fitur & Graf** ===
- Sliding window (seq_len = 6)
- Graf fully-connected (3 node)
- Edge attr: jarak Euclidean / arah angin;

fork
  :=== **Modul Retrieval** ===
  FAISS IndexIVFFlat
  k = 3 nearest neighbors
  Embedding: fitur meteorologis
  ternormalisasi;
fork again
  :=== **Modul Graf Spasio-Temporal** ===
  SpatialGNN (GAT 2-layer, 4 heads)
  → TemporalAttention (4 heads)
  → Graph Embedding [64-dim];
end fork

:=== **Pelatihan Model** ===
Conditional Diffusion Model (DDPM)
- Time embedding (sinusoidal)
- Context conditioning (fitur cuaca)
- Retrieval conditioning (FAISS)
- Graph conditioning (ST-GNN)
Optimizer: AdamW (lr=1e-3, wd=1e-4)
Loss: Standard MSE pada noise;

:=== **Inferensi** ===
- DDIM Sampling (20 steps)
- Ensemble 30 sampel
- Denormalisasi (clip_sample=False)
- Clamp (precip>=0, humidity 0-100);

:=== **Evaluasi** ===
- Deterministik: RMSE, MAE, Correlation
- Probabilistik: CRPS, Brier Score
- Threshold: POD, FAR, CSI;

stop
@enduml
```

---

### Diagram 2 — Skema Pembagian Dataset Temporal

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial

title Skema Pembagian Dataset Temporal\n(Strict Temporal Split — Mencegah Kebocoran Informasi)

scale 1 as 60 pixels

concise "Dataset" as D

@D
0 is {-}

@2005
D is "Training (14 tahun)\n~67% data" #90CAF9

@2019
D is "Validasi (3 tahun)\n~14% data" #FFF59D

@2022
D is "Pengujian (4 tahun)\n~19% data" #EF9A9A

@2026
D is {-}

@D
2005 <-> 2019 : 2005-01-01 s/d 2018-12-31
2019 <-> 2022 : 2019-01-01 s/d 2021-12-31
2022 <-> 2026 : 2022-01-01 s/d 2025-12-31

@enduml
```

**Keterangan:**
- **Training (2005–2018):** Digunakan untuk melatih model dan menghitung statistik normalisasi. Data retrieval FAISS juga hanya berasal dari periode ini.
- **Validasi (2019–2021):** Digunakan untuk early stopping dan tuning hyperparameter. Dinormalisasi dengan statistik dari data training.
- **Pengujian (2022–2025):** Evaluasi performa final. Tidak pernah dilihat selama proses pelatihan.
- **Tidak ada random shuffle** — pembagian berdasarkan waktu untuk menghindari kebocoran informasi temporal.

---

### Diagram 3 — Struktur Graf Spasio-Temporal Antar Node

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial
skinparam componentStyle rectangle

title Struktur Graf Spasio-Temporal\nGunung Gede-Pangrango (Fully-Connected)

package "Graf Spasio-Temporal" {

  node "**Puncak**\n(-6.7698, 106.9636)\nElevasi: ~2958 m" as N1 #B3E5FC
  node "**Lereng Cibodas**\n(-6.7517, 106.9872)\nElevasi: ~1524 m" as N2 #C8E6C9
  node "**Hilir Cianjur**\n(-6.8160, 107.1330)\nElevasi: ~468 m" as N3 #FFF9C4

  N1 <--> N2 : edge (jarak / angin)
  N2 <--> N3 : edge (jarak / angin)
  N1 <--> N3 : edge (jarak / angin)
}

note bottom of N1
  **Fitur per Node (9 fitur):**
  temperature_2m, relative_humidity_2m,
  dewpoint_2m, surface_pressure,
  wind_speed_10m, wind_direction_10m,
  cloudcover, precipitation_lag1,
  elevation
end note

note right
  **Topologi:**
  - Statis: Fully-connected, bobot = jarak Euclidean
  - Dinamis: Berdasarkan arah & kecepatan angin
    (edge hanya jika angin menuju node tujuan ±45°)
end note

@enduml
```

---

### Diagram 4 — Mekanisme Graph Attention Network (GAT)

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial
skinparam packageStyle rectangle

title Mekanisme Graph Attention Network (GAT)\npada Spatio-Temporal GNN

rectangle "**Input: Node Features**\n[Num_Nodes × Input_Dim]" as INPUT #E8F4FD

rectangle "**GAT Layer 1**" as GAT1 #BBDEFB {
  rectangle "Linear Transform\nW · h_i → h'_i" as LIN1
  rectangle "Attention Coefficients\nα_ij = softmax(LeakyReLU(a^T[h'_i ∥ h'_j]))" as ATT1
  rectangle "Multi-Head Aggregation (4 heads)\nh_i^(l+1) = ∥_{k=1}^{4} σ(Σ_j α_ij^k · W^k · h_j)" as AGG1
}

rectangle "ReLU Activation" as RELU #C8E6C9

rectangle "**GAT Layer 2**" as GAT2 #BBDEFB {
  rectangle "Linear Transform\nW · h_i → h'_i" as LIN2
  rectangle "Attention Coefficients\nα_ij = softmax(LeakyReLU(a^T[h'_i ∥ h'_j]))" as ATT2
  rectangle "Single-Head Output\nh_i^(out) = σ(Σ_j α_ij · W · h_j)" as AGG2
}

rectangle "**Global Mean Pooling**\ng = (1/N) Σ_i h_i^(out)\n[Batch × Output_Dim]" as POOL #FFF9C4

INPUT --> LIN1
LIN1 --> ATT1
ATT1 --> AGG1
AGG1 --> RELU
RELU --> LIN2
LIN2 --> ATT2
ATT2 --> AGG2
AGG2 --> POOL

note right of ATT1
  **Attention menentukan:**
  Seberapa besar pengaruh
  node j terhadap node i
  berdasarkan kesamaan
  fitur meteorologis.
  
  Node dengan pola cuaca
  serupa mendapat bobot
  attention lebih tinggi.
end note

note right of POOL
  **Output:**
  Representasi graf tunggal
  per timestep, digunakan
  sebagai input Temporal
  Attention.
end note

@enduml
```

---

### Diagram 5 — Mekanisme Retrieval Historical Analogs

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial

title Mekanisme Retrieval Historical Analogs (FAISS)

rectangle "**Tahap Offline (Training Time)**" as OFFLINE #E8F4FD {
  rectangle "Data Training\n2005–2018\n(ternormalisasi)" as TRAIN_DATA
  rectangle "**WeatherStateEncoder**\nFlatten → Linear → ReLU\n→ Linear → Tanh" as ENCODER
  rectangle "**FAISS IndexIVFFlat**\n~368K vektor\n256 Voronoi cells\nnprobe = 16" as INDEX
  
  TRAIN_DATA --> ENCODER : fitur cuaca\nper timestep
  ENCODER --> INDEX : embedding\n[dim = num_features]
}

rectangle "**Tahap Online (Inference/Training)**" as ONLINE #FFF9C4 {
  rectangle "Kondisi Cuaca\nSaat Ini (t-1)\n[1 × num_features]" as QUERY
  rectangle "**FAISS Search**\nL2 Distance\nk = 3 neighbors" as SEARCH
  rectangle "**Retrieved Analogs**\n[1 × k × features]\n= [1 × 3 × features]" as RESULT
  rectangle "**Retrieval MLP**\nLinear → SiLU → Linear\n→ retrieval embedding" as RET_MLP
  
  QUERY --> SEARCH
  SEARCH --> RESULT : top-3\nterdekat
  RESULT --> RET_MLP : flatten\n[1 × k*features]
}

INDEX <.. SEARCH : query index

rectangle "**Diffusion Model**\n(conditioning input)" as DIFFUSION #C8E6C9

RET_MLP --> DIFFUSION : retrieval\nembedding\n[hidden_dim]

note bottom of SEARCH
  **Prinsip:**
  Mencari kondisi historis
  yang paling mirip dengan
  kondisi saat ini.
  
  Hipotesis: Pola cuaca serupa
  di masa lalu menghasilkan
  outcome serupa.
end note

@enduml
```

---

### Diagram 6 — Arsitektur Conditional Diffusion Model

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial

title Arsitektur Conditional Diffusion Model\n(ConditionalDiffusionModel)

rectangle "**Forward Diffusion (Training)**" as FWD #FFCDD2 {
  rectangle "x₀: Target Asli\n[B × 3]\n(precip, wind, humidity)" as X0
  rectangle "Tambah Noise Gaussian\nxₜ = √(ᾱₜ)·x₀ + √(1-ᾱₜ)·ε\nt ~ U(0, 999)" as ADD_NOISE
  rectangle "xₜ: Target + Noise\n[B × 3]" as XT
  
  X0 --> ADD_NOISE
  ADD_NOISE --> XT
}

rectangle "**Conditioning Inputs**" as COND #E8F4FD {
  rectangle "Time Embedding\nSinusoidal → MLP\n[B × hidden_dim]" as TIME_EMB
  rectangle "Context MLP\nFitur cuaca (t-1)\n[B × hidden_dim]" as COND_MLP
  rectangle "Retrieval MLP\nFAISS k=3 analogs\n[B × hidden_dim]" as RET_MLP
  rectangle "Graph MLP\nST-GNN embedding\n[B × hidden_dim]" as GRAPH_MLP
}

rectangle "**Denoising Network (U-Net-like)**" as UNET #C8E6C9 {
  rectangle "Down Block 1\nLinear(3→128) + SiLU" as DOWN1
  rectangle "Down Block 2\nLinear(128→256) + SiLU" as DOWN2
  rectangle "Mid Block\nLinear(256→256) + SiLU" as MID
  rectangle "Up Block (+ Skip Connection)\nLinear(512→128) + SiLU" as UP1
  rectangle "Output\nLinear(128→3)" as OUT
}

rectangle "**ε̂: Predicted Noise**\n[B × 3]" as NOISE_PRED #FFF9C4

XT --> DOWN1
DOWN1 --> DOWN2
DOWN2 --> MID

' Conditioning injection
TIME_EMB --> DOWN1 : + emb
COND_MLP --> DOWN1 : + emb
RET_MLP --> DOWN1 : + emb
GRAPH_MLP --> DOWN1 : + emb

' Skip connection
DOWN2 --> UP1 : concat\n(skip)
MID --> UP1

UP1 --> OUT
OUT --> NOISE_PRED

note right of NOISE_PRED
  **Loss:**
  Standard MSE(ε̂, ε)
  
  F.mse_loss(noise_pred, noise)
  tanpa weighting — sesuai
  formulasi DDPM standar.
end note

note bottom of UNET
  **Scheduler:**
  - Training: DDPM (T = 1000 steps)
  - Inference: DDIM (20 steps, ~50× lebih cepat)
end note

@enduml
```

---

### Diagram 7 — Proses Pembangkitan Ensemble pada Tahap Inferensi

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial

title Proses Pembangkitan Ensemble pada Inferensi\n(DDIM Sampling → Denormalisasi → Output)

rectangle "**Input Kondisi**" as INPUT #E8F4FD {
  rectangle "Sequence Fitur\n[seq_len=6, features]\n(ternormalisasi)" as SEQ
}

rectangle "**Conditioning Pipeline**" as PIPE #BBDEFB {
  rectangle "ST-GNN\n→ graph_emb [64]" as GNN
  rectangle "FAISS\n→ retrieved [k×feat]" as FAISS
  rectangle "Context\n→ fitur terakhir [feat]" as CTX
}

SEQ --> GNN
SEQ --> FAISS
SEQ --> CTX

rectangle "**DDIM Reverse Sampling**\n(20 denoising steps)" as DDIM #FFF9C4 {
  rectangle "x_T ~ N(0, I)\n[30 x 3]\n(30 sampel independen)" as NOISE
  rectangle "Iterasi t = T → 0:\nx_{t-1} = denoise(x_t, t, cond)" as LOOP
  rectangle "x₀: Raw Predictions\n[30 × 3]" as RAW
  
  NOISE --> LOOP
  LOOP --> RAW
}

GNN --> LOOP : graph_emb
FAISS --> LOOP : retrieval
CTX --> LOOP : context

rectangle "**Denormalisasi**" as DENORM #C8E6C9 {
  rectangle "Precip: expm1(x·σ + μ)\nWind: x·σ + μ\nHumidity: x·σ + μ" as INV
  rectangle "Clamp:\nprecip ≥ 0\nhumidity ∈ [0, 100]" as CLAMP
}

RAW --> INV
INV --> CLAMP

rectangle "**Output Ensemble**\n[30 sampel per variabel]" as OUTPUT #E1BEE7 {
  rectangle "Median → Prediksi Deterministik" as MEDIAN
  rectangle "Spread → Interval Ketidakpastian\n(P10–P90)" as SPREAD
  rectangle "Distribusi → CRPS, Brier Score" as DIST
}

CLAMP --> MEDIAN
CLAMP --> SPREAD
CLAMP --> DIST

@enduml
```

---

### Diagram 8 — Kerangka Evaluasi dan Analisis Trade-Off Model

```plantuml
@startuml
skinparam backgroundColor #FEFEFE
skinparam shadowing false
skinparam defaultFontName Arial

title Kerangka Evaluasi dan Analisis Trade-Off Model

rectangle "**Prediksi Model**\n(Ensemble 30 sampel)" as PRED #E8F4FD

rectangle "**Observasi Aktual**\n(ERA5 Test Set 2022–2025)" as OBS #E8F4FD

rectangle "**Evaluasi Deterministik**\n(Median Ensemble vs Aktual)" as DET #BBDEFB {
  rectangle "RMSE\n(Root Mean Square Error)" as RMSE
  rectangle "MAE\n(Mean Absolute Error)" as MAE
  rectangle "Correlation\n(Pearson r)" as CORR
}

rectangle "**Evaluasi Probabilistik**\n(Distribusi Ensemble vs Aktual)" as PROB #C8E6C9 {
  rectangle "CRPS\n(Continuous Ranked\nProbability Score)" as CRPS
  rectangle "Brier Score\n(Kalibrasi probabilitas\nkejadian biner)" as BRIER
}

rectangle "**Evaluasi Threshold**\n(Deteksi Kejadian Ekstrem)" as THRESH #FFF9C4 {
  rectangle "POD\n(Probability of Detection)\nTP / (TP+FN)" as POD
  rectangle "FAR\n(False Alarm Ratio)\nFP / (TP+FP)" as FAR
  rectangle "CSI\n(Critical Success Index)\nTP / (TP+FP+FN)" as CSI
}

PRED --> DET
PRED --> PROB
PRED --> THRESH
OBS --> DET
OBS --> PROB
OBS --> THRESH

rectangle "**Analisis Trade-Off**" as TRADE #FFCCBC {
  rectangle "Akurasi Rata-Rata\nvs\nSensitivitas Kejadian Ekstrem" as TRADEOFF
  rectangle "Perbandingan dengan\nMLP Baseline (deterministik)\n→ menunjukkan added value\n  pendekatan probabilistik" as BASELINE
}

DET --> TRADEOFF
PROB --> TRADEOFF
THRESH --> TRADEOFF
TRADEOFF --> BASELINE

note bottom of THRESH
  **Threshold per Variabel:**
  - Presipitasi: > 10 mm/jam (hujan lebat)
  - Kecepatan Angin: > 10 m/s (angin kencang)
  - Kelembapan: > 90% (sangat tinggi)
end note

@enduml
```

---

## DAFTAR TABEL

---

### Tabel 1 — Lokasi Node Penelitian

| No | Nama Node | Latitude | Longitude | Elevasi (m) | Karakteristik |
|----|-----------|----------|-----------|-------------|---------------|
| 1 | Puncak | -6.769797 | 106.963583 | ~2958 | Zona puncak gunung, elevasi tertinggi |
| 2 | Lereng Cibodas | -6.751722 | 106.987160 | ~1524 | Zona lereng tengah, kawasan hutan |
| 3 | Hilir Cianjur | -6.816000 | 107.133000 | ~468 | Zona dataran rendah, hilir |

**Catatan:** Koordinat didefinisikan dalam `src/data/ingest.py`. Elevasi diperoleh melalui Open-Meteo Elevation API. Ketiga node membentuk graf fully-connected untuk memodelkan interaksi spasial antar zona ketinggian.

---

### Tabel 2 — Variabel Meteorologis yang Digunakan

| No | Variabel | Satuan | Sumber | Tipe |
|----|----------|--------|--------|------|
| 1 | `precipitation` | mm/jam | ERA5 Reanalysis | Dinamis |
| 2 | `temperature_2m` | °C | ERA5 Reanalysis | Dinamis |
| 3 | `relative_humidity_2m` | % | ERA5 Reanalysis | Dinamis |
| 4 | `dewpoint_2m` | °C | ERA5 Reanalysis | Dinamis |
| 5 | `surface_pressure` | hPa | ERA5 Reanalysis | Dinamis |
| 6 | `wind_speed_10m` | m/s | ERA5 Reanalysis | Dinamis |
| 7 | `wind_direction_10m` | ° (derajat) | ERA5 Reanalysis | Dinamis |
| 8 | `cloudcover` | % | ERA5 Reanalysis | Dinamis |
| 9 | `elevation` | m | Open-Meteo Elevation API | Statis |
| 10 | `land_sea_mask` | biner (0/1) | Turunan dari elevasi | Statis |
| 11 | `precipitation_lag1` | mm/jam | Lag-1 dari presipitasi | Autoregresif |
| 12 | `precipitation_lag3` | mm/jam | Lag-3 dari presipitasi | Autoregresif |

**Catatan:** Data diambil melalui Open-Meteo Archive API untuk periode 2005–2025 dengan resolusi temporal per jam. Variabel lag dihitung per node setelah pengambilan data.

---

### Tabel 3 — Fitur Input dan Variabel Target Model

| Kategori | Variabel | Keterangan |
|----------|----------|------------|
| **Fitur Input (Conditioning)** | `temperature_2m` | Suhu udara 2m |
| | `relative_humidity_2m` | Kelembapan relatif 2m |
| | `dewpoint_2m` | Titik embun (humidity proxy) |
| | `surface_pressure` | Tekanan permukaan |
| | `wind_speed_10m` | Kecepatan angin 10m |
| | `wind_direction_10m` | Arah angin 10m |
| | `cloudcover` | Tutupan awan (convective proxy) |
| | `precipitation_lag1` | Presipitasi 1 jam sebelumnya |
| | `elevation` | Ketinggian node (statis) |
| **Variabel Target** | `precipitation` | Presipitasi (mm/jam) |
| | `wind_speed_10m` | Kecepatan angin (m/s) |
| | `relative_humidity_2m` | Kelembapan relatif (%) |

**Catatan:** Model memprediksi 3 variabel target secara simultan (multi-output). `temperature_2m` awalnya dipertimbangkan sebagai target namun dikeluarkan (excluded) dalam implementasi final.

---

### Tabel 4 — Konfigurasi Pra-pemrosesan Data

| Langkah | Variabel | Metode | Detail |
|---------|----------|--------|--------|
| Transformasi | `precipitation` | Log1p | $x' = \log(1 + x)$ — mengurangi skewness distribusi presipitasi |
| Transformasi | `wind_speed_10m` | Tidak ada | Nilai asli digunakan langsung |
| Transformasi | `relative_humidity_2m` | Tidak ada | Nilai asli digunakan langsung |
| Normalisasi Target | `precipitation` | Z-score standar | $z = \frac{x' - \mu}{\sigma}$ — tanpa multiplier, target range N(0,1) |
| Normalisasi Target | `wind_speed_10m`, `relative_humidity_2m` | Z-score standar | $z = \frac{x - \mu}{\sigma}$ |
| Normalisasi Fitur | Semua fitur input | Z-score standar | $z = \frac{x - \mu_c}{\sigma_c + 10^{-5}}$ |
| Lag Features | `precipitation_lag1` | Shift(1) per node | Presipitasi 1 jam sebelumnya, NaN diisi 0 |
| Lag Features | `precipitation_lag3` | Shift(3) per node | Presipitasi 3 jam sebelumnya, NaN diisi 0 |

**PENTING:** Statistik normalisasi ($\mu$, $\sigma$) dihitung **hanya dari data training (2005–2018)** untuk mencegah kebocoran informasi. Data validasi dan test dinormalisasi dengan statistik yang sama.

---

### Tabel 5 — Konfigurasi Arsitektur Model

| Komponen | Parameter | Nilai |
|----------|-----------|-------|
| **Spatio-Temporal GNN** | | |
| └ SpatialGNN (GAT) | Jumlah layer | 2 |
| | Attention heads (Layer 1) | 4 (concat) |
| | Attention heads (Layer 2) | 1 |
| | Input dim | num_features (9) |
| | Hidden dim | 64 |
| | Output dim | 64 |
| | Dropout | 0.1 |
| └ TemporalAttention | Mekanisme | Multi-Head Self-Attention |
| | Attention heads | 4 |
| | Dropout | 0.1 |
| | Agregasi | Mean over sequence |
| └ Output Projection | Dimensi output (graph_dim) | 64 |
| **Conditional Diffusion Model** | | |
| | Input dim (num_targets) | 3 |
| | Context dim | num_features (9) |
| | Retrieval dim | num_features × k = 9 × 3 = 27 |
| | Graph dim | 64 |
| | Hidden dim | 128 |
| | Time embedding | Sinusoidal positional |
| | Arsitektur backbone | U-Net-like (down→mid→up + skip) |
| | Aktivasi | SiLU (backbone), GELU (time MLP) |
| **Diffusion Scheduler** | | |
| | Training scheduler | DDPM |
| | Timesteps (T) | 1000 |
| | Inference scheduler | DDIM |
| | Inference steps | 20 |
| **Retrieval Module** | | |
| | Indeks FAISS | IndexIVFFlat (L2 distance) |
| | Voronoi cells (nlist) | 256 |
| | Search probe (nprobe) | 16 |
| | k neighbors | 3 |

---

### Tabel 6 — Hyperparameter Pelatihan Model

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Optimizer | AdamW | Dengan weight decay |
| Learning rate | 1 × 10⁻³ | Untuk seluruh parameter (GNN + Diffusion) |
| Weight decay | 1 × 10⁻⁴ | Regularisasi L2 |
| Batch size | 512 | Dioptimalkan untuk RTX 3050 4GB VRAM |
| Epochs (Diffusion) | 20 | Dengan validasi per epoch |
| Epochs (MLP) | 51 | Early stopped (patience=10) |
| Sequence length | 6 | 6 timestep per sliding window |
| Mixed precision | AMP (FP16) | Otomatis jika GPU tersedia |
| Gradient scaling | GradScaler | Untuk stabilitas AMP |
| Loss function | Standard MSE | `F.mse_loss(noise_pred, noise)` — DDPM standar |
| Scheduler (MLP) | CosineAnnealingLR | Untuk MLP baseline |
| Early stopping (MLP) | patience=10 | + best checkpoint selection |
| Best val_loss (Diff) | 0.1210 | Checkpoint terbaik disimpan |

---

### Tabel 7 — Metrik Evaluasi Model

| Kategori | Metrik | Formula | Tujuan |
|----------|--------|---------|--------|
| **Deterministik** | RMSE | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$ | Mengukur rata-rata besarnya kesalahan prediksi; sensitif terhadap outlier |
| | MAE | $\frac{1}{N}\sum_{i=1}^{N}\|\hat{y}_i - y_i\|$ | Mengukur rata-rata kesalahan absolut; lebih robust terhadap outlier |
| | Correlation | $r = \frac{\text{cov}(\hat{y}, y)}{\sigma_{\hat{y}} \cdot \sigma_y}$ | Mengukur kekuatan hubungan linear antara prediksi dan observasi |
| **Probabilistik** | CRPS | $\text{CRPS} = E\|X - y\| - \frac{1}{2}E\|X - X'\|$ | Mengukur kualitas distribusi prediksi; semakin rendah semakin baik |
| | Brier Score | $BS = \frac{1}{N}\sum_{i=1}^{N}(p_i - o_i)^2$ | Mengukur kalibrasi probabilitas untuk kejadian biner (hujan/tidak hujan di atas threshold) |
| **Threshold** | POD | $\frac{TP}{TP + FN}$ | Proporsi kejadian ekstrem yang berhasil terdeteksi oleh model |
| | FAR | $\frac{FP}{TP + FP}$ | Proporsi alarm palsu dari seluruh prediksi positif model |
| | CSI | $\frac{TP}{TP + FP + FN}$ | Indeks gabungan yang mempertimbangkan hit, miss, dan false alarm |

**Keterangan:**
- $\hat{y}$: prediksi (median ensemble untuk metrik deterministik)
- $y$: observasi aktual
- $X, X'$: sampel independen dari distribusi ensemble
- $p_i$: probabilitas prediksi (fraksi ensemble yang melampaui threshold)
- $o_i$: observasi biner (1 jika melampaui threshold, 0 jika tidak)
- TP: True Positive, FP: False Positive, FN: False Negative

---

### Tabel 8 — Ambang Intensitas untuk Evaluasi Threshold

| Variabel | Ambang | Satuan | Kategori | Keterangan |
|----------|--------|--------|----------|------------|
| `precipitation` | > 10.0 | mm/jam | Hujan lebat | Sesuai klasifikasi BMKG untuk intensitas hujan lebat per jam |
| `wind_speed_10m` | > 10.0 | m/s | Angin kencang | Kecepatan angin yang berpotensi berbahaya bagi pendaki |
| `relative_humidity_2m` | > 90.0 | % | Kelembapan sangat tinggi | Indikator potensi kabut tebal dan visibilitas rendah |

**Catatan:**
- Ambang digunakan untuk menghitung metrik threshold: POD, FAR, CSI, dan Brier Score
- Brier Score menggunakan probabilitas ensemble: $p = \frac{\text{jumlah sampel} > \text{threshold}}{\text{total sampel ensemble}}$
- Evaluasi threshold penting untuk konteks keselamatan pendaki di Gunung Gede-Pangrango — deteksi kejadian ekstrem lebih kritis daripada akurasi rata-rata

---
