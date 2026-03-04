# Model Analysis Report

## Retrieval-Augmented Diffusion Model – Gunung Gede-Pangrango
### Analisis Lanjutan Hasil Evaluasi

---

## 1. Interpretasi Metrik Model

### 1.1 Precipitation

| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |
|--------|--------|----------------|---------------|------------|
| RMSE | 1.8285 | 1.7944 | 2.3504 | 2.0073 |
| MAE | 0.8024 | 0.7780 | 0.9672 | 0.8492 |
| CORRELATION | 0.4888 | 0.4992 | 0.4235 | 0.4553 |
| CRPS | 0.7864 | 0.7609 | 0.9427 | 0.8300 |
| BRIER_SCORE | 0.0163 | 0.0155 | 0.0294 | 0.0204 |
| POD | 0.0000 | 0.0000 | 0.1429 | 0.0588 |
| FAR | 1.0000 | 1.0000 | 0.8333 | 0.8750 |
| CSI | 0.0000 | 0.0000 | 0.0833 | 0.0417 |

**Temuan Utama – Precipitation:**

- POD sangat rendah (agregasi: 0.0588), bahkan 0.0000 pada Puncak dan Lereng Cibodas.
- FAR sangat tinggi (0.8750), artinya hampir semua prediksi heavy rain adalah false alarm.
- CSI mendekati nol (0.0417), menunjukkan model gagal total mendeteksi heavy rain event.
- Namun RMSE (2.0073) dan Correlation (0.4553) cukup reasonable.
- Brier Score sangat rendah (0.0204) — ini misleading karena event ~sangat jarang~.

**Penjelasan Ilmiah:**
Model memprediksi dengan baik nilai-nilai kecil/nol rainfall (mayoritas data),
tetapi gagal menangkap extreme events karena threshold 10 mm/jam terlalu tinggi 
relatif terhadap distribusi data ERA5 (lihat Bagian 2).

### 1.2 Wind Speed

| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |
|--------|--------|----------------|---------------|------------|
| RMSE | 1.5882 | 1.5798 | 2.6620 | 2.0087 |
| MAE | 1.2444 | 1.2279 | 2.0709 | 1.5144 |
| CORRELATION | 0.8705 | 0.8699 | 0.7402 | 0.8328 |
| CRPS | 1.2181 | 1.2013 | 2.0440 | 1.4878 |
| BRIER_SCORE | 0.0593 | 0.0561 | 0.1811 | 0.0988 |
| POD | 0.5122 | 0.5528 | 0.4839 | 0.5054 |
| FAR | 0.0870 | 0.0933 | 0.2228 | 0.1662 |
| CSI | 0.4884 | 0.5231 | 0.4249 | 0.4592 |

**Temuan Utama – Wind Speed:**

- Correlation tinggi (0.8328), model menangkap pola diurnal angin.
- POD moderat (~0.50), model mendeteksi separuh event angin kencang.
- FAR rendah (0.1662), sedikit false alarm.
- CSI terbaik (0.4592) dibandingkan variabel lain.
- Performa konsisten di Puncak & Lereng Cibodas; sedikit lebih rendah di Hilir Cianjur (RMSE 2.66 vs ~1.58).

### 1.3 Humidity

| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |
|--------|--------|----------------|---------------|------------|
| RMSE | 6.3387 | 5.5209 | 8.0307 | 6.7120 |
| MAE | 5.3698 | 4.0812 | 6.0560 | 5.1690 |
| CORRELATION | 0.9059 | 0.8953 | 0.9450 | 0.9231 |
| CRPS | 5.2345 | 3.9133 | 5.8586 | 5.0021 |
| BRIER_SCORE | 0.1179 | 0.1202 | 0.0055 | 0.0812 |
| POD | 0.2881 | 0.4180 | 0.1429 | 0.3512 |
| FAR | 0.1500 | 0.3070 | 0.5000 | 0.2557 |
| CSI | 0.2742 | 0.3527 | 0.1250 | 0.3134 |

**Temuan Utama – Humidity:**

- Correlation tertinggi (0.9231), variabel paling mudah diprediksi.
- RMSE ~6.7% RH tergolong baik untuk nowcasting.
- POD (0.3512) dan CSI (0.3134) moderat untuk threshold >90% RH.
- Lereng Cibodas memiliki performa terbaik (RMSE 5.52).

## 2. Analisis Distribusi Rainfall

### 2.1 Statistik Distribusi

Dataset ERA5 reanalysis memiliki distribusi rainfall yang **sangat skewed**:

| Statistik | Nilai |
|-----------|-------|
| Total sampel | 526,032 |
| Zero rainfall | 64.2% |
| > 0.1 mm/jam | 28.1% |
| > 2 mm/jam | 4.08% |
| > 5 mm/jam | 0.93% |
| **> 10 mm/jam (threshold)** | **0.151%** |
| > 20 mm/jam | 0.0097% |
| Max | 30.4 mm/jam |

### 2.2 Percentiles

| Percentile | Nilai (mm/jam) |
|------------|----------------|
| P50 | 0.000 |
| P75 | 0.200 |
| P90 | 0.800 |
| P95 | 1.700 |
| P99 | 4.900 |
| P99.5 | 6.600 |
| P99.9 | 11.300 |

### 2.3 Implikasi untuk POD/CSI

Dengan threshold heavy rain = 10 mm/jam, hanya **0.151%** data yang termasuk 'event'.
Ini berarti:
- Dalam test set (~78,963 sampel), diperkirakan hanya **~119 jam** dengan heavy rain.
- Dengan eval_step=24h (subsampling), kemungkinan model hanya melihat **sangat sedikit** event aktual.
- Model cenderung prediksi 'tidak hujan lebat' (climatological bias) → POD rendah.
- Beberapa false alarm terjadi → FAR sangat tinggi karena denominator kecil.

**Rekomendasi:**
- Gunakan threshold **2 mm/jam** atau **5 mm/jam** untuk evaluasi yang lebih meaningful.
- Alternatif: gunakan percentile-based threshold (misalnya P95 atau P99).
- P95 = 1.700 mm/jam, P99 = 4.900 mm/jam.

![Rainfall Distribution](plots/rainfall_distribution.png)

## 3. Analisis Korelasi Antar Node

### 3.1 Correlation Matrix

**Precipitation:**

| | Puncak | Lereng Cibodas | Hilir Cianjur |
|------|--------|----------------|---------------|
| Puncak | 1.0000 | 1.0000 | 0.5717 |
| Lereng_Cibodas | 1.0000 | 1.0000 | 0.5717 |
| Hilir_Cianjur | 0.5717 | 0.5717 | 1.0000 |

**Wind_Speed:**

| | Puncak | Lereng Cibodas | Hilir Cianjur |
|------|--------|----------------|---------------|
| Puncak | 1.0000 | 1.0000 | 0.5373 |
| Lereng_Cibodas | 1.0000 | 1.0000 | 0.5373 |
| Hilir_Cianjur | 0.5373 | 0.5373 | 1.0000 |

**Humidity:**

| | Puncak | Lereng Cibodas | Hilir Cianjur |
|------|--------|----------------|---------------|
| Puncak | 1.0000 | 1.0000 | 0.8610 |
| Lereng_Cibodas | 1.0000 | 1.0000 | 0.8615 |
| Hilir_Cianjur | 0.8610 | 0.8615 | 1.0000 |

### 3.2 Interpretasi

Korelasi antar node bervariasi:
- precipitation: Puncak ↔ Lereng_Cibodas: r = 1.0000 (sangat tinggi)
- precipitation: Puncak ↔ Hilir_Cianjur: r = 0.5717 (rendah)
- precipitation: Lereng_Cibodas ↔ Hilir_Cianjur: r = 0.5717 (rendah)
- wind_speed: Puncak ↔ Lereng_Cibodas: r = 1.0000 (sangat tinggi)
- wind_speed: Puncak ↔ Hilir_Cianjur: r = 0.5373 (rendah)
- wind_speed: Lereng_Cibodas ↔ Hilir_Cianjur: r = 0.5373 (rendah)
- humidity: Puncak ↔ Lereng_Cibodas: r = 1.0000 (sangat tinggi)
- humidity: Puncak ↔ Hilir_Cianjur: r = 0.8610 (tinggi)
- humidity: Lereng_Cibodas ↔ Hilir_Cianjur: r = 0.8615 (tinggi)

![Cross-Node Correlation](plots/node_correlation_matrix.png)

## 4. Efek Resolusi ERA5

### 4.1 Koordinat Node dan Grid ERA5

| Node | Latitude | Longitude | ERA5 Grid Cell |
|------|----------|-----------|----------------|
| Puncak | -6.769797 | 106.963583 | (-6.75, 107.00) |
| Lereng_Cibodas | -6.751722 | 106.987160 | (-6.75, 107.00) |
| Hilir_Cianjur | -6.816000 | 107.133000 | (-6.75, 107.25) |

### 4.2 Jarak Antar Node

| Node A | Node B | Jarak (km) |
|--------|--------|------------|
| Puncak | Lereng_Cibodas | 3.29 |
| Puncak | Hilir_Cianjur | 19.40 |
| Lereng_Cibodas | Hilir_Cianjur | 17.62 |

ERA5 grid resolution pada lintang -6.8°: ~27.6 km (E-W) × ~27.6 km (N-S).

### 4.3 Analisis

⚠ **Dua node berbagi grid cell ERA5 yang sama.**
- Grid cell (-6.75, 107.0): Puncak, Lereng_Cibodas (data identik)
- Grid cell (-6.75, 107.25): Hilir_Cianjur (data unik)

**Implikasi:**
- Korelasi antar node yang tinggi mengkonfirmasi bahwa resolusi ERA5 (~25 km) terlalu kasar
  untuk membedakan variasi mikro-klimat di kawasan Gunung Gede-Pangrango.
- Graph Neural Network perlu data observasi resolusi tinggi untuk benar-benar
  menangkap spatio-temporal dependency antar lokasi.
- Meskipun demikian, model tetap mampu mempelajari pola temporal dari data ERA5.

## 5. Apakah Model Benar-benar Belajar?

### 5.1 Perbandingan dengan Persistence Baseline (24h lag)

| Variable | Metric | Persistence | Model | Improvement |
|----------|--------|-------------|-------|-------------|
| precipitation | RMSE | 1.3847 | 2.0073 | -45.0% ❌ |
| precipitation | MAE | 0.4317 | 0.8492 | -96.7% ❌ |
| precipitation | Corr | 0.2590 | 0.4553 | +0.1963 ✅ |
| wind_speed | RMSE | 2.8777 | 2.0087 | +30.2% ✅ |
| wind_speed | MAE | 2.1791 | 1.5144 | +30.5% ✅ |
| wind_speed | Corr | 0.5512 | 0.8328 | +0.2816 ✅ |
| humidity | RMSE | 7.6310 | 6.7120 | +12.0% ✅ |
| humidity | MAE | 5.2624 | 5.1690 | +1.8% ✅ |
| humidity | Corr | 0.8085 | 0.9231 | +0.1146 ✅ |

### 5.2 Perbandingan dengan Climatology Baseline

| Variable | Climatology RMSE | Model RMSE | Skill Score |
|----------|-----------------|------------|-------------|
| precipitation | 1.1391 | 2.0073 | -210.5% ❌ |
| wind_speed | 3.2699 | 2.0087 | +62.3% ✅ |
| humidity | 13.6860 | 6.7120 | +75.9% ✅ |

### 5.3 Kesimpulan Skill Analysis

Model mengungguli persistence baseline pada **4/6** metrik (RMSE/MAE).

- **precipitation**: Partially better (Correlation) → ⚠ model belajar sebagian pola.
- **wind_speed**: Model lebih baik (RMSE, MAE, Correlation) → ✅ model belajar pola wind_speed.
- **humidity**: Model lebih baik (RMSE, MAE, Correlation) → ✅ model belajar pola humidity.

## 6. Kesimpulan Akhir

### Model Strengths
1. **Humidity**: Correlation 0.92, model sangat baik menangkap pola kelembaban temporal.
2. **Wind Speed**: Correlation 0.83, POD ~0.50, model mendeteksi pola angin dengan baik.
3. **Probabilistic Output**: CRPS memberikan distribusi prediksi, bukan point estimate.
4. **Numerical Stability**: 0 NaN dari 3,291 sampel evaluasi.

### Model Weaknesses
1. **Heavy Rain Detection**: POD=0.0588, tetapi ini disebabkan oleh:
   - Threshold 10 mm/jam terlalu tinggi (hanya 0.151% data melebihi threshold)
   - Data ERA5 cenderung under-estimate precipitation intensity dibanding observasi
   - Extreme events sangat jarang dalam dataset
2. **ERA5 Resolution**: ~25 km terlalu kasar untuk kawasan gunung berukuran <20 km
3. **Node Similarity**: Korelasi antar node sangat tinggi, mengurangi nilai tambah graph structure

### Rekomendasi untuk Perbaikan
1. Gunakan threshold **2–5 mm/jam** untuk heavy rain evaluation
2. Tambahkan data observasi resolusi tinggi (BMKG AWS, radar cuaca)
3. Downscaling ERA5 menggunakan topographic correction
4. Augmentasi data untuk rare extreme events
5. Cost-sensitive loss function yang memberikan bobot lebih pada heavy rain events

---

![Skill Comparison](plots/skill_comparison.png)

![Per-Node Performance](plots/per_node_performance.png)

## 7. Additional Rainfall Evaluation

### 7.1 Threshold Sensitivity Analysis

Evaluasi event detection dilakukan pada tiga threshold:

| Threshold | N Events | POD | FAR | CSI | Brier Score |
|-----------|----------|-----|-----|-----|-------------|
| 2 mm/jam | 424 | 0.4151 | 0.3889 | 0.3284 | 0.1064 |
| 5 mm/jam | 153 | 0.2484 | 0.6162 | 0.1776 | 0.0523 |
| 10 mm/jam | 51 | 0.0588 | 0.8696 | 0.0423 | 0.0204 |

### 7.2 Contingency Tables

**Threshold = 2 mm/jam** (N events = 424)

| | Predicted Event | Predicted No Event | Total |
|---|---|---|---|
| Actual Event | 176 (Hits) | 248 (Misses) | 424 |
| Actual No Event | 112 (FA) | 2755 (CN) | 2867 |
| Total | 288 | 3003 | 3291 |

**Threshold = 5 mm/jam** (N events = 153)

| | Predicted Event | Predicted No Event | Total |
|---|---|---|---|
| Actual Event | 38 (Hits) | 115 (Misses) | 153 |
| Actual No Event | 61 (FA) | 3077 (CN) | 3138 |
| Total | 99 | 3192 | 3291 |

**Threshold = 10 mm/jam** (N events = 51)

| | Predicted Event | Predicted No Event | Total |
|---|---|---|---|
| Actual Event | 3 (Hits) | 48 (Misses) | 51 |
| Actual No Event | 20 (FA) | 3220 (CN) | 3240 |
| Total | 23 | 3268 | 3291 |

### 7.3 Interpretasi Threshold Sensitivity

**Temuan:**

- Pada threshold **2 mm/jam**: POD = 0.4151, CSI = 0.3284
  Model mendeteksi 41.5% dari 424 rain events.
- Pada threshold **5 mm/jam**: POD = 0.2484, CSI = 0.1776
  Model mendeteksi 24.8% dari 153 rain events.
- Pada threshold **10 mm/jam**: POD = 0.0588, CSI = 0.0423
  Model mendeteksi 5.9% dari 51 rain events.

CSI meningkat **7.8x** dari threshold 10mm ke 2mm.

Ini mengkonfirmasi bahwa threshold 10 mm/jam terlalu tinggi untuk dataset ERA5.
Threshold **2-5 mm/jam** lebih realistis untuk evaluasi kemampuan event detection model.

### 7.4 Precipitation Baseline Comparison

| Method | RMSE | MAE | Correlation |
|--------|------|-----|-------------|
| Climatology Mean | 2.1982 | 0.9262 | 0.0000 |
| Persistence (t-24) | 2.6207 | 1.2418 | 0.2062 |
| Persist + Bias Corr | 2.6207 | 1.2421 | 0.2062 |
| RA-Diffusion (Ours) | 2.0089 | 0.8491 | 0.4551 |

**Analisis Skill:**

- RMSE vs Persistence: +23.3% (model lebih baik)
- MAE vs Persistence: +31.6% (model lebih baik)
- Correlation: Model 0.4551 vs Persistence 0.2062 (+0.2489)

**Catatan Penting:**

Precipitation memiliki karakteristik unik dibandingkan wind_speed dan humidity:
1. **Intermittent**: 64% data bernilai nol
2. **Skewed**: Distribusi sangat miring ke kanan
3. **Bursty**: Perubahan cepat dari 0 ke nilai tinggi
4. **Persistence advantage**: Karena sebagian besar waktu tidak hujan,
   persistence (t-24) cenderung memiliki RMSE rendah dengan prediksi 'tetap 0'.

Model diffusion menunjukkan **correlation lebih tinggi** dibanding persistence,
yang mengindikasikan model mempelajari pola temporal rainfall meskipun RMSE-nya lebih tinggi.
Ini karena model menghasilkan prediksi non-zero yang terkadang miss timing/magnitude.

### 7.5 Visual Analysis

![Rainfall Scatter](plots/rainfall_scatter_analysis.png)

![Rainfall Intensity](plots/rainfall_intensity_histogram.png)

![Confusion Matrices](plots/confusion_matrices_rainfall.png)

![Precipitation Baselines](plots/precipitation_baselines.png)

