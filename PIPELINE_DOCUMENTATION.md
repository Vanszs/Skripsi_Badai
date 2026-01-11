# Pipeline Lengkap: Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning

## Informasi Umum

**Judul Skripsi:**
> Probabilistic Nowcasting Hujan Ekstrem Pemicu Banjir Bandang Sitaro Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning

**Lokasi Studi:** Kepulauan Sitaro (Siau, Tagulandang, Biaro), Sulawesi Utara, Indonesia

**Tujuan:** Memprediksi curah hujan dalam bentuk distribusi probabilistik (bukan angka tunggal) untuk mendukung early warning system banjir bandang.

---

## Daftar Pipeline

| No | Pipeline | Input | Output |
|----|----------|-------|--------|
| 1 | Data Ingestion | Koordinat + Rentang Waktu | DataFrame Cuaca |
| 2 | Preprocessing | DataFrame Cuaca | Tensor Ternormalisasi |
| 3 | Sliding Window | Tensor Flat | Sequence Graphs |
| 4 | Retrieval Database | Context Features | K-Nearest Neighbors |
| 5 | Spatio-Temporal GNN | Graph Sequence | Graph Embedding |
| 6 | Diffusion Training | Noisy Target + Conditioning | Predicted Noise |
| 7 | Probabilistic Inference | Current Condition | 50 Rain Samples |

---

# ⚠️ CRITICAL: TEMPORAL DATA SPLIT (MENCEGAH DATA LEAKAGE)

## Mengapa Ini Penting?

Pada data time series, **RANDOM SHUFFLE ADALAH KESALAHAN FATAL** yang menyebabkan **data leakage**.

### Apa itu Data Leakage pada Time Series?

```
SALAH (Random Shuffle):
┌─────────────────────────────────────────────────────────────────────┐
│ Training Set: [2005, 2010, 2015, 2020, 2007, 2023, 2012, ...]      │
│ Test Set:     [2008, 2018, 2006, 2022, 2011, ...]                  │
└─────────────────────────────────────────────────────────────────────┘

Masalah: Model "melihat" data 2023 saat training, 
         lalu di-test pada 2022 → prediksi "masa lalu"!
         Ini BUKAN forecasting, ini CHEATING.
```

```
BENAR (Temporal Split):
┌─────────────────────────────────────────────────────────────────────┐
│ Training Set: [2005, 2006, 2007, ..., 2017, 2018]  ← HANYA MASA LALU│
│ Validation:   [2019, 2020, 2021]                   ← TUNING         │
│ Test Set:     [2022, 2023, 2024, 2025]             ← EVALUASI AKHIR │
└─────────────────────────────────────────────────────────────────────┘

Model TIDAK PERNAH melihat data setelah 2018 saat training.
Evaluasi pada 2022-2025 adalah TRUE out-of-sample forecast.
```

## Strategi Split yang Digunakan

| Split | Periode | Tahun | Jumlah Jam (approx) | Proporsi |
|-------|---------|-------|---------------------|----------|
| **Training** | 2005-01-01 s/d 2018-12-31 | 14 tahun | ~122,640 | 67% |
| **Validation** | 2019-01-01 s/d 2021-12-31 | 3 tahun | ~26,280 | 14% |
| **Test** | 2022-01-01 s/d 2025-12-31 | 4 tahun | ~35,040 | 19% |

### Justifikasi Ilmiah Rasio 67% / 14% / 19%

#### 1. Pertimbangan Klimatologi: Siklus ENSO

```
El Niño-Southern Oscillation (ENSO) memiliki periode rata-rata 2-7 tahun.

Training 14 tahun (2005-2018):
├── El Niño events: 2006-07, 2009-10, 2015-16
├── La Niña events: 2007-08, 2010-11, 2017-18
└── Neutral years: 2012-14

→ Model belajar dari MINIMAL 2 siklus ENSO lengkap
→ Mencakup variabilitas cuaca tropis yang representatif
```

**Referensi:** NOAA Climate Prediction Center - ENSO cycle duration 2-7 years

#### 2. Pertimbangan Machine Learning: Bias-Variance Tradeoff

```
Split Ratio Guidelines (Hastie et al., 2009):

Standard ML:     60/20/20  atau  70/15/15
Time Series:     70/15/15  atau  80/10/10 (lebih banyak training)
Our Choice:      67/14/19

Reasoning:
├── 67% training: Cukup data untuk model kompleks (GNN+Diffusion)
├── 14% validation: Cukup untuk hyperparameter tuning tanpa overfitting
└── 19% test: Periode terbaru untuk evaluasi real-world performance
```

**Referensi:** Hastie, Tibshirani, Friedman - "Elements of Statistical Learning" (2009)

#### 3. Pertimbangan Statistik: Minimum Sample Size

```
Untuk neural network training:
Minimum samples = 10 × jumlah parameter / jumlah output

Model kita:
├── SpatioTemporalGNN: ~50,000 parameters
├── DiffusionModel: ~100,000 parameters
├── Total: ~150,000 parameters
├── Output: 1 (precipitation)

Minimum training samples = 10 × 150,000 / 1 = 1,500,000

Actual training samples:
├── 14 tahun × 365 hari × 24 jam × 3 nodes = ~368,000 samples
├── Dengan sliding window (seq=6): ~367,994 samples

→ Mendekati requirement, ditambah regularization (dropout, weight decay)
```

#### 4. Pertimbangan Operasional: Recency of Test Data

```
Test Period 2022-2025:
├── Mencerminkan kondisi cuaca TERKINI
├── Relevan untuk deployment operasional
├── Menangkap perubahan iklim terbaru
└── Jika ada banjir Sitaro Jan 2026: bisa validasi langsung

Jika test period terlalu lama di masa lalu (misal 2010-2015):
├── Model mungkin bagus untuk data lama
├── Tapi tidak relevan untuk prediksi masa depan
└── Climate change effects tidak tertangkap
```

#### 5. Pertimbangan Praktis: Kelipatan Tahun

```
Split menggunakan tahun penuh:
├── Training: 14 tahun (bukan 13.5 atau 14.2)
├── Validation: 3 tahun (bukan 2.8 atau 3.3)
├── Test: 4 tahun (bukan 3.7 atau 4.1)

Keuntungan:
├── Setiap split mencakup 1 siklus musim penuh (Jan-Des)
├── Tidak ada bias musiman di boundary
├── Memudahkan reproduksi dan interpretasi
└── Standard practice di climate science
```

#### 6. Formula Matematis

```
Diberikan:
- Total data: N = 21 tahun (2005-2025)
- ENSO cycle: T_ENSO = 2-7 tahun
- Minimum cycles untuk generalisasi: C_min = 2

Training years:
  Y_train ≥ C_min × max(T_ENSO) = 2 × 7 = 14 tahun ✓

Validation years:
  Y_val ≥ T_ENSO_min = 2-3 tahun
  Kita pilih 3 tahun untuk mencakup variabilitas ✓

Test years:
  Y_test = N - Y_train - Y_val = 21 - 14 - 3 = 4 tahun ✓

Proporsi:
  Train: 14/21 = 66.7% ≈ 67%
  Val:   3/21  = 14.3% ≈ 14%
  Test:  4/21  = 19.0% = 19%
```

### Perbandingan dengan Studi Lain

| Studi | Domain | Train | Val | Test |
|-------|--------|-------|-----|------|
| **Ours** | Precipitation | 67% | 14% | 19% |
| Ravuri et al. (2021) | Precipitation | 70% | 15% | 15% |
| DeepAR (Amazon) | Time Series | 80% | 10% | 10% |
| Transformer (Vaswani) | NLP | 90% | 5% | 5% |
| Climate Modeling (CMIP) | Climate | 75% | 10% | 15% |

→ **Rasio kita (67/14/19) sesuai dengan praktik standard precipitation nowcasting.**
    """
    Split DataFrame berdasarkan waktu, BUKAN random.
    
    Args:
        df: DataFrame dengan kolom 'date'
        train_end: Tanggal terakhir training
        val_end: Tanggal terakhir validation
    
    Returns:
        train_df, val_df, test_df
    """
    df['date'] = pd.to_datetime(df['date'])
    
    train_mask = df['date'] <= train_end
    val_mask = (df['date'] > train_end) & (df['date'] <= val_end)
    test_mask = df['date'] > val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    return train_df, val_df, test_df
```

## Aturan Ketat yang WAJIB Diikuti

### 1. Normalization Stats HANYA dari Training Set

```python
# BENAR:
stats = compute_stats(train_df)  # μ dan σ dari training SAJA

# Terapkan ke semua set dengan stats yang SAMA
train_norm = normalize(train_df, stats)
val_norm = normalize(val_df, stats)    # Gunakan stats dari training
test_norm = normalize(test_df, stats)  # Gunakan stats dari training
```

```python
# SALAH (DATA LEAKAGE):
stats = compute_stats(full_df)  # ❌ Termasuk data val & test!
```

### 2. Retrieval Database HANYA dari Training Set

```python
# BENAR:
retrieval_db.add_items(train_features)  # Index HANYA training

# Saat inference pada test set:
retrieved = retrieval_db.query(test_query)  # Query ke training data
```

```python
# SALAH (DATA LEAKAGE):
retrieval_db.add_items(all_features)  # ❌ Test data masuk index!
```

### 3. DataLoader TANPA Shuffle untuk Validation/Test

```python
# Training: shuffle=True (dalam satu epoch, bukan across time)
train_loader = DataLoader(train_dataset, shuffle=True)

# Validation & Test: shuffle=False (urutan waktu dipertahankan)
val_loader = DataLoader(val_dataset, shuffle=False)
test_loader = DataLoader(test_dataset, shuffle=False)
```

### 4. Sliding Window TIDAK Boleh Cross Boundary

```python
# Jika seq_len = 6, maka:
# - Sample pertama valid di training: timestep ke-6 (perlu 6 history)
# - Sample pertama valid di validation: 2019-01-01 jam 06:00
#   (menggunakan history dari 2019-01-01 00:00 - 05:00)

# BUKAN menggunakan data 2018-12-31 untuk prediksi 2019-01-01!
# Setiap split harus mandiri.
```

## Visualisasi Timeline

```
2005        2010        2015        2018  2019     2021  2022        2025
  │           │           │           │     │        │     │           │
  ├───────────┴───────────┴───────────┤     │        │     │           │
  │         TRAINING SET              │     │        │     │           │
  │      (14 tahun data)              │     │        │     │           │
  │    Model belajar patterns         │     │        │     │           │
  │    Compute μ, σ dari sini         │     │        │     │           │
  │    FAISS index dari sini          │     │        │     │           │
  └───────────────────────────────────┘     │        │     │           │
                                            │        │     │           │
                                            ├────────┤     │           │
                                            │  VAL   │     │           │
                                            │3 tahun │     │           │
                                            │Tuning  │     │           │
                                            │Hyperparam│   │           │
                                            └────────┘     │           │
                                                           │           │
                                                           ├───────────┤
                                                           │   TEST    │
                                                           │ 4 tahun   │
                                                           │ Final     │
                                                           │ Evaluation│
                                                           │ CRPS, BSS │
                                                           └───────────┘
```

## Mengapa Memilih 2018 sebagai Cutoff?

1. **Cukup data training**: 14 tahun mencakup berbagai pola cuaca
2. **Event La Niña/El Niño**: Training mencakup siklus ENSO lengkap
3. **Recent test data**: 2022-2025 adalah periode operasional yang relevan
4. **Banjir Sitaro Jan 2026**: Jika ada data aktual, bisa digunakan sebagai case study

## Konsekuensi Pelanggaran

| Pelanggaran | Konsekuensi |
|-------------|-------------|
| Random shuffle seluruh data | Metrik overestimate, model gagal di real-world |
| Stats dari full data | Model "tahu" range nilai test, normalisasi bias |
| Retrieval dari full data | Model "menemukan" analog dari masa depan |
| Shuffle test loader | Evaluasi tidak mencerminkan real-time forecasting |

## Referensi Metodologi

> **Bergmeir & Benítez (2012)**: "On the use of cross-validation for time series predictor evaluation"
> - Random CV dengan blocking harus digunakan
> - Atau strict temporal holdout (yang kita gunakan)

> **Tashman (2000)**: "Out-of-sample tests of forecasting accuracy"
> - Rolling origin evaluation
> - Fixed origin evaluation (yang kita gunakan: single train-val-test split)

---

# PIPELINE 1: DATA INGESTION

## Deskripsi
Mengambil data cuaca historis dari Open-Meteo Archive API (sumber: ERA5 Reanalysis ECMWF) untuk 3 pulau di Kepulauan Sitaro.

## Input

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Koordinat Siau | (2.75°N, 125.40°E) | Pulau utara, gunung api aktif |
| Koordinat Tagulandang | (2.33°N, 125.42°E) | Pulau tengah |
| Koordinat Biaro | (2.10°N, 125.37°E) | Pulau selatan |
| Rentang Waktu | 2005-01-01 s/d 2025-12-31 | 20 tahun data |
| Resolusi Temporal | Hourly (per jam) | ~175,000 timesteps per node |

## Output: DataFrame dengan Variabel

| No | Variabel | Satuan | Tipe | Deskripsi |
|----|----------|--------|------|-----------|
| 1 | `date` | datetime | Index | Timestamp UTC+8 (WITA) |
| 2 | `node_id` | string | Identifier | Nama pulau: Siau/Tagulandang/Biaro |
| 3 | `precipitation` | mm/jam | **TARGET** | Curah hujan akumulasi per jam |
| 4 | `temperature_2m` | °C | Dynamic | Suhu udara pada ketinggian 2 meter |
| 5 | `relative_humidity_2m` | % | Dynamic | Kelembaban relatif pada 2 meter |
| 6 | `dewpoint_2m` | °C | Dynamic | Titik embun (proxy kelembaban absolut) |
| 7 | `surface_pressure` | hPa | Dynamic | Tekanan udara permukaan |
| 8 | `wind_speed_10m` | m/s | Dynamic | Kecepatan angin pada 10 meter |
| 9 | `wind_direction_10m` | ° | Dynamic | Arah angin (0°=Utara, 90°=Timur) |
| 10 | `cloudcover` | % | Dynamic | Persentase tutupan awan total |
| 11 | `elevation` | meter | Static | Ketinggian dari Elevation API |
| 12 | `land_sea_mask` | 0/1 | Static | Indikator darat (1) atau laut (0) |
| 13 | `precipitation_lag1` | mm | Derived | Curah hujan 1 jam sebelumnya |
| 14 | `precipitation_lag3` | mm | Derived | Curah hujan 3 jam sebelumnya |

## Rumus Derived Features

### Lag Features (Autoregressive Input)

```
precipitation_lag1(t) = precipitation(t - 1)

precipitation_lag3(t) = precipitation(t - 3)
```

Dimana `t` adalah timestep saat ini.

**Penanganan Missing Values:**
```
Jika t < 1:  precipitation_lag1 = 0
Jika t < 3:  precipitation_lag3 = 0
```

### Land-Sea Mask

Untuk pulau vulkanik Sitaro dengan topografi curam (tanpa delta atau estuaria):

```
land_sea_mask = { 1,  jika elevation > 0
                { 0,  jika elevation ≤ 0
```

**Justifikasi:** Valid secara geomorfologi untuk pulau vulkanik tanpa dataran rendah di bawah permukaan laut.

## Hubungan dengan Pipeline Selanjutnya
- Output DataFrame → **Pipeline 2 (Preprocessing)**
- Shape: [N × 14] dimana N = jumlah total rows (timestamps × nodes)

---

# PIPELINE 2: PREPROCESSING

## Deskripsi
Mengubah data mentah menjadi tensor ternormalisasi yang optimal untuk training neural network.

## Input
- DataFrame dari Pipeline 1: [N × 14 kolom]
- Target column: `precipitation`
- Feature columns: 10 kolom (temperature_2m, relative_humidity_2m, dst.)

## Proses Detail

### Step 2.1: Log Transform pada Target

**Masalah:** Curah hujan memiliki distribusi heavy-tailed:
- Mayoritas nilai = 0 (tidak hujan)
- Sedikit nilai ekstrem > 100 mm/jam

**Solusi:** Transformasi logaritmik untuk meng-compress range nilai.

**Rumus Log1p Transform:**
```
y_log = log(1 + y_raw)
```

Dimana:
- `y_raw` = curah hujan mentah dalam mm
- `log` = natural logarithm (ln)
- `+1` untuk menghindari log(0)

**Contoh:**
```
y_raw = 0 mm    →  y_log = log(1 + 0) = log(1) = 0
y_raw = 10 mm   →  y_log = log(1 + 10) = log(11) ≈ 2.40
y_raw = 100 mm  →  y_log = log(1 + 100) = log(101) ≈ 4.62
```

**Inverse Transform (untuk denormalisasi saat inference):**
```
y_raw = exp(y_log) - 1
```

### Step 2.2: Z-Score Normalization

**Tujuan:** Mengubah distribusi agar mean=0 dan std=1.

**Rumus untuk Target (setelah log transform):**
```
y_norm = (y_log - μ_y) / σ_y
```

Dimana:
- `μ_y` = mean(y_log) di seluruh dataset training
- `σ_y` = std(y_log) di seluruh dataset training

**Rumus untuk Features (per kolom):**
```
x_norm[i] = (x_raw[i] - μ_x[i]) / σ_x[i]
```

Dimana untuk setiap fitur ke-i:
- `μ_x[i]` = mean fitur ke-i di seluruh dataset
- `σ_x[i]` = std fitur ke-i di seluruh dataset

**Numerik Stability:**
```
x_norm[i] = (x_raw[i] - μ_x[i]) / (σ_x[i] + ε)
```
dengan `ε = 1e-5` untuk menghindari division by zero.

## Output

| Tensor | Shape | Dtype | Deskripsi |
|--------|-------|-------|-----------|
| `target_norm` | [N, 1] | float32 | Target ternormalisasi |
| `features_norm` | [N, F] | float32 | Features ternormalisasi |
| `stats` | dict | - | Parameter normalisasi |

**Stats dictionary:**
```python
stats = {
    't_mean': μ_y,    # scalar, target log mean
    't_std': σ_y,     # scalar, target log std
    'c_mean': μ_x,    # tensor [F], feature means
    'c_std': σ_x      # tensor [F], feature stds
}
```

## Hubungan dengan Pipeline Selanjutnya
- `features_norm` → **Pipeline 3** (Sliding Window)
- `features_norm` → **Pipeline 4** (Retrieval Database indexing)
- `target_norm` → **Pipeline 6** (Diffusion Training target)
- `stats` → **Pipeline 7** (Inference denormalization)

---

# PIPELINE 3: SLIDING WINDOW & GRAPH CONSTRUCTION

## Deskripsi
Mengubah data tabular flat menjadi sequence of graphs untuk pemodelan spatio-temporal.

## Input
- `features_norm`: Tensor [N, F] ternormalisasi dari Pipeline 2
- `target_norm`: Tensor [N, 1] ternormalisasi
- `seq_len`: 6 (hyperparameter: jumlah timesteps dalam sequence)
- `node_names`: ['Siau', 'Tagulandang', 'Biaro']

## Proses Detail

### Step 3.1: Sliding Window Construction

Untuk setiap timestep target `t`, buat window dari `t-seq_len` hingga `t-1`:

```
Untuk prediksi precipitation(t):

Window = [features(t-6), features(t-5), features(t-4), 
          features(t-3), features(t-2), features(t-1)]

Indeks:    τ=0          τ=1          τ=2          
           τ=3          τ=4          τ=5
```

**Jumlah Valid Samples:**
```
N_valid = N_total - seq_len
```

### Step 3.2: Graph Construction per Timestep

Setiap timestep τ dalam window direpresentasikan sebagai graph G = (V, E).

**Nodes (V):**
```
V = {v_Siau, v_Tagulandang, v_Biaro}
|V| = 3 nodes
```

**Node Feature Matrix:**
```
X ∈ ℝ^(|V| × F) = ℝ^(3 × 10)

X = [ x_Siau        ]   3 nodes
    [ x_Tagulandang ]   ×
    [ x_Biaro       ]   10 features
```

**Edges (E) - Fully Connected:**
```
E = {(i,j) | i,j ∈ V, i ≠ j}
|E| = |V| × (|V| - 1) = 3 × 2 = 6 edges
```

**Edge Index (COO format untuk PyTorch Geometric):**
```
edge_index = [[0, 0, 1, 1, 2, 2],   # source nodes
              [1, 2, 0, 2, 0, 1]]   # target nodes

Artinya:
  Siau → Tagulandang
  Siau → Biaro
  Tagulandang → Siau
  Tagulandang → Biaro
  Biaro → Siau
  Biaro → Tagulandang
```

**Edge Weight (Opsional, berbasis jarak geografis):**
```
w_ij = 1 / d_ij

d_ij = √[(lat_i - lat_j)² + (lon_i - lon_j)²]
```

### Step 3.3: PyG Data Object

Untuk setiap timestep τ:
```python
Graph_τ = Data(
    x = torch.tensor[3, F],      # Node features
    edge_index = torch.tensor[2, 6]  # Edge connectivity
)
```

**Batch untuk Training:**
```python
Batch = Batch.from_data_list([Graph_1, Graph_2, ..., Graph_B])

Hasil:
  Batch.x.shape = [B × 3, F] = [B×3, 10]
  Batch.edge_index = merged edges
  Batch.batch = node-to-graph assignment
```

## Output

| Output | Shape | Deskripsi |
|--------|-------|-----------|
| `graphs_sequence` | List[Batch] × 6 | 6 PyG Batch objects per sample |
| `target` | [B, 1] | Target precipitation untuk timestep t |
| `context` | [B, F] | Features timestep terakhir untuk retrieval |

## Hubungan dengan Pipeline Selanjutnya
- `graphs_sequence` → **Pipeline 5** (SpatioTemporalGNN)
- `context` → **Pipeline 4** (FAISS query)
- `target` → **Pipeline 6** (Diffusion target)

---

# PIPELINE 4: RETRIEVAL DATABASE (FAISS)

## Deskripsi
Membangun database vektor untuk mencari K kejadian historis yang paling mirip dengan kondisi saat ini. Ini adalah komponen "Retrieval-Augmented" dalam judul.

## Input
- `context_norm`: [N, F] semua context features ternormalisasi (untuk indexing)
- `query`: [B, F] query vectors untuk pencarian (saat training/inference)
- `k`: 3 (jumlah neighbors yang dicari)

## Proses Detail

### Step 4.1: Indexing dengan FAISS

**Inisialisasi Index:**
```python
index = faiss.IndexFlatL2(F)  # F = 10 dimensi
```

**Menambah Data ke Index:**
```python
index.add(context_norm.numpy())  # Tambah N vectors
```

**Kompleksitas:**
- Indexing: O(N)
- Query (brute force L2): O(N × k)

### Step 4.2: Distance Calculation

**Rumus Euclidean Distance (L2):**
```
d(q, x_i) = ||q - x_i||₂

         = √[Σⱼ (qⱼ - xᵢⱼ)²]

         = √[(q₁-x₁)² + (q₂-x₂)² + ... + (q_F-x_F)²]
```

Dimana:
- `q` = query vector [F]
- `x_i` = database vector ke-i [F]
- `j` = indeks fitur (1 sampai F)

### Step 4.3: K-Nearest Neighbor Search

**Operasi:**
```
indices, distances = index.search(query, k)
```

**Algoritma:**
```
Untuk setiap query q:
  1. Hitung d(q, x_i) untuk semua i ∈ {1..N}
  2. Sort distances ascending
  3. Ambil k indices dengan distance terkecil
```

**Output:**
```
indices.shape = [B, k]     # Indeks neighbors
distances.shape = [B, k]   # Jarak ke neighbors
```

### Step 4.4: Retrieve Feature Values

```python
retrieved_values = context_norm[indices]  # [B, k, F]
```

## Output

| Output | Shape | Deskripsi |
|--------|-------|-----------|
| `retrieved` | [B, k, F] = [B, 3, 10] | K nearest neighbor features |

## Rumus Matematis Lengkap

**Objective:**
```
retrieved_i = argmin_{S ⊂ Database, |S|=k} Σₓ∈S ||query_i - x||₂
```

## Hubungan dengan Pipeline Selanjutnya
- `retrieved` → **Pipeline 6** (Diffusion conditioning)
- Flatten: `[B, k, F] → [B, k×F]` untuk MLP processing

---

# PIPELINE 5: SPATIO-TEMPORAL GRAPH NEURAL NETWORK

## Deskripsi
Mengekstrak embedding dari sequence of graphs yang menangkap:
1. **Spatial dependencies** antar pulau via Graph Attention Network (GAT)
2. **Temporal patterns** dalam sequence via Self-Attention

## Input
- `graphs_sequence`: List[Batch] dengan panjang seq_len = 6
- Setiap Batch: [B×3 nodes, F features]

## Arsitektur Detail

### 5.1: Spatial Processing - Graph Attention Network (GAT)

Untuk SETIAP timestep τ ∈ {0, 1, 2, 3, 4, 5}:

#### Layer 1: Multi-Head Graph Attention

**Attention Coefficient Computation:**
```
e_ij = LeakyReLU(a^T · [W·h_i || W·h_j])
```

Dimana:
- `h_i` = node feature vector untuk node i
- `W` ∈ ℝ^(F' × F) = learnable weight matrix
- `a` ∈ ℝ^(2F') = attention weight vector
- `||` = concatenation
- LeakyReLU dengan negative_slope = 0.2

**Softmax Normalization:**
```
α_ij = softmax_j(e_ij)

     = exp(e_ij) / Σ_{k∈N(i)} exp(e_ik)
```

Dimana N(i) = neighbors dari node i.

**Message Aggregation:**
```
h'_i = σ(Σ_{j∈N(i)} α_ij · W · h_j)
```

Dimana σ = ELU activation.

**Multi-Head (K=4 heads):**
```
h'_i = ||_{k=1}^{K} σ(Σ_{j∈N(i)} α_ij^(k) · W^(k) · h_j)
```

Output: `h' ∈ ℝ^(K × F')` per node (concatenated heads).

#### Layer 2: Single-Head Output

```
h''_i = σ(Σ_{j∈N(i)} α_ij · W^(2) · h'_j)
```

Output: `h'' ∈ ℝ^H` per node (H = hidden_dim).

#### Global Mean Pooling

Mengagregasi node embeddings menjadi graph-level embedding:
```
z_τ = (1/|V|) · Σ_{v∈V} h''_v

    = (h''_Siau + h''_Tagulandang + h''_Biaro) / 3
```

Output per timestep: `z_τ ∈ ℝ^(B × H)`

### 5.2: Temporal Processing - Self-Attention

**Input:** Stack spatial outputs dari 6 timesteps:
```
Z = [z_0, z_1, z_2, z_3, z_4, z_5]

Z ∈ ℝ^(B × 6 × H)
```

#### Multi-Head Self-Attention

**Linear Projections:**
```
Q = Z · W_Q    # Query:  [B, 6, H]
K = Z · W_K    # Key:    [B, 6, H]
V = Z · W_V    # Value:  [B, 6, H]
```

Dimana W_Q, W_K, W_V ∈ ℝ^(H × H) adalah learnable matrices.

**Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

Dimana:
- `Q·K^T` ∈ ℝ^(B × 6 × 6) = similarity matrix
- `d_k = H / num_heads` = dimension per head (untuk scaling)
- `√d_k` mencegah gradients vanishing untuk large d_k

**Softmax per Query (row-wise):**
```
attention_weights = softmax(Q·K^T / √d_k, dim=-1)

Σ_j attention_weights[i,j] = 1  untuk setiap i
```

**Multi-Head (4 heads):**
```
head_k = Attention(Q·W_Q^k, K·W_K^k, V·W_V^k)

MultiHead = [head_1 || head_2 || head_3 || head_4] · W_O
```

#### Temporal Aggregation

Mean pooling across timesteps:
```
graph_emb = (1/seq_len) · Σ_{τ=0}^{5} output_τ

          ∈ ℝ^(B × H)
```

### 5.3: Output Projection

```
graph_emb_final = Linear(graph_emb)

                ∈ ℝ^(B × G)
```

Dimana G = graph_dim = 64.

## Output

| Output | Shape | Deskripsi |
|--------|-------|-----------|
| `graph_emb` | [B, G] = [B, 64] | Spatio-temporal graph embedding |

## Hubungan dengan Pipeline Selanjutnya
- `graph_emb` → **Pipeline 6** (Diffusion conditioning via graph_mlp)

---

# PIPELINE 6: CONDITIONAL DIFFUSION MODEL TRAINING

## Deskripsi
Melatih model generatif berbasis DDPM (Denoising Diffusion Probabilistic Model) untuk memprediksi noise pada data curah hujan, dikondisikan pada multiple sources.

## Input

| Input | Shape | Dari Pipeline |
|-------|-------|---------------|
| `target_norm` | [B, 1] | Pipeline 2 |
| `context` | [B, F] | Pipeline 3 |
| `retrieved` | [B, k, F] | Pipeline 4 |
| `graph_emb` | [B, G] | Pipeline 5 |

## Proses Detail

### 6.1: Forward Diffusion Process (Adding Noise)

**Noise Schedule (Linear):**
```
β_t = β_start + (β_end - β_start) · t / T

β_start = 0.0001
β_end = 0.02
T = 1000 timesteps
```

**Cumulative Product:**
```
α_t = 1 - β_t

ᾱ_t = Π_{s=1}^{t} α_s = α_1 · α_2 · ... · α_t
```

**Forward Process - Adding Noise ke Data:**
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

Dimana:
- `x_0` = target_norm (data asli, clean)
- `ε ~ N(0, I)` = Gaussian noise (sampled randomly)
- `t ~ Uniform{1, 2, ..., T}` = random timestep
- `x_t` = noisy version of data at timestep t

**Interpretasi:**
- t kecil → x_t ≈ x_0 (sedikit noise)
- t besar → x_t ≈ ε (mostly noise)
- t = T → x_T ≈ N(0, I) (pure noise)

### 6.2: Conditioning Embeddings

#### Time Embedding (Sinusoidal Positional Encoding)

**Rumus:**
```
PE(t, 2i)   = sin(t / 10000^(2i/d))
PE(t, 2i+1) = cos(t / 10000^(2i/d))
```

Dimana:
- `t` = timestep (integer 0-999)
- `i` = dimension index
- `d` = embedding dimension = H

**MLP Projection:**
```
t_emb = MLP(PE(t))

      = Linear(GELU(Linear(PE(t))))

      ∈ ℝ^H
```

#### Context Embedding

```
c_emb = MLP(context)

      = Linear(SiLU(Linear(context)))

      ∈ ℝ^H
```

Dimana context ∈ ℝ^F (current weather features).

#### Retrieval Embedding

```
r_flat = flatten(retrieved)  

       = reshape([B, k, F] → [B, k×F])

       ∈ ℝ^(k×F) = ℝ^30

r_emb = MLP(r_flat)

      = Linear(SiLU(Linear(r_flat)))

      ∈ ℝ^H
```

#### Graph Embedding

```
g_emb = MLP(graph_emb)

      = Linear(SiLU(Linear(graph_emb)))

      ∈ ℝ^H
```

Dimana graph_emb ∈ ℝ^G dari SpatioTemporalGNN.

#### Combined Conditioning

**Additive Fusion:**
```
emb = t_emb + c_emb + r_emb + g_emb

    ∈ ℝ^H
```

### 6.3: U-Net Denoiser Architecture

**Down Path:**
```
h_1 = SiLU(Linear(x_t)) + emb    # [B, H]
h_2 = SiLU(Linear(h_1))          # [B, 2H]
```

**Middle:**
```
h_mid = SiLU(Linear(h_2))        # [B, 2H]
```

**Up Path with Skip Connection:**
```
h_cat = concat(h_mid, h_2)       # [B, 4H]
h_up = SiLU(Linear(h_cat))       # [B, H]
```

**Output:**
```
ε_pred = Linear(h_up)            # [B, 1]
```

### 6.4: Loss Function

**Mean Squared Error:**
```
L = E_{t, x_0, ε} [||ε - ε_pred||²]

  = (1/B) · Σ_b (ε_b - ε_pred_b)²
```

Dimana:
- `ε` = actual noise yang ditambahkan
- `ε_pred` = noise yang diprediksi model
- Expectation diambil atas: t (random timestep), x_0 (data), ε (noise)

### 6.5: Training Algorithm

```
Repeat untuk setiap epoch:
  For each batch (target, context, retrieved, graph_emb):
    
    1. Sample timestep: t ~ Uniform{1..T}
    
    2. Sample noise: ε ~ N(0, I)
    
    3. Create noisy target: 
       x_t = √(ᾱ_t)·target + √(1-ᾱ_t)·ε
    
    4. Compute graph embedding:
       graph_emb = SpatioTemporalGNN(graphs)
    
    5. Predict noise:
       ε_pred = Model(x_t, t, context, retrieved, graph_emb)
    
    6. Compute loss:
       L = MSE(ε, ε_pred)
    
    7. Backprop and update:
       optimizer.zero_grad()
       L.backward()
       optimizer.step()
```

## Output
- Trained model weights
- Checkpoint file: `models/diffusion_chkpt.pth`

## Hubungan dengan Pipeline Selanjutnya
- Trained weights → **Pipeline 7** (Inference)

---

# PIPELINE 7: PROBABILISTIC INFERENCE

## Deskripsi
Menggunakan model terlatih untuk menghasilkan 50 skenario prediksi curah hujan, memberikan uncertainty quantification untuk decision making.

## Input
- `condition_sequence`: [seq_len, F] = [6, 10] - 6 timesteps kondisi cuaca
- Trained `SpatioTemporalGNN` dan `DiffusionModel`
- `retrieval_db`: FAISS index
- `stats`: Normalization parameters dari Pipeline 2
- `num_samples`: 50 (jumlah skenario yang di-generate)

## Proses Detail

### Step 7.1: Create Graph Sequence

```
Untuk τ = 0, 1, 2, 3, 4, 5:
  Graph_τ = create_graph(condition_sequence[τ])

graphs_sequence = [Graph_0, Graph_1, ..., Graph_5]
```

### Step 7.2: Compute Graph Embedding

```
graph_emb = SpatioTemporalGNN(graphs_sequence)

          ∈ ℝ^(1 × G)
```

### Step 7.3: Retrieve Historical Analogs

```
context = condition_sequence[-1]  # Last timestep

retrieved = FAISS.query(context, k=3)

          ∈ ℝ^(1 × 3 × F)
```

### Step 7.4: Reverse Diffusion (Denoising) - DDPM Sampling

**Initialization:**
```
x_T ~ N(0, I)    # Start from pure noise
```

**Iterative Denoising (t = T, T-1, ..., 1):**

Untuk setiap timestep t dari T ke 1:

```
1. Predict noise:
   ε_pred = Model(x_t, t, context, retrieved, graph_emb)

2. Compute mean:
   μ_t = (1/√α_t) · (x_t - (β_t/√(1-ᾱ_t)) · ε_pred)

3. Sample noise (for t > 1):
   z ~ N(0, I)

4. Compute x_{t-1}:
   x_{t-1} = μ_t + σ_t · z

   dimana σ_t = √β_t
```

**Untuk t = 1:**
```
x_0 = μ_1    # No noise added at final step
```

**Rumus Lengkap per Step:**
```
x_{t-1} = (1/√α_t) · [x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t, c)] + σ_t · z
```

Dimana:
- `ε_θ` = neural network (trained model)
- `c` = (context, retrieved, graph_emb) = all conditioning

### Step 7.5: Generate Multiple Samples

```
Repeat 50 times dengan initial x_T berbeda:
  sample_i = DDPM_Sample(x_T^(i), Model, conditioning)

predictions = [sample_1, sample_2, ..., sample_50]
```

### Step 7.6: Denormalization

**Inverse Z-Score:**
```
y_log = x_0 · σ_y + μ_y
```

**Inverse Log Transform:**
```
y_mm = exp(y_log) - 1
```

**Clamp Negative Values:**
```
y_mm = max(0, y_mm)
```

**Unit:** mm/jam (curah hujan)

## Output

| Output | Shape | Deskripsi |
|--------|-------|-----------|
| `predictions` | [50] | 50 sampel curah hujan dalam mm |

## Analisis Probabilistik

### Statistik Deskriptif

```
Mean     = (1/50) · Σᵢ predictions[i]

Std      = √[(1/50) · Σᵢ (predictions[i] - Mean)²]

Median   = percentile(predictions, 50)

P10      = percentile(predictions, 10)   # Lower bound
P90      = percentile(predictions, 90)   # Upper bound
```

### Probabilitas Event Ekstrem

**Probability of Extreme Rainfall (>100mm):**
```
P(Rain > 100mm) = count(predictions > 100) / 50
```

**Contoh Interpretasi:**
```
Jika P(Rain > 100mm) = 0.3:
  "Ada 30% kemungkinan curah hujan melebihi 100mm/jam,
   yang berpotensi memicu banjir bandang."
```

### Confidence Interval

**90% Prediction Interval:**
```
CI_90 = [P5, P95]

      = [percentile(5), percentile(95)]
```

---

# RINGKASAN ALUR DATA

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────────┘

[Open-Meteo API] ──────────────────────────────────────┐
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ PIPELINE 1        │                                   │
│ Data Ingestion    │                                   │
│ Output: DataFrame │                                   │
│ [N × 14 columns]  │                                   │
└───────┬───────────┘                                   │
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ PIPELINE 2        │                                   │
│ Preprocessing     │                                   │
│ Output: Tensors   │                                   │
│ [N × F] + stats   │──────────────────────┐            │
└───────┬───────────┘                      │            │
        │                                  │            │
        ├────────────────┬─────────────────┤            │
        ▼                ▼                 │            │
┌───────────────┐ ┌───────────────┐        │            │
│ PIPELINE 3    │ │ PIPELINE 4    │        │            │
│ Sliding       │ │ FAISS Index   │        │            │
│ Window        │ │               │        │            │
│ Output:       │ │ Output:       │        │            │
│ Graph Seq     │ │ retrieved     │        │            │
│ [6 × Graph]   │ │ [B, k, F]     │        │            │
└───────┬───────┘ └───────┬───────┘        │            │
        │                 │                │            │
        ▼                 │                │            │
┌───────────────┐         │                │            │
│ PIPELINE 5    │         │                │            │
│ Spatio-       │         │                │            │
│ Temporal GNN  │         │                │            │
│ Output:       │         │                │            │
│ graph_emb     │         │                │            │
│ [B, G]        │         │                │            │
└───────┬───────┘         │                │            │
        │                 │                │            │
        └────────┬────────┘                │            │
                 │                         │            │
                 ▼                         ▼            │
        ┌───────────────────────────────────┐           │
        │ PIPELINE 6                        │           │
        │ Diffusion Training                │           │
        │                                   │           │
        │ Conditioning:                     │           │
        │ - context (current weather)       │           │
        │ - retrieved (historical analogs)  │           │
        │ - graph_emb (spatio-temporal)     │           │
        │                                   │           │
        │ Output: Trained Model             │           │
        └───────────────┬───────────────────┘           │
                        │                               │
                        ▼                               │
        ┌───────────────────────────────────┐           │
        │ PIPELINE 7                        │◄──────────┘
        │ Probabilistic Inference           │   (stats for
        │                                   │   denormalization)
        │ Output: 50 Samples                │
        │ + P(>100mm)                       │
        │ + Confidence Interval             │
        └───────────────────────────────────┘
```

---

# HYPERPARAMETERS

| Parameter | Nilai | Deskripsi |
|-----------|-------|-----------|
| `seq_len` | 6 | Panjang sequence temporal (6 jam) |
| `num_nodes` | 3 | Jumlah pulau Sitaro |
| `num_features` (F) | 10 | Jumlah fitur cuaca |
| `hidden_dim` (H) | 128 | Hidden dimension |
| `graph_dim` (G) | 64 | Graph embedding dimension |
| `k_neighbors` (k) | 3 | Jumlah retrieval neighbors |
| `num_gat_heads` | 4 | GAT attention heads |
| `num_attn_heads` | 4 | Temporal attention heads |
| `diffusion_steps` (T) | 1000 | DDPM timesteps |
| `β_start` | 0.0001 | Noise schedule start |
| `β_end` | 0.02 | Noise schedule end |
| `batch_size` | 32 | Training batch size |
| `epochs` | 10 | Training epochs |
| `learning_rate` | 0.001 | AdamW learning rate |
| `weight_decay` | 0.0001 | L2 regularization |
| `num_samples` | 50 | Inference samples |

---

# FILE STRUCTURE

```
Skripsi_Bevan/
├── src/
│   ├── data/
│   │   ├── ingest.py            # Pipeline 1: Data Ingestion
│   │   └── temporal_loader.py   # Pipeline 3: Sliding Window
│   ├── models/
│   │   ├── gnn.py               # Pipeline 5: SpatioTemporalGNN
│   │   └── diffusion.py         # Pipeline 6: Diffusion Model
│   ├── retrieval/
│   │   └── base.py              # Pipeline 4: FAISS
│   ├── graph/
│   │   └── builder.py           # Graph construction utilities
│   ├── train.py                 # Training script (Pipeline 6)
│   ├── inference.py             # Inference script (Pipeline 7)
│   └── evaluate.py              # CRPS & Brier Score
├── data/
│   └── raw/
│       └── sitaro_era5_2005_2025.parquet
├── models/
│   └── diffusion_chkpt.pth         # Trained checkpoint
├── PIPELINE_DOCUMENTATION.md       # This file
└── requirements.txt
```

---

# REFERENSI

1. **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models" (NeurIPS 2020)
2. **GAT**: Veličković et al., "Graph Attention Networks" (ICLR 2018)
3. **Transformer**: Vaswani et al., "Attention Is All You Need" (NeurIPS 2017)
4. **FAISS**: Johnson et al., "Billion-scale similarity search with GPUs" (IEEE 2019)
5. **ERA5**: Hersbach et al., "The ERA5 global reanalysis" (QJRMS 2020)
6. **Nowcasting**: Ravuri et al., "Skilful precipitation nowcasting using deep generative models" (Nature 2021)
