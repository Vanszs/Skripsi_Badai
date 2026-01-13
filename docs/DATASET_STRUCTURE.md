# Penjelasan Struktur Dataset Sitaro: Graph-Based Representation

## Ringkasan Singkat

| Aspek | Detail |
|-------|--------|
| **File** | `sitaro_era5_2005_2025.parquet` |
| **Total Baris** | 552,240 |
| **Total Kolom** | 8 |
| **Struktur** | **ALL-IN-ONE** (gabungan 3 lokasi dalam 1 file) |
| **Pembeda Lokasi** | Kolom `node_id` dengan nilai: `Siau`, `Tagulandang`, `Biaro` |
| **Baris per Lokasi** | 184,080 (21 tahun × 365.25 hari × 24 jam ≈ 184,086) |

---

## 1. Mengapa 3 Lokasi?

### 1.1 Latar Belakang Geografis

**Sitaro** adalah singkatan dari **Si**au - **Ta**gulandang - Bia**ro**, sebuah kepulauan di Sulawesi Utara yang terdiri dari 3 pulau utama:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        KEPULAUAN SITARO                                │
│                                                                         │
│                           ▲ N                                          │
│                           │                                            │
│         ┌─────────────────┴──────────────────┐                         │
│         │                                     │                         │
│         │    🌋 SIAU (2.75°N, 125.40°E)      │  ← Pulau Utara          │
│         │       Gunung Karangetang            │    (Gunung api aktif)   │
│         │                                     │                         │
│         │                │                    │                         │
│         │            ~45 km                   │                         │
│         │                │                    │                         │
│         │    🏝️ TAGULANDANG (2.33°N, 125.42°E)│  ← Pulau Tengah         │
│         │                                     │                         │
│         │                │                    │                         │
│         │            ~25 km                   │                         │
│         │                │                    │                         │
│         │    🏝️ BIARO (2.10°N, 125.37°E)     │  ← Pulau Selatan        │
│         │                                     │                         │
│         └─────────────────────────────────────┘                         │
│                                                                         │
│                    LAUT MALUKU                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Alasan Pemilihan 3 Lokasi

#### A. Representasi Spasial Kepulauan

Karena judul skripsi fokus pada **Banjir Bandang Sitaro**, maka kita perlu merepresentasikan **seluruh wilayah** yang berpotensi terdampak:

| Pulau | Koordinat | Karakteristik |
|-------|-----------|---------------|
| **Siau** | (2.75°N, 125.40°E) | Pulau terbesar, ada Gunung Karangetang (aktif) |
| **Tagulandang** | (2.33°N, 125.42°E) | Pulau tengah, menghubungkan utara-selatan |
| **Biaro** | (2.10°N, 125.37°E) | Pulau selatan, terkecil |

#### B. Model Graph Neural Network (GNN)

3 lokasi ini menjadi **NODES** dalam struktur Graph:

```
          SIAU
          (Node 0)
           /    \
          /      \
         /        \
   TAGULANDANG ── BIARO
     (Node 1)    (Node 2)
```

**Mengapa Graph?**

1. **CNN/Grid tidak cocok** untuk kepulauan → piksel laut tidak punya makna
2. **Graph menghubungkan pulau** berdasarkan:
   - Jarak geografis
   - Arah angin (cuaca dari satu pulau mempengaruhi pulau lain)
   - Korelasi hujan antar pulau

#### C. Keterbatasan Resolusi ERA5

ERA5 (sumber data Open-Meteo) memiliki resolusi ~25km. Dengan 3 titik yang berjarak ~25-45km, kita mendapat:

- **Cakupan optimal** untuk wilayah Sitaro
- **Tidak redundan** (jika terlalu dekat, data akan identik)
- **Menangkap variabilitas spasial** hujan di kepulauan

---

## 2. Struktur Dataset: All-in-One

### 2.1 Format Data

Dataset **BUKAN** terpisah per lokasi, melainkan **DIGABUNG** dalam 1 file dengan kolom pembeda:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    sitaro_era5_2005_2025.parquet                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  SIAU DATA (184,080 rows)                                           │   │
│  │  date                  | precipitation | ... | node_id              │   │
│  │  2005-01-01 00:00:00   | 0.5           | ... | Siau                 │   │
│  │  2005-01-01 01:00:00   | 0.3           | ... | Siau                 │   │
│  │  ...                   | ...           | ... | ...                  │   │
│  │  2025-12-31 23:00:00   | 0.0           | ... | Siau                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  TAGULANDANG DATA (184,080 rows)                                    │   │
│  │  date                  | precipitation | ... | node_id              │   │
│  │  2005-01-01 00:00:00   | 0.2           | ... | Tagulandang          │   │
│  │  2005-01-01 01:00:00   | 0.4           | ... | Tagulandang          │   │
│  │  ...                   | ...           | ... | ...                  │   │
│  │  2025-12-31 23:00:00   | 0.1           | ... | Tagulandang          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  BIARO DATA (184,080 rows)                                          │   │
│  │  date                  | precipitation | ... | node_id              │   │
│  │  2005-01-01 00:00:00   | 0.0           | ... | Biaro                │   │
│  │  2005-01-01 01:00:00   | 0.1           | ... | Biaro                │   │
│  │  ...                   | ...           | ... | ...                  │   │
│  │  2025-12-31 23:00:00   | 0.0           | ... | Biaro                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  TOTAL: 552,240 rows = 3 lokasi × 184,080 timesteps                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Kolom Dataset

| No | Kolom | Tipe | Deskripsi |
|----|-------|------|-----------|
| 1 | `date` | datetime64[ns, UTC] | Timestamp per jam |
| 2 | `precipitation` | float32 | **TARGET** - Curah hujan (mm/jam) |
| 3 | `temperature_2m` | float32 | Suhu udara di 2m (°C) |
| 4 | `relative_humidity_2m` | float32 | Kelembaban relatif (%) |
| 5 | `surface_pressure` | float32 | Tekanan udara (hPa) |
| 6 | `wind_speed_10m` | float32 | Kecepatan angin di 10m (m/s) |
| 7 | `wind_direction_10m` | float32 | Arah angin (derajat) |
| 8 | `node_id` | string | **PEMBEDA LOKASI**: `Siau`, `Tagulandang`, `Biaro` |

### 2.3 Mengapa Digabung (All-in-One)?

#### Keuntungan Format Gabungan:

1. **Efisiensi Storage**: 1 file vs 3 file
2. **Kemudahan Processing**: Filter dengan `df[df['node_id'] == 'Siau']`
3. **Graph Construction**: Mudah men-pivot data untuk membuat multi-node graph per timestep
4. **Konsistensi Temporal**: Semua node memiliki timestamp yang sama

#### Cara Akses Data per Lokasi:

```python
import pandas as pd

# Load dataset
df = pd.read_parquet('data/raw/sitaro_era5_2005_2025.parquet')

# Akses data Siau saja
df_siau = df[df['node_id'] == 'Siau']

# Akses data Tagulandang saja
df_tagu = df[df['node_id'] == 'Tagulandang']

# Akses data Biaro saja
df_biaro = df[df['node_id'] == 'Biaro']
```

---

## 3. Bagaimana Data Ini Digunakan dalam Pipeline

### 3.1 Visualisasi Alur Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW PIPELINE                                  │
└─────────────────────────────────────────────────────────────────────────────┘

                            ┌───────────────────┐
                            │ sitaro_era5_      │
                            │ 2005_2025.parquet │
                            │   (552,240 rows)  │
                            └────────┬──────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │         STEP 1: FILTER BY DATE        │
                 │  (Temporal Split: Train/Val/Test)     │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: PIVOT BY TIMESTAMP                          │
│                                                                            │
│  Dari format LONG:                                                         │
│  ┌──────────────────────────────────────┐                                  │
│  │ date       | precip | node_id        │                                  │
│  │ 2005-01-01 | 0.5    | Siau           │                                  │
│  │ 2005-01-01 | 0.2    | Tagulandang    │                                  │
│  │ 2005-01-01 | 0.0    | Biaro          │                                  │
│  └──────────────────────────────────────┘                                  │
│                                                                            │
│  Menjadi format GRAPH (per timestep):                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Timestep: 2005-01-01 00:00                                          │  │
│  │                                                                      │  │
│  │     Node 0 (Siau)         Node 1 (Tagu)       Node 2 (Biaro)        │  │
│  │  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐        │  │
│  │  │ temp: 28.5    │    │ temp: 27.8    │    │ temp: 28.1    │        │  │
│  │  │ humid: 85%    │    │ humid: 82%    │    │ humid: 84%    │        │  │
│  │  │ precip: 0.5   │────│ precip: 0.2   │────│ precip: 0.0   │        │  │
│  │  │ ...           │    │ ...           │    │ ...           │        │  │
│  │  └───────────────┘    └───────────────┘    └───────────────┘        │  │
│  │         ↑                    ↑                    ↑                  │  │
│  │         └────────────────────┴────────────────────┘                  │  │
│  │                    EDGES (koneksi antar node)                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└───────────────────────────────────────────────────────────────────────────-┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │      STEP 3: SLIDING WINDOW           │
                 │  6 timesteps → 1 prediction           │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                 ┌───────────────────────────────────────┐
                 │ STEP 4: INPUT KE GRAPH NEURAL NETWORK │
                 │ Sequence of 6 graphs → GNN → Embedding│
                 └───────────────────────────────────────┘
```

### 3.2 Kode Konstruksi Graph

Dari file `src/data/temporal_loader.py`:

```python
# Default node names
node_names: List[str] = ['Siau', 'Tagulandang', 'Biaro']

# Setiap timestep, buat 1 graph dengan 3 nodes
# Nodes terhubung fully-connected (semua saling terhubung)
```

---

## 4. Matematika Konstruksi Graph

### 4.1 Definisi Graph

Untuk setiap timestep `t`, kita mendefinisikan graph `G_t = (V, E)`:

**Nodes (V):**
```
V = {v_Siau, v_Tagulandang, v_Biaro}
|V| = 3
```

**Node Features:** Untuk setiap node `v_i` pada timestep `t`:
```
x_i^(t) = [temp, humid, pressure, wind_speed, wind_dir, precip_lag1]
```

**Edges (E) - Fully Connected:**
```
E = {(i,j) | i,j ∈ V, i ≠ j}
|E| = 3 × 2 = 6 (directed edges)
```

### 4.2 Edge Index (Format COO untuk PyTorch Geometric)

```python
edge_index = [
    [0, 0, 1, 1, 2, 2],  # Source nodes
    [1, 2, 0, 2, 0, 1]   # Target nodes
]
```

Artinya:
- Siau → Tagulandang
- Siau → Biaro
- Tagulandang → Siau
- Tagulandang → Biaro
- Biaro → Siau
- Biaro → Tagulandang

---

## 5. Contoh Data Aktual

### 5.1 Sample Data untuk 1 Timestep

```
Timestamp: 2005-01-01 00:00:00 UTC

┌───────────────┬──────────────┬─────────────┬───────────────┬─────────────┐
│ node_id       │ precipitation│ temperature │ rel_humidity  │ wind_speed  │
├───────────────┼──────────────┼─────────────┼───────────────┼─────────────┤
│ Siau          │ 0.52 mm      │ 26.3°C      │ 87%           │ 3.2 m/s     │
│ Tagulandang   │ 0.31 mm      │ 26.1°C      │ 85%           │ 2.8 m/s     │
│ Biaro         │ 0.15 mm      │ 26.5°C      │ 83%           │ 3.5 m/s     │
└───────────────┴──────────────┴─────────────┴───────────────┴─────────────┘

        → Ini menjadi 1 GRAPH dengan 3 NODES pada timestep ini
```

### 5.2 Sliding Window (6 Timesteps → 1 Prediction)

```
Input: 6 consecutive graphs (t-6 sampai t-1)
Target: Precipitation at timestep t for ALL 3 nodes

┌─────────────────────────────────────────────────────────────────────────┐
│         SLIDING WINDOW EXAMPLE                                          │
│                                                                         │
│   Graph(t-6)  Graph(t-5)  Graph(t-4)  Graph(t-3)  Graph(t-2)  Graph(t-1)│
│      ▼           ▼           ▼           ▼           ▼           ▼      │
│   ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐    ┌─────┐       │
│   │ S   │    │ S   │    │ S   │    │ S   │    │ S   │    │ S   │       │
│   │/ \ │    │/ \ │    │/ \ │    │/ \ │    │/ \ │    │/ \ │       │
│   │T─B │    │T─B │    │T─B │    │T─B │    │T─B │    │T─B │       │
│   └─────┘    └─────┘    └─────┘    └─────┘    └─────┘    └─────┘       │
│                                                                         │
│                           ┌───────────────────────┐                     │
│                           │  Spatio-Temporal GNN  │                     │
│                           └───────────┬───────────┘                     │
│                                       ▼                                 │
│                           ┌───────────────────────┐                     │
│                           │ Prediction: precip(t) │                     │
│                           │ untuk Siau, Tagu, Biaro│                    │
│                           └───────────────────────┘                     │
│                                                                         │
│   Legend: S = Siau, T = Tagulandang, B = Biaro                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. FAQ (Frequently Asked Questions)

### Q1: Apakah bisa pakai lebih dari 3 lokasi?

**A:** Ya, secara teknis bisa. Namun:
- Resolusi ERA5 (~25km) membatasi kepadatan titik
- 3 titik sudah representatif untuk Sitaro yang kecil (~50km utara-selatan)
- Lebih banyak titik = lebih kompleks tanpa tambahan informasi signifikan

### Q2: Apakah harus pakai ketiga lokasi, atau bisa prediksi 1 lokasi saja?

**A:** Model saat ini memprediksi **semua 3 lokasi sekaligus**. Ini karena:
- GNN mempelajari korelasi antar pulau
- Hujan di Siau bisa "migrate" ke Tagulandang dalam beberapa jam
- Informasi spasial meningkatkan akurasi

### Q3: Bagaimana jika ingin menambah pulau baru?

**A:** Update `SITARO_NODES` di `src/data/ingest.py`:

```python
SITARO_NODES = pd.DataFrame({
    'name': ['Siau', 'Tagulandang', 'Biaro', 'PulauBaru'],
    'lat': [2.75, 2.33, 2.10, X.XX],
    'lon': [125.40, 125.42, 125.37, XXX.XX]
})
```

### Q4: Mengapa tidak pakai resolusi lebih tinggi (misal per-kilometer)?

**A:** Keterbatasan sumber data:
- ERA5 (Open-Meteo) resolusi ~25km
- Untuk resolusi tinggi perlu GPM IMERG (~10km) atau radar lokal
- Diffusion Model dalam pipeline ini justru bertujuan "super-resolution" dari data kasar

---

## 7. Kesimpulan

| Aspek | Jawaban |
|-------|---------|
| **Mengapa 3 lokasi?** | Merepresentasikan 3 pulau utama Sitaro untuk model Graph Neural Network |
| **Struktur data?** | **All-in-One** - 1 file dengan kolom `node_id` sebagai pembeda lokasi |
| **Bagaimana digunakan?** | Di-pivot per timestep → menjadi 1 graph dengan 3 nodes → sequence 6 graphs → GNN → prediksi |

> **IMPORTANT:** Dataset ini sudah siap digunakan langsung untuk pipeline GNN. Kolom `node_id` adalah kunci untuk memisahkan dan menggabungkan data per lokasi.
