# Penjelasan Struktur Dataset Gunung Gede-Pangrango: Graph-Based Representation

## Ringkasan Singkat

| Aspek | Detail |
|-------|--------|
| **File** | `pangrango_era5_2005_2025.parquet` |
| **Total Baris** | ~526,032 |
| **Total Kolom** | 13 |
| **Struktur** | **ALL-IN-ONE** (gabungan 3 lokasi dalam 1 file) |
| **Pembeda Lokasi** | Kolom `node` dengan nilai: `Puncak`, `Lereng_Cibodas`, `Hilir_Cianjur` |
| **Baris per Lokasi** | ~175,344 (21 tahun x 365.25 hari x 24 jam) |

---

## 1. Mengapa 3 Lokasi?

### 1.1 Latar Belakang Geografis

**Gunung Gede-Pangrango** adalah gunung berapi kembar di Jawa Barat, Indonesia, yang menjadi destinasi pendakian populer. Tiga titik observasi mewakili zona elevasi berbeda:

```
                    GUNUNG GEDE-PANGRANGO

                         /\  Puncak
                        /  \  (~3000 mdpl)
                       /    \
                      / Node \
                     /   0    \
                    /----------\
                   /  Lereng    \
                  /  Cibodas     \
                 /   (~1400m)    \
                /    Node 1       \
               /-------------------\
              /  Hilir Cianjur      \
             /   (~500 mdpl)        \
            /    Node 2              \
           /--------------------------\
```

### 1.2 Alasan Pemilihan 3 Lokasi

| Zona | Koordinat | Karakteristik |
|------|-----------|---------------|
| **Puncak** | (-6.75 S, 106.98 E) | Zona alpine, elevasi tinggi, angin kencang |
| **Lereng_Cibodas** | (-6.74 S, 107.00 E) | Zona hutan montane, jalur pendakian utama |
| **Hilir_Cianjur** | (-6.70 S, 107.10 E) | Zona dataran rendah, aliran air hujan |

3 lokasi ini menjadi **NODES** dalam struktur Graph Neural Network:

```
        PUNCAK
        (Node 0)
         /    \
        /      \
       /        \
  LERENG    -- HILIR
  CIBODAS      CIANJUR
  (Node 1)    (Node 2)
```

---

## 2. Struktur Dataset: All-in-One

### 2.1 Kolom Dataset

| No | Kolom | Tipe | Deskripsi |
|----|-------|------|-----------|
| 1 | `date` | datetime64 | Timestamp per jam |
| 2 | `precipitation` | float32 | **TARGET** - Curah hujan (mm/jam) |
| 3 | `temperature_2m` | float32 | Suhu udara di 2m (C) |
| 4 | `relative_humidity_2m` | float32 | **TARGET** - Kelembaban relatif (%) |
| 5 | `dewpoint_2m` | float32 | Titik embun (C) |
| 6 | `surface_pressure` | float32 | Tekanan udara (hPa) |
| 7 | `wind_speed_10m` | float32 | **TARGET** - Kecepatan angin di 10m (m/s) |
| 8 | `wind_direction_10m` | float32 | Arah angin (derajat) |
| 9 | `cloud_cover` | float32 | Tutupan awan (%) |
| 10 | `precipitation_lag1` | float32 | Curah hujan 1 jam sebelumnya |
| 11 | `elevation` | float32 | Elevasi lokasi (m) |
| 12 | `node` | string | Pembeda lokasi |

### 2.2 Cara Akses Data per Lokasi

```python
import pandas as pd
df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')

df_puncak = df[df['node'] == 'Puncak']
df_lereng = df[df['node'] == 'Lereng_Cibodas']
df_hilir  = df[df['node'] == 'Hilir_Cianjur']
```

---

## 3. Temporal Split

| Split | Periode | Tahun | Kegunaan |
|-------|---------|-------|----------|
| **Training** | 2005-2018 | 14 tahun | Training + normalisasi |
| **Validation** | 2019-2021 | 3 tahun | Early stopping & tuning |
| **Test** | 2022-2025 | 4 tahun | Evaluasi final |

> Normalisasi mean/std dihitung **hanya dari training set**.

---

## 4. Pipeline Data Flow

```
  pangrango_era5_2005_2025.parquet
              |
              v
     Temporal Split (Train/Val/Test)
              |
              v
     Pivot by Timestamp -> Graph per timestep (3 nodes)
              |
              v
     Sliding Window (6 timesteps -> 1 prediction)
              |
              v
     Spatio-Temporal GNN -> Graph Embedding
              |
              v
     Conditional Diffusion Model -> Ensemble Predictions
```

---

## 5. Kesimpulan

| Aspek | Jawaban |
|-------|---------|
| **Mengapa 3 lokasi?** | Merepresentasikan 3 zona elevasi Gunung Gede-Pangrango untuk GNN |
| **Struktur data?** | All-in-One, kolom `node` sebagai pembeda lokasi |
| **Bagaimana digunakan?** | Pivot per timestep -> graph 3 nodes -> sequence 6 graphs -> GNN -> prediksi |
