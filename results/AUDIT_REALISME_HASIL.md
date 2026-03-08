# AUDIT REALISME HASIL MODEL
## Retrieval-Augmented Diffusion Model — Gunung Gede-Pangrango

**Tanggal Audit:** 8 Maret 2026  
**Status:** ⚠️ HASIL TEKNIS BENAR, NAMUN PERLU INTERPRETASI KRITIS

---

## Ringkasan Eksekutif

Hasil evaluasi **secara teknis benar** (tidak ada bug perhitungan atau data leakage pada split). Namun, kesan visual bahwa "prediksi hampir menempel dengan aktual" **BUKAN karena model diffusion yang superior**, melainkan didominasi oleh **persistence forecast** (nilai jam sebelumnya) melalui mekanisme hybrid weighting yang sangat tinggi.

---

## 1. TEMUAN UTAMA: Hybrid Weights = Persistence Forecast

### Lokasi Masalah
`src/inference.py`, fungsi `run_inference_hybrid()`, baris ~295:

```python
weights = {
    'precipitation': 0.90,   # 90% lag, 10% model
    'wind_speed':    0.90,   # 90% lag, 10% model
    'humidity':      0.70    # 70% lag, 30% model
}
```

### Formula yang Digunakan
```
hybrid = (1 - w) × model_prediction + w × lag_value_jam_sebelumnya
```

| Variabel | Kontribusi Model | Kontribusi Persistence | Implikasi |
|----------|:---:|:---:|---|
| Curah Hujan | **10%** | **90%** | Prediksi = 90% nilai jam lalu |
| Kecepatan Angin | **10%** | **90%** | Prediksi = 90% nilai jam lalu |
| Kelembapan | **30%** | **70%** | Prediksi = 70% nilai jam lalu |

### Mengapa Plot Terlihat "Menempel"?

Variabel cuaca memiliki **autokorelasi temporal sangat tinggi** pada resolusi per-jam. Suhu, angin, dan kelembapan jarang berubah drastis dalam 1 jam. Maka, jika prediksi = 90% nilai jam lalu, otomatis prediksi akan sangat dekat dengan aktual — **bukan karena model pintar, tapi karena cuaca jarang berubah dalam 1 jam**.

Ini setara dengan baseline paling sederhana dalam meteorologi: **persistence forecast** ("besok cuaca sama seperti hari ini").

---

## 2. Perbandingan: Model Diffusion vs MLP Baseline

| Metrik | MLP Baseline | Diffusion (Hybrid) | Pemenang |
|--------|:---:|:---:|:---:|
| Precip RMSE | **0.878** | 2.002 | MLP ✓ |
| Wind RMSE | **1.088** | 2.003 | MLP ✓ |
| Humidity RMSE | **2.744** | 6.005 | MLP ✓ |
| Precip Corr | 0.455 | 0.456 | ≈ Sama |
| Wind Corr | **0.898** | 0.833 | MLP ✓ |
| Humidity Corr | **0.974** | 0.923 | MLP ✓ |

> **Catatan penting:** MLP baseline rata-rata node per timestamp, diffusion evaluasi per-node. Perbandingan tidak 100% apple-to-apple, namun perbedaannya substansial.

**Kesimpulan:** Model diffusion yang jauh lebih kompleks (GNN + Retrieval + Diffusion) memiliki RMSE **2-3x lebih besar** dari MLP 3-layer sederhana. Bahkan DENGAN 90% persistence weight, hasilnya masih kalah.

---

## 3. Kegagalan Deteksi Extreme Event

Tujuan inti skripsi: **"Nowcasting Probabilistik Hujan Lebat untuk Keselamatan Pendaki"**

| Node | Heavy Rain POD | FAR | CSI |
|------|:---:|:---:|:---:|
| Puncak | **0.000** | 1.000 | 0.000 |
| Lereng_Cibodas | **0.000** | 1.000 | 0.000 |
| Hilir_Cianjur | 0.143 | 0.833 | 0.083 |
| **Agregasi** | **0.059** | **0.864** | **0.043** |

- **POD 0%** di Puncak dan Lereng: Model **sama sekali tidak mendeteksi** hujan lebat (>10 mm/jam)
- **FAR 86%**: Dari semua alarm hujan lebat yang dikeluarkan, 86% adalah false alarm
- **CSI 4%**: Secara keseluruhan, kemampuan deteksi hujan lebat nyaris nol

Reliability diagram juga menunjukkan semua probabilitas forecast terkumpul di dekat 0 — model tidak pernah memberikan probabilitas tinggi untuk hujan lebat.

---

## 4. Hal-Hal yang BENAR (Tidak Ada Masalah)

### ✅ Temporal Split — Benar
- Training: 2005-2018, Validation: 2019-2021, Test: 2022-2025
- Tidak ada random shuffle pada time series
- Normalization stats dihitung dari training set saja

### ✅ Retrieval Database — Benar
- Dibangun hanya dari training data
- Tidak ada data test/validation yang bocor ke retrieval index

### ✅ Evaluasi Metrik — Benar
- CRPS, Brier Score, POD, FAR, CSI dihitung dengan formula yang benar
- RMSE, MAE, Correlation implementasinya benar

### ✅ Feature Engineering — Acceptable
- Features termasuk `relative_humidity_2m` dan `wind_speed_10m` yang juga adalah target
- Namun ini diambil dari sequence [t-6 hingga t-1] untuk memprediksi target di waktu t
- Ini adalah praktik standar autoregressive dan BUKAN data leakage

---

## 5. Diagnosis Detail

### 5a. Mengapa Korelasi Curah Hujan Rendah (r ≈ 0.45)?
- Curah hujan bersifat **intermittent** (banyak jam dengan 0 mm, sesekali spike tinggi)
- Distribusi sangat **skewed** — didominasi nol
- Persistence forecast buruk untuk precipitation karena spike datang tiba-tiba
- Bahkan dengan 90% persistence, korelasi hanya 0.45

### 5b. Mengapa Humidity dan Wind Korelasinya Tinggi?
- Kelembapan dan kecepatan angin **berubah perlahan** (smooth time series)
- Autokorelasi lag-1 jam sangat tinggi secara alami
- Persistence forecast sudah cukup untuk mendapatkan r > 0.8
- Yang kita lihat di plot adalah **keberhasilan persistence, bukan model**

### 5c. Learning Curve Analysis
- Training loss turun dari ~0.31 ke ~0.15 (konvergen)
- Validation loss turun dari ~0.11 ke ~0.095 (tidak overfitting)
- **Namun**: val loss LEBIH RENDAH dari train loss — ini menunjukkan weighted loss yang digunakan saat training (5x-10x weight untuk extreme events) membuat train loss terinflasi secara artifisial

---

## 6. Rekomendasi

### Opsi A: Evaluasi Kontribusi Model yang Jujur
Jalankan evaluasi **TANPA hybrid** (weights = 0) untuk melihat performa murni diffusion model. Bandingkan dengan:
1. Pure persistence baseline (prediksi = nilai jam lalu)
2. MLP baseline
3. Diffusion model murni (tanpa hybrid)
4. Diffusion model + hybrid

Ini akan menunjukkan secara transparan berapa **added value** dari model diffusion di atas persistence.

### Opsi B: Turunkan Hybrid Weights
Jika hybrid tetap digunakan, gunakan weights yang lebih seimbang:
```python
weights = {
    'precipitation': 0.3,   # 30% lag, 70% model
    'wind_speed':    0.3,
    'humidity':      0.3
}
```
Lalu evaluasi ulang. Hasilnya mungkin lebih buruk secara metrik, tapi lebih jujur.

### Opsi C: Perbaiki Evaluasi Baseline agar Apple-to-Apple
Buat evaluasi MLP baseline per-node (bukan rata-rata) sehingga perbandingan fair.

### Opsi D: Presentasikan Hasil dengan Jujur
Dalam skripsi, jelaskan bahwa:
1. Hybrid approach digunakan dengan persistence weight
2. Tunjukkan hasil **dengan dan tanpa** hybrid
3. Akui bahwa sebagian besar prediksi didominasi persistence
4. Fokus pembahasan pada kontribusi **probabilistic** (CRPS, ensemble spread) yang memang added value dari diffusion model

---

## 7. Kesimpulan Audit

| Aspek | Status | Keterangan |
|-------|:---:|---|
| Data Leakage | ✅ Aman | Temporal split benar, stats dari training |
| Implementasi Metrik | ✅ Benar | CRPS, Brier, POD, FAR, CSI valid |
| Implementasi Model | ✅ Benar | Diffusion + GNN + Retrieval berfungsi |
| Hybrid Weights | ⚠️ Masalah | 90% persistence mendominasi prediksi |
| Deteksi Hujan Lebat | ❌ Gagal | POD ≈ 0%, CSI ≈ 4% |
| "Too Good to Be True" | ⚠️ Ya | Plot menempel karena persistence, bukan model |
| Perbandingan Baseline | ⚠️ Kalah | Diffusion RMSE 2-3x lebih besar dari MLP |

**Jawaban atas pertanyaan Anda:** Ya, intuisi Anda benar — hasilnya terlihat "terlalu bagus" karena 90% dari prediksi hanyalah nilai jam sebelumnya (persistence). Model diffusion sesungguhnya berkontribusi sangat kecil (10-30%) dan justru **memperburuk** akurasi dibandingkan baseline MLP.

---

*Audit ini dihasilkan dari analisis kode sumber lengkap: `src/inference.py`, `src/train.py`, `src/data/temporal_loader.py`, `src/data/ingest.py`, `src/models/diffusion.py`, `src/evaluation/probabilistic_metrics.py`, dan seluruh file hasil evaluasi.*
