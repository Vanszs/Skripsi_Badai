# PANDUAN REVISI PROPOSAL SKRIPSI

**Dokumen Acuan**: `Draft Bepan (2).docx` (Pra-Skripsi / Proposal)
**Berdasarkan**: Laporan Audit Konsistensi Komprehensif (Juni 2025)
**Tujuan**: Menyesuaikan narasi dokumen agar selaras dengan implementasi kode yang sudah berjalan

> **Catatan Penting**: Kode tidak perlu diubah secara substansial — implementasi sudah solid, berjalan, dan menghasilkan metrik. Yang perlu disesuaikan hanyalah **narasi dan klaim di dokumen proposal**.

---

## 1 · Ringkasan Perubahan

### 🔴 Perubahan WAJIB

> Harus diperbaiki agar dokumen konsisten dengan kode. Tanpa perubahan ini, penguji dapat menemukan kontradiksi langsung antara klaim proposal dan implementasi.

| No | Masalah | Lokasi Dokumen | Perubahan yang Harus Dilakukan |
|----|---------|----------------|-------------------------------|
| W1 | Judul tesis mengklaim "Hujan Lebat" / "Cuaca Ekstrem", padahal model tidak mampu mendeteksi hujan lebat (POD 5.9%) | **Halaman Judul**, **Abstrak**, **BAB I – Latar Belakang**, **BAB I – Tujuan Penelitian** | Ganti framing dari "Hujan Lebat"/"Cuaca Ekstrem" → **"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki"** |
| W2 | RQ3 mengklaim "deteksi spike events" yang tidak tercapai | **BAB I – Pendahuluan** → Subbab Rumusan Masalah → RQ3 | Ubah RQ3 menjadi: *"Bagaimana kemampuan model probabilistik dalam memberikan estimasi ketidakpastian prakiraan cuaca multi-variabel melalui kerangka ensemble nowcasting?"* |
| W3 | Dokumen mengklaim "horizon prediksi 0–6 jam" (multi-step), kode hanya prediksi 1 jam | **BAB I** → Subbab Batasan Masalah → kalimat tentang "horizon prediksi" | Ubah menjadi: *"...dengan memanfaatkan jendela observasi 6 jam terakhir untuk memprediksi kondisi cuaca satu jam ke depan (single-step hourly nowcasting)."* |
| W4 | Dokumen menjanjikan 4 skenario ablasi, kode hanya mengimplementasi 2 | **BAB III – Metodologi** → Subbab Alur Eksperimen / Desain Eksperimen → paragraf yang menjelaskan skenario model | Revisi menjadi 2 skenario: *(1) Baseline deterministik MLP, dan (2) Model probabilistik lengkap.* Tambahkan analisis kualitatif kontribusi komponen di BAB Pembahasan |
| W5 | Dokumen hanya menyebut 3 variabel, kode menggunakan 9 fitur input | **BAB III – Metodologi** → Subbab Dataset / Variabel Penelitian → paragraf yang membahas variabel | Tambah subbab baru **"Fitur Input dan Variabel Target"** yang mendokumentasikan seluruh 9 fitur input dan 3 variabel target secara terpisah |
| W6 | Weighted denoising loss (5×/10×) tidak terdokumentasi | **BAB III – Metodologi** → Subbab Prosedur Pelatihan → paragraf tentang loss function | Tambahkan paragraf yang mendokumentasikan skema weighted MSE: bobot 5× untuk \|z\| > 1σ dan 10× untuk \|z\| > 3σ |
| W7 | Dokumen mengklaim "early stopping", kode hanya best-model selection | **BAB III – Metodologi** → Subbab Prosedur Pelatihan → kalimat tentang "early stopping" | Ganti *"mekanisme early stopping"* menjadi: *"Model terbaik dipilih berdasarkan loss validasi terendah selama 20 epoch pelatihan (best model selection)."* |
| W8 | Dokumen mengklaim "graf berbobot", kode menggunakan graf tanpa bobot | **BAB III – Metodologi** → Subbab Representasi Graf → kalimat *"graf tak berarah berbobot"* | Ganti menjadi: *"graf tak berarah fully-connected. Bobot relasi antar node dipelajari secara implisit melalui mekanisme Graph Attention (GAT) dengan 4 attention heads."* |
| W9 | Hybrid weights diklaim "dari validasi", kode hardcoded 0.90/0.90/0.70 | **BAB III – Metodologi** → Subbab Hybrid Persistence Post-Processing → kalimat tentang penentuan bobot | Tampilkan nilai eksplisit (0.90/0.90/0.70) dan jelaskan proses penentuan empiris. Lihat detail ➜ GAP-03 |
| W10 | Tidak ada subbab Keterbatasan Penelitian | **BAB V – Pembahasan** (atau BAB terakhir sebelum Kesimpulan) | Tambahkan subbab baru: **Keterbatasan Penelitian** — bahas resolusi ERA5 ~25 km, 2 node pada grid cell yang sama, target rata-rata antar node |

### 🟡 Perubahan DISARANKAN

> Tidak wajib, tetapi memperjelas metodologi dan menghindari pertanyaan reviewer.

| No | Saran Perbaikan | Lokasi Dokumen | Mengapa Ini Penting |
|----|-----------------|----------------|---------------------|
| D1 | Dokumentasikan T_STD_MULTIPLIER = 5.0 | **BAB III** → Subbab Pra-pemrosesan → setelah paragraf log-transform | Teknik stabilisasi diffusion yang berpengaruh signifikan — reviewer mungkin bertanya mengapa presipitasi dinormalisasi berbeda |
| D2 | Jelaskan EVAL_STEP = 24 (daily subsampling) | **BAB III** → Subbab Metode Evaluasi → di akhir penjelasan metrik | Jumlah sampel evaluasi ~1/24 dari total — reviewer mungkin mempertanyakan representativitas |
| D3 | Jelaskan target prediksi = rata-rata antar 3 node | **BAB III** → Subbab Desain Model / Arsitektur → setelah penjelasan node | Reviewer mungkin mengira model memprediksi per-node; perlu klarifikasi bahwa target adalah rata-rata |
| D4 | Jelaskan 2 node yang share grid cell ERA5 | **BAB III** → Subbab Wilayah Studi → setelah deskripsi node | Puncak & Lereng berjarak ~3 km (resolusi ERA5 ~25 km) — ini desain yang disengaja, bukan kekeliruan |
| D5 | Sebutkan hyperparameter spesifik | **BAB III** → Subbab Prosedur Pelatihan → setelah penjelasan training | SEQ_LEN=6, BATCH_SIZE=512, LR=1e-3, EPOCHS=20, NUM_ENSEMBLE=30 — transparansi reproduktibilitas |
| D6 | Jelaskan DDIM 20-step vs DDPM 1000-step | **BAB III** → Subbab Inferensi → paragraf tentang sampling | Reviewer yang paham diffusion akan bertanya mengapa langkah inference berbeda dari training |
| D7 | Bahas dominansi persistence secara kritis | **BAB V – Pembahasan** → paragraf tentang hybrid persistence | Hybrid weight 0.90 berarti 90% persistence — perlu framing bahwa ini wajar pada skala 1 jam |
| D8 | Bersihkan komentar kode stale "4 variabel" | **Kode**: `diffusion.py`, `train.py`, `inference.py` | Komentar masih menyebut "4 variables" padahal sudah 3 — bisa membingungkan reviewer yang melihat kode |
| D9 | Tambah threshold evaluasi 2 mm & 5 mm | **Kode**: `final_proven_eval.py` | Threshold 10 mm/jam terlalu tinggi untuk ERA5 — threshold lebih rendah akan menunjukkan performa lebih baik |
| D10 | Tabel perbandingan pure persistence vs hybrid vs diffusion-only | **BAB V – Pembahasan** → subbab baru / paragraf baru | Memperkuat argumen bahwa diffusion model memberikan added value di dimensi probabilistik |

---

## 2 · Detail Perubahan

### GAP-01 — Horizon Prediksi "0–6 jam" vs Prediksi Single-Step

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB I – Pendahuluan** → Subbab **Batasan Masalah**
- Cari kalimat: *"Skala waktu prediksi dibatasi pada skala jam-an (hourly nowcasting) dengan horizon prediksi 0–6 jam ke depan"*
- Periksa juga setiap penyebutan "horizon 0–6 jam" di seluruh dokumen (termasuk BAB III – Desain Penelitian)

**Masalah**
Dokumen menyatakan "horizon prediksi 0–6 jam", mengimplikasikan model mampu menghasilkan prakiraan untuk 1–6 jam ke depan. Kode hanya memprediksi **satu langkah berikutnya** (t+1 = 1 jam ke depan). SEQ_LEN=6 adalah jendela **input** (6 jam ke belakang), bukan horizon output.

**Implementasi Kode**
- `train.py`: `SEQ_LEN = 6` → jendela input (bukan output)
- `temporal_loader.py`: `target = self.targets_norm[t].mean(dim=0)` → hanya satu timestep berikutnya

**Perbaikan yang Disarankan**
Ganti kalimat asli dengan:

> *"Model menggunakan jendela observasi 6 jam terakhir sebagai input untuk memprediksi kondisi cuaca satu jam ke depan (single-step hourly nowcasting). Pendekatan ini sesuai dengan definisi nowcasting menurut WMO, yaitu prakiraan dengan rentang waktu 0–6 jam, di mana model dijalankan secara iteratif pada setiap jam."*

Pastikan seluruh penyebutan "horizon 0–6 jam" di dokumen diubah menjadi "jendela input 6 jam → prediksi 1 jam ke depan".

---

### GAP-02 — Empat Skenario Ablasi vs Dua Skenario Terimplementasi

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Alur Eksperimen** atau **Desain Eksperimen**
- Cari paragraf yang mendaftar 4 skenario: (1) Baseline deterministik, (2) Model probabilistik tanpa retrieval, (3) Model probabilistik dengan retrieval, (4) Model probabilistik dengan retrieval dan hybrid

**Masalah**
Dokumen menjanjikan 4 skenario ablasi. Kode hanya mengimplementasi 2 skenario ekstrem: baseline MLP dan model penuh. Tidak ada skrip untuk skenario 2 dan 3.

**Implementasi Kode**
- `train_baseline.py` → Skenario 1: MLP baseline
- `train.py` → Skenario 4: full model (Retrieval + GNN + Hybrid)
- Skenario 2 & 3 → **tidak ada implementasi**

**Perbaikan yang Disarankan**
Revisi daftar skenario menjadi:

> *"Eksperimen terdiri dari dua skenario utama: (1) Baseline deterministik menggunakan Multi-Layer Perceptron (MLP) dengan MSE loss, dan (2) Model probabilistik lengkap menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal GNN dan Hybrid Persistence Post-Processing. Perbandingan antara kedua skenario ini memungkinkan evaluasi sejauh mana pendekatan probabilistik dan komponen-komponen tambahan (retrieval, graf, hybrid) memberikan peningkatan terhadap baseline deterministik."*

Untuk menjawab RQ2 tentang kontribusi per-komponen, tambahkan **analisis kualitatif** di BAB Pembahasan yang membahas peran masing-masing modul berdasarkan karakteristik output.

---

### GAP-03 — Bobot Hybrid "dari Validasi" vs Hardcoded

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Hybrid Persistence Post-Processing**
- Cari kalimat: *"Parameter bobot w ditentukan menggunakan data validasi dan tidak menggunakan data pengujian"*

**Masalah**
Dokumen mengklaim bobot ditentukan dari validasi secara sistematis. Kode menggunakan nilai hardcoded 0.90/0.90/0.70 tanpa proses optimisasi yang terdokumentasi.

**Implementasi Kode**
- `inference.py`: bobot hardcoded `{'precipitation': 0.90, 'wind_speed': 0.90, 'humidity': 0.70}`
- Tidak ada skrip grid-search atau optimisasi

**Perbaikan yang Disarankan**

> *"Parameter bobot hybrid persistence ditetapkan melalui eksplorasi empiris pada data validasi dengan tujuan meminimalkan RMSE gabungan. Nilai akhir yang digunakan: w\_precipitation = 0.90, w\_wind\_speed = 0.90, w\_humidity = 0.70. Bobot yang tinggi untuk presipitasi dan angin (0.90) mencerminkan dominansi pola persistensi pada skala waktu 1 jam untuk variabel-variabel tersebut, sementara kelembapan relatif memiliki bobot lebih rendah (0.70) karena model diffusion menunjukkan kontribusi prediktif yang lebih signifikan pada variabel ini."*

---

### GAP-04 — Fitur Input Tidak Terdokumentasi Lengkap

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Dataset** atau **Variabel Penelitian**
- Cari paragraf yang hanya menyebutkan 3 variabel (curah hujan, kecepatan angin, kelembapan)
- Lokasi penambahan: di antara subbab Pra-pemrosesan dan subbab Representasi Graf

**Masalah**
Dokumen tidak membedakan variabel target (3 output) dan fitur input (9 conditioning features). Pembaca mengira model hanya menerima 3 variabel.

**Implementasi Kode**
- `train.py`: 9 fitur input → `temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover`, `precipitation_lag1`, `elevation`
- 3 variabel target → `precipitation`, `wind_speed_10m`, `relative_humidity_2m`

**Perbaikan yang Disarankan**
Tambahkan subbab baru **"Fitur Input dan Variabel Target"**:

> *"Model menerima 9 fitur input sebagai conditioning: temperature\_2m (°C), relative\_humidity\_2m (%), dewpoint\_2m (°C), surface\_pressure (hPa), wind\_speed\_10m (m/s), wind\_direction\_10m (°), cloud\_cover (%), precipitation\_lag1 (lag curah hujan 1-jam, mm), dan elevation (m, fitur statis per node). Variabel target prediksi terdiri dari 3 variabel: curah hujan (precipitation), kecepatan angin 10m (wind\_speed\_10m), dan kelembapan relatif 2m (relative\_humidity\_2m). Pemisahan antara fitur input (conditioning) dan variabel target penting untuk memahami arsitektur conditional diffusion model yang digunakan."*

---

### GAP-05 — Weighted Loss Tidak Terdokumentasi

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Prosedur Pelatihan**
- Cari kalimat: *"Loss utama yang digunakan dalam diffusion adalah objective denoising"*
- Tambahkan paragraf baru setelah kalimat tersebut

**Masalah**
Kode mengimplementasi weighted MSE yang memberikan bobot lebih besar pada sampel ekstrem. Ini adalah kontribusi teknis penting yang sama sekali tidak disebutkan di dokumen.

**Implementasi Kode**
- `train.py` & `diffusion.py`: `weights[targets.abs() > 1.0] = 5.0` dan `weights[targets.abs() > 3.0] = 10.0`

**Perbaikan yang Disarankan**

> *"Untuk mengatasi bias prediksi terhadap nilai rata-rata pada distribusi heavy-tailed, digunakan skema pembobotan kerugian (weighted denoising loss): sampel dengan nilai target ternormalisasi |z| > 1σ diberi bobot 5×, dan |z| > 3σ diberi bobot 10×. Strategi ini mendorong model untuk lebih sensitif terhadap deviasi signifikan dari kondisi rata-rata, yang penting untuk menangkap variasi presipitasi yang memiliki distribusi sangat miring (right-skewed)."*

---

### GAP-06 — Evaluasi Subsampling Setiap 24 Jam Tidak Terdokumentasi

**Status Perubahan**: 🟡 DISARANKAN

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Metode Evaluasi**
- Tambahkan di akhir subbab, setelah daftar metrik

**Masalah**
Evaluasi menggunakan subsampling setiap 24 jam (EVAL_STEP=24), mengurangi sampel evaluasi ~1/24 dari total. Tidak disebutkan di dokumen.

**Implementasi Kode**
- `final_proven_eval.py`: `EVAL_STEP = 24`

**Perbaikan yang Disarankan**

> *"Evaluasi pada data pengujian dilakukan dengan subsampling setiap 24 jam (daily subsampling) untuk mengurangi autokorelasi temporal antar sampel berurutan dan fokus pada representasi harian. Total sampel evaluasi yang digunakan adalah sekitar 1/24 dari keseluruhan data pengujian, mencakup periode 2022–2025."*

---

### GAP-07 — Early Stopping Diklaim tapi Tidak Diimplementasi

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Prosedur Pelatihan**
- Cari kalimat: *"mekanisme early stopping berdasarkan performa pada data validasi"*

**Masalah**
Tidak ada early stopping di kode. Training berjalan penuh 20 epoch. Hanya best-model saved.

**Implementasi Kode**
- `train.py`: Loop penuh hingga `EPOCHS=20`, checkpoint best model berdasarkan val loss

**Perbaikan yang Disarankan**
Ganti kalimat *"mekanisme early stopping"* menjadi:

> *"Model terbaik dipilih berdasarkan loss validasi terendah selama 20 epoch pelatihan (best model selection). Pendekatan ini secara fungsional serupa dengan early stopping, namun pelatihan tetap berjalan hingga selesai untuk memastikan eksplorasi loss landscape yang optimal."*

---

### GAP-08 — Graf "Berbobot" vs Implementasi Tanpa Bobot

**Status Perubahan**: 🔴 WAJIB DIUBAH

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Representasi Graf**
- Cari kalimat: *"Ketiga titik elevasi direpresentasikan sebagai node dalam suatu graf tak berarah berbobot."*

**Masalah**
Kode menggunakan graf fully-connected tanpa bobot eksplisit. Modul `builder.py` yang mengimplementasi graf berbobot tidak pernah diimpor dalam pipeline.

**Implementasi Kode**
- `temporal_loader.py`: `_build_fully_connected_edges()` tanpa atribut bobot
- `gnn.py`: `GATConv` 2-layer, 4 heads — bobot dipelajari implisit via attention
- `builder.py`: Ada di kode tapi **tidak digunakan**

**Perbaikan yang Disarankan**
Ganti kalimat *"graf tak berarah berbobot"* menjadi:

> *"Ketiga titik elevasi direpresentasikan sebagai node dalam suatu graf tak berarah fully-connected. Bobot relasi antar node dipelajari secara implisit melalui mekanisme Graph Attention (GAT) dengan 4 attention heads, yang secara otomatis menentukan pentingnya informasi dari setiap node tetangga berdasarkan kesamaan fitur meteorologis."*

---

### GAP-09 — Komentar Kode Stale ("4 variabel")

**Status Perubahan**: 🟡 DISARANKAN (perubahan kode, bukan dokumen)

**Lokasi**
- `src/models/diffusion.py` → komentar *"MULTI-OUTPUT: Predicts 4 variables"*
- `src/train.py` → komentar *"Now supports MULTI-OUTPUT: 4 target variables"*
- `src/inference.py` → komentar *"samples is [num_samples, 4]"*

**Masalah**
Komentar masih menyebut "4 variables/targets" padahal `NUM_TARGETS = 3`. Temperature sudah dikeluarkan dari target.

**Perbaikan yang Disarankan**
Search-and-replace semua komentar "4 variables"/"4 target" → "3 variables"/"3 target". Estimasi ~5 menit.

---

### GAP-10 — T_STD_MULTIPLIER = 5.0 Tidak Terdokumentasi

**Status Perubahan**: 🟡 DISARANKAN

**Lokasi Dokumen**
- **BAB III – Metodologi** → Subbab **Pra-pemrosesan**
- Tambahkan setelah paragraf tentang log-transform dan standard scaling

**Masalah**
Standar deviasi presipitasi dikalikan faktor 5× sebelum normalisasi. Teknik ini penting untuk stabilisasi diffusion tetapi tidak disebutkan.

**Implementasi Kode**
- `train.py`: `T_STD_MULTIPLIER = 5.0`

**Perbaikan yang Disarankan**

> *"Untuk menstabilkan proses diffusion pada variabel presipitasi yang telah di-log-transform (log1p), standar deviasi normalisasi diperbesar dengan faktor pengali 5.0 (T\_STD\_MULTIPLIER). Hal ini mempersempit rentang nilai ternormalisasi presipitasi agar proses denoising pada diffusion model menjadi lebih stabil."*

---

### GAP-11 — Performa Deteksi Presipitasi Ekstrem Sangat Rendah

**Status Perubahan**: 🔴 WAJIB DIUBAH (terkait reframe judul W1+W2)

**Lokasi Dokumen — Perubahan tersebar di beberapa bagian:**

1. **Halaman Judul** — ubah judul tesis
2. **BAB I – Pendahuluan** → Subbab **Latar Belakang** — ubah narasi dari "hujan lebat" ke "presipitasi probabilistik"
3. **BAB I – Pendahuluan** → Subbab **Rumusan Masalah** → **RQ3** — ubah dari "deteksi spike" ke "kuantifikasi ketidakpastian"
4. **BAB V – Pembahasan** → paragraf tentang hasil presipitasi — reframe narasi
5. **BAB V – Pembahasan** → Subbab baru **Keterbatasan Penelitian**

**Masalah**
POD presipitasi >10 mm/jam hanya 5.9%, CSI 4.2%, FAR 86.4%. Model gagal mendeteksi hujan lebat. Namun ini bukan kegagalan arsitektur — ERA5 ~25 km melakukan smoothing signifikan, dan kejadian >10 mm/jam hanya 0.15% data.

**Hasil Aktual**
- Presipitasi: POD=0.059, CSI=0.042, FAR=0.864 (sangat buruk untuk deteksi spike)
- Angin: Corr=0.833, POD=0.552 (cukup baik)
- Kelembapan: Corr=0.931, POD=0.517 (baik)
- CRPS: presipitasi 0.557, angin 1.010, kelembapan 3.275

**Perbaikan yang Disarankan**

**(a) Judul baru:**
> **"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki di Gunung Gede–Pangrango menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**

**(b) RQ3 baru:**
> *"Bagaimana kemampuan model probabilistik dalam memberikan estimasi ketidakpastian prakiraan cuaca multi-variabel melalui kerangka ensemble nowcasting?"*

**(c) Narasi pembahasan:**
Tekankan bahwa model memberikan distribusi probabilistik, bukan point forecast. Performa angin dan kelembapan menunjukkan efektivitas untuk nowcasting multi-variabel. Threshold 10 mm/jam terlalu tinggi untuk ERA5 — threshold 2–5 mm/jam lebih representatif.

**(d) Subbab Keterbatasan Penelitian** harus membahas:
1. Resolusi spasial ERA5 vs realitas orografis
2. Bias negatif ERA5 pada presipitasi ekstrem di wilayah pegunungan tropis
3. Dua node (Puncak & Lereng_Cibodas) pada grid cell ERA5 yang sama
4. Implikasi terhadap generalisasi model

---

## 3 · Bagian yang Perlu Klarifikasi agar Reviewer Tidak Bingung

Butir-butir berikut secara teknis bukan kesalahan, tetapi berpotensi menimbulkan pertanyaan dari penguji jika tidak dijelaskan dengan tepat.

### 3.1 Horizon Prediksi

**Potensi Pertanyaan**: *"Bukankah tesis ini tentang nowcasting 0–6 jam?"*

**Cara Menjelaskan di Dokumen**:
- Model menggunakan **jendela input** 6 jam (ke belakang) untuk memprediksi **1 jam** ke depan
- Ini masih termasuk nowcasting menurut definisi WMO (0–6 jam), karena model dijalankan iteratif per jam
- Lokasi: **BAB I** → Batasan Masalah, dan **BAB III** → Desain Penelitian

### 3.2 Hybrid Persistence Mendominasi (90%)

**Potensi Pertanyaan**: *"Jika bobot persistence 0.90, bukankah model Anda sebenarnya hanya persistence?"*

**Cara Menjelaskan di Dokumen**:
- Pada skala 1 jam, autokorelasi cuaca memang sangat tinggi — persistence kuat adalah hal yang **diharapkan**
- Kontribusi model diffusion ada pada **dimensi probabilistik**: ensemble spread, CRPS, uncertainty quantification
- Pure persistence tidak bisa memberikan informasi ketidakpastian — ini lah added value model
- Sertakan tabel perbandingan: pure persistence vs hybrid vs diffusion-only
- Lokasi: **BAB V** → Pembahasan

### 3.3 Dua Node pada Grid Cell ERA5 yang Sama

**Potensi Pertanyaan**: *"Kalau 2 node datanya sama, apa gunanya?"*

**Cara Menjelaskan di Dokumen**:
- Puncak (-6.77, 106.96) dan Lereng_Cibodas (-6.75, 106.99) berjarak ~3 km; ERA5 resolusi ~25 km
- Kedua node berbagi data dinamis yang sama, tetapi berbeda pada **fitur statis `elevation`**
- Ini disengaja untuk menguji apakah GNN dapat menangkap gradien elevasi
- Lokasi: **BAB III** → Wilayah Studi

### 3.4 Evaluasi Subsampling Setiap 24 Jam

**Potensi Pertanyaan**: *"Mengapa tidak evaluasi setiap jam?"*

**Cara Menjelaskan di Dokumen**:
- Subsampling mengurangi autokorelasi temporal — sampel berurutan sangat mirip
- Fokus pada representasi **harian** lebih bermakna
- Jumlah sampel tetap cukup untuk signifikansi statistik (~1,100+ sampel dari 3 tahun)
- Lokasi: **BAB III** → Metode Evaluasi

### 3.5 Target Prediksi = Rata-rata Antar Node

**Potensi Pertanyaan**: *"Mengapa tidak prediksi per-node?"*

**Cara Menjelaskan di Dokumen**:
- Target adalah rata-rata 3 node: `targets.mean(dim=0)`
- Tujuan utama: estimasi kondisi cuaca **agregat** di kawasan jalur pendakian
- Pada resolusi ERA5, 3 node merepresentasikan grid cell yang sama/berdekatan — rata-rata adalah representasi yang wajar
- Lokasi: **BAB III** → Desain Model / Arsitektur

---

## 4 · Bagian yang Tidak Lazim Secara Akademik (Tapi Masih Valid)

### 4.1 Hybrid Weight Sangat Tinggi (0.90)

**Persepsi**: Seolah model tidak berguna — 90% hasil adalah persistence biasa.

**Framing yang Tepat**:
- Pada skala temporal 1 jam, autokorelasi cuaca >0.95. Hybrid weight 0.90 adalah **rasional dan selaras** dengan literatur nowcasting
- Nilai tambah model ada pada **kuantifikasi ketidakpastian** melalui ensemble sampling
- Sertakan sitasi: literatur nowcasting yang menunjukkan persistence sebagai baseline kuat pada skala <3 jam

### 4.2 Hanya 2 Skenario (Bukan 4)

**Persepsi**: Studi ablasi tidak lengkap.

**Framing yang Tepat**:
- Perbandingan baseline deterministik vs model probabilistik lengkap sudah cukup kuat
- Analisis kualitatif kontribusi per-komponen (retrieval, GNN, hybrid) ditambahkan di BAB Pembahasan
- Banyak skripsi hanya membandingkan 1 model vs baseline

### 4.3 Evaluasi Daily Subsampling

**Persepsi**: Sampel evaluasi terlalu sedikit.

**Framing yang Tepat**:
- ~1,100+ sampel evaluasi (3 tahun × 365 hari) masih memadai secara statistik
- Mengurangi bias dari autokorelasi temporal yang tinggi pada data jam-an
- Metode ini lazim dalam evaluasi time-series forecasting

### 4.4 Target Rata-rata Node

**Persepsi**: Kehilangan informasi spasial.

**Framing yang Tepat**:
- Pada resolusi ERA5, variasi spasial antar 3 node sangat kecil (2 node pada grid cell sama)
- Rata-rata memberikan estimasi kondisi umum kawasan pendakian — sesuai tujuan mitigasi risiko
- Prediksi per-node membutuhkan data resolusi lebih tinggi — sebutkan sebagai saran riset lanjutan

### 4.5 Adam vs AdamW

**Persepsi**: Dokumen salah menyebut optimizer.

**Framing yang Tepat**:
- AdamW adalah varian Adam dengan decoupled weight decay — perbedaan minor
- Revisi di dokumen: *"optimizer AdamW (Adam with decoupled weight decay, lr=1×10⁻³, wd=1×10⁻⁴)"*
- Lokasi: **BAB III** → Prosedur Pelatihan

---

## 5 · Checklist Editing Proposal

> Gunakan checklist ini saat membuka `Draft Bepan (2).docx`. Centang setiap item setelah selesai.

### 🔴 Perubahan WAJIB (P0 — Prioritas Tertinggi)

- [ ] **W1** — Reframe judul tesis di halaman judul, abstrak, latar belakang, tujuan
  - Dari "Hujan Lebat"/"Cuaca Ekstrem" → "Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki"
  - Estimasi: ~2 jam

- [ ] **W2** — Revisi RQ3 di BAB I → Rumusan Masalah
  - Dari "deteksi spike events" → "kuantifikasi ketidakpastian prakiraan multi-variabel"
  - Estimasi: ~15 menit

- [ ] **W3** — Revisi klaim "horizon 0–6 jam" di BAB I → Batasan Masalah
  - Ubah menjadi "jendela input 6 jam + prediksi 1 jam ke depan"
  - Periksa juga BAB III untuk penyebutan serupa
  - Estimasi: ~30 menit

- [ ] **W4** — Revisi skenario eksperimen di BAB III → Alur Eksperimen
  - Ubah dari 4 skenario → 2 skenario + analisis kualitatif
  - Estimasi: ~30 menit

- [ ] **W10** — Tambah subbab "Keterbatasan Penelitian" di BAB V (Pembahasan)
  - Bahas: resolusi ERA5, 2 node pada grid cell sama, target rata-rata, threshold 10mm
  - Estimasi: ~1 jam

### 🔴 Perubahan WAJIB (P1 — Perlu Dilakukan)

- [ ] **W5** — Tambah subbab "Fitur Input dan Variabel Target" di BAB III
  - Dokumentasikan 9 fitur input dan 3 variabel target secara terpisah
  - Estimasi: ~30 menit

- [ ] **W6** — Dokumentasikan weighted denoising loss di BAB III → Prosedur Pelatihan
  - Bobot 5× untuk |z|>1σ dan 10× untuk |z|>3σ
  - Estimasi: ~15 menit

- [ ] **W7** — Ganti "early stopping" → "best model selection" di BAB III → Prosedur Pelatihan
  - Estimasi: ~5 menit

- [ ] **W8** — Ganti "graf berbobot" → "graf fully-connected + GAT attention" di BAB III → Representasi Graf
  - Estimasi: ~10 menit

- [ ] **W9** — Tampilkan hybrid weights (0.90/0.90/0.70) eksplisit di BAB III → Hybrid Persistence
  - Estimasi: ~15 menit

### 🟡 Perubahan DISARANKAN (P2 — Memperkuat Kualitas)

- [ ] **D1** — Dokumentasikan T_STD_MULTIPLIER=5.0 di BAB III → Pra-pemrosesan
- [ ] **D2** — Jelaskan EVAL_STEP=24 di BAB III → Metode Evaluasi
- [ ] **D3** — Jelaskan target rata-rata antar 3 node di BAB III → Desain Model
- [ ] **D4** — Jelaskan 2 node pada grid cell yang sama di BAB III → Wilayah Studi
- [ ] **D5** — Sebutkan hyperparameter spesifik (SEQ_LEN=6, BATCH_SIZE=512, dll)
- [ ] **D6** — Jelaskan DDIM 20-step vs DDPM 1000-step di BAB III → Inferensi
- [ ] **D7** — Bahas dominansi persistence secara kritis di BAB V → Pembahasan

### 🟢 Perubahan Kode Opsional (P3)

- [ ] **D8** — Ganti komentar "4 variables" → "3 variables" di kode (~5 menit)
- [ ] **D9** — Tambah threshold evaluasi 2 mm & 5 mm di `final_proven_eval.py` (~15 menit)

---

## 6 · Elemen yang Sudah Konsisten (Tidak Perlu Diubah)

Untuk referensi, 13 aspek berikut sudah **selaras antara dokumen dan kode**:

| # | Aspek | Status |
|---|-------|--------|
| 1 | Variabel target = 3 (curah hujan, angin, kelembapan) | ✅ |
| 2 | Sumber data ERA5 via Open-Meteo API | ✅ |
| 3 | Tiga node: Puncak, Lereng, Hilir | ✅ |
| 4 | Temporal split: Train 2005–2018, Val 2019–2021, Test 2022–2025 | ✅ |
| 5 | Log-transform presipitasi | ✅ |
| 6 | Arsitektur conditional diffusion | ✅ |
| 7 | Retrieval via FAISS (IndexFlatL2) | ✅ |
| 8 | GNN dengan GATConv + global_mean_pool | ✅ |
| 9 | Baseline MLP dengan MSE loss | ✅ |
| 10 | Ensemble sampling (num_samples parameter) | ✅ |
| 11 | Metrik evaluasi lengkap (MAE, RMSE, Corr, CRPS, Brier, POD, FAR, CSI) | ✅ |
| 12 | Normalisasi dari training data saja | ✅ |
| 13 | Triple conditioning (temporal + retrieval + graph) | ✅ |

---

## 7 · Kekuatan yang Menonjol (Bisa Ditonjolkan ke Reviewer)

1. **Anti-leakage procedures** — temporal split kronologis, normalisasi dari training saja, FAISS index dari training saja. Ini melampaui standar rata-rata skripsi
2. **Metrik probabilistik** — CRPS dan Brier Score jarang digunakan di skripsi Indonesia. Ini menunjukkan kedalaman metodologi
3. **Arsitektur triple-conditioning** — integrasi retrieval, graph, dan temporal attention dalam satu framework coherent
4. **Weighted denoising loss** — kontribusi teknis yang mengatasi class imbalance pada distribusi heavy-tail (wajib didokumentasikan!)
5. **Performa angin dan kelembapan** — Wind Corr=0.833, Humidity Corr=0.931 menunjukkan model efektif untuk nowcasting multi-variabel

---

## 8 · Estimasi Effort

| Prioritas | Perkiraan Waktu | Item |
|-----------|----------------|------|
| 🔴 P0 | ~4.5 jam | W1, W2, W3, W4, W10 |
| 🔴 P1 | ~1.5 jam | W5, W6, W7, W8, W9 |
| 🟡 P2 | ~1 jam | D1–D7 |
| 🟢 P3 | ~20 menit | D8, D9 |
| **Total** | **~7.5 jam** | |

> **Strategi editing**: Kerjakan semua item P0 terlebih dahulu (terutama reframe judul W1 — ini mengubah tone seluruh dokumen). Setelah P0 selesai, lanjutkan P1. Item P2 dan P3 bisa dikerjakan setelah seluruh perubahan wajib selesai.

---

*Panduan ini disusun berdasarkan cross-referencing line-by-line antara dokumen proposal dan seluruh source code (>3000 baris), termasuk evaluation results dan model configurations.*
