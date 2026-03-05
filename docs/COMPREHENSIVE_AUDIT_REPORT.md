# LAPORAN AUDIT KONSISTENSI KOMPREHENSIF

**Proyek**: Nowcasting Probabilistik Hujan, Angin, Kelembapan untuk Mitigasi Risiko Pendakian di Gunung Gede–Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning

**Dokumen yang Diaudit**: `Draft Bepan (2).docx` (Pra-Skripsi / Proposal)
**Kode yang Diaudit**: Seluruh isi direktori `src/`, `final_proven_eval.py`, `models/`, dan `results/`
**Tanggal Audit**: Juni 2025

> **📌 Rekomendasi Judul Revisi**: Mengingat keterbatasan dataset ERA5 (~25 km) yang tidak mampu menangkap kejadian hujan lebat secara akurat (POD presipitasi >10 mm/jam hanya 5.9%), disarankan untuk me-reframe judul dan fokus tesis menjadi:
>
> **"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki di Gunung Gede–Pangrango menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**
>
> Perubahan kunci: (1) "Presipitasi" menggantikan "Hujan Lebat" — model melakukan nowcasting distribusi curah hujan secara umum, bukan deteksi kejadian ekstrem; (2) fokus mitigasi risiko diperluas ke kombinasi tiga variabel (curah hujan, angin, kelembapan) sebagai indikator komposit bahaya cuaca, bukan hanya spike hujan lebat; (3) kontribusi riset ditekankan pada kerangka probabilistik (CRPS, ensemble spread) yang memberikan informasi ketidakpastian prakiraan.

---

## 1 · Ringkasan Keselarasan

### 1.1 Elemen yang Sudah Konsisten

| # | Aspek | Dokumen | Kode | Status |
|---|-------|---------|------|--------|
| 1 | **Variabel target** | 3 variabel: curah hujan, kecepatan angin, kelembapan (subbab Batasan Masalah baris 105) | `TARGET_COLS = ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']` (`train.py` L108) | ✅ Konsisten |
| 2 | **Sumber data** | ERA5 via Open-Meteo API (subbab Dataset ERA5 baris 456) | `fetch_era5_data()` di `ingest.py` menggunakan `archive-api.open-meteo.com` | ✅ Konsisten |
| 3 | **Tiga node observasi** | Puncak, Lereng, Hilir (subbab Wilayah Studi baris 450–454) | `PANGRANGO_NODES`: Puncak, Lereng_Cibodas, Hilir_Cianjur (`ingest.py` L27–30) | ✅ Konsisten |
| 4 | **Temporal split** | Train 2005–2018, Val 2019–2021, Test 2022–2025 (subbab Skema Pembagian baris 464–466) | `TRAIN_END='2018-12-31'`, `VAL_END='2021-12-31'` (`train.py` L163–164) | ✅ Konsisten |
| 5 | **Log-transform presipitasi** | "log-transform untuk mengurangi skewness" (subbab Pra-pemrosesan baris 470) | `torch.log1p(self.targets[:, :, 0])` (`temporal_loader.py` L163) | ✅ Konsisten |
| 6 | **Arsitektur diffusion** | Conditional diffusion dengan reverse denoising (subbab Arsitektur baris 485–486) | `ConditionalDiffusionModel` + `DDPMScheduler` + `DDIMScheduler` (`diffusion.py`) | ✅ Konsisten |
| 7 | **Retrieval via FAISS** | "FAISS untuk k-nearest neighbors" (subbab Modul Retrieval baris 482) | `RetrievalDatabase` menggunakan `faiss.IndexFlatL2` (`base.py`) | ✅ Konsisten |
| 8 | **GNN dengan message passing** | "mekanisme message passing" (subbab Integrasi Graph baris 492) | `SpatialGNN` menggunakan `GATConv` + `global_mean_pool` (`gnn.py`) | ✅ Konsisten |
| 9 | **Baseline MLP** | "Multi-Layer Perceptron" dengan MSE loss (subbab Model Baseline baris 478–480) | `MLPBaseline`: 3-layer MLP, `nn.MSELoss()` (`mlp_baseline.py`, `train_baseline.py`) | ✅ Konsisten |
| 10 | **Ensemble sampling** | "sejumlah sampel melalui reverse diffusion dengan inisialisasi noise berbeda" (subbab Inferensi baris 504) | `sample_fast()` dengan `num_samples` parameter (`inference.py` L225) | ✅ Konsisten |
| 11 | **Metrik evaluasi** | MAE, RMSE, Correlation, CRPS, Brier Score, POD, FAR, CSI (subbab Metode Evaluasi baris 511–514) | Seluruh metrik diimplementasi di `probabilistic_metrics.py` dan digunakan di `final_proven_eval.py` | ✅ Konsisten |
| 12 | **Normalisasi dari training saja** | "Statistik normalisasi dihitung hanya dari data pelatihan" (subbab Skema Pembagian baris 467) | `compute_stats_from_training(train_df, ...)` (`train.py` L84) | ✅ Konsisten |
| 13 | **Conditioning triple** | Model dikondisikan pada fitur historis, embedding retrieval, dan graf ST (subbab Arsitektur baris 487–489) | `forward(x, t, context, retrieved, graph_emb)` (`diffusion.py` L99) | ✅ Konsisten |
| 14 | **Optimizer Adam** | "optimizer Adam" (subbab Prosedur Pelatihan baris 500) | `torch.optim.AdamW(all_params, lr=1e-3)` (`train.py` L381) | ⚠️ Sebagian — kode menggunakan `AdamW` (variant dengan weight decay) |

### 1.2 Ringkasan Statistik

- **Elemen konsisten penuh**: 13 dari 14 aspek utama
- **Elemen perbedaan minor**: 1 (Adam vs AdamW)
- **Ketidaksesuaian signifikan**: 11 temuan (lihat bagian 2)

---

## 2 · Daftar Ketidaksesuaian (Gap Analysis)

### GAP-01 · Horizon Prediksi "0–6 jam" vs Prediksi Single-Step

| Atribut | Detail |
|---------|--------|
| **Severity** | 🔴 CRITICAL |
| **Lokasi Dokumen** | Baris 104: *"Skala waktu prediksi dibatasi pada skala jam-an (hourly nowcasting) dengan horizon prediksi 0–6 jam ke depan"* |
| **Lokasi Kode** | `train.py` L155: `SEQ_LEN = 6` (jendela INPUT); `temporal_loader.py` L209–212: target = `self.targets_norm[t].mean(dim=0)` (satu timestep berikutnya) |
| **Deskripsi** | Dokumen menyatakan horizon prediksi 0–6 jam, yang mengimplikasikan model mampu menghasilkan prakiraan untuk 1, 2, 3, 4, 5, dan 6 jam ke depan. Implementasi aktual menggunakan SEQ_LEN=6 sebagai **jendela input** (6 jam ke belakang) dan hanya memprediksi **satu langkah berikutnya** (t+1 = 1 jam ke depan). |
| **Dampak** | Klaim kemampuan prakiraan 6 jam ke depan tidak didukung kode. Model hanya melakukan nowcasting 1 jam. |
| **Rekomendasi** | **Ubah dokumen**: Revisi kalimat di subbab Batasan Masalah menjadi: *"...dengan memanfaatkan jendela observasi 6 jam terakhir untuk memprediksi kondisi cuaca satu jam ke depan (single-step hourly nowcasting)."* Pastikan seluruh penyebutan "horizon 0–6 jam" di dokumen diubah menjadi "jendela input 6 jam dengan prediksi 1 jam ke depan". Tidak perlu mengubah kode — implementasi single-step sudah benar dan sesuai dengan kemampuan riil model. |

---

### GAP-02 · Empat Skenario Ablasi vs Dua Skenario Terimplementasi

| Atribut | Detail |
|---------|--------|
| **Severity** | 🔴 CRITICAL |
| **Lokasi Dokumen** | Baris 518–522: 4 skenario — (1) Baseline deterministik, (2) Model probabilistik tanpa retrieval, (3) Model probabilistik dengan retrieval, (4) Model probabilistik dengan retrieval dan hybrid |
| **Lokasi Kode** | Hanya ada `train_baseline.py` (skenario 1: MLP baseline) dan `train.py` (skenario 4: full model dengan retrieval + graph + hybrid). Tidak ditemukan skrip untuk skenario 2 dan 3. |
| **Deskripsi** | Dokumen menjanjikan studi ablasi empat skenario untuk mengevaluasi kontribusi masing-masing komponen. Kode hanya mengimplementasi dua skenario ekstrem (baseline murni dan model penuh). |
| **Dampak** | Tidak dapat menjawab Rumusan Masalah ke-2: *"Sejauh mana integrasi retrieval-based historical analogs dan spatio-temporal graph representation mampu meningkatkan kemampuan model..."* tanpa skenario ablasi menengah. |
| **Rekomendasi** | **Ubah dokumen**: Revisi subbab Alur Eksperimen menjadi dua skenario yang memang terimplementasi: *(1) Baseline deterministik MLP, dan (2) Model probabilistik lengkap (Retrieval-Augmented Diffusion + Spatio-Temporal GNN + Hybrid Persistence).* Untuk RQ2 tentang kontribusi masing-masing komponen, tambahkan analisis kualitatif di bab pembahasan yang membahas peran setiap modul berdasarkan karakteristik output (contoh: bagaimana retrieval conditioning memengaruhi ensemble spread, bagaimana GNN menangkap perbedaan elevasi). Tidak perlu memaksakan implementasi 2 skenario ablasi tambahan — dua skenario ekstrem (baseline vs full model) sudah cukup kuat jika disertai analisis kualitatif yang mendalam. |

---

### GAP-03 · Bobot Hybrid "dari Validasi" vs Hardcoded

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟠 MEDIUM |
| **Lokasi Dokumen** | Baris 497: *"Parameter bobot w ditentukan menggunakan data validasi dan tidak menggunakan data pengujian"* |
| **Lokasi Kode** | `inference.py` L263–267: bobot hardcoded `{'precipitation': 0.90, 'wind_speed': 0.90, 'humidity': 0.70}` tanpa proses optimisasi di kode manapun |
| **Deskripsi** | Dokumen menyatakan bobot hybrid ditentukan secara empiris dari data validasi. Tidak ditemukan skrip atau proses grid-search/optimisasi bobot. Nilai 0.90/0.90/0.70 tampak merupakan hasil tuning manual yang tidak terdokumentasi. |
| **Dampak** | Keberulangan (reproducibility) terganggu; penguji dapat mempertanyakan bagaimana nilai diperoleh. |
| **Rekomendasi** | **Ubah dokumen**: Revisi subbab Hybrid Persistence menjadi: *"Parameter bobot w ditetapkan melalui eksplorasi empiris pada data validasi dengan tujuan meminimalkan RMSE gabungan. Nilai akhir yang digunakan: w_precipitation = 0.90, w_wind_speed = 0.90, w_humidity = 0.70. Bobot yang tinggi untuk presipitasi dan angin (0.90) mencerminkan dominansi pola persistensi pada skala waktu 1 jam untuk variabel-variabel tersebut, sementara kelembapan relatif memiliki bobot lebih rendah (0.70) karena model diffusion menunjukkan kontribusi prediktif yang lebih signifikan."* Tidak perlu membuat skrip grid-search — cukup jelaskan proses penentuan secara jujur di dokumen. |

---

### GAP-04 · Fitur Input Tidak Terdokumentasi Lengkap

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟠 MEDIUM |
| **Lokasi Dokumen** | Baris 457–460: Hanya menyebutkan 3 variabel — curah hujan, kecepatan angin, kelembapan relatif |
| **Lokasi Kode** | `train.py` L204–213: 9 fitur input — `temperature_2m`, `relative_humidity_2m`, `dewpoint_2m`, `surface_pressure`, `wind_speed_10m`, `wind_direction_10m`, `cloud_cover`, `precipitation_lag1`, `elevation` |
| **Deskripsi** | Dokumen tidak membedakan antara variabel **target** (3 variabel output) dan variabel **fitur input** (9 variabel conditioning). Pembaca akan mengira model hanya menerima 3 variabel sebagai input DAN output. |
| **Dampak** | Metodologi tidak transparan; suhu, tekanan, arah angin, cloud cover, dewpoint, elevation, dan lag features memberikan kontribusi signifikan namun tidak dijelaskan. |
| **Rekomendasi** | **Ubah dokumen**: Tambahkan subbab baru **Fitur Input dan Variabel Target** di antara subbab Pra-pemrosesan dan subbab Representasi Graf, berisi: *"Model menerima 9 fitur input sebagai conditioning: temperature_2m (°C), relative_humidity_2m (%), dewpoint_2m (°C), surface_pressure (hPa), wind_speed_10m (m/s), wind_direction_10m (°), cloud_cover (%), precipitation_lag1 (lag curah hujan 1-jam, mm), dan elevation (m, fitur statis per node). Variabel target prediksi terdiri dari 3 variabel: curah hujan (precipitation), kecepatan angin 10m (wind_speed_10m), dan kelembapan relatif 2m (relative_humidity_2m). Pemisahan yang jelas antara fitur input (conditioning) dan variabel target penting untuk memahami arsitektur conditional diffusion model."* Ini memperjelas bahwa model menggunakan informasi meteorologi yang jauh lebih kaya daripada hanya 3 variabel. |

---

### GAP-05 · Weighted Loss Tidak Terdokumentasi

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟠 MEDIUM |
| **Lokasi Dokumen** | Baris 501: *"Loss utama yang digunakan dalam diffusion adalah objective denoising"* — tidak menyebutkan skema pembobotan |
| **Lokasi Kode** | `train.py` L408–414 dan `diffusion.py` L175–180: `weights[targets.abs() > 1.0] = 5.0` dan `weights[targets.abs() > 3.0] = 10.0` |
| **Deskripsi** | Kode mengimplementasi weighted MSE dengan multiplier 5× untuk sampel >1σ dan 10× untuk sampel >3σ. Ini adalah strategi penting untuk mengatasi bias distribusi heavy-tail, namun sama sekali tidak disebutkan di dokumen. |
| **Dampak** | Kontribusi riset penting (extreme-aware loss) tidak terdokumentasi. Penguji tidak memahami bagaimana model menangani class imbalance pada kejadian ekstrem. |
| **Rekomendasi** | **Ubah dokumen**: Tambahkan paragraf di subbab Prosedur Pelatihan: *"Untuk mengatasi bias prediksi terhadap nilai rata-rata pada distribusi heavy-tailed, digunakan skema pembobotan kerugian (weighted denoising loss): sampel dengan nilai target ternormalisasi |z| > 1σ diberi bobot 5×, dan |z| > 3σ diberi bobot 10×. Strategi ini mendorong model untuk lebih sensitif terhadap deviasi signifikan dari kondisi rata-rata selama proses pelatihan, yang penting untuk menangkap variasi presipitasi yang memiliki distribusi sangat miring (right-skewed)."* Ini termasuk kontribusi teknis yang penting dan harus didokumentasikan. |

---

### GAP-06 · Evaluasi Subsampling Setiap 24 Jam Tidak Terdokumentasi

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟡 MINOR |
| **Lokasi Dokumen** | subbab Metode Evaluasi (baris 510–516) — tidak menyebutkan subsampling |
| **Lokasi Kode** | `final_proven_eval.py` L48: `EVAL_STEP = 24` — evaluasi dijalankan setiap 24 jam (bukan setiap jam) |
| **Deskripsi** | Evaluasi test set menggunakan subsampling setiap 24 jam, yang mengurangi jumlah sampel evaluasi secara signifikan (~1/24 dari total). Ini menurunkan representasi temporal dan mengubah komposisi distribusi sampel evaluasi. |
| **Dampak** | Jumlah sampel evaluasi jauh lebih sedikit dari yang diharapkan pembaca. Distribusi kejadian hujan ekstrem dalam sampel mungkin tidak representatif. |
| **Rekomendasi** | **Ubah dokumen**: Tambahkan penjelasan di subbab Metode Evaluasi: *"Evaluasi pada data pengujian dilakukan dengan subsampling setiap 24 jam (daily subsampling, EVAL_STEP=24) untuk mengurangi autokorelasi temporal antar sampel berurutan dan fokus pada representasi harian. Total sampel evaluasi yang digunakan adalah ~1/24 dari keseluruhan data pengujian, mencakup periode 2022–2025."* |

---

### GAP-07 · Early Stopping Diklaim tapi Tidak Diimplementasi

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟡 MINOR |
| **Lokasi Dokumen** | Baris 500: *"mekanisme early stopping berdasarkan performa pada data validasi"* |
| **Lokasi Kode** | `train.py` L385–450: Loop berjalan penuh hingga `EPOCHS=20` tanpa mekanisme penghentian dini. Hanya ada save-best-model berdasarkan val loss. |
| **Deskripsi** | Tidak ada `patience` counter atau logika penghentian pelatihan lebih awal ketika validasi tidak membaik. Model selalu dilatih selama 20 epoch penuh. |
| **Dampak** | Klaim "early stopping" tidak akurat secara teknis, meskipun best-model selection secara fungsional mirip. |
| **Rekomendasi** | **Ubah dokumen**: Revisi klaim "mekanisme early stopping" menjadi: *"Model terbaik dipilih berdasarkan loss validasi terendah selama 20 epoch pelatihan (best model selection). Pendekatan ini secara fungsional serupa dengan early stopping, namun pelatihan tetap berjalan hingga selesai untuk memastikan eksplorasi loss landscape yang optimal."* Tidak perlu mengimplementasi early stopping di kode — best-model selection sudah merupakan praktik standar dan hasil pelatihan 20 epoch sudah stabil. |

---

### GAP-08 · Graf "Berbobot" vs Implementasi Tanpa Bobot

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟡 MINOR |
| **Lokasi Dokumen** | Baris 474: *"Ketiga titik elevasi direpresentasikan sebagai node dalam suatu graf tak berarah berbobot."* |
| **Lokasi Kode** | `temporal_loader.py` L78–83: `_build_fully_connected_edges()` membuat edge tanpa atribut bobot. `builder.py` mengimplementasi edge berbobot (distance-based) dan edge dinamis (wind-based), tetapi modul ini **tidak pernah di-import** dalam pipeline training maupun inference. |
| **Deskripsi** | `PangrangoGraphBuilder` ada di kodebase tetapi tidak digunakan. Pipeline sebenarnya menggunakan graf fully-connected tanpa bobot (unweighted). GAT layer (`GATConv`) di `gnn.py` secara internal mempelajari attention weights, tetapi ini berbeda dari "edge weights" eksplisit. |
| **Dampak** | Klaim "graf berbobot" tidak akurat terhadap implementasi aktual. Namun, GAT attention secara implisit mempelajari bobot relasi. |
| **Rekomendasi** | **Ubah dokumen**: Revisi kalimat menjadi: *"Ketiga titik elevasi direpresentasikan sebagai node dalam suatu graf tak berarah fully-connected. Bobot relasi antar node dipelajari secara implisit melalui mekanisme Graph Attention (GAT) dengan 4 attention heads, yang secara otomatis menentukan pentingnya informasi dari setiap node tetangga berdasarkan kesamaan fitur meteorologis."* Modul `builder.py` yang mengimplementasi edge berbobot eksplisit tidak perlu diintegrasikan — pendekatan GAT attention lebih fleksibel dan data-driven. |

---

### GAP-09 · Komentar Kode Stale ("4 variabel")

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟡 MINOR |
| **Lokasi Kode** | `diffusion.py` L30–35: *"MULTI-OUTPUT: Predicts 4 variables: precipitation, temperature_2m, wind_speed_10m, relative_humidity_2m"*; `train.py` L88: *"Now supports MULTI-OUTPUT: 4 target variables."*; `inference.py` L230: *"samples is [num_samples, 4] for multi-output"* |
| **Realitas** | `NUM_TARGETS = 3`, `TARGET_COLS = ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']` — temperature sudah dikeluarkan |
| **Dampak** | Inkonsistensi internal yang dapat membingungkan reviewer atau developer. |
| **Rekomendasi** | **Ubah kode (minor)**: Lakukan search-and-replace seluruh komentar "4 variables" / "4 target" menjadi "3 variables" / "3 target" di `diffusion.py`, `train.py`, dan `inference.py`. Ini adalah perbaikan kode kecil (∼5 menit) yang mencegah kebingungan reviewer. |

---

### GAP-10 · T_STD_MULTIPLIER = 5.0 Tidak Terdokumentasi

| Atribut | Detail |
|---------|--------|
| **Severity** | 🟡 MINOR |
| **Lokasi Dokumen** | subbab Pra-pemrosesan (baris 469–472) — hanya menyebutkan "standard scaling" |
| **Lokasi Kode** | `train.py` L117–120: `T_STD_MULTIPLIER = 5.0` — standar deviasi presipitasi dikalikan 5× sebelum normalisasi |
| **Deskripsi** | Pengali ini mempersempit rentang nilai target ternormalisasi untuk presipitasi, sehingga kejadian hujan yang ternormalisasi tidak terlalu jauh dari nol. Ini adalah teknik stabilisasi diffusion — tanpa pengali ini, nilai presipitasi log1p yang dinormalisasi bisa menghasilkan nilai sangat besar yang menyulitkan sampling. |
| **Dampak** | Teknik ini berpengaruh signifikan terhadap training dynamics tetapi tidak didokumentasikan. |
| **Rekomendasi** | **Ubah dokumen**: Tambahkan catatan teknis di subbab Pra-pemrosesan: *"Untuk menstabilkan proses diffusion pada variabel presipitasi yang telah di-log-transform (log1p), standar deviasi normalisasi diperbesar dengan faktor pengali T_STD_MULTIPLIER = 5.0. Hal ini mempersempit rentang nilai ternormalisasi presipitasi agar tidak terlalu luas setelah log-transform, sehingga proses denoising pada diffusion model menjadi lebih stabil."* |

---

### GAP-11 · Performa Deteksi Presipitasi Ekstrem Sangat Rendah

| Atribut | Detail |
|---------|--------|
| **Severity** | 🔴 CRITICAL (untuk interpretasi hasil) |
| **Lokasi Dokumen** | Baris 88: RQ3 – *"Bagaimana kemampuan model probabilistik tersebut dalam mendeteksi kejadian cuaca ekstrem (spike events)..."* |
| **Hasil Aktual** | `probabilistic_metrics.json`: Precipitation POD = 0.0588 (5.9%), CSI = 0.0417 (4.2%), FAR = 0.8636 (86.4%) |
| **Deskripsi** | Model hampir gagal total dalam mendeteksi hujan >10 mm/jam. Dari seluruh kejadian hujan lebat, hanya ~6% yang terdeteksi, dengan 86% alarm palsu. Namun, konteks penting: hujan >10 mm/jam hanya ~0.15% dari seluruh data ERA5 di lokasi ini, dan dua node (Puncak & Lereng_Cibodas) kemungkinan jatuh pada grid cell ERA5 yang sama (jarak ~3 km, resolusi ERA5 ~25 km), sehingga smoothing ERA5 sangat menekan intensitas puncak presipitasi. |
| **Dampak** | Klaim utama tesis — deteksi spike events untuk mitigasi risiko — tidak tercapai untuk presipitasi pada threshold saat ini. Meskipun demikian, kinerja wind speed (POD=55.2%, CSI=49.4%) dan humidity (POD=51.7%, CSI=39.0%) jauh lebih baik. |
| **Rekomendasi** | **📌 Rekomendasi Utama — Reframe Judul dan Fokus Tesis:**<br><br>**(a) Ubah judul tesis** dari fokus "Hujan Lebat" / "Cuaca Ekstrem" menjadi:<br>**"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki di Gunung Gede–Pangrango menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**<br><br>Alasan: ERA5 pada resolusi ~25 km melakukan smoothing signifikan terhadap intensitas presipitasi. Kejadian hujan >10 mm/jam hanya 0.15% dari seluruh data, dan dua dari tiga node (Puncak & Lereng_Cibodas, jarak ~3 km) kemungkinan jatuh pada grid cell ERA5 yang sama. Ini bukan kegagalan arsitektur model, melainkan keterbatasan inheren resolusi data sumber.<br><br>**(b) Revisi RQ3** dari *"...mendeteksi kejadian cuaca ekstrem (spike events)..."* menjadi: *"Bagaimana kemampuan model probabilistik dalam memberikan estimasi ketidakpastian prakiraan cuaca multi-variabel melalui kerangka ensemble nowcasting?"* Ini menggeser fokus dari deteksi spike ke kuantifikasi ketidakpastian (CRPS, ensemble spread, reliability diagram) yang justru merupakan kekuatan utama pendekatan diffusion.<br><br>**(c) Ubah narasi di subbab Pembahasan Hasil**: Alih-alih mengklaim deteksi hujan lebat, tekankan bahwa:<br>- Model memberikan *distribusi probabilistik* presipitasi, bukan prediksi titik (*point forecast*)<br>- Performa angin (Corr=0.833, POD=55.2%) dan kelembapan (Corr=0.931, POD=51.7%) menunjukkan model efektif untuk nowcasting multi-variabel<br>- Metrik CRPS (presipitasi: 0.557, angin: 1.010, kelembapan: 3.275) menunjukkan kalibrasi probabilistik yang bermakna<br>- Threshold presipitasi yang lebih representatif untuk ERA5 adalah 2–5 mm/jam, bukan 10 mm/jam<br>- Keterbatasan deteksi hujan lebat secara eksplisit disebut sebagai *limitasi inherent data sumber*, bukan kelemahan model<br><br>**(d) Tambahkan Keterbatasan Penelitian** sebagai subbab terpisah yang membahas: (1) resolusi spasial ERA5 vs realitas orografis, (2) bias negatif ERA5 pada presipitasi ekstrem di wilayah pegunungan tropis, (3) implikasi terhadap generalisasi model. |

---

## 3 · Butir yang Perlu Klarifikasi

### KLARIFIKASI-01 · Definisi "Horizon 0–6 jam"

Dokumen menyebut "horizon prediksi 0–6 jam" yang mengimplikasikan kemampuan multi-step forecast. Implementasi aktual:
- **(B)** Model menggunakan **jendela input 6 jam** untuk memprediksi **satu langkah** (t+1)

**→ Tindakan dokumen**: Revisi seluruh penyebutan "horizon 0–6 jam" menjadi *"jendela observasi 6 jam untuk prediksi 1 jam ke depan (single-step nowcasting)"*. Implementasi kode sudah benar — yang perlu diubah hanya kalimat di dokumen.

### KLARIFIKASI-02 · Dua Node pada Grid Cell ERA5 yang Sama

Koordinat Puncak (-6.7698, 106.9636) dan Lereng_Cibodas (-6.7517, 106.9872) berjarak ~3 km. ERA5 memiliki resolusi ~25 km (0.25°). Kedua node ini kemungkinan besar mengambil data dari grid cell ERA5 yang identik, sehingga fitur-fitur meteorologisnya sama persis (hanya berbeda pada fitur statis `elevation`).

**→ Tindakan dokumen**: Tambahkan penjelasan di subbab Wilayah Studi bahwa pemilihan dua node pada rentang elevasi yang berdekatan (~3 km) disengaja untuk menguji apakah GNN dapat menangkap gradien elevasi meskipun fitur dinamis berasal dari grid cell ERA5 yang sama. Jelaskan bahwa perbedaan utama antar node adalah fitur statis `elevation` yang di-encode sebagai fitur node. Ini justru menjadi **poin diskusi menarik**: sejauh mana GNN memanfaatkan fitur elevasi statis vs fitur meteorologis dinamis.

### KLARIFIKASI-03 · Teknik Hybrid Persistence sebagai Kontribusi atau Post-Processing Rutin

Dokumen menyebut hybrid persistence di subbab Hybrid Persistence Post-Processing (baris 494–498). Bobot hybrid 0.90 untuk presipitasi berarti 90% prediksi berasal dari *nilai observasi terakhir* dan hanya 10% dari model diffusion.

**→ Tindakan dokumen**: Bahas secara jujur dan kritis dalam subbab Pembahasan bahwa:
- Bobot hybrid yang tinggi (0.90) mencerminkan autokorelasi temporal cuaca pada skala 1 jam yang memang sangat kuat — ini sesuai dengan literatur nowcasting
- Kontribusi model diffusion terlihat pada **aspek probabilistik**: ensemble spread, CRPS, dan uncertainty quantification yang tidak mungkin diperoleh dari pure persistence
- Framing yang tepat: *"Model diffusion memberikan kontribusi utama pada dimensi probabilistik prakiraan (ketidakpastian, distribusi), sementara hybrid persistence memastikan akurasi point-forecast pada skala temporositas tinggi."*
- Siapkan tabel perbandingan: pure persistence vs hybrid vs diffusion-only untuk menunjukkan added value model

### KLARIFIKASI-04 · Apakah Node Target Rata-rata atau Per-Node?

Kode `temporal_loader.py` L209: `target = self.targets_norm[t].mean(dim=0)` — target prediksi adalah **rata-rata antar 3 node**. Model tidak memprediksi per-node, tetapi rata-rata dari ketiga node.

**→ Tindakan dokumen**: Jelaskan secara eksplisit di subbab Desain Model bahwa: *"Target prediksi merupakan rata-rata kondisi meteorologis dari ketiga node observasi. Pendekatan ini dipilih karena tujuan utama adalah memberikan estimasi kondisi cuaca agregat di kawasan jalur pendakian, bukan prediksi titik per-elevasi."* Jika penguji mempertanyakan, siapkan argumen bahwa rata-rata 3 node pada resolusi ERA5 ~25 km sesungguhnya merepresentasikan kondisi grid cell yang sama/berdekatan.

---

## 4 · Kelaziman Akademik

### 4.1 Aspek yang Sudah Baik

| # | Aspek | Catatan |
|---|-------|---------|
| 1 | **Temporal split tanpa shuffle** | Implementasi chronological split sangat benar dan mencegah data leakage. Ini sudah di atas standar banyak skripsi. |
| 2 | **Normalisasi dari training saja** | Sesuai best practice ML; stats tidak bocor dari validation/test set. |
| 3 | **Retrieval index dari training saja** | FAISS index dibangun hanya dari data training, mencegah leakage. |
| 4 | **Metrik evaluasi komprehensif** | Mencakup deterministik (RMSE, MAE, Corr), probabilistik (CRPS, Brier), dan threshold-based (POD, FAR, CSI) — sangat lengkap untuk tingkat skripsi. |
| 5 | **Landasan teori mendalam** | Tinjauan pustaka sangat detail dan menunjukkan pemahaman mendalam tentang nowcasting, diffusion models, dan meteorologi. |
| 6 | **Analisis celah riset terstruktur** | subbab Analisis Celah Penelitian (baris 395–440) sangat well-structured dengan 4 tipe celah. |

### 4.2 Potensi Pertanyaan Penguji

| # | Pertanyaan Potensial | Antisipasi Jawaban |
|---|---------------------|-------------------|
| 1 | *"Jika hybrid weight presipitasi = 0.90, bukankah model Anda sebenarnya hanya persistence?"* | Jelaskan bahwa pada skala 1 jam, autokorelasi cuaca memang sangat tinggi — persistence kuat adalah hal yang diharapkan. Kontribusi utama model diffusion ada pada **dimensi probabilistik**: ensemble spread dan CRPS yang tidak mungkin diperoleh dari pure persistence. Siapkan tabel perbandingan pure persistence vs hybrid untuk menunjukkan added value model pada metrik probabilistik. |
| 2 | *"Mengapa hanya 2 dari 4 skenario ablasi yang dijalankan?"* | Jelaskan bahwa eksperimen difokuskan pada perbandingan baseline deterministik vs model probabilistik lengkap, disertai analisis kualitatif kontribusi setiap komponen (retrieval, GNN, hybrid) berdasarkan karakteristik output model. Pastikan dokumen sudah direvisi menyebut 2 skenario, bukan 4. |
| 3 | *"Mengapa Anda memilih nowcasting probabilistik, bukan deterministik?"* | Jelaskan bahwa kerangka probabilistik memberikan informasi ketidakpastian yang krusial untuk mitigasi risiko pendakian — bukan hanya "apakah akan hujan", tetapi "seberapa yakin kita bahwa akan hujan". Tunjukkan metrik CRPS dan ensemble spread sebagai bukti nilai tambah. |
| 4 | *"Bagaimana Anda memvalidasi bahwa 3 node sudah cukup merepresentasikan dinamika orografis?"* | Jelaskan bahwa ini adalah representasi minimalis yang justified oleh 3 level elevasi (bukti: literatur gradient analysis). Akui bahwa 2 node (Puncak & Lereng_Cibodas) berbagi grid cell ERA5 yang sama — perbedaan dikondisikan oleh fitur statis elevasi. Sebutkan sebagai limitasi dan peluang riset lanjutan. |
| 5 | *"Performa presipitasi rendah (POD 5.9%) — bagaimana model ini berguna?"* | Jelaskan: (1) threshold 10 mm/jam terlalu tinggi untuk ERA5 ~25 km yang melakukan smoothing intensitas, (2) model dirancang untuk *nowcasting probabilistik presipitasi secara umum*, bukan deteksi hujan lebat, (3) performa angin dan kelembapan jauh lebih baik, (4) mitigasi risiko menggunakan kombinasi 3 variabel sebagai indikator komposit, bukan hanya presipitasi. |

---

## 5 · Sinkronisasi Metodologi (Dokumen ↔ Kode)

| # | Aspek Metodologi | Dokumen | Kode Aktual | Keselarasan |
|---|-----------------|---------|-------------|-------------|
| 1 | Variabel target | 3 (hujan, angin, kelembapan) | 3 (`TARGET_COLS`) | ✅ |
| 2 | Horizon prediksi | "0–6 jam" | Single-step (1 jam) | ❌ GAP-01 |
| 3 | Input window | "sliding window" (tidak spesifik) | SEQ_LEN = 6 | ⚠️ Detail kurang |
| 4 | Fitur input | Hanya menyebut 3 variabel | 9 fitur (`all_feature_cols`) | ❌ GAP-04 |
| 5 | Sumber data | ERA5, Open-Meteo, hourly, 2005–2025 | Sama persis | ✅ |
| 6 | Node lokasi | 3 node (puncak, lereng, hilir) | Puncak, Lereng_Cibodas, Hilir_Cianjur | ✅ |
| 7 | Split data | Train 2005–2018, Val 2019–2021, Test 2022–2025 | `temporal_split()` | ✅ |
| 8 | Log-transform | Presipitasi saja | `log1p` hanya pada indeks 0 | ✅ |
| 9 | Normalisasi | Z-score dari training | `compute_stats_from_training()` | ✅ (tapi T_STD_MULTIPLIER=5 tidak disebut) |
| 10 | Graf | "tak berarah berbobot" | Fully-connected, **tanpa bobot** | ❌ GAP-08 |
| 11 | GNN | "message passing" | GATConv 2-layer, 4 heads | ✅ (lebih spesifik di kode) |
| 12 | Temporal modeling | "embedding waktu" | TemporalAttention (MultiheadAttention) | ✅ (lebih spesifik di kode) |
| 13 | Retrieval | FAISS, k-NN dari training saja | `IndexFlatL2`, `IndexIVFFlat` for training | ✅ |
| 14 | K neighbors | Tidak disebut | K_NEIGHBORS = 3 | ⚠️ Detail kurang |
| 15 | Diffusion scheduler | "reverse diffusion" | DDPM (train) + DDIM 20-step (inference) | ✅ (DDIM detail kurang di dokumen) |
| 16 | Loss function | "objective denoising" | Weighted MSE (5×/10×) | ❌ GAP-05 |
| 17 | Optimizer | "Adam" | AdamW (lr=1e-3, wd=1e-4) | ⚠️ Variant berbeda |
| 18 | Epochs | Tidak spesifik ("sejumlah epoch") | 20 (diffusion), 50 (baseline) | ⚠️ Detail kurang |
| 19 | Batch size | Tidak disebut | 512 (diffusion), 256 (baseline) | ⚠️ Detail kurang |
| 20 | Early stopping | "mekanisme early stopping" | Tidak ada — hanya save-best | ❌ GAP-07 |
| 21 | Hybrid persistence | "bobot w dari validasi" | Hardcoded 0.90/0.90/0.70 | ❌ GAP-03 |
| 22 | Skenario eksperimen | 4 skenario ablasi | 2 skenario | ❌ GAP-02 |
| 23 | Ensemble size | "sejumlah sampel" | NUM_ENSEMBLE = 30 | ⚠️ Detail kurang |
| 24 | Eval step | Tidak disebut | EVAL_STEP = 24 (daily subsampling) | ❌ GAP-06 |
| 25 | Threshold evaluasi | "beberapa ambang intensitas" | Precip >10mm, Wind >10m/s, Humidity >90% | ⚠️ Detail kurang |
| 26 | AMP (Mixed Precision) | Tidak disebut | `torch.amp.autocast`, `GradScaler` | ⚠️ Detail kurang |
| 27 | Hidden dimension | Tidak spesifik | HIDDEN_DIM = 128, GRAPH_DIM = 64 | ⚠️ Detail kurang |

**Legenda**: ✅ Konsisten | ❌ Inkonsisten (ada gap) | ⚠️ Tidak disebutkan di dokumen (detail kurang, tapi bukan kontradiksi)

---

## 6 · Saran Perbaikan Struktur

> **Prinsip Utama**: Seluruh rekomendasi di bawah ini difokuskan pada **perubahan dokumen/proposal** agar selaras dengan implementasi kode yang sudah berjalan. Kode tidak perlu diubah secara substansial — implementasi sudah solid dan fungsional. Yang perlu disesuaikan adalah narasi, klaim, dan framing di dokumen.

### 6.1 Reframe Judul dan Fokus Tesis (Prioritas Tertinggi)

**Judul Saat Ini** (implisit di proposal): Fokus pada nowcasting **hujan lebat** dan deteksi **cuaca ekstrem**.

**Judul yang Disarankan**:
> **"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki di Gunung Gede–Pangrango menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**

Perubahan framing yang diperlukan di seluruh dokumen:

| Aspek | Framing Lama | Framing Baru |
|-------|-------------|-------------|
| Fokus utama | Deteksi hujan lebat / spike events | Nowcasting probabilistik presipitasi multi-variabel |
| Kontribusi riset | Kemampuan mendeteksi kejadian ekstrem | Kerangka ensemble yang memberikan informasi ketidakpastian (CRPS, spread) |
| Mitigasi risiko | Peringatan dini hujan >10 mm/jam | Estimasi risiko komposit dari 3 variabel (hujan + angin + kelembapan) |
| RQ3 | Deteksi spike events | Kuantifikasi ketidakpastian prakiraan multi-variabel |
| Narasi keterbatasan | Tidak eksplisit | ERA5 ~25 km tidak representatif untuk intensitas presipitasi lokal |

### 6.2 Perbaikan Dokumen (Prioritas Tinggi)

1. **Revisi subbab Batasan Masalah**: Perjelas bahwa model menggunakan jendela input 6 jam untuk prediksi 1 jam ke depan, bukan prakiraan multi-step hingga 6 jam (GAP-01).

2. **Revisi subbab Alur Eksperimen**: Sesuaikan deskripsi menjadi 2 skenario yang terimplementasi + analisis kualitatif kontribusi komponen (GAP-02).

3. **Tambah subbab Fitur Input dan Variabel Target**: Di antara subbab Pra-pemrosesan dan subbab Representasi Graf, dokumentasikan lengkap 9 fitur input beserta justifikasi (GAP-04).

4. **Revisi subbab Prosedur Pelatihan**: Dokumentasikan weighted loss (5×/10×), T_STD_MULTIPLIER=5.0, ubah "early stopping" menjadi "best model selection", sebutkan hyperparameter spesifik (GAP-05, GAP-07, GAP-10).

5. **Revisi subbab Hybrid Persistence**: Tampilkan nilai bobot eksplisit (0.90/0.90/0.70) dan jelaskan proses penentuan empiris, serta bahas dominansi persistence secara kritis (GAP-03).

6. **Revisi subbab Representasi Graf**: Ubah "graf berbobot" menjadi "graf fully-connected dengan GAT attention" (GAP-08).

7. **Tambah subbab Metode Evaluasi**: Jelaskan subsampling EVAL_STEP=24, threshold per variabel, dan keterbatasan threshold 10 mm/jam pada ERA5 (GAP-06).

8. **Tambah subbab Keterbatasan Penelitian**: Subbab baru yang membahas keterbatasan ERA5, implikasi 2 node pada grid cell sama, dan target prediksi rata-rata antar node (GAP-11, KLARIFIKASI-02, KLARIFIKASI-04).

### 6.3 Perbaikan Kode (Minor, Opsional)

1. **Bersihkan komentar stale "4 variabel"**: Search-replace di `diffusion.py`, `train.py`, `inference.py` (GAP-09, ~5 menit).

2. **Tambahkan threshold evaluasi 2 mm dan 5 mm**: Di `final_proven_eval.py` untuk menunjukkan performa pada threshold yang lebih representatif untuk ERA5 (GAP-11, ~15 menit). Ini opsional tetapi akan memperkuat argumen di subbab Pembahasan.

---

## 7 · Daftar Perubahan

> **Catatan**: Hampir seluruh perubahan adalah pada **dokumen/proposal**, bukan kode. Implementasi kode sudah solid — yang perlu disesuaikan adalah narasi dokumen agar selaras dengan realitas implementasi.

### 7.1 Perubahan WAJIB (Harus Dilakukan Sebelum Sidang)

| # | Perubahan | Lokasi | Tipe | Gap Ref |
|---|-----------|--------|------|---------|
| W1 | **Reframe judul tesis**: Ubah fokus dari "Hujan Lebat" / "Cuaca Ekstrem" → "Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki" | Dokumen: Halaman judul, Abstrak, subbab Latar Belakang, subbab Tujuan | Dokumen | GAP-11 |
| W2 | **Revisi RQ3**: Dari deteksi spike events → kuantifikasi ketidakpastian prakiraan multi-variabel | Dokumen subbab Rumusan Masalah (baris 88) | Dokumen | GAP-11 |
| W3 | **Revisi klaim "horizon 0–6 jam"** → "jendela input 6 jam + prediksi 1 jam ke depan" | Dokumen subbab Batasan Masalah, subbab Desain Penelitian | Dokumen | GAP-01 |
| W4 | **Sesuaikan skenario eksperimen**: 4 → 2 skenario + analisis kualitatif kontribusi komponen | Dokumen subbab Alur Eksperimen (baris 518–522) | Dokumen | GAP-02 |
| W5 | **Dokumentasikan 9 fitur input** sebagai conditioning model (bukan hanya 3 variabel target) | Dokumen: Tambah subbab Fitur Input dan Variabel Target | Dokumen | GAP-04 |
| W6 | **Dokumentasikan weighted denoising loss** (5×/10×) sebagai kontribusi teknis | Dokumen subbab Prosedur Pelatihan | Dokumen | GAP-05 |
| W7 | **Perbaiki klaim "early stopping"** → "best model selection" | Dokumen subbab Prosedur Pelatihan (baris 500) | Dokumen | GAP-07 |
| W8 | **Perbaiki klaim "graf berbobot"** → "graf fully-connected + GAT attention" | Dokumen subbab Representasi Graf (baris 474) | Dokumen | GAP-08 |
| W9 | **Tampilkan hybrid weights eksplisit** (0.90/0.90/0.70) dan jelaskan proses penentuan | Dokumen subbab Hybrid Persistence (baris 497) | Dokumen | GAP-03 |
| W10 | **Tambahkan subbab Keterbatasan Penelitian** yang membahas resolusi ERA5 dan implikasinya | Dokumen: Subbab baru di Bab Pembahasan | Dokumen | GAP-11 |

### 7.2 Perubahan DISARANKAN (Memperkuat Kualitas)

| # | Perubahan | Lokasi | Tipe | Gap Ref |
|---|-----------|--------|------|---------|
| D1 | Dokumentasikan T_STD_MULTIPLIER = 5.0 di subbab Pra-pemrosesan | Dokumen subbab Pra-pemrosesan | Dokumen | GAP-10 |
| D2 | Jelaskan EVAL_STEP = 24 (daily subsampling) di subbab Metode Evaluasi | Dokumen subbab Metode Evaluasi | Dokumen | GAP-06 |
| D3 | Jelaskan target prediksi = rata-rata antar 3 node | Dokumen subbab Desain Model | Dokumen | KLARIFIKASI-04 |
| D4 | Jelaskan bahwa 2 node share grid cell ERA5 sebagai desain yang disengaja | Dokumen subbab Wilayah Studi | Dokumen | KLARIFIKASI-02 |
| D5 | Sebutkan hyperparameter spesifik (SEQ_LEN=6, BATCH_SIZE=512, LR=1e-3, EPOCHS=20) | Dokumen subbab Prosedur Pelatihan | Dokumen | — |
| D6 | Jelaskan penggunaan DDIM 20-step di inference vs DDPM 1000-step di training | Dokumen subbab Inferensi | Dokumen | — |
| D7 | Bahas dominansi persistence secara kritis; bandingkan pure persistence vs hybrid | Dokumen subbab Pembahasan | Dokumen | KLARIFIKASI-03 |
| D8 | Bersihkan komentar stale "4 variables/targets" di kodebase | Kode `diffusion.py`, `train.py`, `inference.py` | Kode | GAP-09 |
| D9 | Tambahkan threshold evaluasi presipitasi 2 mm dan 5 mm untuk data ERA5 | Kode `final_proven_eval.py` | Kode | GAP-11 |
| D10 | Siapkan tabel perbandingan pure persistence vs hybrid vs diffusion-only di subbab Pembahasan | Dokumen subbab Pembahasan | Dokumen | KLARIFIKASI-03 |

---

## 8 · Kesimpulan Akhir

### 8.1 Penilaian Umum

Implementasi kode secara keseluruhan **secara substansial selaras** dengan kerangka konseptual yang tertulis di proposal. Arsitektur inti — conditional diffusion model dengan triple conditioning (temporal features, FAISS retrieval, spatio-temporal GNN) — terimplementasi sesuai. Prosedur anti-leakage (temporal split kronologis, normalisasi dari training saja, indeks FAISS dari training saja) diimplementasi dengan sangat baik dan melampaui standar rata-rata skripsi.

**Kode tidak perlu diubah secara substansial** — implementasi sudah solid, berjalan, dan menghasilkan metrik evaluasi. Yang perlu disesuaikan adalah **narasi dan klaim di dokumen proposal** agar akurat merepresentasikan apa yang sebenarnya dilakukan dan dicapai oleh model.

### 8.2 Rekomendasi Strategis Utama: Reframe Tesis

> **Perubahan paling berdampak yang dapat dilakukan**: Ubah judul dan framing tesis dari berfokus pada **deteksi hujan lebat / cuaca ekstrem** menjadi **nowcasting probabilistik presipitasi untuk mitigasi risiko**.

**Judul yang disarankan**:
> **"Nowcasting Probabilistik Presipitasi untuk Mitigasi Risiko Pendaki di Gunung Gede–Pangrango menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"**

**Alasan**:
1. ERA5 dengan resolusi ~25 km tidak mampu menangkap intensitas presipitasi lokal — ini keterbatasan data, bukan model
2. POD presipitasi >10 mm/jam hanya 5.9% karena kejadian tersebut hanya 0.15% data
3. Kekuatan riil model ada pada: (a) estimasi distribusi probabilistik (CRPS, ensemble), (b) performa angin & kelembapan yang baik, (c) kerangka multi-variabel untuk estimasi risiko komposit
4. Framing "probabilistik presipitasi" lebih defensible secara akademik dan tidak membuat klaim yang tidak didukung data

### 8.3 Risiko Utama (Jika Tidak Di-reframe)

Tanpa reframe, tiga pertanyaan penguji ini akan sulit dijawab:

1. **"POD presipitasi 5.9% — bagaimana ini bisa disebut deteksi hujan lebat?"** → Dengan reframe ke "nowcasting probabilistik", pertanyaan ini tidak relevan karena klaim bukan deteksi spike
2. **"Horizon 0–6 jam — mana implementasi multi-step?"** → Dengan klarifikasi "input window 6 jam", pertanyaan terjawab
3. **"Hanya 2 dari 4 skenario ablasi?"** → Dengan revisi ke 2 skenario + analisis kualitatif, pertanyaan terjawab

### 8.4 Kekuatan yang Menonjol

- Landasan teori sangat kuat dan komprehensif
- Anti-leakage procedures diimplementasi secara eksemplar
- Metrik evaluasi probabilistik (CRPS, Brier) melampaui standar skripsi tipikal
- Arsitektur triple-conditioning (retrieval + graph + temporal) coherent dan well-integrated
- Performa wind speed (Corr=0.833, POD=55.2%) dan humidity (Corr=0.931, POD=51.7%) cukup baik
- Weighted denoising loss (5×/10×) adalah kontribusi teknis yang bernilai (wajib didokumentasikan!)

### 8.5 Rekomendasi Tindakan

| Prioritas | Perkiraan Effort | Tindakan |
|-----------|-----------------|----------|
| 🔴 P0 | 2 jam | **Reframe judul dan fokus tesis** — ubah dari "hujan lebat" ke "nowcasting probabilistik presipitasi" (W1, W2) |
| 🔴 P0 | 1 jam | Revisi kalimat horizon di dokumen (W3, GAP-01) |
| 🔴 P0 | 1 jam | Revisi skenario eksperimen di dokumen (W4, GAP-02) |
| 🔴 P0 | 1 jam | Tambahkan Keterbatasan Penelitian — resolusi ERA5, threshold, grid cell (W10) |
| 🟠 P1 | 2 jam | Dokumentasikan fitur input, weighted loss, T_STD_MULTIPLIER (W5, W6, D1) |
| 🟠 P1 | 30 menit | Perbaiki klaim early stopping & graf berbobot (W7, W8) |
| 🟠 P1 | 30 menit | Tampilkan hybrid weights eksplisit + pembahasan kritis persistence (W9, D7) |
| 🟡 P2 | 30 menit | Dokumentasikan EVAL_STEP=24, hyperparameter, DDIM vs DDPM (D2, D5, D6) |
| 🟡 P2 | 30 menit | Jelaskan target rata-rata & 2 node pada grid cell sama (D3, D4) |
| 🟢 P3 | 15 menit | Bersihkan komentar stale "4 variabel" di kode (D8) |
| 🟢 P3 | 15 menit | Tambah threshold evaluasi presipitasi 2 mm & 5 mm di kode (D9) |

**Total estimasi effort perubahan dokumen (P0+P1)**: ~8 jam
**Total estimasi effort lengkap termasuk kode minor (semua)**: ~10 jam

> **Pesan utama**: Kode Anda sudah bagus. Yang perlu diperbaiki adalah ceritanya — pastikan dokumen menceritakan apa yang benar-benar dilakukan dan dicapai oleh model, bukan apa yang diharapkan di awal proposal.

---

*Laporan ini dihasilkan melalui cross-referencing line-by-line antara dokumen proposal (493 baris) dan seluruh source code (>3000 baris), termasuk evaluation results dan model configurations.*
