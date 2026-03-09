# PANDUAN SEMINAR PROPOSAL SKRIPSI
## Nowcasting Probabilistik Cuaca Multi-Variabel untuk Mitigasi Risiko Pendakian di Gunung Gede-Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning

**Konteks**: Seminar Proposal adalah presentasi BAB 1–3 (ide + rencana). Belum ada hasil eksperimen. Penguji mengevaluasi **reasoning** dan **kesiapan metodologi**, bukan akurasi model.

---

## 1. APA YANG DIBAHAS DI SEMINAR PROPOSAL?

Seminar proposal mencakup **BAB 1–3 saja**. Tidak ada BAB 4 (Hasil) atau BAB 5 (Pembahasan) karena penelitian belum dilakukan.

| BAB | Isi | Level Detail |
|-----|-----|--------------|
| **BAB 1** | Latar belakang, rumusan masalah, tujuan, batasan, manfaat | Lengkap |
| **BAB 2** | Tinjauan pustaka, landasan teori, penelitian terdahulu | Lengkap |
| **BAB 3** | Metodologi: data, arsitektur model, pipeline, rencana evaluasi | Detail — tapi tanpa hasil |

> **Tidak perlu disebutkan**: target akurasi spesifik, hyperparameter final, hasil eksperimen,
> atau perbandingan metrik (semua itu BAB 4).

---

## 2. SEJAUH APA JUDUL & ALGORITMA DIBAHAS (BAB 1–3)?

### BAB 1 — Judul dan Motivasi
- Judul didekomposisi menjadi motivasi: *mengapa diffusion model? mengapa GNN? mengapa retrieval?*
- Sebutkan **masalah nyata** yang dijawab (nowcasting cuaca untuk keselamatan pendaki di Gede-Pangrango, observasi cuaca lokal terbatas di kawasan gunung)
- Rumusan masalah → pertanyaan penelitian yang akan dijawab
- Tujuan = versi jawaban dari rumusan masalah
- **Batasan penelitian**: area geografis (Gunung Gede-Pangrango, 3 titik ketinggian berbeda), sumber data (ERA5), variabel target (3 variabel), periode (2005–2025)

### BAB 2 — Algoritma Secara Teoritis
Bahas masing-masing komponen sebagai teori, **bukan implementasi**:

| Komponen | Depth di BAB 2 |
|----------|---------------|
| **Diffusion Model (DDPM/DDIM)** | Proses forward (q) dan reverse (p), noise schedule, loss L_simple |
| **Graph Neural Network** | Konsep graph, pesan-passing, GAT (attention mechanism) |
| **Analog Retrieval** | Konsep retrieval analog historis, FAISS similarity search, k-NN retrieval |
| **Spatio-Temporal Modeling** | Kenapa spasial penting, temporal sequence, kombinasi keduanya |
| **Data ERA5** | Karakteristik resolusi 25km, variabel yang tersedia, keterbatasan |
| **Penelitian Terdahulu** | ≥5 paper terkait: CoDiCast, Prediff, GenCast, WFM, analog retrieval forecasting |

### BAB 3 — Metodologi (Detail tapi Tanpa Hasil)
- **Gambaran umum arsitektur**: ConditionalDiffusionModel + SpatioTemporalGNN + FAISS Retrieval
- **Pipeline secara menyeluruh** dari data raw hingga output prediksi
- **Spesifikasi teknis yang SUDAH diputuskan** (bukan yang masih ditentukan):
  - Input: seq_len=6 jam, 3 node ketinggian (Puncak 3019m, Lereng 1800m, Hilir 500m), 9 fitur
  - Target: 3 variabel (precipitation, wind_speed_10m, relative_humidity_2m)
  - Output: ensemble sampel probabilistik
- **Rencana evaluasi**: metrik yang akan digunakan (RMSE, MAE, Corr, CRPS), skenario perbandingan (persistence, MLP baseline, ablation)
- **TIDAK perlu**: nilai final hyperparameter, hasil training, grafik evaluasi

---

## 3. KEY POINTS — INFORMASI WAJIB DI PRESENTASI

### 3.1 Latar Belakang (5 poin utama)
1. **Urgensi**: Gunung Gede-Pangrango adalah destinasi pendakian tersibuk di Indonesia — cuaca ekstrem (petir, hipotermia, angin kencang) menjadi risiko utama pendaki
2. **Gap data**: Tidak ada stasiun cuaca otomatis di jalur pendakian; observasi lokal sangat terbatas di kawasan gunung
3. **Gap metode**: Model deterministik kehilangan informasi ketidakpastian — pendaki butuh probabilitas kejadian cuaca ekstrem, bukan satu angka prediksi
4. **Peluang**: Diffusion model terbukti superior untuk probabilistic generation (GenCast, Prediff)
5. **Novelty**: Kombinasi tiga komponen (Diffusion + GNN Spasial + Retrieval Analog) untuk nowcasting cuaca gunung belum pernah dilakukan

### 3.2 Rumusan Masalah (3 RQ)
1. Bagaimana arsitektur Retrieval-Augmented Diffusion Model dengan ST-GNN dapat melakukan nowcasting cuaca multi-variabel secara probabilistik untuk mitigasi risiko pendakian di Gunung Gede-Pangrango?
2. Seberapa besar kontribusi masing-masing komponen (Retrieval, GNN) terhadap akurasi prediksi?
3. Apakah model menghasilkan prediksi lebih baik dari baseline (persistence, MLP)?

### 3.3 Kontribusi / Novelty
- Integrasi **retrieval analog historis** (FAISS) sebagai kondisi tambahan pada diffusion model — bukan generasi teks melainkan kondisi cuaca
- **ST-GNN sebagai kondisi spasial** yang menangkap hubungan antar ketinggian di gunung (Puncak 3019m → Lereng 1800m → Hilir 500m)
- **Evaluasi probabilistik** dengan CRPS dan reliability diagram — lebih kaya dari RMSE saja

### 3.4 Batasan Penelitian
- Data: ERA5 reanalysis 2005–2025, resolusi ~25km (bukan observasi langsung)
- Area: 3 titik ketinggian di Gunung Gede-Pangrango — Puncak (3019m, -6.77°S 106.96°E), Lereng Cibodas (1800m, -6.75°S 106.99°E), Hilir Cianjur (500m, -6.82°S 107.13°E)
- Prediksi: Nowcasting satu langkah ke depan (t+1 jam) dengan input 6 jam sebelumnya
- Hardware: Tidak dirancang untuk real-time deployment

---

## 4. DIAGRAM WAJIB DAN OPSIONAL

### 4.1 Wajib Ada (BAB 1–3)

| # | Diagram | Letak | Isi |
|---|---------|-------|-----|
| 1 | **Diagram Lokasi Penelitian** | BAB 1 | Peta Gunung Gede-Pangrango, 3 titik ketinggian + koordinat + elevasi |
| 2 | **Kerangka Konsep / Kerangka Berpikir** | BAB 1 | Hubungan variabel input → model → output |
| 3 | **Arsitektur Model Utama** | BAB 3 | Full pipeline: Data → GNN → Retrieval → Diffusion → Output |
| 4 | **Diagram Alir Penelitian (Flowchart)** | BAB 3 | Tahapan: Studi literatur → Data → Implementasi → Evaluasi |
| 5 | **Diagram Alir Pipeline Data** | BAB 3 | ERA5 → Preprocessing → Normalisasi → Temporal Windows |
| 6 | **Arsitektur GNN** | BAB 3 | GAT 2-layer + TemporalAttention — input/output node |

### 4.2 Direkomendasikan

| # | Diagram | Isi |
|---|---------|-----|
| 7 | **Ilustrasi DDPM Forward/Reverse** | Proses q(x_t\|x_{t-1}) dan p_θ(x_{t-1}\|x_t) |
| 8 | **Ilustrasi Retrieval Process** | Query → FAISS Index → Top-k analog → Retrieved context |
| 9 | **Diagram Skenario Ablation** | 6 skenario dalam tabel/diagram |
| 10 | **Contoh Data Time Series** | 1 sample ERA5 showing 3 variables across 3 nodes |

---

## 5. PERTANYAAN PENGUJI YANG DIANTISIPASI

### Kelompok A — Justifikasi Algoritma
| Pertanyaan | Jawaban Kunci |
|-----------|---------------|
| Mengapa Diffusion Model dan bukan LSTM/Transformer biasa? | LSTM deterministik — tidak menghasilkan distribusi probabilitas. Diffusion menghasilkan ensemble 30 sampel → ketidakpastian cuaca bisa dikuantifikasi (CRPS, reliability) |
| Mengapa butuh GNN? Data hanya 3 titik, tidak terlalu kompleks? | 3 titik ketinggian berbeda (500m–3019m) memiliki hubungan spasial vertikal — cuaca di puncak gunung memengaruhi lereng dan kaki gunung. GAT belajar bobot attention antar ketinggian secara data-driven |
| Mengapa Retrieval? Biasanya dipakai untuk NLP | Analog retrieval untuk forecasting ada presedennya (CoDiCast IJCAI 2025). Ide: pola cuaca historis yang mirip kondisi sekarang adalah kondisi yang informatif untuk model generatif. Ini bukan RAG (NLP), melainkan retrieval analog cuaca |
| Apa perbedaan dengan pure diffusion tanpa komponen tambahan? | Dijawab melalui ablation study (6 skenario). Tanpa retrieval/GNN, model kehilangan konteks historis dan spasial |

### Kelompok B — Data dan Preprocessing  
| Pertanyaan | Jawaban Kunci |
|-----------|---------------|
| Kenapa pakai ERA5 bukan data observasi lokal? | Tidak ada stasiun cuaca otomatis di jalur pendakian. ERA5 adalah reanalysis global validated, kontinyu 2005–2025, hourly, dengan 9 variabel konsisten per titik ketinggian |
| ERA5 resolusi 25km — seberapa representatif untuk gunung? | Keterbatasan yang diakui di batasan penelitian. ERA5 menangkap sinyal mesoscale di kawasan pegunungan. Studi ini proof-of-concept sebelum integrasi data observasi lokal |
| Missing value, outlier? | ERA5 reanalysis tidak memiliki missing value karena proses assimilasi data global |

### Kelompok C — Metodologi dan Evaluasi
| Pertanyaan | Jawaban Kunci |
|-----------|---------------|
| Train/val/test split bagaimana? | Temporal split — tidak random. Train: 2005–2018, Val: 2019–2021, Test: 2022–2025. Mencegah data leakage |
| Bagaimana menghindari overfitting? | Dropout, weight decay (AdamW), best checkpoint selection, early stopping dengan patience |
| Metrik apa yang digunakan? Kenapa CRPS? | RMSE/MAE untuk perbandingan deterministik, CRPS (Continuous Ranked Probability Score) untuk evaluasi probabilistik — metrik standar untuk ensemble forecasting |
| Bagaimana membuktikan model lebih baik dari baseline? | Perbandingan 6 skenario: persistence (naive), MLP (learning baseline), ablation 3 konfigurasi, full model. RMSE dan CRPS lebih rendah = lebih baik |

### Kelompok D — Relevansi dan Kontribusi
| Pertanyaan | Jawaban Kunci |
|-----------|---------------|
| Kontribusi ilmiah utamanya apa? | (1) Arsitektur novel: kombinasi Diffusion+Retrieval+GNN untuk nowcasting cuaca gunung; (2) Evaluasi probabilistik CRPS — lebih informatif untuk mitigasi risiko pendakian dari RMSE saja |
| Siapa yang akan menggunakan hasilnya? | BTNGP (Balai Taman Nasional Gunung Gede-Pangrango), Basarnas/SAR, komunitas pendaki — untuk early warning cuaca di jalur pendakian |
| Apakah bisa diimplementasikan secara real-time? | Bukan target utama skripsi ini (batasan penelitian). Ini studi feasibility probabilistic nowcasting untuk kawasan gunung. Deployment sebagai early warning system bisa menjadi pengembangan lanjutan |

---

## 6. STRUKTUR SLIDE PRESENTASI (15 Menit)

| Slide | Konten | Waktu |
|-------|--------|-------|
| 1 | Judul, Nama, Prodi | 30 detik |
| 2 | Outline presentasi | 30 detik |
| 3–4 | Latar belakang + urgensi masalah | 2 menit |
| 5 | Rumusan masalah + tujuan | 1 menit |
| 6 | Tinjauan pustaka ringkas (tabel penelitian terdahulu) | 2 menit |
| 7 | Landasan teori: Diffusion + GNN + Retrieval (1 slide each, konsep) | 2 menit |
| 8 | **Diagram arsitektur model utama** (paling penting) | 1.5 menit |
| 9 | **Flowchart pipeline data + metodologi** | 1.5 menit |
| 10 | Rencana evaluasi: metrik + 6 skenario | 1 menit |
| 11 | Timeline penelitian | 30 detik |
| 12 | Kesimpulan sementara + kontribusi | 1 menit |
| 13 | Q&A | 45 menit |

---

## 7. HAL YANG TIDAK PERLU DISIAPKAN

Jangan habiskan waktu menyiapkan ini untuk seminar proposal:

- ❌ Hasil eksperimen atau grafik akurasi
- ❌ Hyperparameter final (learning rate, batch size, epoch optimal)
- ❌ Kode implementasi
- ❌ BAB 4 (Hasil), BAB 5 (Pembahasan), BAB 6 (Kesimpulan)
- ❌ Perbandingan metrik aktual antar model
- ❌ Ablation study results

Yang perlu disiapkan adalah **reasoning yang kuat**: mengapa arsitektur ini dipilih, mengapa data ini dipakai, dan bagaimana rencana evaluasinya sudah cukup untuk menjawab RQ.

---

## 8. CHECKLIST KESIAPAN SEMINAR PROPOSAL

### Dokumen
- [ ] BAB 1 lengkap: latar belakang, RQ ×3, tujuan, batasan, manfaat
- [ ] BAB 2 lengkap: teori Diffusion, GNN, Analog Retrieval, ERA5, ≥5 penelitian terdahulu
- [ ] BAB 3 lengkap: pipeline data, arsitektur model, rencana evaluasi, timeline

### Diagram (Wajib)
- [ ] Diagram lokasi penelitian (peta Gunung Gede-Pangrango + 3 titik ketinggian + koordinat)
- [ ] Kerangka berpikir / kerangka konsep
- [ ] Diagram alir penelitian (flowchart tahapan)
- [ ] Arsitektur model utama (full pipeline)
- [ ] Arsitektur GNN detail
- [ ] Diagram alir pipeline data (ERA5 → preprocessing → training)

### Jawaban Siap
- [ ] Justifikasi pilihan Diffusion Model vs LSTM/Transformer
- [ ] Justifikasi ERA5 vs BMKG
- [ ] Penjelasan kontribusi novelty (3 poin)
- [ ] Rencana evaluasi: metrik + skenario ablation (6 skenario)
- [ ] Train/val/test split temporal (tidak random)
- [ ] Batasan penelitian (4 poin)

---

## 9. REFERENSI KUNCI UNTUK DIKUTIP

| Paper | Relevansi |
|-------|-----------|
| Ho et al. (2020) — DDPM | Fondasi diffusion model |
| Song et al. (2020) — DDIM | Fast sampling inference |
| Veličković et al. (2018) — GAT | Graph Attention Network |
| Johnson et al. — Analog Forecasting | Konsep analog retrieval untuk prediksi cuaca |
| Yuan et al. (2024) — CoDiCast IJCAI 2025 | Retrieval + diffusion untuk NWP |
| Price et al. (2023) — GenCast | Diffusion ensembles untuk cuaca |
| Bi et al. (2023) — Pangu-Weather | Transformer untuk NWP, baseline |
| ERA5 (Hersbach et al. 2020) | Sumber data reanalysis |
| Johnson & Tippett (2022) — Analog Ensembles | Analog retrieval pada weather forecasting |
