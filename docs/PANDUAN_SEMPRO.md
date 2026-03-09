# PANDUAN SEMINAR PROPOSAL SKRIPSI

## Prinsip Utama

Seminar Proposal = presentasi **ide dan rencana** (BAB 1–3). Belum ada hasil. Penguji menilai **reasoning dan kesiapan metodologi**, bukan akurasi atau performa model.

**Yang dinilai**: Apakah masalah jelas? Apakah metode yang dipilih masuk akal? Apakah rencana penelitian bisa dieksekusi?

**Yang TIDAK dinilai**: Hasil eksperimen, angka akurasi, perbandingan metrik, hyperparameter optimal.

---

## 1. APA YANG DIBAHAS DI SEMINAR PROPOSAL?

| BAB | Isi | Level Detail |
|-----|-----|--------------|
| **BAB 1** | Latar belakang, rumusan masalah, tujuan, batasan, manfaat | Lengkap |
| **BAB 2** | Tinjauan pustaka, landasan teori, penelitian terdahulu | Lengkap |
| **BAB 3** | Metodologi: gambaran data, arsitektur konseptual, rencana evaluasi | Konseptual — tanpa detail implementasi |

> **TIDAK perlu di seminar proposal:**
> - Nama spesifik variabel (cukup "variabel cuaca multi-variabel")
> - Angka akurasi atau target metrik
> - Hyperparameter (learning rate, batch size, epoch, dll)
> - Kode atau pseudo-code implementasi
> - Hasil training atau grafik loss
> - Koordinat spesifik atau detail teknis dataset

---

## 2. SEJAUH APA JUDUL & ALGORITMA DIBAHAS?

### BAB 1 — Dekomposisi Judul menjadi Motivasi

Judul skripsi harus bisa "dibongkar" menjadi alasan-alasan:

| Kata Kunci di Judul | Yang Harus Dijelaskan |
|---------------------|----------------------|
| **Nowcasting Probabilistik** | Mengapa butuh prediksi jangka pendek? Mengapa probabilistik, bukan deterministik? |
| **Cuaca Multi-Variabel** | Mengapa multi-variabel penting untuk konteks ini? |
| **Mitigasi Risiko Pendakian** | Apa masalah nyata yang terjadi? Siapa yang terdampak? |
| **Gunung Gede-Pangrango** | Mengapa lokasi ini? Apa signifikansinya? |
| **Retrieval-Augmented** | Mengapa butuh informasi historis tambahan? |
| **Diffusion Model** | Mengapa generative model? Apa kelebihan dibanding model prediktif biasa? |
| **Spatio-Temporal Graph** | Mengapa hubungan antar lokasi penting? Mengapa perlu temporal? |

**Depth BAB 1**: Cukup narasi konseptual. Tidak perlu menyebutkan nama variabel spesifik, cukup "beberapa variabel cuaca kritis untuk keselamatan pendaki".

### BAB 2 — Teori Konseptual (Bukan Implementasi)

Bahas **konsep dan prinsip** setiap komponen, bukan arsitektur teknis detail:

| Komponen | Yang Dibahas (✅) | Yang TIDAK Dibahas (❌) |
|----------|-------------------|------------------------|
| **Diffusion Model** | Konsep forward/reverse process, ide denoising, mengapa bisa menghasilkan distribusi | Noise schedule formula, loss function detail, jumlah timestep |
| **Graph Neural Network** | Konsep graph representation, message passing, attention mechanism | Jumlah layer, dimensi hidden, head count |
| **Analog Retrieval** | Konsep pencarian pola historis mirip, similarity search | Implementasi FAISS, jumlah k, dimensi embedding |
| **Spatio-Temporal** | Mengapa data cuaca punya dependensi ruang dan waktu | Panjang sequence, window size |
| **ERA5** | Apa itu reanalysis data, mengapa cocok untuk penelitian ini | Resolusi teknis, jumlah variabel spesifik |
| **Penelitian Terdahulu** | ≥5 paper terkait, posisi penelitian ini dibanding state-of-the-art | Detail reproduksi paper |

**Depth BAB 2**: Level textbook + literature review. Pembaca harus paham "apa itu" dan "mengapa relevan", bukan "bagaimana implementasinya".

### BAB 3 — Metodologi (Rencana, Bukan Implementasi)

BAB 3 menjelaskan **rencana** bagaimana penelitian akan dilakukan:

| Yang Dibahas (✅) | Yang TIDAK Dibahas (❌) |
|-------------------|------------------------|
| Gambaran umum arsitektur (diagram blok) | Detail layer/dimensi/parameter |
| Alur pipeline: data → preprocessing → model → evaluasi | Kode atau pseudo-code |
| Sumber data dan periode umum | Koordinat GPS, nama variabel spesifik |
| Jenis metrik evaluasi yang akan dipakai | Target angka akurasi |
| Skenario perbandingan (baseline vs proposed) | Hasil eksperimen |
| Timeline / jadwal penelitian | - |

**Depth BAB 3**: Seperti "resep masakan" — bahan dan langkah-langkah umum, tapi belum dimasak.

---

## 3. KEY POINTS — Informasi Wajib di Seminar Proposal

### 3.1 Latar Belakang & Urgensi
- **Masalah nyata**: Risiko cuaca ekstrem bagi pendaki di kawasan gunung (petir, hipotermia, angin kencang)
- **Gap observasi**: Keterbatasan data cuaca real-time di kawasan pegunungan
- **Gap metode**: Model prediksi deterministik tidak memberikan informasi ketidakpastian — padahal untuk mitigasi risiko, probabilitas kejadian ekstrem lebih berguna
- **Peluang**: Perkembangan generative AI (diffusion model) untuk domain cuaca menunjukkan hasil menjanjikan

### 3.2 Novelty / Kontribusi
- Integrasi tiga pendekatan (generative model + graph spatial + retrieval historis) dalam satu arsitektur untuk nowcasting cuaca
- Pendekatan probabilistik — menghasilkan distribusi prediksi, bukan satu angka
- Aplikasi pada konteks mitigasi risiko pendakian gunung

### 3.3 Rumusan Masalah (3 RQ)
1. Bagaimana merancang arsitektur yang mengintegrasikan diffusion model, GNN, dan retrieval analog untuk nowcasting cuaca probabilistik?
2. Seberapa besar kontribusi masing-masing komponen terhadap kualitas prediksi? (via ablation study)
3. Apakah model yang diajukan lebih baik dari baseline konvensional?

### 3.4 Tujuan Penelitian
- Merancang dan mengimplementasikan arsitektur retrieval-augmented diffusion model dengan graph conditioning
- Mengevaluasi kontribusi setiap komponen melalui ablation study
- Membandingkan performa model terhadap baseline

### 3.5 Batasan Penelitian (Umum)
- Sumber data: ERA5 reanalysis (bukan observasi langsung)
- Lokasi: Kawasan Gunung Gede-Pangrango
- Fokus: Nowcasting jangka pendek
- Bukan sistem real-time deployment

### 3.6 Manfaat Penelitian
- **Akademis**: Kontribusi arsitektur novel untuk probabilistic weather nowcasting
- **Praktis**: Potensi dasar pengembangan early warning system untuk keselamatan pendaki

---

## 4. DIAGRAM YANG DIPERLUKAN

### 4.1 Wajib Ada

| # | Diagram | BAB | Fungsi |
|---|---------|-----|--------|
| 1 | **Kerangka Berpikir** | BAB 1 | Alur logika: masalah → solusi yang diajukan → expected outcome |
| 2 | **Diagram Lokasi Penelitian** | BAB 1/3 | Peta kawasan penelitian (umum, tidak perlu koordinat tepat) |
| 3 | **Arsitektur Model (High-Level)** | BAB 3 | Diagram blok: Data → Komponen A → Komponen B → Komponen C → Output |
| 4 | **Diagram Alir Penelitian** | BAB 3 | Flowchart tahapan: Studi Literatur → Pengumpulan Data → Preprocessing → Implementasi → Evaluasi → Penulisan |
| 5 | **Pipeline Data** | BAB 3 | Alur data dari sumber hingga siap diproses model |

### 4.2 Direkomendasikan

| # | Diagram | Fungsi |
|---|---------|--------|
| 6 | **Ilustrasi Konsep Diffusion** | Forward (menambah noise) → Reverse (menghilangkan noise) — konseptual |
| 7 | **Ilustrasi Konsep Graph** | Titik-titik lokasi sebagai node, hubungan sebagai edge |
| 8 | **Ilustrasi Konsep Retrieval** | Kondisi sekarang → cari yang mirip di historis → gunakan sebagai referensi |
| 9 | **Tabel Penelitian Terdahulu** | Perbandingan 5+ paper: metode, data, hasil, gap |
| 10 | **Timeline / Gantt Chart** | Jadwal rencana penelitian per bulan |

### Catatan tentang Diagram
- Semua diagram harus **konseptual**, bukan teknis
- Arsitektur model: cukup kotak-kotak dengan label komponen dan panah alur data
- Tidak perlu menampilkan dimensi tensor, jumlah layer, atau parameter apapun
- Diagram lokasi: cukup peta umum yang menunjukkan area penelitian

---

## 5. PERTANYAAN PENGUJI YANG DIANTISIPASI

### A — Justifikasi Metode
| Pertanyaan | Poin Jawaban |
|-----------|-------------|
| Mengapa diffusion model, bukan LSTM/Transformer? | Diffusion menghasilkan distribusi probabilitas (ensemble), bukan satu prediksi. Untuk mitigasi risiko, informasi ketidakpastian sangat penting |
| Mengapa perlu GNN? | Data cuaca memiliki dependensi spasial — kondisi di satu lokasi dipengaruhi lokasi sekitarnya. GNN menangkap hubungan ini secara learnable |
| Mengapa retrieval? Biasanya untuk NLP | Konsep analog forecasting sudah lama ada di meteorologi. Ini bukan RAG (NLP) — ini pencarian pola cuaca historis yang mirip sebagai referensi tambahan |
| Terlalu kompleks — kenapa tidak pakai satu model saja? | Ablation study akan menjawab apakah setiap komponen benar-benar berkontribusi. Kompleksitas dijustifikasi jika ada peningkatan |

### B — Data
| Pertanyaan | Poin Jawaban |
|-----------|-------------|
| Kenapa ERA5 bukan data observasi? | Tidak ada stasiun cuaca otomatis di jalur pendakian; ERA5 menyediakan data kontinu dan konsisten untuk kawasan manapun |
| ERA5 resolusinya rendah, representatif? | Keterbatasan yang diakui. Studi ini proof-of-concept; integrasi data observasi lokal bisa jadi pengembangan lanjutan |
| Bagaimana menangani ketidakseimbangan data cuaca? | Cuaca ekstrem memang langka — pendekatan probabilistik lebih cocok menangkap tail distribution dibanding metode deterministik |

### C — Evaluasi
| Pertanyaan | Poin Jawaban |
|-----------|-------------|
| Metrik apa yang dipakai? | Metrik deterministik (RMSE, MAE) untuk komparabilitas + metrik probabilistik (CRPS) yang lebih relevan untuk ensemble forecasting |
| Baseline apa yang akan dibandingkan? | Persistence (naive), model sederhana (MLP), dan ablation skenario (menghilangkan komponen satu per satu) |
| Sudah ada target akurasi? | Belum — itu hasil penelitian nanti. Yang penting rencana evaluasi sudah komprehensif |

### D — Relevansi
| Pertanyaan | Poin Jawaban |
|-----------|-------------|
| Siapa yang akan menggunakan? | Pengelola taman nasional, SAR, komunitas pendaki — sebagai dasar pengembangan early warning |
| Bisa real-time? | Bukan fokus skripsi ini. Ini studi feasibility arsitektur — deployment adalah pengembangan lanjutan |
| Kontribusi ilmiahnya apa? | Arsitektur novel yang menggabungkan 3 pendekatan + evaluasi probabilistik untuk nowcasting cuaca pegunungan |

---

## 6. STRUKTUR SLIDE PRESENTASI (~15 Menit)

| Slide | Konten | Waktu |
|-------|--------|-------|
| 1 | Judul, Nama, Pembimbing | 30 detik |
| 2 | Outline presentasi | 30 detik |
| 3–4 | Latar belakang + urgensi masalah | 2 menit |
| 5 | Rumusan masalah + tujuan + batasan | 1.5 menit |
| 6 | Tinjauan pustaka (tabel penelitian terdahulu) | 2 menit |
| 7 | Landasan teori: konsep 3 komponen utama | 2 menit |
| 8 | **Diagram arsitektur model (high-level)** | 1.5 menit |
| 9 | **Flowchart metodologi + pipeline data** | 1.5 menit |
| 10 | Rencana evaluasi + skenario perbandingan | 1 menit |
| 11 | Timeline penelitian | 30 detik |
| 12 | Kontribusi + penutup | 1 menit |
| — | Sesi tanya jawab | ~45 menit |

---

## 7. YANG TIDAK PERLU DISIAPKAN

- ❌ Nama spesifik variabel input/output
- ❌ Angka akurasi, RMSE, atau metrik apapun
- ❌ Hyperparameter (learning rate, batch size, jumlah layer, dimensi, epoch)
- ❌ Kode implementasi atau pseudo-code detail
- ❌ Hasil eksperimen atau grafik training
- ❌ BAB 4 (Hasil), BAB 5 (Pembahasan), BAB 6 (Kesimpulan)
- ❌ Screenshot output model

Yang disiapkan adalah **reasoning**: mengapa masalah ini penting, mengapa metode ini dipilih, dan bagaimana rencana menjawab pertanyaan penelitian.

---

## 8. CHECKLIST KESIAPAN

### Dokumen
- [ ] BAB 1: latar belakang, 3 RQ, tujuan, batasan, manfaat
- [ ] BAB 2: teori diffusion, GNN, retrieval analog, ERA5, ≥5 penelitian terdahulu
- [ ] BAB 3: pipeline data (konseptual), arsitektur model (high-level), rencana evaluasi, timeline

### Diagram Wajib
- [ ] Kerangka berpikir
- [ ] Diagram lokasi penelitian (peta umum)
- [ ] Arsitektur model (diagram blok high-level)
- [ ] Diagram alir penelitian (flowchart)
- [ ] Pipeline data (konseptual)

### Kesiapan Menjawab
- [ ] Justifikasi diffusion model vs alternatif (LSTM, Transformer)
- [ ] Justifikasi ERA5 vs observasi lokal
- [ ] Penjelasan novelty (3 poin)
- [ ] Rencana evaluasi: jenis metrik + skenario perbandingan
- [ ] Batasan penelitian

---

## 9. REFERENSI UTAMA

| Paper | Relevansi |
|-------|-----------|
| Ho et al. (2020) — DDPM | Fondasi teori diffusion model |
| Song et al. (2020) — DDIM | Accelerated sampling |
| Veličković et al. (2018) — GAT | Graph Attention Network |
| Yuan et al. (2024) — CoDiCast | Retrieval + diffusion untuk weather prediction |
| Price et al. (2023) — GenCast | Diffusion ensembles untuk cuaca global |
| Bi et al. (2023) — Pangu-Weather | Transformer untuk NWP |
| Hersbach et al. (2020) — ERA5 | Sumber data reanalysis |
| Delle Monache et al. (2013) — Analog Ensemble | Analog retrieval framework untuk forecasting |
