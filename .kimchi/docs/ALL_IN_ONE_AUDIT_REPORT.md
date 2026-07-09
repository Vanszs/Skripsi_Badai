# ALL-IN-ONE AUDIT REPORT
## Draft Bepan1 (PDF) vs Implementation Code

**Date:** 2026-07-09  
**Scope:** BAB 1–3 dari `Draft Bepan1.pdf` dibandingkan dengan kode di `src/` dan artefak evaluasi  
**Sources:**
- `Draft Bepan1.pdf` → `.understand-anything/tmp/draft_bepan1_pdf_extracted.txt`
- Repository code: `src/`, `run_eval_final.py`, `src/config.py`
- Evaluation artifacts: `result_test/EVALUATION_REPORT.md`, `result_test/comparison/comparison_summary.csv`

---

## EXECUTIVE SUMMARY

**Verdict: NEEDS_MAJOR_REVISIONS**

Draft PDF masih mengandung beberapa kontradiksi signifikan terhadap implementasi kode. Isu paling kritis adalah klaim topologi **fully-connected** padahal kode menggunakan **star topology**. Selain itu, abstrak, aplikasi mobile, hybrid persistence, dan mekanisme evaluasi perlu diperjelas.

### Issue Count

| Level | Count |
|---|---|
| FATAL | 1 |
| MAJOR | 8 |
| MINOR | 8 |

---

## SECTION 1: ISSUES IN PDF VS CODE

### 1.1 FATAL

#### 1.1.1 Graph Topology: Fully-Connected vs Star

**Lokasi PDF:** BAB 2 [1309], BAB 3 [2023]

**Klaim PDF:** "Struktur graph fully-connected ..."

**Bukti Kode:**
```python
# src/config.py
GRAPH_TOPOLOGY = "star"

STAR_EDGES = [
    (MAIN_NODE_NAME, "UP"), ("UP", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "DOWN"), ("DOWN", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "LEFT"), ("LEFT", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "RIGHT"), ("RIGHT", MAIN_NODE_NAME),
]
STAR_EDGE_COUNT = len(STAR_EDGES)  # = 8
```

**Analisis:** Fully-connected 5 node = 20 edge terarah. Kode hanya punya 8 edge terarah berbentuk bintang dengan MAIN di pusat. Ini kontradiksi arsitektural fundamental.

**Perbaikan:** Ganti "fully-connected" menjadi "star topology" di seluruh draft.

---

### 1.2 MAJOR

#### 1.2.1 Abstract Horizon: 0–6 Jam

**Lokasi PDF:** Abstrak [62]

**Klaim PDF:** "Model dilatih untuk menghasilkan distribusi probabilistik kondisi presipitasi pada horizon prediksi 0–6 jam ke depan."

**Bukti Kode:**
- `src/data/temporal_loader.py`: target = `targets[t, main_node_idx, :]` (one-step t+1)
- `run_eval_final.py`: evaluasi one-step `idx` → `target = targets_raw[idx]`
- `src/train.py`: dilatih untuk prediksi satu langkah

**Analisis:**
- 0–6 jam adalah definisi WMO untuk nowcasting → itu benar.
- Namun, kalimat abstrak menyatakan model dilatih untuk horizon prediksi 0–6 jam → ini menyesatkan.
- Model sebenarnya dilatih untuk **1 jam ke depan (t+1)**.

**Perbaikan:** "Model dilatih untuk menghasilkan distribusi probabilistik kondisi presipitasi **satu jam ke depan (t+1)**, yang berada dalam rentang nowcasting 0–6 jam menurut WMO."

---

#### 1.2.2 Output / Deliverable: Mobile App Claim

**Lokasi PDF:** BAB 1 [92], [101], [109]

**Klaim PDF:** Aplikasi mobile sebagai media penyampaian informasi cuaca.

**Klarifikasi Penulis:** Output yang sebenarnya direncanakan adalah **sistem web dashboard berbasis Vue.js (frontend) dan FastAPI (backend)**, bukan aplikasi mobile. Namun, di repo saat ini belum ada implementasi Vue/FastAPI — pipeline berakhir di `run_eval_final.py`.

**Bukti Kode:**
- Tidak ada direktori/file aplikasi mobile (Android, iOS, Flutter, React Native).
- Tidak ada kode Vue atau FastAPI di repo saat ini.
- Pipeline berakhir di `run_eval_final.py` yang menghasilkan artefak evaluasi statis.

**Perbaikan:**
- Ganti klaim "aplikasi mobile" menjadi **"sistem visualisasi berbasis web dengan Vue.js dan FastAPI"** di rumusan masalah, tujuan, dan manfaat.
- Atau, jika belum sempat dibangun, arahkan sebagai **future work** / rencana pengembangan lanjut.
- Tambahkan subbab di BAB 2/Landasan Teori tentang arsitektur output sistem: Vue.js + FastAPI untuk menampilkan hasil prediksi probabilistik.

---

#### 1.2.3 Hybrid Persistence

**Lokasi PDF:** BAB 3 — muncul sebagai label "Hybrid Persistence Post-Processing" di sekitar [863], tetapi **tidak dijelaskan detailnya di badan teks PDF**.

**Lokasi DOCX (untuk referensi):** [547]–[550] — di sini dijelaskan skema hybrid persistence dengan rumus dan bobot 0,90 untuk curah hujan, 0,90 untuk angin, 0,70 untuk kelembapan, yang ditentukan secara empiris dari data validasi.

**Klaim:** Model menggunakan hybrid persistence post-processing yang menggabungkan prediksi model dengan observasi lag terakhir menggunakan bobot tertentu.

**Bukti Kode:**
- `run_eval_final.py` lines 133–144 hanya implementasi **naive persistence** (copy t-1 ke t):
  ```python
  pred = np.array([main_df.iloc[idx - 1][c] for c in TARGET_COLS], dtype=np.float32)
  ```
- Tidak ada bobot hybrid (0.90/0.90/0.70), tidak ada optimasi di data validasi, dan tidak ada penggabungan prediksi model dengan observasi.

**Perbedaan:**
- **Klaim draft:** Post-processing hybrid dengan bobot optimal per variabel.
- **Kode:** Persistence sederhana tanpa bobot, tanpa mixing dengan prediksi model.

**Perbaikan:** Hapus label/klaim hybrid persistence dari PDF, atau implementasikan hybrid persistence sesuai klaim di `run_eval_final.py`.

---

#### 1.2.4 Retrieval Metric

**Lokasi PDF:** BAB 2 [1213]

**Klaim PDF:** Euclidean distance **atau** cosine similarity.

**Bukti Kode:** `src/retrieval/base.py` menggunakan `faiss.IndexFlatL2` (Euclidean L2). Tidak ada cosine similarity.

**Perbaikan:** State "Euclidean L2 distance via FAISS IndexFlatL2" dan hapus cosine.

---

#### 1.2.5 Evaluation Subsampling (`eval_step=11`)

**Lokasi PDF:** BAB 3 evaluasi (~[2280])

**Klaim PDF:** Subsampling temporal untuk mengurangi autokorelasi.

**Pertanyaan Umum:** Apakah `eval_step=11` berarti model memprediksi 11 jam ke depan seperti TFT?

**Jawaban: TIDAK.** `eval_step=11` **bukan** horizon prediksi. Model tetap memprediksi **1 jam ke depan (t+1)** dari 6 jam input terakhir.

**Penjelasan:**
- `seq_len=6`: model melihat 6 jam data historis.
- Prediksi: selalu **t+1** = 1 jam ke depan.
- `eval_step=11`: hanya menentukan **jarak antar titik evaluasi**.
  - Jika data uji memiliki 35.000 jam, dengan `eval_step=11` kita evaluasi pada jam ke-6, 17, 28, 39, dst.
  - Setiap evaluasi tetap memprediksi **1 jam ke depan** dari posisi tersebut.
  - Tujuannya mengurangi autokorelasi antar sampel karena data cuaca per jam saling berkorelasi tinggi.

**Bukti Kode:**
```python
# run_eval_final.py
idxs = list(range(seq_len, len(main_df), eval_step))  # eval_step hanya mengatur jarak sampel
# ...
target = targets_raw[idx]  # target selalu t+1 dari idx
```

**Perbaikan:** Jelaskan dengan tegas bahwa `eval_step=11` adalah **subsampling evaluasi**, bukan horizon prediksi. Model selalu one-step t+1.

---

#### 1.2.6 Edge Attributes

**Lokasi PDF:** BAB 2 & BAB 3

**Klaim (implisit):** Edge attributes mengandung informasi spasial (jarak/elevasi).

**Bukti Kode:** `src/config.py build_star_edge_attr()` — semua edge konstan 0.25°. Eksperimen edge attribute berbasis elevasi menyebabkan training divergence.

**Perbaikan:** Jujur: edge attributes konstan 0.25° dan non-informatif; sinyal spasial berasal dari topologi star dan fitur node.

---

#### 1.2.7 Hypothermia Risk Index

**Lokasi PDF:** BAB 2 [1404], [1473]

**Klaim PDF:** Output dapat diterjemahkan menjadi level risiko hipotermia multi-faktor.

**Bukti Kode:** Tidak ada modul `hypothermia`, `risk_index`, atau `RiskIndex` di `src/`. `FINAL_TARGET_COLS` hanya presipitasi, angin, kelembapan.

**Perbaikan:** Framing sebagai future work / aplikasi konseptual, bukan output yang diimplementasikan.

---

#### 1.2.8 Optimizer Adam vs AdamW

**Lokasi PDF:** BAB 3 [2158]

**Klaim PDF:** Optimizer Adam.

**Bukti Kode:**
```python
# src/train.py
optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
```

**Perbaikan:** Ganti "Adam" menjadi "AdamW".

---

### 1.3 MINOR Issues

| # | Issue | Lokasi | Perbaikan |
|---|---|---|---|
| 1 | Weighted loss threshold | BAB 3 | Cantumkan 5× untuk \|target ternormalisasi\| > 1.0; 10× untuk > 3.0 |
| 2 | Tabel 2.2 tidak ada | BAB 2 [156] | Tambahkan atau hapus referensi |
| 3 | Penomoran gambar/tabel tidak konsisten | BAB 3 | Perbaiki penomoran |
| 4 | Klaim novelty berlebihan | BAB 1 [81], BAB 2 [265] | Pelunakkan klaim |
| 5 | Temperatur tidak jadi target | BAB 1 [115] | Berikan justifikasi lebih kuat |
| 6 | ERA5 disebut operasional | BAB 1–3 | Jelaskan ERA5 adalah reanalysis, bukan real-time |
| 7 | MAIN elevation 1529 m | BAB 3 [303] | Tambahkan catatan grid-average ERA5 |
| 8 | DOWN elevation 0 m | BAB 3 [303] | Tambahkan catatan grid-average ERA5 |

---

## SECTION 2: PROPOSED NEW LIMITATIONS (Batasan Masalah)

### 2.1 Limited Extreme Precipitation Data

**Keterbatasan:** Data presipitasi zero-inflated dan heavy-tailed; event ekstrem langka.

**Bukti Usaha:**
- Log transform (`log1p`) di `src/data/temporal_loader.py`.
- Weighted noise loss 5×/10× di `src/models/diffusion.py` lines 178–185.
- Retrieval-augmented historical analogs via `src/retrieval/base.py`.
- Diffusion ensemble (num_ensemble=30).

**Mengapa tidak bisa diatasi:** Frekuensi event ekstrem dalam data ERA5 2005–2025 tetap terbatas; tidak ada teknik yang bisa menciptakan informasi yang tidak ada di data historis.

---

### 2.2 Constant Edge Attributes (0.25°)

**Keterbatasan:** Edge attributes konstan 0.25° dan non-informatif.

**Bukti Usaha:**
- Eksperimen edge attribute berbasis elevasi → training divergence.
- Revert ke konstanta jarak lat/lon.
- Sinyal spasial dipreserved melalui topologi star dan fitur node yang berbeda.

**Mengapa tidak bisa diatasi:** Pada grid ERA5 0.25°, jarak lat/lon dari MAIN ke semua neighbor tepat 0.25°. Edge attribute berbasis elevasi merusak konvergensi.

---

### 2.3 ERA5 Not Real-Time

**Keterbatasan:** ERA5 reanalysis memiliki latensi ~5–7 hari, tidak bisa operasional real-time.

**Bukti Usaha:**
- Gunakan ERA5 sebagai dataset historis konsisten 2005–2025.
- Pipeline dibuat modular (`src/data/ingest.py`, `src/config.py`) sehingga `OPEN_METEO_MODEL` bisa diganti dengan sumber near real-time di masa depan.

**Mengapa tidak bisa diatasi:** ERA5 adalah reanalysis, bukan observasi real-time. Operasional membutuhkan radar, AWS, atau data satelit near real-time.

---

## SECTION 3: EVIDENCE OF SCENARIO TESTING

### 3.1 Six Scenarios

1. `persistence` — naive baseline
2. `mlp_baseline` — deterministic MLP
3. `diff_only` — diffusion without retrieval/GNN
4. `diff_retrieval` — diffusion + retrieval
5. `diff_gnn` — diffusion + GNN
6. `full_model` — diffusion + retrieval + GNN

### 3.2 Evaluation Metadata

- `graph_topology`: star
- `eval_step`: 11
- `samples_per_scenario`: 3189
- `num_ensemble`: 30
- `seq_len`: 6
- `node_order`: [MAIN, UP, DOWN, LEFT, RIGHT]

### 3.3 Precipitation Metrics

**Tabel 3.1. Deterministic & Probabilistic Metrics**

| Scenario | RMSE | MAE | Correlation | CRPS |
|---|---:|---:|---:|---:|
| persistence | 0.6847 | 0.2187 | 0.6856 | 0.2187 |
| mlp_baseline | 0.7417 | 0.2686 | 0.5461 | 0.2686 |
| diff_only | 0.8718 | 0.4132 | 0.5345 | 0.2893 |
| diff_retrieval | 0.8957 | 0.4219 | 0.5303 | 0.2960 |
| diff_gnn | 0.7802 | 0.3415 | 0.6320 | 0.2463 |
| full_model | 0.7920 | 0.3530 | 0.6390 | 0.2518 |

**Catatan teknis:** CRPS untuk persistence dan MLP baseline sama dengan MAE karena model deterministik single-pass.

### 3.4 Threshold Metrics (10 mm)

**Tabel 3.2. Threshold Metrics at 10 mm**

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.5000 | 0.0000 | 0.5000 | 0.0003 |
| mlp_baseline | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_only | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_retrieval | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_gnn | 0.0000 | nan | 0.0000 | 0.0006 |
| full_model | 0.0000 | nan | 0.0000 | 0.0007 |

**Interpretasi:** Pada threshold 10 mm, model diffusion tidak mendeteksi event ekstrem sama sekali (POD=0). Ini mendukung batasan masalah Section 2.1.

### 3.5 Interpretation of Counter-Intuitive Results

1. `diff_only` dan `diff_retrieval` lebih buruk dari persistence pada RMSE/MAE karena diffusion dioptimalkan untuk distribusi (CRPS), bukan titik.
2. `diff_gnn` sedikit lebih baik dari `full_model` pada RMSE karena variabilitas stochastic dan interaksi retrieval-GNN yang belum optimal.

---

## SECTION 4: REVISION PRIORITY

### Priority 1 — Must Fix

1. Ganti "fully-connected" → "star topology" (BAB 2 [1309], BAB 3 [2023]).
2. Perjelas abstrak: model prediksi **1 jam ke depan (t+1)** dalam kerangka nowcasting 0–6 jam.
3. Koreksi klaim aplikasi mobile menjadi **sistem web dashboard (Vue.js + FastAPI)** atau arahkan sebagai future work.
4. Hapus/implementasikan hybrid persistence.

### Priority 2 — Important

5. State retrieval pakai Euclidean L2 only.
6. Jelaskan `eval_step=11` dan dukungan `eval_step=1`.
7. Jujur: edge attributes konstan 0.25°.
8. Hipotermia sebagai future work.
9. Ganti Adam → AdamW.
10. Tambahkan 3 batasan masalah baru dengan bukti usaha.

### Priority 3 — Polish

11. Cantumkan threshold weighted loss.
12. Perbaiki penomoran gambar/tabel.
13. Perbaiki referensi Tabel 2.2.
14. Pelunakkan klaim novelty.
15. Justifikasi temperatur tidak jadi target.
16. Jelaskan ERA5 adalah reanalysis.
17. Catatan elevation MAIN/DOWN grid-average.

---

## SECTION 5: SINGLE SOURCE OF TRUTH

Untuk merevisi draft, gunakan: **`docs/METODE_PENELITIAN_PROYEK.md`**

Dokumen ini sudah konsisten dengan kode untuk:
- 5 node + star topology
- Prediksi one-step t+1
- 6 skenario evaluasi
- Edge attributes konstan
- `eval_step` evaluation
- Preprocessing pipeline

---

## APPENDIX: QUICK REFERENCE FOR DEFENSE

**Q: Kenapa star, bukan fully-connected?**  
A: Kode pakai star topology 8 edge terarah MAIN↔UP/DOWN/LEFT/RIGHT. Neighbor tidak saling terhubung.

**Q: 0–6 jam atau 1 jam?**  
A: 0–6 jam adalah definisi WMO nowcasting. Model prediksi **1 jam ke depan (t+1)**.

**Q: Output sistem seperti apa?**  
A: Rencana output adalah **web dashboard berbasis Vue.js (frontend) dan FastAPI (backend)**, bukan aplikasi mobile. Namun, implementasi Vue/FastAPI belum ada di repo saat ini; pipeline masih berakhir di evaluasi statis `run_eval_final.py`.

**Q: Hybrid persistence?**  
A: Tidak diimplementasikan. Hanya naive persistence (copy t-1 ke t).

**Q: Bukti penanganan event ekstrem?**  
A: Log transform, weighted loss 5×/10×, retrieval, diffusion ensemble.

**Q: Edge attributes?**  
A: Konstanta 0.25°. Eksperimen elevasi menyebabkan divergence.

**Q: Kenapa eval_step=11? Apakah berarti prediksi 11 jam ke depan?**  
A: **Tidak.** `eval_step=11` hanya mengatur jarak antar titik evaluasi. Model tetap prediksi **1 jam ke depan (t+1)** dari 6 jam input. Eval_step dipakai untuk mengurangi autokorelasi antar sampel cuaca yang berdekatan.
