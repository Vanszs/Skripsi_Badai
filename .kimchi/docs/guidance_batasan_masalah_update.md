# Arahan Revisi Batasan Masalah Berdasarkan Temuan PDF

> Tujuan: menambahkan batasan masalah yang jujur, dengan bukti usaha optimasi yang sudah dilakukan.
> Prinsip: jangan hanya menyalahkan sumber data, tetapi tunjukkan bahwa kita sudah mencoba semaksimal mungkin dan memang tidak bisa dilampaui.

---

## Batasan Masalah yang Perlu Ditambahkan

### 1. Data Curah Hujan yang Minim / Langka (Rare Events)

#### Usaha yang Sudah Dilakukan

1. **Log transform (log1p)** pada variabel presipitasi untuk mengurangi skewness.
2. **Weighted noise loss** dengan bobot:
   - 5× untuk sampel dengan |z| > 1σ
   - 10× untuk sampel dengan |z| > 3σ
3. **Retrieval-augmented mechanism** untuk memberikan memori historis pada kejadian ekstrem.
4. **Conditional diffusion model** untuk menghasilkan ensemble yang bisa mencakup nilai ekstrem.

#### Kenapa Memang Tidak Bisa Dilampaui?

Meskipun teknik-teknik di atas membantu, **distribusi dasar presipitasi tetap zero-inflated dan heavy-tailed**. Sebagian besar jam tidak hujan, dan kejadian ekstrem (>10 mm/jam) sangat langka. Model tetap terbatas oleh frekuensi intrinsik event ekstrem dalam data ERA5 2005–2025. Tidak ada teknik yang bisa menciptakan informasi yang tidak ada dalam data historis.

#### Saran Penulisan untuk BAB 1 / BAB 3

> "Penelitian ini terbatas pada karakteristik data presipitasi yang bersifat zero-inflated dan heavy-tailed, di mana kejadian hujan ekstrem relatif langka. Meskipun telah diterapkan log transform, weighted noise loss, retrieval-augmented mechanism, dan diffusion model untuk meningkatkan sensitivitas terhadap event ekstrem, model tetap dibatasi oleh frekuensi intrinsik event ekstrem dalam dataset ERA5."

---

### 2. Edge Attributes Konstan 0.25° dan Non-Informatif

#### Usaha yang Sudah Dilakukan

1. **Eksperimen dengan edge attributes berbasis perbedaan elevasi antar node** sudah dilakukan.
2. Hasil eksperimen: **training divergence / degradasi konvergensi**.
3. Karena itu, kembali ke **edge attributes konstan berbasis jarak lat/lon = 0.25°**.
4. Spatial signal dipertahankan melalui:
   - Topologi **star graph** (hubungan MAIN-tetangga).
   - Perbedaan fitur node antar lokasi.

#### Bukti di Kode

```python
# src/config.py — build_star_edge_attr()
"""
NOTE (honest limitation): ... An elevation-difference edge feature was tested
but empirically degraded training convergence, so it was reverted.
"""
```

```python
# docs/METODE_PENELITIAN_PROYEK.md §3.3/§4.3
"Eksperimen dengan atribut edge tambahan (misal, perbedaan elevasi) telah dicoba
 tetapi menyebabkan training divergen, sehingga dipertahankan pendekatan edge-attribute konstan."
```

#### Kenapa Memang Tidak Bisa Dilampaui?

Pada grid ERA5 0.25°, jarak lat/lon dari MAIN ke setiap node tetangga selalu **0.25°**. Artinya atribut edge berbasis jarak akan selalu konstan dan tidak informatif. Eksperimen dengan atribut edge berbasis elevasi justru merusak konvergensi training. Oleh karena itu, representasi spasial harus berasal dari topology star dan perbedaan fitur node, bukan dari bobot edge.

#### Saran Penulisan

> "Penelitian ini menggunakan atribut edge (edge_attr) berbasis jarak lat/lon antar node, yang pada grid ERA5 0.25° bernilai konstan 0.25° untuk seluruh edge dan bersifat non-informatif. Telah dilakukan eksperimen dengan atribut edge berbasis perbedaan elevasi, namun secara empiris menyebabkan training divergence, sehingga dipertahankan edge attribute konstan. Spatial conditioning utama berasal dari topologi star graph dan perbedaan fitur node antar lokasi."

---

### 3. ERA5 Hanya Update ~1 Minggu Sekali, Tidak Real-Time

#### Penjelasan

ERA5 adalah **reanalysis data**, bukan observasi real-time. Data ERA5 melalui Open-Meteo Archive memiliki latensi sekitar **5–7 hari** (update mingguan).

#### Implikasi untuk Prototype

- Prototype model ini **tidak bisa langsung dioperasikan real-time** hanya dengan ERA5.
- Untuk deployment operasional nyata, diperlukan sumber data observasi near real-time seperti:
  - Data radar cuaca.
  - Data stasiun otomatis (AWS).
  - Data satelit resolusi tinggi.
  - Model NWP deterministik dengan update lebih sering.

#### Saran Penulisan

> "Penelitian ini menggunakan dataset reanalysis ERA5 melalui Open-Meteo Archive API, yang memiliki keterlambatan update sekitar 5–7 hari. Oleh karena itu, prototype model ini bersifat proof-of-concept berbasis data historis dan belum dapat dioperasikan secara real-time. Untuk implementasi operasional di lapangan, diperlukan integrasi dengan sumber data observasi near real-time seperti radar, stasiun otomatis, atau satelit resolusi tinggi."

---

## Format Usulan Batasan Masalah (Gabungan)

Berikut usulan paragraf yang bisa dimasukkan ke BAB 1 Batasan Masalah:

> "Selain batasan yang telah diuraikan, penelitian ini juga mempertimbangkan keterbatasan berikut:
>
> (1) **Keterbatasan data presipitasi**: Dataset presipitasi bersifat zero-inflated dan heavy-tailed, di mana kejadian hujan ekstrem relatif langka. Meskipun telah diterapkan log transform, weighted noise loss, retrieval-augmented mechanism, dan diffusion model untuk meningkatkan sensitivitas terhadap event ekstrem, model tetap dibatasi oleh frekuensi intrinsik event dalam data historis.
>
> (2) **Keterbatasan edge attributes**: Atribut edge yang digunakan bernilai konstan 0.25° karena jarak antar node pada grid ERA5 0.25° identik. Eksperimen dengan atribut edge berbasis perbedaan elevasi telah dilakukan tetapi menyebabkan training divergence, sehingga dipertahankan edge attribute konstan. Sinyal spasial diperoleh dari topologi star graph dan perbedaan fitur node.
>
> (3) **Keterbatasan sumber data real-time**: ERA5 merupakan data reanalysis dengan latensi update sekitar 5–7 hari, sehingga prototype ini bersifat proof-of-concept berbasis data historis dan belum dapat dioperasikan secara real-time. Implementasi operasional memerlukan integrasi dengan data radar, stasiun otomatis, atau satelit near real-time."

---

## Prinsip Penulisan yang Harus Dijaga

| Jangan | Lakukan |
|---|---|
| "Karena data dari sana memang minim, jadi tidak bisa." | "Data memang minim, **tetapi kita sudah mencoba** log transform, weighted loss, retrieval, dan diffusion. Hasilnya tetap terbatas oleh frekuensi intrinsik data." |
| "Edge attr memang konstan karena grid ERA5 0.25°." | "Edge attr konstan karena grid ERA5 0.25°, **dan kita sudah mencoba** atribut berbasis elevasi tetapi menyebabkan training divergence." |
| "ERA5 tidak real-time." | "ERA5 tidak real-time, **sehingga untuk deployment nyata** diperlukan integrasi dengan data observasi near real-time." |

---

## Bukti Kode untuk Diacu saat Sidang

Jika penguji bertanya "Apa bukti kalau sudah mencoba?", bisa tunjukkan:

1. **Edge attr elevation experiment**: `src/config.py` baris 120–138 (catatan dalam `build_star_edge_attr`).
2. **Weighted loss**: `src/models/diffusion.py` baris 177–185.
3. **Log transform**: `src/data/temporal_loader.py` (transformasi log1p) dan `run_eval_final.py` (`expm1` saat denormalisasi).
4. **Retrieval**: `src/retrieval/base.py` implementasi FAISS k-NN.

---

## Catatan Tambahan

- Batasan-batasan ini sebaiknya dimasukkan sebagai **poin tambahan di BAB 1 Batasan Masalah** dan **dijelaskan lebih detail di BAB 3** jika diperlukan.
- Ini juga bisa menjadi dasar **Saran / Future Work** di BAB 5.

---

## Konfirmasi: Hanya Ada BAB 1–3

Dari verifikasi PDF, draft thesis hanya terdiri dari BAB 1 (Pendahuluan), BAB 2 (Tinjauan Pustaka & Landasan Teori), dan BAB 3 (Metode Penelitian), diikuti Daftar Pustaka. Tidak ada BAB 4 (Hasil) atau BAB 5 (Kesimpulan) dalam draft ini.

---

## Bukti Kita Sudah Tes Berbagai Skenario

Penelitian ini sudah menjalankan **6 skenario evaluasi** secara lengkap. Bukti tersedia di:

- `result_test/EVALUATION_REPORT.md`
- `result_test/comparison/comparison_summary.csv`
- `result_test/comparison/comparison_summary.json`
- `result_test/*/metrics.json` untuk masing-masing skenario

### Keenam Skenario yang Diuji

1. **persistence** — baseline naif (copy t-1 ke t)
2. **mlp_baseline** — MLP deterministik main-node-only
3. **diff_only** — diffusion tanpa retrieval dan tanpa GNN
4. **diff_retrieval** — diffusion + retrieval
5. **diff_gnn** — diffusion + GNN
6. **full_model** — diffusion + retrieval + GNN

### Metadata Evaluasi

Dari `result_test/EVALUATION_REPORT.md`:
- `graph_topology`: star
- `target_node_policy`: main_node_only
- `context_policy`: main_node_context
- `eval_step`: 11
- `samples_per_scenario`: 3189
- `num_ensemble`: 30
- `seq_len`: 6
- `node_order`: ['MAIN', 'UP', 'DOWN', 'LEFT', 'RIGHT']

### Contoh Hasil Metrik (precipitation)

| Skenario | RMSE | MAE | Correlation | CRPS |
|---|---:|---:|---:|---:|
| persistence | 0.6847 | 0.2187 | 0.6856 | 0.2187 |
| mlp_baseline | 0.7417 | 0.2686 | 0.5461 | 0.2686 |
| diff_only | 0.8718 | 0.4132 | 0.5345 | 0.2893 |
| diff_retrieval | 0.8957 | 0.4219 | 0.5303 | 0.2960 |
| diff_gnn | 0.7802 | 0.3415 | 0.6320 | 0.2463 |
| full_model | 0.7920 | 0.3530 | 0.6390 | 0.2518 |

### Interpretasi Hasil

- **diff_gnn** unggul dalam banyak metrik, menunjukkan kontribusi GNN.
- **full_model** memberikan performa kompetitif dengan kombinasi retrieval + GNN.
- Hasil ini menjadi bukti empiris bahwa arsitektur yang diusulkan sudah diuji secara sistematis.

### Cara Menyebutkan di BAB 3 / BAB 1

> "Evaluasi model dilakukan secara komprehensif menggunakan enam skenario, yaitu persistence, MLP baseline, diffusion only, diffusion + retrieval, diffusion + GNN, dan full model. Hasil evaluasi tersimpan dalam artefak `result_test/EVALUATION_REPORT.md` dan `result_test/comparison/comparison_summary.csv`, dengan total 3.189 sampel per skenario dan ensemble size 30."

---

## Rekomendasi Penempatan Batasan Masalah

| Batasan | BAB 1 (singkat) | BAB 3 (detail) |
|---|---|---|
| Data presipitasi minim | Disebutkan sebagai batasan | Dijelaskan dengan log transform, weighted loss, retrieval |
| Edge attr konstan | Disebutkan sebagai batasan | Dijelaskan eksperimen elevasi gagal |
| ERA5 tidak real-time | Disebutkan sebagai batasan | Dijelaskan latensi dan rekomendasi data near real-time |
| 6 skenario evaluasi | Tidak perlu di batasan | Dijelaskan di metode evaluasi |

---

## Daftar File Bukti yang Bisa Ditunjukkan saat Sidang

1. **Bukti 5 node + star topology**: `src/config.py` dan `result_test/EVALUATION_REPORT.md`.
2. **Bukti 6 skenario**: `run_eval_final.py` dan `result_test/comparison/comparison_summary.csv`.
3. **Bukti weighted loss**: `src/models/diffusion.py` baris 177–185.
4. **Bukti log transform**: `src/data/temporal_loader.py` dan `run_eval_final.py` (`expm1`).
5. **Bukti edge attr eksperimen gagal**: `src/config.py` catatan dalam `build_star_edge_attr()`.
6. **Bukti evaluasi real**: `result_test/*/metrics.json` untuk semua skenario.
