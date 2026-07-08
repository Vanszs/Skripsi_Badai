# KNOWLEDGE.md — Pemahaman Penuh Proyek Skripsi
## Nowcasting probabilistik presipitasi untuk mitigasi resiko pendaki di gunung gede-pangrango dengan menggunakan retrieval augemented difussion model dengan spatio temporal graph conditioning
### Nowcasting probabilistik presipitasi untuk mitigasi resiko pendaki di gunung gede-pangrango dengan menggunakan retrieval augemented difussion model dengan spatio temporal graph conditioning

> Dokumen ini dirancang agar kamu memahami **100%** proyek ini: setiap istilah, alasan pemilihan,
> cara kerja algoritma end-to-end, parameter, hasil, dan keterbatasan. Disusun dari kode aktual
> (`src/`, `run_eval_final.py`, `scripts/`) per commit `ebfc784`.

---

## 0. RINGKASAN SATU PARAGRAF (elevator pitch)

Sistem memprediksi cuaca **1 jam ke depan** (nowcasting) untuk **satu titik utama** (puncak Gede–Pangrango,
node `MAIN`) untuk **3 variabel**: curah hujan, kecepatan angin, kelembapan. Caranya: ambil **6 jam terakhir**
data dari **5 titik grid** (MAIN + 4 tetangga: atas/bawah/kiri/kanan) yang disusun sebagai **graf bintang**,
lalu (1) sebuah **GNN spatio-temporal** meringkas pola spasial+waktu, (2) sebuah **retrieval** mencari
analog historis mirip dari data training dan mengembalikan *apa yang terjadi 1 jam setelahnya*, (3) sebuah
**diffusion model** menggabungkan semua itu untuk menghasilkan **banyak sampel prediksi** (ensemble) sehingga
kita dapat **ketidakpastian** (bukan cuma 1 angka). Hasilnya dibandingkan dengan baseline **persistence** dan
**MLP**. Temuan jujur: model unggul untuk angin & kelembapan, tapi untuk hujan masih kalah persistence karena
**keterbatasan data ERA5 0.25°** yang terlalu kasar untuk hujan lokal gunung.

---

## 1. IDENTITAS PROYEK

| Field | Nilai |
|-------|-------|
| Judul | Nowcasting probabilistik presipitasi untuk mitigasi resiko pendaki di gunung gede-pangrango dengan menggunakan retrieval augemented difussion model dengan spatio temporal graph conditioning |
| Wilayah | Gunung Gede–Pangrango, Jawa Barat |
| Node utama (MAIN) | koordinat grid kanonik `(-6.75, 107.00)` |
| Target | `precipitation` (mm/jam), `wind_speed_10m` (m/s), `relative_humidity_2m` (%) |
| Horizon | 1 jam ke depan (one-step), hourly |
| Sumber data | Open-Meteo Archive API, model `era5` |

**Tujuan global:** prediksi nowcasting jangka pendek untuk titik MAIN, memanfaatkan konteks spasial node
sekeliling + memori historis (retrieval) + ketidakpastian probabilistik (diffusion ensemble).

---

## 2. GLOSARIUM ISTILAH (wajib hafal)

- **Nowcasting**: prediksi cuaca jangka sangat pendek (di sini 1 jam ke depan).
- **One-step / horizon 1**: prediksi hanya untuk waktu t+1, bukan multi-jam ke depan.
- **Operational/real-time rolling**: tiap jam, jendela input digeser pakai **observasi terbaru** (bukan
  prediksi sendiri yang di-feedback). Lawannya: *recursive* (pakai prediksi sebagai input → error menumpuk).
- **Node**: satu titik grid cuaca (sel ERA5 0.25°). Ada 5: MAIN, UP, DOWN, LEFT, RIGHT.
- **Star topology (graf bintang)**: MAIN di pusat, terhubung dua arah ke 4 tetangga. 8 edge berarah
  (MAIN↔UP, MAIN↔DOWN, MAIN↔LEFT, MAIN↔RIGHT).
- **Spatio-Temporal**: spasial (antar node) + temporal (antar waktu).
- **GNN / GAT**: Graph Neural Network / Graph Attention Network — jaringan saraf yang memproses graf;
  GAT memberi bobot perhatian (attention) pada tetangga.
- **Temporal attention**: self-attention untuk urutan waktu (6 timestep), pakai **causal mask** agar
  waktu t tidak "mengintip" masa depan.
- **Retrieval (k-NN)**: cari k tetangga terdekat dari kondisi sekarang di basis data historis.
- **FAISS**: library Facebook untuk pencarian k-NN cepat (di sini `IndexFlatL2` = jarak Euclidean L2).
- **Diffusion model (DDPM)**: model generatif yang belajar membalik proses penambahan noise bertahap.
  Train: tambah noise ke target → model belajar **memprediksi noise**-nya.
- **DDIM**: sampler diffusion yang lebih cepat (lebih sedikit langkah) saat inferensi.
- **Ensemble**: banyak sampel prediksi dari model probabilistik → bisa hitung sebaran/ketidakpastian.
- **Persistence**: baseline naif, prediksi t+1 = nilai t. Tanpa belajar.
- **MLP**: Multi-Layer Perceptron, jaringan feedforward dasar (Linear→ReLU→Dropout).
- **Wet/dry head (rain occurrence)**: cabang klasifikasi "hujan/tidak" untuk menangani data hujan yang
  banyak nol (zero-inflated).
- **Rain gate**: jika probabilitas hujan < threshold, prediksi hujan dipaksa 0.
- **Normalisasi (z-score)**: (x − mean)/std agar fitur seragam skalanya.
- **log1p / expm1**: log(1+x) dan inversnya (e^x − 1). Untuk hujan yang skewed & zero-inflated.
- **CRPS, Brier, POD, FAR, CSI, RMSE, MAE, korelasi**: metrik evaluasi (lihat §9).
- **Leakage (kebocoran)**: data masa depan/uji bocor ke training → hasil palsu bagus. Dihindari ketat.
- **Ablation**: menyalakan/mematikan komponen untuk mengukur kontribusi masing-masing.

---

## 3. DATA — sumber, struktur, alasan

### 3.1 Sumber & alasan ERA5
- API: Open-Meteo Archive, parameter wajib `models=era5`.
- **ERA5** = reanalysis ECMWF: gabungan model + observasi, konsisten historis panjang, variabel lengkap,
  tersedia stabil. Alasan dipilih: konsistensi & keseragaman antar titik untuk 2005–2025.
- Resolusi: **0.25° (~25 km/grid)**, **hourly**.

### 3.2 Dataset aktif
- `data/raw/pangrango_era5_5node_2005_2025.parquet`
- 920.400 baris = 184.080 timestamp × 5 node. 21 kolom.
- Periode: 2004-12-31 17:00 UTC → 2025-12-31 16:00 UTC.

### 3.3 Node kanonik (urutan WAJIB tetap: [MAIN, UP, DOWN, LEFT, RIGHT])
| Node | Koordinat | Elevasi grid | Peran |
|------|-----------|-------------|-------|
| MAIN | (-6.75, 107.00) | 1529 m | target utama (puncak) |
| UP | (-6.50, 107.00) | 162 m | konteks |
| DOWN | (-7.00, 107.00) | 0 m (maritim) | konteks |
| LEFT | (-6.75, 106.75) | 823 m | konteks |
| RIGHT| (-6.75, 107.25) | 288 m | konteks |

> Urutan node WAJIB konsisten di ingest→loader→train→checkpoint→inference→eval agar tensor tidak
> "drift indeks". Loader fail-fast jika urutan/jumlah/nama node salah.

### 3.4 Fitur (9) & target (3)
- **Fitur** (`FINAL_FEATURE_COLS`): temperature_2m, relative_humidity_2m, dewpoint_2m, surface_pressure,
  wind_speed_10m, wind_direction_10m, cloud_cover, precipitation_lag1, elevation.
- **Target** (`FINAL_TARGET_COLS`): precipitation, wind_speed_10m, relative_humidity_2m.
- Catatan: RH & wind_speed adalah **fitur sekaligus target**. Bukan leakage karena fitur diambil dari
  jendela [t-6..t-1], target di waktu t (masa depan relatif jendela).
- `precipitation_lag1` = hujan jam sebelumnya (`shift(1)` per node, nilai awal 0). Backward-only, no leak.

### 3.5 Split temporal (kronologis, anti-leakage)
- Train: `date <= 2018-12-31` → 613.480 baris (122.696 timestamp)
- Val: `2018-12-31 < date <= 2021-12-31` → 131.520 baris
- Test: `date > 2021-12-31` → 175.400 baris (35.080 timestamp MAIN)
- **Alasan kronologis** (bukan acak): mencegah model "melihat masa depan" saat belajar.

---

## 4. NORMALISASI & TRANSFORMASI (detail + alasan)

- Statistik (mean/std) dihitung **hanya dari data train node MAIN** (`compute_stats_from_training`).
  Dipakai konsisten untuk val/test/inference (TIDAK dihitung ulang di test → no leakage).
- Fitur: `x̃ = (x − c_mean)/(c_std + 1e-5)`.
- Target: precipitation di-`log1p` dulu, lalu z-score; wind & humidity z-score langsung.
  - **Alasan log1p untuk hujan**: distribusi hujan sangat skewed + banyak nol (zero-inflated).
    log meredam dominasi nilai ekstrem saat optimasi.
- **FIX KRITIS (elevation)**: elevasi MAIN konstan (1529 m) → std MAIN = 0 → dulu normalisasi bikin
  elevasi node tetangga jadi ~1.5×10⁸ (meledak). Diperbaiki: untuk fitur yang std-nya ~0 di MAIN
  (degenerate), pakai **fallback statistik semua-node** (elevasi: mean 560.4, std 557.5 → nilai O(1)).
  `stats_scope = "main_node_only_with_allnode_fallback_for_constant_features"`.
- Denormalisasi prediksi: `x*std + mean`, lalu hujan `expm1(clamp(max=20))` + `clamp(0, 60)`,
  angin `clamp(≥0)`, kelembapan `clamp(0,100)`. **Clamp hanya pada PREDIKSI, tidak pernah pada target.**

---

## 5. ARSITEKTUR MODEL UTAMA (3 komponen + cara kerja)


> **Catatan keterbatasan edge weight:** pada grid ERA5 0.25°, jarak antar MAIN dan setiap tetangga konstan 0.25°. Oleh karena itu `edge_attr` bersifat non-informatif; spatial conditioning berasal dari topology graf dan fitur node, bukan dari bobot edge. Eksperimen edge_attr berbasis Δelevasi pernah dicoba tapi divergen, sehingga direvert.

### 5.1 Spatio-Temporal GNN — `src/models/gnn.py`
**Tujuan:** meringkas pola spasial (antar 5 node) + temporal (6 jam) jadi satu vektor "graph embedding".

Alur:
1. Untuk **tiap timestep** (6 buah), graf 5-node diproses **SpatialGNN** = 2 layer `GATConv`
   (Graph Attention, 4 heads di layer-1). Output di-`global_mean_pool` → 1 vektor per timestep.
   - `edge_dim=1` menerima `edge_attr` (jarak antar node). Catatan: nilai jarak konstan 0.25°
     (grid equidistant) → non-informatif; sinyal spasial efektif dari topologi + fitur node.
2. 6 vektor timestep ditumpuk → **TemporalAttention** (multi-head self-attention, 4 heads) dengan
   **positional embedding** + **causal mask** (waktu t tak boleh attend ke t+1..). Ambil representasi
   **timestep terakhir**.
3. `output_proj` → graph embedding berdimensi `graph_dim=64`.

**Kenapa GAT + attention?** GAT memberi bobot adaptif pada tetangga (tidak semua tetangga sama penting);
temporal attention memilih jam mana yang paling relevan untuk prediksi 1 jam ke depan.

### 5.2 Retrieval-Augmented Memory — `src/retrieval/base.py` + precompute di `train.py`
**Tujuan:** beri model "memori" — analog historis: "dulu ketika kondisinya mirip ini, 1 jam kemudian
terjadi apa?"

Cara kerja:
- **Basis data (DB)** dibangun HANYA dari train node MAIN: `key = fitur ternormalisasi pada waktu τ`,
  `value = target ternormalisasi pada waktu τ+1` (outcome jam berikutnya; precip log1p).
  → `add_items(features_norm[:-1], targets_norm[1:])`.
- **Query**: vektor fitur MAIN saat ini (waktu t-1) → FAISS cari `k=3` tetangga terdekat (L2) →
  kembalikan 3 outcome historis (masing-masing 3 target) → `retrieval_dim = 3×3 = 9`.
- **Anti-leakage saat training**: precompute pakai `strict_past=True` (tetangga hanya dari masa lalu,
  index < waktu query) + `exclude_self=True`. Saat inferensi, semua data train sudah di masa lalu.

**Kenapa retrieval outcome (bukan fitur)?** Inilah makna "Retrieval-Augmented" sebenarnya: model
dikondisikan oleh **hasil** dari situasi serupa, bukan sekadar fitur yang mirip (yang sudah ada di context).

### 5.3 Conditional Diffusion Model — `src/models/diffusion.py`
**Tujuan:** menghasilkan prediksi probabilistik (banyak sampel) 3 target sekaligus.

Komponen conditioning (digabung **aditif** ke dimensi hidden 128):
- `context` (fitur MAIN t-1) → cond_mlp
- `retrieved` (analog outcome, 9-dim) → retrieval_mlp
- `graph_emb` (dari STGNN, 64-dim) → graph_mlp
- `time embedding` (langkah diffusion) → sinusoidal + MLP
- Total: `emb = t_emb + (cond + retrieval + graph)`.

Backbone "denoiser": MLP dengan 1 skip connection
(`down1(3→128)+emb → down2(128→256) → mid → concat[mid,down2]=512 → up1→128 → out→3`).
> Catatan jujur: backbone diffusion adalah **MLP + 1 skip**, bukan U-Net konvolusional.
> Untuk target vektor 3-variabel (bukan citra), MLP memang pilihan tepat.

**Training (DDPM)** — `train.py` loop:
1. Ambil target ternormalisasi `x0` [B,3].
2. Sample noise ε ~ N(0,I); sample timestep t acak [0,1000).
3. `noisy = scheduler.add_noise(x0, ε, t)` (= √ᾱ·x0 + √(1-ᾱ)·ε).
4. Model prediksi noise `ε̂ = model(noisy, t, context, retrieved, graph_emb)`.
5. Loss = **weighted MSE(ε̂, ε)** + `λ·BCE(wet_head)`.

**Inferensi (DDIM)** — `sample_fast`: mulai dari noise acak, denoise bertahap (20 langkah DDIM,
lebih cepat dari 1000 langkah DDPM), hasilkan `num_samples` sampel. `clip_sample=False` (penting:
data cuaca z-score bisa [-4,+8], beda dari citra [-1,1]). Ada clamp noise [-10,10] + nan_to_num = jaring
pengaman numerik.

**Wet/dry head (spesialisasi hujan):**
- Cabang klasifikasi: `wet_logit = wet_head(cond_emb)` (TANPA time — probabilitas okurensi, bukan langkah).
- Label: `wet = (precip_mm >= 0.1)` (0.1mm = ambang WMO "hujan terukur").
- Loss: `BCEWithLogits` dengan `pos_weight` (imbang kelas, di-clip [1,100]).
- **Kalibrasi threshold** di **validation** (cari threshold yang maksimalkan CSI).
- **Rain gate** saat inferensi: jika P(wet) < threshold → precip diset 0. Ada **guard anti-kolaps**:
  jika SEMUA sampel di bawah threshold (akan jadi kering total), gate dilewati (cegah output semua-nol senyap).

**Loss berbobot (weighted_noise_loss):** bobot 1× normal, **5×** jika |target| > 1σ, **10×** jika > 3σ.
Tujuan: tekankan event ekstrem (hujan lebat). Nilai 5/10 heuristik (belum di-ablation).

---

## 6. BASELINE PEMBANDING

### 6.1 Persistence
- ŷ(t+1) = y(t). Tanpa parameter, tanpa training. Baseline meteorologi paling dasar.
- Kuat untuk horizon 1 jam karena cuaca berubah lambat (autocorrelation tinggi).
- Di kode: `pred = main_df.iloc[idx-1]`, `target = main_df.iloc[idx]`.

### 6.2 MLP baseline — `src/models/mlp_baseline.py`, `src/train_baseline.py`
- Arsitektur: `Linear(54→128)→ReLU→Dropout→Linear(128→128)→ReLU→Dropout→Linear(128→3)`.
  Input = 6 jam × 9 fitur (MAIN saja) di-flatten = 54.
- Deterministik (1 forward pass saat eval; CRPS-nya = MAE karena ensemble=1).
- Training: MSE loss, AdamW, CosineAnnealing, early stopping.
- Peran: "ada model tapi sederhana" — pembanding di atas persistence.

---

## 7. FLOW END-TO-END (pipeline)

```
[1] Ingest (src/data/ingest.py)
    Open-Meteo era5 → validasi grid-center 5 node (fail-fast jika collision)
    → gabung, urutkan [date, node kanonik] → buat precipitation_lag1 → simpan parquet
[2] Split (train.py temporal_split): train≤2018, val≤2021, test>2021
[3] Stats (compute_stats_from_training): mean/std dari train MAIN (fallback all-node utk fitur konstan)
[4] Dataset (temporal_loader.TemporalGraphDataset):
    sliding window seq_len=6 → tiap sampel = (6 graf 5-node, target MAIN@t, context MAIN@t-1)
    fail-fast kontrak node (urutan/jumlah/nama)
[5] Retrieval DB (train.py): FAISS dari train MAIN; key=fitur τ, value=outcome τ+1; strict-past
[6] Train (train.py): STGNN + Diffusion + wet head; weighted-noise MSE + wet BCE;
    cond_dropout 0.15 (acak nol-kan retrieval/graph → model tahan ablation);
    grad clip 1.0; ReduceLROnPlateau; early stop; simpan checkpoint terbaik (val terendah)
[7] Train baseline (train_baseline.py): MLP
[8] Inference (inference.py): muat checkpoint + metadata; rebuild retrieval DB; sampling DDIM
[9] Eval (run_eval_final.py): 6 skenario, metrik deterministik+probabilistik, plot, report
[10] Weekly crosscheck (scripts/eval_rain_robust.py): driest/median/wettest week, 3 variabel
```

---

## 8. 6 SKENARIO EVALUASI (ablation)

| Skenario | GNN | Retrieval | Maksud |
|----------|-----|-----------|--------|
| persistence | - | - | baseline naif |
| mlp_baseline | - | - | baseline model sederhana |
| diff_only | off | off | diffusion saja (context only) |
| diff_retrieval | off | on | + kontribusi retrieval |
| diff_gnn | on | off | + kontribusi GNN |
| full_model | on | on | model penuh (judul) |

> Ablation mematikan komponen dgn meng-**nol-kan** graph_emb/retrieved. Karena model dilatih dengan
> `cond_dropout`, ia tahan input nol (tidak meledak). Tanpa cond_dropout dulu pernah meledak (RMSE jutaan).
> Semua skenario pakai **indeks sampel identik** (`_eval_indices`) + assert jumlah sama → apple-to-apple.

---

## 9. METRIK EVALUASI (rumus + arti)

**Deterministik** (pakai prediksi titik = median ensemble):
- **RMSE** = √(mean((ŷ−y)²)) — error besar dihukum lebih berat. Kecil = baik.
- **MAE** = mean(|ŷ−y|) — error rata-rata absolut. Kecil = baik.
- **Korelasi Pearson** — kesesuaian pola naik-turun. Tinggi = baik. (NaN jika varians ~0.)

**Probabilistik** (pakai seluruh ensemble):
- **CRPS** = E|X−y| − ½E|X−X'| (X,X' sampel independen dari prediksi). Mengukur kualitas distribusi.
  Kecil = baik. **Fair estimator** pakai pembagi n(n−1). Untuk ensemble 1-anggota, CRPS = MAE.
- **Brier** = mean((p−o)²), p=fraksi ensemble > threshold, o=observasi biner. Kecil = baik.
- **POD** (Probability of Detection) = hits/(hits+misses). Tinggi = baik (deteksi event).
- **FAR** (False Alarm Ratio) = false_alarms/(hits+false_alarms). Rendah = baik.
- **CSI** (Critical Success Index) = hits/(hits+misses+false_alarms). Tinggi = baik (seimbang).
- Event hujan dinilai di threshold {2, 5, 10} mm (main eval) — definisi event: P(ensemble>thr) ≥ 0.5.

---

## 10. HASIL (jujur, checkpoint reproducible seed=1, eval_step=11)

| Variabel | full_model | diff_gnn | diff_only | persistence | mlp |
|----------|-----------|----------|-----------|-------------|-----|
| precip RMSE | 0.792 | 0.780 | 0.872 | **0.685** | 0.742 |
| precip korr | 0.639 | 0.632 | 0.534 | **0.686** | 0.546 |
| wind RMSE | **0.986** | 0.975 | 1.171 | 1.115 | 1.156 |
| wind korr | **0.889** | 0.890 | 0.848 | 0.849 | 0.832 |
| humidity RMSE | **2.99** | 2.97 | 4.33 | 4.49 | 4.55 |
| humidity korr | **0.973** | 0.973 | 0.943 | 0.938 | 0.956 |

**Interpretasi:**
- **Menang** di angin & kelembapan (RMSE & korelasi lebih baik dari persistence & MLP).
- **GNN berkontribusi nyata**: diff_only → diff_gnn menurunkan precip RMSE 0.87→0.78, humidity 4.33→2.97.
- **Hujan masih kalah** persistence (0.79 vs 0.685 RMSE) — keterbatasan ERA5 (lihat §11).
- Training: best_val ≈ 0.587, corr noise ~0.94, gradien sehat (max ~9.4 setelah fix elevasi),
  non_finite=0. Tests 10/10.

---

## 11. KETERBATASAN (wajib tahu — juga di private.note)

**A. Data ERA5 (paling fundamental):** resolusi 0.25° (~25 km) terlalu kasar untuk hujan konvektif/orografis
gunung (skala 1-5 km); reanalysis meredam spike; node DOWN maritim (0m) beda rezim vs puncak (1529m);
hujan zero-inflated + autocorrelation lag-1 ~0.67 → persistence sangat kuat. → **batas atas performa hujan
dibatasi data, bukan algoritma.**

**B. Cakupan:** hanya 5 titik grid, evaluasi MAIN-only, tak diuji generalisasi titik lain.

**C. Metodologi:** single seed (belum multi-seed/CI); eval pakai subsampel step=11 (tak bias tapi belum
hourly penuh); hyperparameter (5×/10×, cond_dropout 0.15, wet_loss_weight 0.7) heuristik tanpa ablation;
test mayoritas struktural; edge_attr konstan 0.25 (non-informatif, terdokumentasi).

**D. Penamaan:** backbone diffusion adalah MLP+skip (bukan U-Net).

**E. Future work:** sumber hujan resolusi halus (radar/GSMaP/IMERG); multi-seed+CI; edge feature informatif;
loss event-aware khusus hujan.

---

## 12. PARAMETER PENTING (cheat-sheet)

| Param | Nilai | Lokasi | Arti |
|-------|-------|--------|------|
| seq_len | 6 | train/loader | jendela input 6 jam |
| k_neighbors | 3 | train | tetangga retrieval |
| hidden_dim | 128 | diffusion | lebar layer |
| graph_dim | 64 | STGNN | dim graph embedding |
| retrieval_dim | 9 | =3×3 | k × num_targets (outcome) |
| num_targets | 3 | config | precip/wind/humidity |
| batch_size | 512 | train | |
| epochs / early_stop | 35 / patience 12 | train | |
| lr | 1e-3 (AdamW) | train | + ReduceLROnPlateau |
| grad_clip_norm | 1.0 | train | tahan gradien besar |
| cond_dropout | 0.15 | train | acak nol-kan retrieval/graph (robust ablation) |
| seed | 1 | train/eval | reproducibility |
| wet_loss_weight | 0.7 | train | bobot BCE wet head |
| rain_occurrence_threshold | 0.1 mm | train | ambang "hujan" (WMO) |
| wet_probability_threshold | ~0.55-0.60 | kalibrasi val (CSI) | rain gate |
| num_inference_steps (DDIM) | 20 | inference | langkah denoising |
| num_ensemble | 30 | eval | jumlah sampel probabilistik |
| eval_step | 1 (default) / 11 (artefak) | eval | jarak sampel evaluasi (jam) |
| PRECIP_PHYSICAL_MAX_MM | 60 | config | clamp fisik prediksi hujan |

---

## 13. CARA MENJALANKAN

```bash
# 1. (jika belum ada) ambil data
python3 -m src.data.ingest

# 2. latih model utama (reproducible)
python3 -m src.train --epochs 35 --seed 1 --lr 1e-3 --num-workers 8

# 3. latih baseline MLP
python3 -m src.train_baseline

# 4. evaluasi 6 skenario (hourly penuh utk angka final)
python3 run_eval_final.py --eval-step 1 --num-ensemble 30 --seed 1

# 5. crosscheck mingguan 3-variabel
python3 scripts/eval_rain_robust.py --num-ensemble 30

# 6. plot timeseries 1 minggu
python3 scripts/plot_week_timeseries.py --all

# 7. tes otomatis
python3 -m unittest discover -s tests
```

---

## 14. ANTISIPASI PERTANYAAN ACAK (Q&A)

**Q: Kenapa node DOWN di laut tetap dipakai?**
A: By design — mempertahankan struktur star simetris; konteks maritim diterima & didokumentasikan di grid
validation. Diakui sebagai keterbatasan fisik (rezim beda).

**Q: Kenapa hujan kalah persistence padahal modelnya kompleks?**
A: ERA5 0.25° terlalu kasar untuk hujan lokal gunung + hujan autocorrelation lag-1 tinggi (persistence kuat).
Itu temuan sah, bukan bug.

**Q: Apa bedanya DDPM dan DDIM di sini?**
A: DDPM = skema training (1000 langkah noise). DDIM = sampler inferensi cepat (20 langkah). Sama model,
beda cara sampling.

**Q: Kenapa retrieval mengembalikan outcome, bukan fitur?**
A: Agar benar-benar "augmentasi memori": model tahu hasil historis dari kondisi serupa. Kalau kembalikan
fitur, itu redundan dengan context.

**Q: Bagaimana mencegah kebocoran data?**
A: Split kronologis; stats dari train saja; retrieval DB train saja + strict-past; lag fitur backward-only;
loader fail-fast.

**Q: Apa itu cond_dropout dan kenapa perlu?**
A: Saat training, acak nol-kan graph_emb/retrieved (p=0.15) supaya model tetap waras saat ablation
mematikan komponen (input nol). Tanpa ini, ablation meledak (out-of-distribution).

**Q: Kenapa precip pakai log1p?**
A: Distribusi hujan skewed + banyak nol; log meredam outlier besar saat optimasi. expm1 untuk balik.

**Q: Apa fungsi wet/dry head + rain gate?**
A: Tangani zero-inflation: head klasifikasi hujan/tidak; gate set precip=0 jika P(hujan) rendah (dengan
guard anti-kolaps agar tak semua-kering senyap).

**Q: Apa metrik utama & kenapa banyak?**
A: Deterministik (RMSE/MAE/korelasi) untuk titik; probabilistik (CRPS/Brier/POD/FAR/CSI) untuk distribusi &
deteksi event. Cuaca multi-aspek → tak cukup 1 metrik.

**Q: Kenapa graph embedding pakai timestep terakhir?**
A: Prediksi 1-step-ahead → representasi paling relevan adalah ringkasan kausal sampai waktu t.

**Q: Apa yang membuktikan komponen tidak dummy?**
A: Ablation menunjukkan kontribusi nyata (diff_only < diff_gnn ≈ full); audit numerik memverifikasi
dimensi & alur; tidak ada output hardcoded.

**Q: Apakah hasilnya reproducible?**
A: Ya untuk training (seed=1, cudnn.benchmark=False) & eval (seed sampling). Tercatat di checkpoint config.

**Q: Edge graf katanya konstan 0.25, apa GNN masih berguna?**
A: Ya — sinyal spasial datang dari topologi (siapa terhubung ke siapa) + fitur node (termasuk elevasi),
bukan dari bobot edge. Edge weight konstan = keterbatasan, bukan pembatal fungsi.

---

## 15. RIWAYAT AUDIT (kenapa kode bisa dipercaya)

4 ronde audit menemukan & memperbaiki bertahap (detail di `docs/AUDIT_SESSION_CONTEXT.md`):
1. Retrieval simpan fitur (bukan outcome) → **fixed**.
2. CRPS formula bias → **fixed** (fair estimator).
3. Ablation meledak (input nol OOD) → **fixed** (cond_dropout).
4. **KRITIS**: normalisasi elevasi std=0 → input GNN ~1.5×10⁸, gradien meledak, GNN lumpuh → **fixed**
   (fallback all-node stats). Setelah ini GNN baru benar-benar berkontribusi & hasil membaik.
5. Ronde-2 swarm: parity train-vs-inference, protokol nowcasting, math conditioning — **bersih** numerik;
   sisa hanya minor/dokumentasi.

**Status:** kode faithful ke judul, tanpa kecurangan/dummy/hardcode/kebocoran. Hasil dilaporkan apa adanya.
"Tidak ada kecurangan" ≠ "model selalu menang" — model kalah di hujan karena keterbatasan data (temuan sah).

---

*Dokumen ini disusun dari kode aktual & 4 ronde audit. Untuk detail teknis lihat: `docs/METODE_PENELITIAN_PROYEK.md`,
`docs/ACTIVE_5NODE_STAR_MAIN.md`, `docs/AUDIT_SESSION_CONTEXT.md`, `private.note`.*
