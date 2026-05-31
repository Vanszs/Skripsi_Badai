# Konteks Sesi Audit & Perbaikan — RA-Diffusion 5-Node Star

Disimpan: 2026-05-30 ~23:06 WIB
Proyek: Nowcasting Probabilistik Cuaca Multi-Variabel, Gunung Gede–Pangrango,
Retrieval-Augmented Diffusion + Spatio-Temporal Graph Conditioning (5-node star, main-node-only).

Dokumen ini merekam alur diskusi, temuan audit, perbaikan, hasil re-evaluasi, dan status terkini
agar sesi berikutnya bisa lanjut tanpa kehilangan konteks.

---

## 1. Permintaan user (kronologis)

1. Audit detail end-to-end (algoritma, dataset, flow, nowcasting, target, fitur, parameter). Cari
   yang tidak konsisten/ambigu/dummy/asal-fix/output palsu-hardcoded/tidak sesuai judul. Pakai sub-agent.
2. Buat plan MD robust (file yang diedit, dampak, logika), buat todo list, fix semua sekaligus,
   lalu evaluasi ulang persis seperti audit pertama; lapor hanya bila robust. (Buat plan MD dulu.)
3. "Sudah saya edit" → periksa git changes, cek sesuai plan, cek issue tersisa / lebih buruk,
   audit ulang detail seperti semula.
4. Pertanyaan: apakah dataset 5-node + gunung jadi pusat + algoritma hybrid benar-benar sesuai
   tujuan & robust (terlepas dari 1 error baru).
5. Penegasan: batasan di luar kode (ERA5 jelek, terrain) = wajar & itu temuan sah. Yang ditanya:
   **apakah semua kode sudah menerapkan judul & algoritma yang ditentukan**, terlepas berhasil/tidak.
6. "Jadi sisa 1 saja kemudian bersih?"
7. User menjalankan training sendiri (cond-dropout). Tanya progres/epoch & cara kerja patience.
8. Simpan konteks percakapan ke MD (dokumen ini).

---

## 2. Sistem & lingkungan (terverifikasi)

- OS Linux. Python kerja: `python3` = pyenv 3.11.9, torch 2.7.1+cu118, torch_geometric, faiss,
  diffusers tersedia. (`.venv/` berisi `.exe` Windows — TIDAK dipakai di Linux.)
- GPU: NVIDIA RTX 3050 Laptop (CUDA ada).
- Dataset aktif: `data/raw/pangrango_era5_5node_2005_2025.parquet` (920.400 baris, 5 node, hourly).
- Split: train ≤ 2018-12-31, val ≤ 2021-12-31, test > 2021-12-31. Test MAIN = 35.080 baris.
- Node & elevasi grid (ERA5 0.25°): MAIN 1529 m (-6.75,107.0), UP 162 m, DOWN 0 m (maritim),
  LEFT 823 m, RIGHT 288 m. Topologi star, 8 edge terarah, target main-node-only.

---

## 3. AUDIT AWAL — 15 isu (ringkas)

KRITIS:
1. Retrieval menyimpan FITUR sebagai value (`add_items(feat,feat)`) → bukan analog outcome;
   redundan dgn context. Ablation: retrieval malah memperburuk precip.
2. Model utama KALAH baseline pada precip (full RMSE 1.69 vs persistence 1.38 vs MLP 1.30).
3. Persistence off-by-one di eval (`range(seq_len+1,...)` vs `seq_len`) → sampel beda → tidak
   apple-to-apple.
4. Klaim hourly tapi `eval_step=24` (1 sampel/hari, selalu jam 07:00).
5. Rain gate dikalibrasi val (thr 0.55) tapi di test wet_prob max 0.305 → seluruh precip di-nol-kan.

MAYOR:
6. `edge_attr` (jarak/angin) dihitung tapi tak pernah masuk GATConv (loader buat Data tanpa edge_attr).
7. Dead code menyiratkan fitur tak ada: WeatherStateEncoder, SimpleGraphEncoder,
   create_pangrango_graph, PangrangoGraphBuilder/build_dynamic_edges, create_temporal_graphs(NotImplemented).
8. Artefak hasil tanpa skrip penghasil (audit_nowcasting_protocols.json, audit_mlp_week_metrics.json,
   latest_1week_3feature_*, actual_vs_pred_nowcasting_full_model.csv) → tidak reproducible.
9. Angka jendela sama saling bertentangan antar file (RMSE 1.09 vs 1.14).
10. recursive_closed_loop degenerate (precip n_unique=1; humidity RMSE meledak 14.6).

MINOR:
11. CRPS bias: `/n^2` (harusnya `/(n(n-1))`), `abs(term2)` tak perlu; persistence CRPS=NaN (1 member).
12. Dua definisi "event hujan" (gate wet-head vs ensemble-fraction) tak direkonsiliasi.
13. MLP "ensemble" via MC-dropout = pseudo-uncertainty, dibandingkan setara dgn ensemble diffusion.
14. Test 7/7 hanya struktural, tak uji kebenaran numerik.
15. Doc usang: respon.md (val 0.0966, path 3-node), default wet_loss_weight 0.5 vs run 0.7.

Yang BENAR sejak awal: kontrak 5-node (ordering/fail-fast/grid), split kronologis tanpa leakage,
stats train-only, retrieval train-only strict-past+exclude-self, DDPM/DDIM nyata, wet/dry head terlatih.

Kesimpulan audit awal: 3 pilar judul (Retrieval-Augmented, Graph Conditioning, hourly) tidak benar-benar
terbukti di kode/hasil.

---

## 4. RENCANA PERBAIKAN

- Ditulis ke `docs/AUDIT_FIX_PLAN.md` (216 baris): keputusan per isu, perubahan per-file, matriks dampak,
  urutan A→B→C→D→E, risiko, Definition of Done, catatan integritas (tak boleh hardcode agar menang).
- Todo list 12 task (Tahap A1..A8 edit, B validasi statis, C retrain 1×, D re-eval, E re-audit).

---

## 5. PERBAIKAN YANG DILAKUKAN USER (git changes — terverifikasi sesuai plan)

22 file berubah. Inti:
- #1 `src/train.py` + `src/inference.py`: DB = `add_items(features[:-1], targets[1:])`,
  value = outcome target τ+1 (precip log1p lalu normalisasi). `retrieval_dim = num_targets*k = 9`.
  Verifikasi: stored_value_dim (122695, 3); denorm analog plausibel (precip≥0, RH~96%).
- #6 `src/config.py` `build_star_edge_attr` (jarak euclidean lat/lon); `src/models/gnn.py`
  `GATConv(edge_dim=1)` + teruskan edge_attr; `temporal_loader.py` & `inference.py` sertakan edge_attr.
- #7 hapus dead code (WeatherStateEncoder, SimpleGraphEncoder, create_pangrango_graph,
  src/graph/builder.py dihapus). Grep referencer = nihil.
- #11 `probabilistic_metrics.py` CRPS fair `/(n(n-1))`, hapus abs, n>=1 → single-member = MAE.
- #3/#4/#5 `run_eval_final.py`: `_eval_indices` satu sumber; assert sample-count identik;
  default `eval_step=1` + `--max-eval-samples`; rain-gate post-loop dgn guard anti-kolaps.
- #8/#10 hapus artefak yatim; `scripts/eval_rain_robust.py` ditulis ulang jadi weekly one-step 3-var
  (model+persistence+mlp), seed tetap, output `weekly_crosscheck_3var.csv` + meta JSON; recursive dibuang.
- #14 tests: retrieval_dim==num_targets*k; inference smoke cek edge_attr & retrieved shape [*,k,3];
  CRPS known-case (single→MAE, 2-member fair=1.5).
- #15 `respon.md` ditandai USANG; `docs/METODE_PENELITIAN_PROYEK.md` update retrieval-outcome & edge_attr;
  default `wet_loss_weight` 0.5→0.7. Tidak ada angka hasil di-hardcode di docs.

Validasi statis: `py_compile` OK; **unit test 10/10 PASS**.

---

## 6. RETRAIN #1 (checkpoint 20:02) & RE-EVAL (eval_step=11, 3189 sampel, menutupi semua jam + rentang penuh)

Checkpoint: retrieval_dim=9, GATConv edge_dim=1, best val ~0.6197 (epoch 13), rain thr 0.5.

comparison_summary (test):
| Var | full_model | persistence | mlp |
|---|---|---|---|
| precip RMSE | 0.969 | 0.685 | 0.734 |
| precip corr | 0.542 | 0.686 | 0.548 |
| wind RMSE | 1.032 | 1.115 | 1.130 |
| wind corr | 0.878 | 0.849 | 0.832 |
| humidity RMSE | 3.200 | 4.494 | 3.832 |
| humidity corr | 0.968 | 0.938 | 0.958 |

- full_model MENANG di wind & humidity (dulu kalah/imbang) → fix #1/#6 berdampak nyata.
- Precip tetap kalah persistence (keterbatasan ERA5 0.25° utk hujan lokal gunung — temuan sah).
- Alignment OK (semua skenario 3189 sampel). CRPS fair (persistence punya CRPS 0.219).
- Weekly crosscheck_3var ter-regen reproducible (driest/median/wettest), full unggul humidity tiap minggu.

### 🔴 REGRESI BARU (#16) — lebih buruk dari sebelumnya
- diff_only precip RMSE = 8.652.663 ; diff_retrieval = 2.051.899 (CRPS 48M/38M). Dulu ~1.5.
- Root cause: ablation mematikan GNN/retrieval dgn meng-input NOL ke model yang dilatih selalu-nyala
  (out-of-distribution) + `expm1(clamp(max=20))` mengizinkan precip s/d 485 juta mm (tanpa clamp fisik).
- full_model & diff_gnn SEHAT (conditioning lengkap). Tapi tabel ablation + bar chart jadi tak terbaca.
- Akibatnya kontribusi murni GNN tak bisa dibaca → klaim "tiap pilar berkontribusi" belum terbukti bersih.

---

## 7. KESIMPULAN AUDIT (kode vs judul/algoritma) — apa adanya

Sudah faithful menerapkan judul (5 dari 6 komponen):
- 5-node star + MAIN pusat ✅
- Spatio-Temporal Graph (GAT edge_dim=1 + temporal attn kausal) ✅ (edge kini benar dipakai)
- Retrieval-Augmented (FAISS kNN train-only strict-past, value=outcome τ+1) ✅
- Conditional Diffusion (DDPM train + DDIM sample, 3 conditioning, wet/dry head) ✅
- Probabilistik (CRPS fair, Brier, POD/FAR/CSI) ✅
- No leakage (split kronologis, stats & retrieval train-only) ✅

Belum bersih:
- 🔴 #16 ablation degenerate (artefak kode, bukan keterbatasan data). WAJIB diperbaiki agar tabel ablation valid.

Catatan ilmiah (disepakati user, bukan kegagalan): ERA5 0.25° terlalu kasar utk hujan titik di gunung;
node DOWN maritim; precip kalah persistence = temuan sah ("ERA5 buruk utk kasus ini").

Caveat transparansi: artefak final saat ini `eval_step=11` (subsample diurnal-covering), bukan hourly penuh
(=1, 35074 sampel) demi waktu; kode mendukung penuh. Untuk laporan final idealnya 1× run penuh.

---

## 8. STATUS TERKINI (saat dokumen ini ditulis)

- User SEDANG menjalankan retrain #2 untuk memperbaiki #16:
  `python3 -m src.train --epochs 35 --batch-size 512 --hidden-dim 128 --k-neighbors 3
   --grad-clip-norm 1.0 --train-end 2018-12-31 --val-end 2021-12-31 --cond-dropout 0.15
   --early-stop-patience 12`  (pid 3622337)
- `--cond-dropout 0.15` = perbaikan #16 (acak nol-kan retrieval/graph saat training → model robust
  terhadap ablation, sehingga diff_only/diff_retrieval tak meledak lagi).
- Progres terakhir terpantau: epoch 30/35 (~44% batch). Best val = 0.6200 (epoch 28); epoch 29 = 0.6209.
  cond-dropout TIDAK merusak kualitas (setara checkpoint lama 0.6197). corr noise ~0.93.
- Early stopping: berhenti bila val tak membaik 12 epoch berturut-turut, lalu pakai best; namun kemungkinan
  batas 35 epoch tercapai dulu.

### LANGKAH SETELAH TRAINING SELESAI (TODO Tahap D→E ulang)
1. `python3 run_eval_final.py --eval-step 11 --num-ensemble 30` (atau `--eval-step 1` utk final penuh).
   → CEK: diff_only/diff_retrieval precip RMSE turun dari jutaan ke wajar (< ~10) = bukti #16 RESOLVED.
2. `python3 scripts/eval_rain_robust.py --num-ensemble 20 --num-inference-steps 20`.
3. `python3 -m unittest discover -s tests -v` (harus tetap 10/10).
4. RE-AUDIT angka baru (semua berubah krn checkpoint baru): konfirmasi #16 resolved, rain-gate tak kolaps,
   alignment OK, tak ada output janggal/hardcoded. Lapor "bersih" HANYA bila semua KRITIS/MAYOR resolved.

---

## 8b. RETRAIN #2 (cond-dropout) & RE-AUDIT FINAL — #16 RESOLVED

Retrain #2 selesai (checkpoint 23:16):
`--epochs 35 --batch-size 512 --hidden-dim 128 --k-neighbors 3 --grad-clip-norm 1.0 --cond-dropout 0.15 --early-stop-patience 12`
- Selesai 35/35 epoch. **Best val = 0.6085 (epoch 34)** → lebih baik dari run sebelumnya (0.6197).
- Config tersimpan: `cond_dropout=0.15`, `retrieval_dim=9`, edge_dim aktif, rain thr 0.55, CSI val 0.637.
- Catatan: `max_grad_norm` sempat 8.9e8 (ditahan grad-clip 1.0); val tetap turun, hasil sehat.

Re-eval (eval_step=1, 1000 sampel) — comparison_summary 23:34:

| skenario | precip RMSE (SEBELUM #16) | precip RMSE (SEKARANG) |
|---|---|---|
| diff_only | 8.652.663 🔴 | **0.724** ✅ |
| diff_retrieval | 2.051.899 🔴 | **0.817** ✅ |
| diff_gnn | 0.943 | 0.735 |
| full_model | 0.969 | 0.820 |

Tabel lengkap (test):
| Var | full_model | persistence | mlp |
|---|---|---|---|
| precip RMSE | 0.820 | 0.548 | 0.517 |
| precip corr | 0.523 | 0.557 | 0.466 |
| wind RMSE | 0.982 | 1.064 | 1.097 |
| wind corr | 0.885 | 0.858 | 0.836 |
| humidity RMSE | 2.645 | 4.542 | 3.694 |
| humidity corr | 0.974 | 0.925 | 0.951 |

- **#16 RESOLVED**: semua skenario waras (tak ada jutaan). cond-dropout bikin model tahan ablation.
- Ablation kini terbaca: **GNN kontributor terbesar** (wind 1.10→0.97, humidity 4.22→2.67);
  retrieval marginal untuk precip. full_model MENANG di wind & humidity; precip kalah (ERA5 0.25° = temuan sah).
- Tests **10/10 PASS**. weekly_crosscheck_3var regen 23:37 (reproducible).

STATUS FINAL: 15 isu awal + #16 = **SEMUA RESOLVED**. Kode faithful terhadap judul & algoritma.

Caveat laporan final (bukan blocker):
1. Grad spike besar saat training (ditahan clip) — catat di laporan.
2. Artefak ini pakai sampel di-cap (1000@step1 atau 3189@step11), BUKAN hourly penuh 35.074.
   Untuk angka definitif skripsi: jalankan `python3 run_eval_final.py --eval-step 1` tanpa `--max-eval-samples`.

Checkpoint cond-dropout: log `results/training_logs/retrain_dropout2_20260530_214508.out.log`.
Bug transien `config.py` "NameError: ou" sempat muncul saat ada edit live — kini compile OK.

---

## 9. File kunci
- Plan: `docs/AUDIT_FIX_PLAN.md`
- Metode: `docs/METODE_PENELITIAN_PROYEK.md`, `docs/ACTIVE_5NODE_STAR_MAIN.md`
- Kode inti: `src/train.py`, `src/inference.py`, `src/models/{diffusion,gnn}.py`,
  `src/data/temporal_loader.py`, `src/retrieval/base.py`, `src/config.py`,
  `src/evaluation/probabilistic_metrics.py`, `run_eval_final.py`, `scripts/eval_rain_robust.py`
- Hasil: `result_test/comparison/comparison_summary.csv`, `result_test/*/metrics.json`,
  `result_test/nowcasting_hourly_week/weekly_crosscheck_3var.csv`
- Checkpoint: `models/diffusion_chkpt.pth`, `models/mlp_baseline_chkpt.pth`
- Log retrain #2: `results/training_logs/retrain_dropout2_20260530_214508.out.log`

## 8c. AUDIT INTEGRITAS (cheating check) + PNG mingguan — 2026-05-30 ~23:48

Cek SEMUA git changes untuk kecurangan/pemaksaan hasil/keluar judul. HASIL: BERSIH.
- `cond_dropout=0.15`: hanya di `src/train.py` (regularisasi train-time, seperti feature dropout).
  TIDAK ada di inference/eval → tidak menyentuh prediksi test maupun actual. Sah.
- `PRECIP_PHYSICAL_MAX_MM=60.0` (config): clamp HANYA pada PREDIKSI (`samples_denorm`/`mc_denorm`),
  TIDAK pernah pada target/actual (actual dibaca mentah). Dataset max precip = 21.5 (MAIN)/23.1 (all),
  cap 60 ≈ 2.6× → tidak bisa menggelembungkan hasil; murni numerical hygiene anti expm1-blowup OOD.
- `private.note`: hanya tempel analisis (catatan), tidak memengaruhi kode/hasil.
- Persistence/MLP/diffusion semua pakai `_eval_indices` sama → sampel identik (terverifikasi assert).
- Grad spike: efek `weighted_noise_loss` (5×/10× event ekstrem) — design choice, ditahan grad-clip,
  bukan cacat kode. Sampel di-cap: pilihan runtime, kode dukung full. Keduanya bukan kecurangan.

PNG 1-minggu dibuat (reproducible) via `scripts/plot_week_timeseries.py` dari weekly CSV:
- `result_test/nowcasting_hourly_week/timeseries_1week_{driest,median,wettest}_week.png` (162 jam each)
- 3 panel (precip/wind/humidity), garis: Actual / Full model / Persistence / MLP.
- Pengamatan jujur: wind & humidity model menempel actual (bagus); precip sering over-predict spike
  (konsisten dgn precip kalah persistence — keterbatasan ERA5, bukan curang).

VERDICT INTEGRITAS: tidak ada hardcode hasil, tidak ada pemaksaan, tidak keluar dari judul.

## 8d. AUDIT PENUH SUB-AGENT (persistence + MLP + main) + 3 FIX BARU — 2026-05-31 ~00:10

User minta: jelaskan hakikat persistence & MLP; audit penuh pakai sub-agent (bukan hanya algoritma judul).

TL;DR algoritma:
- Persistence = prediksi t+1 = nilai aktual t (y_hat = y_t). Tanpa parameter, deterministik. Baseline naif.
- MLP = feedforward NN dasar (Linear->ReLU->Dropout x2). Belajar 6 jam terakhir x 9 fitur -> 3 target. Deterministik.

Sub-agent (3 paralel + sintesis) menilai "semua faithful", TAPI verifikasi numerik mandiri saya
MENGOREKSI itu — ditemukan 3 isu nyata:

### FIX #11b (MAYOR, terkonfirmasi numerik) — bug formula CRPS
- `compute_crps` salah: `crps = term1 - 0.5*term2`, padahal `term2` (=sum((2i-n-1)x)/ (n(n-1)))
  SUDAH sama dengan 0.5*E|X-X'|. Jadi seharusnya `crps = term1 - term2`.
- Bukti: 40-member, true fair CRPS=0.65309, kode lama=0.99091 (terinflasi ~50%).
- Dampak: CRPS semua model ensemble (diffusion) digelembungkan; persistence (n=1) tak terpengaruh →
  persistence diuntungkan tak adil pada CRPS (bias MELAWAN model utama, bukan mencurangi agar menang).
- FIX: `src/evaluation/probabilistic_metrics.py` → `crps = term1 - half_exx` (half_exx = 0.5 E|X-X'|),
  n==1 → crps=term1 (=MAE). Test `test_crps_two_member` dikoreksi 1.5→1.0. Verifikasi: got=0.65309 = true.

### FIX (MLP eval) — MC-dropout diganti deterministik
- Sebelumnya `run_mlp_baseline` pakai `model.train()` + 30 forward pass + median (MC-dropout) →
  menyimpang dari hakikat MLP deterministik & bikin CRPS MLP "dapat spread" tak setara.
- FIX: `run_eval_final.py run_mlp_baseline` → `model.eval()`, 1 forward pass, ensemble size=1
  (CRPS reduce ke MAE, fair seperti persistence). Metadata `mlp_crps_type` → "deterministic_single_pass".

### FIX #6b — edge_attr dibuat INFORMATIF (sebelumnya konstan 0.25)
- Ditemukan: `build_star_edge_attr` lama = jarak lat/lon euclidean → SEMUA edge = 0.25 (star equidistant)
  → edge weight non-informatif; GAT pakai edge_dim=1 tapi nilainya konstan.
- FIX: edge_attr jadi [num_edges, 2] = [jarak(deg), |selisih elevasi|(km)].
  Elevasi ditambahkan ke NODE_DEFINITIONS (MAIN 1529, UP 162, DOWN 0, LEFT 823, RIGHT 288 m; dari grid validation).
  Nilai elev-diff km: UP 1.367, DOWN 1.529, LEFT 0.706, RIGHT 1.241 → informatif (beda per edge).
  GATConv edge_dim 1→2 (`src/models/gnn.py`). Test edge_attr & smoke diupdate ke shape [.,2].
- Konsekuensi: arsitektur GAT berubah → WAJIB RETRAIN.

Verifikasi statis pasca-fix: py_compile OK; edge_attr informatif terbukti; CRPS true match.
Test: 2 gagal SEMENTARA (smoke load checkpoint lama edge_dim=1; edge_attr shape) — akan hijau pasca-retrain
(test edge_attr shape sudah diupdate ke [8,2]).

### RETRAIN #3 (edge_dim=2) — SEDANG BERJALAN
- `python3 -m src.train --epochs 35 --cond-dropout 0.15 --early-stop-patience 12`
- Log: `results/training_logs/retrain_edge2_20260531_000951.out.log` (pid python 1050461)
- Status terpantau: retrieval DB 122,695 OK, model init OK (edge_dim=2 tak crash), epoch 2 berjalan.

### LANGKAH SETELAH RETRAIN #3 SELESAI
1. `python3 run_eval_final.py --eval-step 11 --num-ensemble 30` (atau step 1 utk final).
2. `python3 scripts/eval_rain_robust.py`.
3. `python3 scripts/plot_week_timeseries.py --all` (regen PNG mingguan).
4. `python3 -m unittest discover -s tests -v` → harus 10/10 (checkpoint baru edge_dim=2).
5. RE-AUDIT: konfirmasi CRPS benar (model CRPS turun ~ vs sebelumnya), MLP deterministik (CRPS≈MAE),
   edge_attr informatif benar dipakai, semua skenario waras, tak ada blowup/leakage/hardcode.

### Catatan integritas (dikonfirmasi BERSIH)
- cond_dropout hanya train; PRECIP clamp hanya prediksi (bukan target); actual dibaca mentah;
  semua skenario _eval_indices identik. Tak ada hardcode/pemaksaan/keluar judul.

## 8e. FIX 3 ISU + REPRODUCIBILITY — FINAL (2026-05-31 ~02:xx)

User minta fix semua 3 isu dari audit sub-agent. Dilakukan:

### FIX CRPS (#11b) — SELESAI, terverifikasi
- `crps = term1 - half_exx` (half_exx = 0.5 E|X-X'|). Sebelumnya salah `term1 - 0.5*term2`.
- Verifikasi numerik: 40-member CRPS got=0.65309 = true fair 0.65309. Test 2-member dikoreksi 1.0.
- Dampak: persistence & MLP (n=1) CRPS = MAE persis (0.2187=0.2187 di hasil final).

### FIX MLP eval — SELESAI
- `run_mlp_baseline` kini `model.eval()` 1 forward pass (deterministik), ensemble=1 → CRPS=MAE.
  Buang MC-dropout. Metadata `mlp_crps_type="deterministic_single_pass"`.

### FIX edge_attr (#6b) — DICOBA elevasi, GAGAL, DI-REVERT (jujur)
- Coba edge_attr 2-dim [jarak, |Δelevasi|km] + GAT edge_dim=2. Retrain → DIVERGEN (val stuck ~2.2,
  corr 0.71 vs 0.93). Empiris memperburuk → DI-REVERT ke edge_attr 1-dim (jarak konstan 0.25).
- Keputusan jujur: edge weight konstan didokumentasikan sebagai keterbatasan (grid equidistant),
  BUKAN dipaksakan. Elevasi disimpan di NODE_ELEVATIONS + docstring, tidak diinject sbg edge weight.
- Ini contoh "tidak memaksakan fix yang merusak" — sesuai prinsip integritas.

### TEMUAN BARU PENTING: training NON-DETERMINISTIK & seed-sensitive
- Saat retrain berulang, val loss bervariasi liar antar run (0.61 / 0.76 / 2.2 / 10) dgn config SAMA.
- Akar: (a) TIDAK ada seed di train.py; (b) `cudnn.benchmark=True` bikin algoritma non-deterministik;
  (c) weighted_noise_loss (5x/10x) + lr 1e-3 → gradien sangat besar (grad spike), training di tepi stabil.
- FIX: tambah `--seed` (default 1) + `random/np/torch manual_seed` + `cudnn.benchmark=False` +
  `--lr` configurable (default 1e-3). Seed & lr disimpan di checkpoint config.
- CATATAN JUJUR: probe pendek sempat terlihat "stuck 2.6" — itu KARENA dimatikan terlalu dini
  (sebelum descent mid-epoch-1), BUKAN bug. Full run converge normal.

### CHECKPOINT FINAL (reproducible)
- best_val 0.7648 (epoch 24), seed 1, edge_dim=1, cond_dropout 0.15, lr 1e-3, non_finite=0.
- Lebih TINGGI dari run lucky sebelumnya (0.61) — tapi REPRODUCIBLE & jujur. grad norm tetap besar
  (artefak weighted loss, ditahan clip) — dicatat sbg keterbatasan.

### HASIL EVAL FINAL (eval_step=11, 3189 sampel) — checkpoint reproducible
| Var | full | persistence | mlp | diff_only | diff_gnn |
|-----|------|-------------|-----|-----------|----------|
| precip RMSE | 0.847 | 0.685 | 0.732 | 0.925 | 0.771 |
| wind RMSE   | 1.112 | 1.115 | 1.129 | 1.082 | 1.078 |
| humidity RMSE | 4.391 | 4.494 | 3.803 | 4.335 | 4.345 |
- Ablation WARAS (tak ada jutaan). CRPS fair (persistence/mlp CRPS=MAE).
- Catatan: dgn checkpoint reproducible ini, keunggulan full model atas baseline MENGECIL/HILANG
  (wind ~setara, humidity kalah MLP). Run "menang" sebelumnya adalah seed beruntung.
  → Kesimpulan jujur: pada konfig+data ini, model hybrid TIDAK konsisten unggul; ERA5 0.25° terbatas.
  Ini temuan penelitian yang SAH, dilaporkan apa adanya (bukan dipaksa menang).
- Tests 10/10 OK. PNG mingguan diregenerasi.

### STATUS: kode jujur & faithful ke judul; hasil dilaporkan apa adanya (tidak dicurangi).

## 8f. SWARM AUDIT (8 agen) — KRITIS BUG DITEMUKAN & DIFIX: elevation normalization (2026-05-31)

Setelah push commit 275ba18, dijalankan swarm 8 sub-agent (data, retrieval, diffusion, gnn,
training, metrics, results, sintesis). Menemukan 1 BUG KRITIS yang TERLEWAT semua audit sebelumnya:

### KRITIS — elevation std=0 → input GNN ~1e8 (INILAH penyebab grad spike, bukan weighted loss)
- `compute_stats_from_training` hitung stats dari MAIN node saja. Elevation MAIN konstan 1529 m
  → c_std[elevation]=0. Normalisasi `(x-mean)/(std+1e-5)` untuk node sekeliling:
  UP -1.367e8, DOWN -1.529e8, LEFT -7.06e7, RIGHT -1.241e8.
- Nilai ~1e8 ini masuk ke GATConv → gradien astronomis (max 8.9e8). Grad clip menahan update tapi
  GNN dipaksa mengabaikan/melawan fitur elevasi → spatial conditioning lumpuh.
- INI mengoreksi penjelasan saya sebelumnya yang SALAH ("grad spike = weighted loss"). Akar
  sebenarnya = bug normalisasi elevasi. Juga menjelaskan seed-sensitivity & instabilitas.

### FIX
- `compute_stats_from_training`: untuk fitur dgn std main-node ~0 (degenerate), fallback ke
  stats ALL-training-node. Elevasi kini c_mean=560.4 c_std=557.5 → normalisasi O(1)
  (MAIN +1.74, UP -0.72, DOWN -1.01, LEFT +0.47, RIGHT -0.49). Fitur cuaca tetap main-node.
- stats_scope = "main_node_only_with_allnode_fallback_for_constant_features".

### RETRAIN (elevfix, seed 1, reproducible) — HASIL TERBAIK & SEHAT
- best_val 0.5867 (terbaik dari semua run). **mean_grad_norm 1.72, max 9.40** (dulu 8.9e8!) → sehat.
  non_finite=0. wet CSI 0.647.

### HASIL EVAL FINAL (eval_step=11, 3189 sampel) — checkpoint elevfix
| Var | full | diff_gnn | diff_only | persistence | mlp |
|-----|------|----------|-----------|-------------|-----|
| precip RMSE | 0.736 | 0.741 | 0.831 | 0.685 | 0.742 |
| precip CRPS | 0.224 | 0.221 | 0.268 | 0.219 | 0.269 |
| precip corr | 0.640 | 0.633 | 0.519 | 0.686 | 0.546 |
| wind RMSE | 0.960 | 0.956 | 1.155 | 1.115 | 1.156 |
| wind corr | 0.892 | 0.894 | 0.848 | 0.849 | 0.832 |
| humidity RMSE | 2.910 | 2.860 | 4.298 | 4.494 | 4.548 |
| humidity corr | 0.974 | 0.975 | 0.944 | 0.938 | 0.956 |

PERUBAHAN PENTING (jujur):
- GNN sekarang BENAR berkontribusi: diff_only->diff_gnn precip RMSE 0.831->0.741, corr 0.519->0.633;
  wind 1.155->0.956; humidity 4.298->2.860. Sebelum fix, GNN lumpuh. INI bukti spatial conditioning
  kini berfungsi nyata.
- full_model MENANG telak vs persistence & MLP di wind (0.960 vs 1.115/1.156) & humidity
  (2.910 vs 4.494/4.548). Precip: RMSE 0.736 masih sedikit kalah persistence (0.685) tapi
  JAUH membaik (sebelum fix 0.847) & corr 0.640 mendekati; CRPS precip 0.224 ~ setara persistence 0.219.
- Ablation waras & monotonik (diff_only < diff_gnn ~ full). CRPS fair. Tests 10/10.

### TEMUAN LAIN SWARM (MINOR, belum difix - bukan blocker):
- correlation return 0.0 (bukan NaN) saat std~0 (probabilistic_metrics.py) - semantik, tak kena di data nyata.
- Brier/POD/FAR/CSI tidak NaN-safe - tak kena krn output di-clamp.
- run_eval_final.py tak set seed sampling diffusion - hasil tak bit-reproducible (eval_rain_robust seed 1234).
- RainForecaster.optimizer/criterion + train_step() = DEAD CODE (tak dipakai train.py).
- wind speed tak di-clamp >=0 di denorm (konsisten dgn baseline; jarang negatif).
- tz_localize(None) shift batas split ~7 jam (negligible, tetap kronologis).
- Default grad_clip_norm=0.0 & early_stop_patience=6 di signature (run aktual pakai CLI 1.0/12).
- edge_attr konstan 0.25 (terdokumentasi, bukan bug).
- num_ensemble 30 (main) vs 20 (weekly) - beda parameter antar skrip.

### STATUS: bug KRITIS elevasi DIFIX → model kini sehat, reproducible, GNN berfungsi, hasil membaik & jujur.
