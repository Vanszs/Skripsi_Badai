# RENCANA PERBAIKAN ROBUST — Audit Skripsi RA-Diffusion 5-Node Star

> Tujuan: menyelaraskan **kode / flow / algoritma** dengan **judul & metodologi** yang dideklarasikan
> ("Nowcasting probabilistik presipitasi untuk mitigasi resiko pendaki di gunung gede-pangrango dengan menggunakan retrieval augemented difussion model dengan spatio temporal graph conditioning, 5-node star, main-node-only, hourly nowcasting").
> Dokumen ini adalah rencana untuk di-review SEBELUM eksekusi. Belum ada kode yang diubah.

Tanggal: 2026-05-30
Basis: temuan audit (15 isu) pada sesi sebelumnya.

---

## 0. Prinsip & batasan

1. **Integritas di atas hasil.** Perbaikan membuat komponen benar-benar berfungsi sesuai klaim. Jika setelah diperbaiki model tetap kalah dari persistence pada suatu variabel, itu **dilaporkan apa adanya** (metodologi §5.3 sudah menyatakan tidak mengasumsikan model selalu unggul). Tidak ada hardcode/penyetelan agar "menang".
2. **Reproducible.** Setiap artefak hasil harus punya skrip penghasil di repo. Artefak yatim dihapus/diarsipkan.
3. **Minimal & terkontrol.** Hanya mengubah logika yang perlu; tidak menambah abstraksi.
4. **Satu kali retrain.** Perubahan #1 dan #6 mengubah dimensi/arsitektur model → digabung jadi **satu** retrain agar konsisten dengan checkpoint.

---

## 1. Ringkasan isu → keputusan perbaikan

| # | Isu (audit) | Severity | Keputusan |
|---|---|---|---|
| 1 | Retrieval menyimpan **fitur** sebagai value (`add_items(feat,feat)`) → bukan analog outcome | KRITIS | Ubah: simpan **outcome target langkah berikut** sebagai value. `retrieval_dim = num_targets*k` |
| 6 | `edge_attr` (jarak/angin) dihitung tapi tak pernah masuk GAT | MAYOR | Alirkan `edge_attr` jarak (static star) ke `GATConv(edge_dim=1)` |
| 3 | Persistence off-by-one → sampel eval tak sama dgn skenario lain | KRITIS | Samakan indeks eval semua skenario (`range(seq_len, N, step)`) |
| 4 | Klaim hourly tapi `eval_step=24` (1 sampel/hari) | KRITIS | Default `eval_step=1` (hourly penuh) + cap opsional |
| 5 | Rain gate kalibrasi val → matikan SEMUA precip di test | KRITIS | Recalibrate pasca-retrain + guard anti-degenerate |
| 8 | Artefak yatim (tak ada skrip penghasil) | MAYOR | Hapus/arsip; ganti dengan skrip reproducible |
| 9 | Angka jendela sama saling bertentangan antar file | MAYOR | Hilang otomatis stlh #8 (satu sumber) |
| 10 | `recursive_closed_loop` degenerate (out of scope §1) | MAYOR | Hapus mode recursive dari artefak |
| 11 | CRPS bias (`/n²`, `abs` tak perlu); persistence CRPS NaN | MINOR | Perbaiki ke `/(n(n-1))`; single-member → CRPS=MAE |
| 12 | Dua definisi "event hujan" (gate vs ensemble-fraction) | MINOR | Dokumentasikan pemisahan peran (by design) |
| 13 | MLP "ensemble" via MC-dropout = pseudo-uncertainty | MINOR | Beri label eksplisit di output/laporan |
| 14 | Test 7/7 hanya struktural, tak uji kebenaran numerik | MINOR | Tambah regresi: retrieval-dim, alignment, CRPS |
| 15 | Doc/usang: `respon.md`, val loss, default `wet_loss_weight` | MINOR | Sinkronkan doc & default |
| 2 | Model utama kalah baseline di precip | (akibat) | Diharapkan membaik stlh #1/#6; laporkan jujur |

---

## 2. Perubahan per-file (apa yang diedit + logika)

### FIX #1 — Retrieval mengembalikan OUTCOME, bukan fitur (KRITIS)

**Semantik baru:** DB analog = pasangan `(key = fitur MAIN pada waktu τ, value = target MAIN pada waktu τ+1, ternormalisasi)`.
Query dengan konteks saat ini (fitur MAIN `t-1`) → kembalikan outcome jam-berikut dari analog historis. Inilah "retrieval-augmented" yang sebenarnya.

- **`src/train.py`**
  - Hitung `train_targets_norm` dari `main_train[FINAL_TARGET_COLS]`: `precip → log1p`, lalu `(g(y)-t_mean)/(t_std+eps)`.
  - Bangun DB: `retrieval_db.add_items(train_features_norm[:-1], train_targets_norm[1:])` (drop baris terakhir; align τ→τ+1).
  - `_build_precomputed_retrieval`: parameter `data` menjadi `train_targets_norm[1:]` (nilai), index FAISS tetap dibangun dari `train_features_norm[:-1]` (key). `query_context_indices` tetap `valid_indices-1` (ruang waktu τ konsisten dgn key index). Filter `strict_past`/`exclude_self` tetap valid (key index j = waktu τ; value τ+1 < t terjaga karena j ≤ t-2).
  - `retrieval_dim = num_targets * k_neighbors` (3*3=9), dipakai saat init `ConditionalDiffusionModel` dan disimpan di `config`.
- **`src/inference.py`** (`load_model_and_stats`)
  - Bangun DB cara sama: hitung `train_targets_norm` dari main-train, `add_items(features[:-1], targets[1:])`.
  - `query(...)` otomatis mengembalikan `[1,k,num_targets]`; cocok dgn `retrieval_dim` baru.
- **`run_eval_final.py`** (`run_diffusion_scenario`, cabang `use_retrieval=False`)
  - `feat_dim = config["retrieval_dim"] // k` → otomatis = `num_targets`; `torch.zeros(1,k,num_targets)`. Tak perlu ubah manual (sudah derive dari config).
- **`scripts/eval_rain_robust.py`**: pakai `retrieval_db.query` dari `load_model_and_stats` → otomatis benar.

**Terdampak:** `src/models/diffusion.py` (tak ubah kode; `retrieval_mlp` input = `retrieval_dim` dari config). Checkpoint lama tak kompatibel → **retrain wajib**.

### FIX #6 — edge_attr (jarak) masuk ke GAT (MAYOR)

- **`src/config.py`**: tambah helper `build_star_edge_attr(node_names)` → tensor `[num_edges,1]` jarak euclidean lat/lon per edge (static, urutan = `STAR_EDGES`).
- **`src/models/gnn.py`**:
  - `SpatialGNN.__init__`: `GATConv(in, hidden, heads, edge_dim=1)` dan `GATConv(hidden*heads, out, heads=1, edge_dim=1)`.
  - `forward(x, edge_index, edge_attr=None, batch=None)`: teruskan `edge_attr` ke kedua `conv`.
  - `SpatioTemporalGNN.forward`: ambil `graph.edge_attr` dan teruskan.
- **`src/data/temporal_loader.py`**: `Data(x=..., edge_index=..., edge_attr=edge_attr)` — sediakan `edge_attr` static (dibangun sekali di `__init__` via config helper). `collate` (PyG `Batch`) otomatis menggabung `edge_attr`.
- **`src/inference.py`** (`create_inference_graphs`): sertakan `edge_attr` pada tiap `Data`.

**Terdampak:** `run_eval_final.py` & `scripts/eval_rain_robust.py` memakai `create_inference_graphs` → otomatis ikut. Arsitektur GAT berubah (param `edge_dim`) → **retrain wajib** (digabung dgn #1).

> Catatan: `build_dynamic_edges` (bobot angin) tetap **tidak** dipakai di pipeline aktif (loader pakai jarak static). Pilihan: hapus `build_dynamic_edges` + `PangrangoGraphBuilder` jika tak dipakai di mana pun, ATAU biarkan tapi tandai legacy. Keputusan: **hapus** `PangrangoGraphBuilder`, `SimpleGraphEncoder`, `create_pangrango_graph`, `create_temporal_graphs`, `WeatherStateEncoder` (semua dead code) agar tidak menyiratkan fitur yang tak ada (#7).

### FIX #3 + #4 — Alignment & hourly (KRITIS)

- **`run_eval_final.py`**
  - `run_persistence`: ubah loop ke `range(seq_len, len(main_df), eval_step)` (sama dgn skenario lain). Pred = `iloc[idx-1]`, target = `iloc[idx]` (idx≥seq_len≥1 aman).
  - Tambah assert: semua skenario menghasilkan jumlah sampel identik (`len` sama) → fail-fast bila tidak.
  - Default `eval_step=1` (hourly). Tambah arg `--max-eval-samples` (opsional) utk membatasi bila perlu; default tanpa batas. Dokumentasikan estimasi runtime.

### FIX #5 — Rain gate anti-degenerate (KRITIS)

- **`src/inference.py`** & **`run_eval_final.py`** & **`scripts/eval_rain_robust.py`**
  - Setelah recalibrate (otomatis saat retrain di `src/train.py`), tambah **guard**: jika pada split evaluasi `max(wet_prob) < wet_threshold` untuk seluruh jendela, jangan zero-kan semua; catat warning & lewati gating (gate non-aktif untuk jendela itu) agar tidak ada "selalu kering" senyap.
  - Logika gate tetap per-sampel; guard hanya mencegah kolaps total + transparan.
- **`src/train.py`**: kalibrasi threshold tetap di validation berbasis CSI (sudah benar); pasca-retrain dgn conditioning yang benar (#1/#6) distribusi wet-prob diharapkan membaik. Verifikasi di re-audit.

### FIX #8/#9/#10 — Reproducibility artefak (MAYOR)

- **Hapus/arsipkan** artefak yatim & kontradiktif:
  - `result_test/nowcasting_hourly_week/audit_nowcasting_protocols.json`
  - `result_test/nowcasting_hourly_week/audit_mlp_week_metrics.json`
  - `result_test/nowcasting_hourly_week/latest_1week_3feature_*` (json/csv/png)
  - `result_test/nowcasting/actual_vs_pred_nowcasting_full_model.csv` (regenerasi dari skrip)
- **`scripts/eval_rain_robust.py`**: perluas jadi **one-step hourly weekly, 3 variabel** (precip+wind+RH), model vs persistence vs MLP, output CSV+JSON deterministik. **Hapus** mode `recursive_closed_loop` (di luar scope §1). Ini menjadi satu-satunya penghasil artefak nowcasting mingguan.
- Hasil 6-skenario tetap dari `run_eval_final.py` (satu sumber).

### FIX #11 — CRPS benar (MINOR)

- **`src/evaluation/probabilistic_metrics.py`** `compute_crps`:
  - `term2 = sum((2i-n-1) x_(i)) / (n*(n-1))` (fair estimator), hapus `abs()`.
  - Izinkan `n>=1`: jika `n==1` → `crps = term1` (= MAE), sehingga **persistence punya CRPS** (bukan NaN).
- **Terdampak:** semua `metrics.json` & `comparison_summary` di-refresh saat re-eval.

### FIX #13/#12 — Pelabelan jujur (MINOR)

- `run_eval_final.py` / laporan: tandai CRPS MLP sebagai `mc_dropout` di metadata; jelaskan event-threshold gate vs scoring berbeda peran. Tanpa ubah angka.

### FIX #14 — Regresi numerik (MINOR)

- **`tests/`** (extend file existing, bukan file scratch — sesuai aturan repo, tidak membuat lalu menghapus):
  - `test_checkpoint_contract.py`: assert `retrieval_dim == num_targets * k_neighbors`.
  - `test_inference_smoke.py`: assert retrieved tensor shape `[*, k, num_targets]` & graph punya `edge_attr`.
  - tambah `test_metrics.py`? → hindari file baru bila aturan melarang; taruh assert CRPS known-case di `test_config_topology.py` atau extend yang ada. (Keputusan final saat eksekusi.)

### FIX #15 — Sinkronisasi dokumen (MINOR)

- **`src/train.py`**: default `wet_loss_weight` 0.5 → 0.7 (samakan dgn run aktual) atau dokumentasikan eksplisit.
- **`docs/METODE_PENELITIAN_PROYEK.md`** §4.4: perbarui deskripsi retrieval (outcome analog, bukan fitur) agar cocok kode baru.
- **`.github/rules/respon.md`** / context bridge: tandai usang (val loss 0.0966 & path 3-node legacy) atau update ke nilai aktual.

---

## 3. Matriks dampak file

| File | #1 | #6 | #3 | #4 | #5 | #8 | #11 | #14 | #15 | Retrain? |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `src/config.py` |  | ✏️ |  |  |  |  |  |  |  | ⟳ |
| `src/models/gnn.py` |  | ✏️ |  |  |  |  |  |  |  | ⟳ |
| `src/models/diffusion.py` | (cfg) |  |  |  |  |  |  |  |  | ⟳ |
| `src/data/temporal_loader.py` |  | ✏️ |  |  |  |  |  |  |  | ⟳ |
| `src/train.py` | ✏️ | (cfg) |  |  | ✏️ |  |  |  | ✏️ | ⟳ |
| `src/inference.py` | ✏️ | ✏️ |  |  | ✏️ |  |  |  |  | — |
| `run_eval_final.py` | (auto) | (auto) | ✏️ | ✏️ | ✏️ |  | (refresh) |  | ✏️ | — |
| `scripts/eval_rain_robust.py` | (auto) | (auto) |  | ✏️ | ✏️ | ✏️ |  |  |  | — |
| `src/evaluation/probabilistic_metrics.py` |  |  |  |  |  |  | ✏️ |  |  | — |
| `src/retrieval/base.py` | (hapus dead) |  |  |  |  |  |  |  |  | — |
| `src/graph/builder.py` |  | (hapus dead) |  |  |  |  |  |  |  | — |
| `tests/*` | ✏️ | ✏️ |  |  |  |  | ✏️ | ✏️ |  | — |
| `docs/*`, `respon.md` |  |  |  |  |  |  |  |  | ✏️ | — |
| artefak `result_test/*` |  |  |  |  |  | 🗑️/⟳ | ⟳ |  |  | — |

✏️=edit logika · (cfg)=ikut perubahan config · (auto)=ikut otomatis · 🗑️=hapus/arsip · ⟳=regenerasi/retrain

---

## 4. Urutan eksekusi (dependensi)

```
Tahap A — Edit model & data (tak jalankan apa pun)
  A1 #1 retrieval outcome  → train.py, inference.py, (diffusion cfg)
  A2 #6 edge_attr          → config.py, gnn.py, temporal_loader.py, inference.py
  A3 #7 hapus dead code    → retrieval/base.py, graph/builder.py, gnn.py
  A4 #11 CRPS              → probabilistic_metrics.py
  A5 #3/#4/#5 eval logic   → run_eval_final.py
  A6 #8/#10 reproducible   → scripts/eval_rain_robust.py + hapus artefak yatim
  A7 #14 tests             → tests/*
  A8 #15 docs/default      → train.py default, docs, respon.md

Tahap B — Validasi statis (tanpa training)
  B1 import semua modul (py_compile) — pastikan tak ada syntax/contract break
  B2 unit tests struktural (config/topology/loader) harus PASS
  B3 smoke: bangun dataset kecil → 1 forward STGNN+diffusion (cek shape, edge_attr, retrieval dim)

Tahap C — Retrain (satu kali, depends A1+A2)
  C1 train_baseline (MLP) — hanya bila perlu refresh (tak terdampak arsitektur; opsional)
  C2 train main: src/train.py (epochs default, rain_specialization on)
     → models/diffusion_chkpt.pth baru (retrieval_dim=9, GAT edge_dim=1)

Tahap D — Re-evaluasi (depends C2)
  D1 run_eval_final.py (eval_step=1 hourly; 6 skenario) → metrics.json, comparison, report, plots
  D2 scripts/eval_rain_robust.py (weekly one-step 3-var, reproducible)
  D3 tests numerik (retrieval dim, alignment count, CRPS) PASS

Tahap E — RE-AUDIT (ulangi audit awal persis)
  E1 telusuri ulang 15 isu vs kode+hasil baru
  E2 konfirmasi tiap isu: RESOLVED / PARTIAL / OPEN + bukti file:line/angka
  E3 jika ada KRITIS/MAYOR masih OPEN → kembali ke Tahap A (iterasi)
  E4 hanya bila semua KRITIS & MAYOR RESOLVED → laporkan
```

---

## 5. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| Retrain butuh GPU + dataset penuh (920K baris); `.venv` tampak Windows | Verifikasi environment dulu (B1); bila tak bisa train di sini, laporkan blocker, jangan fabrikasi hasil |
| `eval_step=1` hourly ~35k sampel × 4 skenario diffusion → lama | Sediakan `--max-eval-samples`; tetap default hourly utk laporan, beri estimasi waktu |
| Setelah fix, model tetap kalah persistence di precip | Laporkan jujur per-variabel; ini temuan sah, bukan kegagalan perbaikan |
| Rain gate masih degenerate di test | Guard anti-kolaps + dokumentasi; pertimbangkan gate "soft" (skala wet_prob) bila perlu, dgn persetujuan |
| Menghapus dead code memutus import tersembunyi | grep referensi dulu sebelum hapus (sudah dicek: tak ada pemakai aktif) |

---

## 6. Definition of Done (kriteria robust)

1. `#1`: tes membuktikan retrieved value = outcome (`dim==num_targets`), DB dibangun `features[:-1]→targets[1:]`, strict-past terjaga.
2. `#6`: `edge_attr` mengalir ke `GATConv`; graph di loader & inference punya `edge_attr`; tak ada dead code graf.
3. `#3`: semua skenario eval punya **jumlah sampel identik** (assert lulus).
4. `#4`: laporan utama hourly (`eval_step=1`) atau eksplisit terdokumentasi.
5. `#5`: tak ada jendela test yang ter-zero precip 100% secara senyap; guard + log aktif.
6. `#8/#9/#10`: setiap artefak punya skrip penghasil; tak ada angka kontradiktif; tak ada recursive degenerate.
7. `#11`: CRPS fair-estimator; persistence punya CRPS numerik.
8. `#14`: regresi numerik hijau.
9. `#15`: doc & default selaras implementasi.
10. **Re-audit** ulang (Tahap E) menyatakan semua KRITIS & MAYOR = RESOLVED, dengan bukti.

---

## 7. Catatan integritas

Perbaikan ini membuat tiga pilar judul benar-benar berfungsi (Retrieval-Augmented = analog outcome; Graph Conditioning = edge berbobot jarak; hourly nowcasting = evaluasi per-jam). Perbaikan **tidak** menjamin dan **tidak** akan dipaksa agar model unggul di semua metrik. Hasil akhir dilaporkan apa adanya per-variabel dan per-jenis-metrik, sesuai metodologi §5.3.
