# ALL-IN-ONE AUDIT REPORT (REVISED)
## Draft Bepan1 (DOCX & PDF) vs Implementation Code

**Date:** 2026-07-09  
**Scope:** BAB 1 (Pendahuluan), BAB 2 (Tinjauan Pustaka & Landasan Teori), BAB 3 (Metode Penelitian)  
**Sources:**
- `Draft Bepan1.docx` extracted to `.understand-anything/tmp/draft_bepan1_extracted.txt`
- `Draft Bepan1.pdf` extracted to `.understand-anything/tmp/draft_bepan1_pdf_extracted.txt`
- Repository code under `src/`, `run_eval_final.py`, `src/config.py`, etc.
- Active docs: `docs/METODE_PENELITIAN_PROYEK.md`, `docs/ACTIVE_5NODE_STAR_MAIN.md`
- Evaluation artifacts: `result_test/EVALUATION_REPORT.md`, `result_test/comparison/comparison_summary.csv`

---

## EXECUTIVE SUMMARY

**Verdict: NEEDS_MAJOR_REVISIONS**

Both `Draft Bepan1.docx` and `Draft Bepan1.pdf` contain significant inconsistencies with the actual implementation code. The PDF version has fixed some critical errors from the DOCX (notably the 3-node claim and 2-scenario claim), but several serious contradictions remain, especially regarding graph topology, abstract horizon, mobile application, hybrid persistence, and evaluation subsampling.

### Severity Definitions

| Level | Definition |
|---|---|
| **FATAL** | Fundamental contradiction that invalidates a core methodological claim. Must be fixed before defense. |
| **MAJOR** | Significant inaccuracy that must be corrected before defense. |
| **MINOR** | Precision, numbering, wording, or polish issue that should be addressed. |

### Issue Count

- **Unresolved in PDF:** 1 FATAL + 8 MAJOR + 8 MINOR
- **Additionally fixed in PDF vs DOCX:** 3 historical MAJOR issues (Section A)

---

## SECTION A: IMPROVEMENTS FROM DOCX TO PDF

The PDF version appears to be a newer revision that corrected the following MAJOR errors present in the DOCX:

| # | Issue | DOCX Status | PDF Status |
|---|---|---|---|
| 1 | BAB 3 claims **3 nodes** | ❌ Wrong | ✅ Fixed — PDF consistently says 5 nodes |
| 2 | BAB 3 [574] claims **2 scenarios** | ❌ Wrong | ✅ Fixed — PDF mentions 6 scenarios |
| 3 | MLP baseline described as **0–6 hour horizon & 3 nodes** | ❌ Wrong | ✅ Fixed — no longer appears |

**Evidence for PDF fixes:**
- PDF repeatedly uses "lima node" (5 nodes) throughout BAB 2 and BAB 3.
- PDF mentions "enam skenario" (6 scenarios) in the experiment section.
- No occurrences of "tiga node", "tiga titik", "dua skenario", or "0–6 jam" for MLP.

**Note:** The DOCX also contained an unverified claim about "scaling factor 5.0 for precipitation" ([520]). The PDF has replaced this with the correct concept of weighted denoising loss (5×/10×), which matches the code.

---

## SECTION B: REMAINING ISSUES IN PDF

### B.1 FATAL Issues

#### B.1.1 Graph Topology: Fully-Connected vs Star

**Locations in PDF:**
- BAB 2 [1309]: "Struktur graph fully-connected memungkinkan informasi mengalir secara efisien antar kelima node tersebut."
- BAB 3 [2023]: "Struktur graf yang digunakan bersifat fully-connected dengan node MAIN sebagai pusat perhatian."

**Claim:** The graph is fully-connected, meaning every node connects to every other node.

**Code Evidence:**
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

**Analysis:**
- A fully-connected 5-node directed graph would have 5 × 4 = 20 edges.
- The code uses only 8 directed edges forming a star centered on MAIN.
- Neighbor nodes (UP, DOWN, LEFT, RIGHT) do NOT connect to each other.

**Impact:** This is a fundamental architectural contradiction. Message passing in the code only flows between MAIN and neighbors, not between neighbors.

**Recommended Fix:** Replace all "fully-connected" with "star topology" and describe the 8 directed edges.

---

### B.2 MAJOR Issues

#### B.2.1 Abstract Horizon: 0–6 Hours vs One-Step

**Location in PDF:** Abstrak [62]

**Claim:** "Model dilatih untuk menghasilkan distribusi probabilistik kondisi presipitasi pada horizon prediksi 0–6 jam ke depan."

**Code Evidence:**
- `src/data/temporal_loader.py`: target is `targets[t, main_node_idx, :]` (one timestep ahead).
- `run_eval_final.py`: evaluation predicts `targets_raw[idx]` where `idx` is one step ahead.
- `src/train.py`: trained for one-step prediction.

**Analysis:**
- **0–6 jam memang definisi WMO untuk nowcasting**, dan penulis kemungkinan bermaksud menyatakan bahwa penelitian ini berada dalam rentang nowcasting 0–6 jam.
- Namun, kalimat abstrak "*Model dilatih untuk ... horizon prediksi 0–6 jam ke depan*" secara eksplisit menyatakan bahwa model dilatih untuk horizon prediksi 0–6 jam. Ini **menyesatkan** karena model sebenarnya dilatih untuk prediksi **1 jam ke depan (t+1)**.
- Perlu perbedaan antara: (a) domain nowcasting secara umum (0–6 jam), dan (b) horizon spesifik yang diprediksi model (1 jam).

**Recommended Fix:** Ubah abstrak menjadi: "Model dilatih untuk menghasilkan distribusi probabilistik kondisi presipitasi **satu jam ke depan (t+1)**, yang berada dalam rentang nowcasting 0–6 jam menurut WMO."

---

#### B.2.2 Mobile Application Claim Without Implementation

**Locations in PDF:**
- BAB 1 [92] Rumusan Masalah: "aplikasi mobile sebagai media penyampaian informasi cuaca"
- BAB 1 [101] Tujuan Khusus: "Mengimplementasikan hasil prediksi ... ke dalam aplikasi mobile"
- BAB 1 [109] Manfaat: "aplikasi mobile sebagai sarana penyampaian informasi cuaca"

**Claim:** The research will implement a mobile application for hikers.

**Code Evidence:**
- No mobile app directory, API backend, Flutter, React Native, Android, or iOS code exists.
- `grep` for `mobile`, `android`, `ios`, `flutter`, `apk` in `src/` returns only documentation references.
- Pipeline ends at `run_eval_final.py` producing CSV/JSON/plots in `result_test/`.

**Analysis:** A mobile application is promised as a deliverable in tujuan/manfaat/rumusan, but no supporting code exists. This is distinct from (but related to) the batasan masalah [119] stating that evaluation does not include operational early warning systems. The core problem is that no mobile app or backend prototype exists at all.

**Recommended Fix:** Remove mobile app from tujuan/manfaat/rumusan, or build a prototype backend/API and document it.

---

#### B.2.3 Hybrid Persistence Post-Processing

**Location in PDF:** BAB 3 [863] "Hybrid Persistence"

**Claim:** Hybrid persistence with weights 0.90 (precipitation), 0.90 (wind), 0.70 (humidity), optimized on validation data.

**Code Evidence:**
- `run_eval_final.py` lines 133–144 implement only **naive persistence** (copy t-1 to t).
- No `hybrid`, `alpha`, `beta`, or 0.90/0.70 weights exist in `src/`, `run_eval_final.py`, or `src/inference.py`.
- `grep "hybrid"` only finds `run_inference_hybrid` function name in archived code, not active pipeline.

**Recommended Fix:** Remove hybrid persistence section, or implement it in the active evaluation pipeline.

---

#### B.2.4 Retrieval Metric: Euclidean or Cosine

**Location in PDF:** BAB 2 [1213]

**Claim:** "menggunakan metrik kesamaan seperti Euclidean distance atau cosine similarity"

**Code Evidence:**
- `src/retrieval/base.py` uses `faiss.IndexFlatL2` (Euclidean L2).
- No cosine similarity implementation or metric selection option exists.

**Recommended Fix:** State "Euclidean L2 distance via FAISS IndexFlatL2" and remove cosine.

---

#### B.2.5 Evaluation Subsampling: `eval_step` Not Clearly Declared

**Location in PDF:** BAB 3 evaluation section (~[2280])

**Claim:** "dilakukan subsampling temporal untuk mengurangi pengaruh autokorelasi yang tinggi pada data meteorologis"

**Code Evidence:**
- `run_eval_final.py` uses `eval_step` parameter (default 1, actual artifact uses 11).
- `_eval_indices()` returns `range(seq_len, len(main_df), eval_step)`.
- `result_test/EVALUATION_REPORT.md` shows `eval_step: 11` with 3,189 samples from ~35,000 test hours.

**Note:** The DOCX version stated "subsampling setiap 24 jam (daily subsampling)" ([893]), but this wording does not appear in the PDF. The PDF is less specific but still fails to declare the actual `eval_step=11` mechanism.

**Recommended Fix:** Explain that evaluation uses `eval_step=11` (or `eval_step=1` for full evaluation) to reduce temporal autocorrelation while maintaining sufficient sample size.

---

#### B.2.6 Edge Attributes: Adaptive vs Constant 0.25°

**Location in PDF:** BAB 2 and BAB 3

**Claim (implicit):** Edge attributes carry spatial information (distance/elevation-based weights).

**Code Evidence:**
```python
# src/config.py build_star_edge_attr()
# "this distance is constant across edges and therefore non-informative"
# "An elevation-difference edge feature was tested but empirically degraded training convergence, so it was reverted."
```

**Analysis:**
- All edges from MAIN to neighbors are exactly 0.25° apart on the ERA5 grid.
- `edge_attr` is constant 0.25° for all 8 edges.
- Elevation-based edge attributes were tested but caused training divergence.

**Recommended Fix:** Transparently state that edge attributes are constant and non-informative; spatial signal comes from topology and node features.

---

#### B.2.7 Hypothermia Risk Index Not Implemented

**Location in PDF:** BAB 2 [1404], [1473]

**Claim:** Output can be translated into "level risiko hipotermia berbasis multi-faktor" and that the research fills the gap of connecting probabilistic output to hypothermia risk index.

**Code Evidence:**
- No `hypothermia`, `risk_index`, or `RiskIndex` module in `src/` or `docs/`.
- `FINAL_TARGET_COLS` only includes precipitation, wind_speed, humidity.
- Evaluation ends at weather metrics (RMSE, MAE, CRPS, CSI, etc.).

**Recommended Fix:** Frame hypothermia risk as a conceptual application or future work, not as implemented output.

---

#### B.2.8 Optimizer: Adam vs AdamW

**Location in PDF:** BAB 3 [2158]

**Claim:** "Optimasi parameter dilakukan menggunakan algoritma Adam"

**Code Evidence:**
```python
# src/train.py line 416
optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)

# src/train_baseline.py line 155
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
```

**Recommended Fix:** Change "Adam" to "AdamW".

---

### B.3 MINOR Issues

| # | Issue | Location | Recommended Fix |
|---|---|---|---|
| 1 | Weighted loss thresholds described as "\|z\| > 1σ, \|z\| > 3σ" | BAB 3 | Clarify that thresholds apply to normalized target values, not z-scores |
| 2 | Weighted loss values (5×, 10×) not disclosed | BAB 3 | Add the specific thresholds and weights |
| 3 | Tabel 2.2 referenced but not present | BAB 2 [156] | Insert the table or remove reference |
| 4 | Figure/table numbering inconsistent/duplicated | BAB 3 | Fix numbering (e.g., Gambar 3.5 vs 3.2, Tabel 3.1 used twice) |
| 5 | Overstatement novelty "pertama kali menggabungkan" | BAB 1 [81], BAB 2 [265] | Soften to context-specific novelty |
| 6 | Temperature not a target despite hypothermia focus | BAB 1 [115] | Add stronger justification or make temperature a target |
| 7 | ERA5 reanalysis described as operational | Throughout | Clarify ERA5 is historical reanalysis, not real-time |
| 8 | MAIN alias "Puncak" elevation 1529 m vs actual ~3000 m | BAB 3 [303] | Add note that MAIN is grid-average, not actual peak |
| 9 | DOWN node elevation 0 m (coastal) | BAB 3 [303] | Add note about ERA5 grid-average limitation |

---

## SECTION C: WHAT IS CORRECT IN THE PDF

The following aspects are consistent between the PDF draft and the code:

| Aspect | Status | Evidence |
|---|---|---|
| 5 nodes (MAIN, UP, DOWN, LEFT, RIGHT) | ✅ Correct in PDF | `src/config.py` NODE_DEFINITIONS |
| Node order [MAIN, UP, DOWN, LEFT, RIGHT] | ✅ Correct | `src/config.py`, `run_eval_final.py` |
| 9 input features | ✅ Correct | `src/config.py` FINAL_FEATURE_COLS |
| 3 target variables | ✅ Correct | `src/config.py` FINAL_TARGET_COLS |
| ERA5 via Open-Meteo hourly | ✅ Correct | `src/config.py`, `src/data/ingest.py` |
| Period 2005–2025 | ✅ Correct | `data/raw/pangrango_era5_5node_2005_2025.parquet` |
| seq_len=6 | ✅ Correct | `src/config.py`, `src/train.py` |
| One-step prediction (t+1) in limitations/method | ✅ Correct | `src/data/temporal_loader.py` |
| Log transform for precipitation | ✅ Correct | `src/data/temporal_loader.py`, `run_eval_final.py` |
| Standard scaling | ✅ Correct | `src/data/temporal_loader.py` |
| Temporal train/val/test split | ✅ Correct | `docs/METODE_PENELITIAN_PROYEK.md` |
| FAISS retrieval | ✅ Correct | `src/retrieval/base.py` |
| GAT for graph attention | ✅ Correct | `src/models/gnn.py` |
| Conditional diffusion architecture | ✅ Correct | `src/models/diffusion.py` |
| 6 evaluation scenarios | ✅ Correct in PDF | `run_eval_final.py` |
| Evaluation metrics (MAE, RMSE, Pearson, CRPS, Brier, POD, FAR, CSI) | ✅ Correct | `src/evaluation/probabilistic_metrics.py`, `run_eval_final.py` |
| Weighted denoising loss concept | ✅ Correct | `src/models/diffusion.py` |
| Wet/dry auxiliary head | ✅ Correct | `src/models/diffusion.py` |

---

## SECTION D: PROPOSED NEW LIMITATIONS (Batasan Masalah)

These limitations should be added to BAB 1 and elaborated in BAB 3. Each includes evidence of attempted optimization.

### D.1 Limited Extreme Precipitation Data

**Limitation:** Precipitation data is zero-inflated and heavy-tailed; extreme events are rare.

**Efforts made:**
- Log transform (`log1p`) to reduce skewness.
- Weighted noise loss: 5× for normalized target absolute value > 1.0, 10× for > 3.0.
- Retrieval-augmented historical analogs.
- Diffusion model ensemble generation.

**Why it cannot be fully overcome:** The intrinsic frequency of extreme events in the ERA5 2005–2025 dataset is fixed. No technique can create information that does not exist in historical data.

**Evidence:** `src/models/diffusion.py` lines 177–185.

---

### D.2 Constant Edge Attributes (0.25°)

**Limitation:** Edge attributes are constant 0.25° and non-informative.

**Efforts made:**
- Tested elevation-difference edge attributes.
- Result: training divergence.
- Reverted to constant distance-based edge attributes.
- Preserved spatial signal through star topology and differentiated node features.

**Why it cannot be fully overcome:** On the ERA5 0.25° grid, the lat/lon distance from MAIN to every neighbor is exactly 0.25°. Elevation-based edge features degraded convergence.

**Evidence:** `src/config.py` docstring in `build_star_edge_attr()`.

---

### D.3 ERA5 Not Real-Time

**Limitation:** ERA5 reanalysis has ~5–7 day latency and cannot support real-time operation.

**Efforts made:**
- Used ERA5 as a consistent long-term historical dataset (2005–2025).
- Built reproducible proof-of-concept pipeline.
- Designed modular pipeline (`src/data/ingest.py`, `src/config.py`) so that `OPEN_METEO_MODEL` can later be replaced with near-real-time sources.

**Why it cannot be fully overcome:** ERA5 is reanalysis, not real-time observation. Operational deployment would require radar, automatic weather stations, or near-real-time satellite data.

**Evidence:** `src/config.py` (`OPEN_METEO_MODEL = "era5"`); `src/data/ingest.py` uses `archive-api.open-meteo.com`.

---

## SECTION E: EVIDENCE OF SCENARIO TESTING

The research has systematically tested **6 scenarios**.

### E.1 Scenarios

1. **persistence** — naive baseline (copy t-1 to t)
2. **mlp_baseline** — deterministic MLP, main-node-only
3. **diff_only** — diffusion without retrieval, without GNN
4. **diff_retrieval** — diffusion + retrieval
5. **diff_gnn** — diffusion + GNN
6. **full_model** — diffusion + retrieval + GNN

### E.2 Evaluation Metadata

From `result_test/EVALUATION_REPORT.md`:
- graph_topology: star
- target_node_policy: main_node_only
- context_policy: main_node_context
- eval_step: 11
- samples_per_scenario: 3189
- num_ensemble: 30
- seq_len: 6
- node_order: [MAIN, UP, DOWN, LEFT, RIGHT]

### E.3 Deterministic/Probabilistic Metrics: Precipitation

**Tabel E.1. Sample Results: Precipitation**

| Scenario | RMSE | MAE | Correlation | CRPS |
|---|---:|---:|---:|---:|
| persistence | 0.6847 | 0.2187 | 0.6856 | 0.2187 |
| mlp_baseline | 0.7417 | 0.2686 | 0.5461 | 0.2686 |
| diff_only | 0.8718 | 0.4132 | 0.5345 | 0.2893 |
| diff_retrieval | 0.8957 | 0.4219 | 0.5303 | 0.2960 |
| diff_gnn | 0.7802 | 0.3415 | 0.6320 | 0.2463 |
| full_model | 0.7920 | 0.3530 | 0.6390 | 0.2518 |

**Technical note:** CRPS for persistence and MLP baseline equals MAE because these are deterministic single-pass models. Direct CRPS comparison should be made among diffusion-based scenarios.

### E.4 Threshold Metrics for Rare Events

**Tabel E.2. Threshold Metrics at 10 mm Precipitation**

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.5000 | 0.0000 | 0.5000 | 0.0003 |
| mlp_baseline | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_only | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_retrieval | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_gnn | 0.0000 | nan | 0.0000 | 0.0006 |
| full_model | 0.0000 | nan | 0.0000 | 0.0007 |

**Interpretation:** At the 10 mm threshold (rare extreme event), none of the diffusion-based models detect any event (POD = 0). This directly supports the limitation in Section D.1 about rare events.

### E.5 Interpretation of Counter-Intuitive Results

1. **diff_only and diff_retrieval worse than persistence on RMSE/MAE:** This is expected because diffusion models optimize for distribution quality (CRPS), not point error. Persistence is very strong for 1-hour-ahead precipitation due to high autocorrelation.
2. **diff_gnn slightly better than full_model on RMSE:** This reflects stochastic variability and suggests that the interaction between retrieval and GNN may not be fully optimal for deterministic metrics.

### E.6 Artifact Locations

- `result_test/EVALUATION_REPORT.md`
- `result_test/comparison/comparison_summary.csv`
- `result_test/comparison/comparison_summary.json`
- `result_test/*/metrics.json` for each scenario

---

## SECTION F: RECOMMENDED REVISION PRIORITY

### Priority 1: Must Fix Before Defense

1. **[B.1.1]** Replace all "fully-connected" with "star topology" (BAB 2 [1309], BAB 3 [2023]).
2. **[B.2.1]** Correct abstract: "0–6 jam" → "1 jam ke depan (one-step)".
3. **[B.2.2]** Remove or implement **mobile application** claims; align Batasan Masalah [119] with tujuan/manfaat.
4. **[B.2.3]** Remove or implement **hybrid persistence**.

### Priority 2: Important

5. **[B.2.4]** State retrieval uses **Euclidean L2 only**.
6. **[B.2.5]** Explain evaluation uses `eval_step=11` (or `eval_step=1` for full evaluation), not fixed 24-hour subsampling.
7. **[B.2.6]** Disclose that edge attributes are constant 0.25° and non-informative.
8. **[B.2.7]** Clarify hypothermia risk index as future work/conceptual application.
9. **[B.2.8]** Change optimizer from **Adam** to **AdamW**.
10. **[D.1–D.3]** Add the 3 new limitations (rare events, constant edge attr, ERA5 latency) with evidence of effort.

### Priority 3: Polish

11. **[B.3.1]** Clarify weighted loss thresholds apply to normalized target values, not z-scores.
12. **[B.3.2]** Disclose weighted loss values: 5× and 10×.
13. **[B.3.3]** Add missing Table 2.2 or remove reference.
14. **[B.3.4]** Fix figure/table numbering.
15. **[B.3.5]** Soften novelty claims.
16. **[B.3.6]** Add stronger justification for temperature not being a target.
17. **[B.3.7]** Clarify ERA5 is reanalysis, not operational real-time data.
18. **[B.3.8–B.3.9]** Add notes about MAIN elevation 1529 m and DOWN elevation 0 m.

---

## SECTION G: GLOBAL RECOMMENDATION

Use `docs/METODE_PENELITIAN_PROYEK.md` as the **single source of truth** when revising the thesis draft. This active methodology document already correctly describes:
- 5 nodes and star topology
- One-step t+1 prediction
- 6 evaluation scenarios
- Constant non-informative edge attributes
- eval_step-based evaluation
- Correct preprocessing pipeline

---

## SECTION H: FILES GENERATED BY THIS AUDIT

- `.kimchi/docs/ALL_IN_ONE_AUDIT_REPORT.md`
- `.kimchi/docs/ALL_IN_ONE_AUDIT_REPORT_REVISED.md` (this file)
- `.kimchi/docs/bab1_pdf_verification_report.md`
- `.kimchi/docs/bab2_pdf_verification_report.md`
- `.kimchi/docs/bab3_pdf_verification_report.md`
- `.kimchi/docs/anomaly_summary_draft_vs_kode_updated_pdf.md`
- `.kimchi/docs/guidance_batasan_masalah_update.md`
- `.kimchi/docs/perbandingan_draft_docx_vs_pdf.md`
- `.kimchi/docs/all_in_one_crosscheck_technical.md` (if completed)
- `.kimchi/docs/all_in_one_crosscheck_completeness.md`
- `.kimchi/docs/all_in_one_crosscheck_consistency.md`

---

## APPENDIX: QUICK REFERENCE FOR DEFENSE

### If examiner asks: "Why star topology, not fully-connected?"

> "Kode menggunakan star topology dengan 8 edge terarah antara MAIN dan keempat tetangga (UP, DOWN, LEFT, RIGHT). Neighbor tidak saling terhubung. Ini tercatat di `src/config.py` dengan `GRAPH_TOPOLOGY = 'star'`. Klaim fully-connected di draft adalah kesalahan penulisan yang perlu dikoreksi."

### If examiner asks: "Is the model predicting 0–6 hours or 1 hour?"

> "Model dilatih untuk prediksi one-step satu jam ke depan (t+1). Frasa 0–6 jam di abstrak mengacu pada definisi WMO untuk nowcasting, bukan horizon prediksi model."

### If examiner asks: "Where is the mobile app?"

> "Aplikasi mobile tidak diimplementasikan di kode. Pipeline berakhir di evaluasi model (`run_eval_final.py`). Jika tetap disebut di tujuan/manfaat, harus dibangun prototipe atau dihapus."

### If examiner asks: "What is hybrid persistence?"

> "Hybrid persistence tidak diimplementasikan di pipeline aktif. Pipeline hanya menggunakan naive persistence (copy t-1 ke t). Klaim hybrid persistence di BAB 3 perlu dihapus atau diimplementasikan terlebih dahulu."

### If examiner asks: "What did you try to handle rare events?"

> "Kami menerapkan log transform, weighted noise loss (5× dan 10× untuk nilai target ternormalisasi), retrieval-augmented analogs, dan diffusion ensemble. Namun model tetap dibatasi oleh frekuensi intrinsik event ekstrem dalam data ERA5."

### If examiner asks: "What about edge attributes?"

> "Kami mencoba edge attributes berbasis perbedaan elevasi, tetapi menyebabkan training divergence. Karena jarak lat/lon antar node pada grid ERA5 0.25° selalu 0.25°, atribut edge berbasis jarak menjadi konstan dan non-informatif. Sinyal spasial berasal dari topologi star dan fitur node."

### If examiner asks: "Why eval_step=11?"

> "eval_step=11 dipilih untuk mengurangi autokorelasi temporal antar sampel sambil mempertahankan ukuran sampel yang cukup besar (n=3,189 per skenario). Kode juga mendukung eval_step=1 untuk evaluasi penuh."


---

# APPENDIX A: DETAILED BAB-BY-BAB VERIFICATION REPORTS

Berikut adalah verifikasi detail per BAB yang menjadi dasar ringkasan di atas.


## A.1 BAB 1: Pendahuluan
# Laporan Verifikasi BAB 1 (PENDAHULUAN) PDF terhadap Kode Implementasi

**Auditor:** Kimchi Review Agent (technical auditor)  
**Tanggal:** 2026-07-09  
**Ruang Lingkup:** BAB 1 PDF paragraf [68] sampai sebelum [121] TINJAUN PUSTAKA  
**Sumber yang Diverifikasi:**
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/.understand-anything/tmp/draft_bepan1_pdf_extracted.txt`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/.understand-anything/tmp/draft_bepan1_extracted.txt` (perbandingan DOCX)
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/data/ingest.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/data/temporal_loader.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/models/gnn.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/models/diffusion.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/train.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/inference.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/run_eval_final.py`
- `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/METODE_PENELITIAN_PROYEK.md`

---

## Verdict: NEEDS_FIXES

BAB 1 PDF secara keseluruhan sudah konsisten dengan implementasi kode pada komponen metodologi inti: lima node, topologi graf bintang (star), sumber data ERA5 via Open-Meteo, variabel target (presipitasi, kecepatan angin, kelembapan relatif), jendela observasi 6 jam, prediksi one-step t+1, model difusi probabilistik, retrieval k-NN, dan evaluasi dengan metrik deterministik serta probabilistik. Namun, terdapat klaim **implementasi aplikasi mobile** yang muncul di tiga tempat (rumusan masalah, tujuan khusus, dan manfaat) tetapi sama sekali tidak diimplementasikan di repositori. Temuan ini memerlukan perbaikan sebelum BAB 1 dapat dianggap selaras secara teknis dengan kode.

---

## Aspek yang Terverifikasi Konsisten dengan Kode

Sebelum menyajikan temuan ketidaksesuaian, berikut ringkasan klaim BAB 1 yang telah diverifikasi sesuai dengan implementasi:

| Aspek | Lokasi PDF | Bukti Kode |
|---|---|---|
| 5 titik lokasi (node) | [113] | `src/config.py`: `NODE_DEFINITIONS` berisi 5 node (MAIN, UP, DOWN, LEFT, RIGHT); `NUM_NODES = 5` |
| Jendela observasi 6 jam | [114] | `src/data/temporal_loader.py`: default `seq_len=6`; `src/train.py`: `seq_len=6` |
| Prediksi satu jam ke depan (single-step) | [114] | `src/data/temporal_loader.py`: target diambil dari `t` (main-node-only); `run_eval_final.py`: evaluasi one-step `idx` → `target = targets_raw[idx]` |
| Variabel target: curah hujan, angin, kelembapan | [115] | `src/config.py`: `FINAL_TARGET_COLS = ["precipitation", "wind_speed_10m", "relative_humidity_2m"]` |
| Sumber data ERA5 via Open-Meteo hourly | [116] | `src/config.py`: `OPEN_METEO_MODEL = "era5"`; `src/data/ingest.py`: `models=OPEN_METEO_MODEL`, `interval="hourly"` |
| Tidak menggunakan radar/satellit/stasiun lokal | [117] | `src/data/ingest.py` hanya memanggil `archive-api.open-meteo.com` dengan parameter ERA5 |
| Model probabilistik generatif + retrieval + graph | [81], [83], [118] | `src/models/diffusion.py`: `ConditionalDiffusionModel`; `src/train.py`: `RetrievalDatabase`; `src/models/gnn.py`: `SpatioTemporalGNN` dengan `GATConv` |
| Evaluasi metrik deterministik & probabilistik | [119] | `src/evaluation/probabilistic_metrics.py` dan `run_eval_final.py`: RMSE, MAE, korelasi, CRPS, Brier, POD, FAR, CSI |

---

## Temuan Ketidaksesuaian

### 1. Aplikasi Mobile Diklaim di Tiga Tempat tetapi Tidak Ada Implementasi

- **Lokasi PDF:**
  - [92] Rumusan Masalah ke-5: "Bagaimana implementasi hasil prediksi nowcasting probabilistik ke dalam **aplikasi mobile** sebagai media penyampaian informasi cuaca jangka pendek untuk mendukung pengambilan keputusan dalam aktivitas pendakian di wilayah pegunungan?"
  - [101] Tujuan Khusus ke-5: "Mengimplementasikan hasil prediksi nowcasting probabilistik ke dalam **aplikasi mobile** sebagai sarana penyampaian informasi cuaca jangka pendek yang adaptif untuk mendukung mitigasi risiko pendakian."
  - [109] Manfaat Penelitian ke-5: "Mengimplementasikan hasil prediksi nowcasting probabilistik ke dalam **aplikasi mobile** sebagai sarana penyampaian informasi cuaca jangka pendek yang lebih adaptif dan mudah diakses untuk mendukung mitigasi risiko pendakian."

- **Klaim:** Penelitian ini akan mengimplementasikan hasil prediksi ke dalam aplikasi mobile sebagai media penyampaian informasi cuaca jangka pendek untuk pendaki.

- **Bukti Kode yang Menyanggah:**
  - Tidak ditemukan direktori, file, atau kode aplikasi mobile (Android, iOS, Flutter, React Native, maupun API backend untuk mobile) di seluruh repositori.
  - Pencarian `grep -i` terhadap kata kunci `mobile`, `android`, `ios`, `flutter`, `apk` pada direktori `src/` dan `docs/` tidak menghasilkan file implementasi (hanya muncul di dokumen audit/draft).
  - Pipeline implementasi berakhir di `run_eval_final.py` yang menghasilkan artefak evaluasi berupa `result_test/*/metrics.json`, `result_test/comparison/comparison_summary.csv`, dan visualisasi statis (PNG). Tidak ada endpoint, antarmuka pengguna, maupun artefak deployable aplikasi.
  - `docs/METODE_PENELITIAN_PROYEK.md` menjelaskan pipeline dari data mentah hingga evaluasi; tidak menyebutkan aplikasi mobile.

- **Severity:** MAJOR

- **Saran Perbaikan:**
  - **Pilihan A (disarankan untuk skripsi):** Hapus seluruh narasi aplikasi mobile dari rumusan masalah, tujuan khusus, dan manfaat. Ganti dengan fokus pada "kerangka keputusan berbasis probabilitas" atau "rekomendasi arah pengembangan aplikasi mobile".
  - **Pilihan B:** Bangun minimal prototipe aplikasi/mobile dashboard dan jadikan bagian dari metode serta evaluasi.

---

### 2. Kontradiksi Internal antara Aplikasi Mobile dengan Batasan Masalah

- **Lokasi PDF:**
  - [92], [101], [109]: klaim implementasi aplikasi mobile (lihat temuan #1).
  - [119] Batasan Masalah ke-7: "Evaluasi tidak mencakup implementasi sistem peringatan dini operasional atau pengujian langsung di lapangan, sehingga hasil penelitian difokuskan pada aspek metodologis dan analitis."

- **Klaim:** Tujuan dan manfaat menjanjikan sebuah aplikasi mobile (sistem penyampaian informasi operasional), sementara batasan masalah secara eksplisit mengecualikan implementasi sistem peringatan dini operasional dan pengujian lapangan.

- **Bukti Kode yang Menyanggah:**
  - Kode tidak mengandung implementasi aplikasi mobile maupun sistem peringatan dini operasional.
  - `run_eval_final.py` hanya melakukan evaluasi statis pada data uji; tidak ada modul notifikasi, threshold peringatan operasional, atau antarmuka pengguna.
  - Kontradiksi ini membuat pembaca/reviewer mempertanyakan apakah penelitian bersifat metodologis-analitis (sesuai [119]) atau aplikatif-operasional (sesuai [92]/[101]/[109]).

- **Severity:** MAJOR

- **Saran Perbaikan:**
  - Selaraskan batasan masalah dengan tujuan/manfaat. Jika aplikasi mobile dipertahankan, hapus kalimat "tidak mencakup implementasi sistem peringatan dini operasional" di [119] dan tambahkan subbab metode pengembangan aplikasi. Jika aplikasi mobile dihapus (Pilihan A di temuan #1), maka [119] sudah konsisten.

---

## Catatan Tambahan (Bukan Ketidaksesuaian, tetapi Perlu Kejelasan)

### Horizon Prediksi 0–6 Jam vs Single-Step t+1

- **Lokasi PDF:** [114] menyatakan: "Model menggunakan jendela observasi 6 jam terakhir sebagai input untuk memprediksi kondisi cuaca **satu jam ke depan** (single-step hourly nowcasting). Pendekatan ini sesuai dengan definisi nowcasting menurut WMO, yaitu prakiraan dengan rentang waktu 0–6 jam, di mana model dijalankan secara iteratif pada setiap jam."
- **Status:** **KONSISTEN** dengan kode. Model memang dilatih dan dievaluasi untuk prediksi single-step t+1. Frasa "0–6 jam" di sini merujuk pada definisi WMO, bukan horizon keluaran model. `docs/METODE_PENELITIAN_PROYEK.md` secara eksplisit menjelaskan bahwa prediksi operasional diperbarui secara rolling: "prediksi pada waktu t+1 selalu dibuat menggunakan jendela observasi hingga waktu t; saat observasi baru tersedia, jendela input diperbarui". Pendekatan ini sejalan dengan [114].

---

## Perbandingan DOCX vs PDF pada BAB 1

Berdasarkan pemeriksaan terhadap `draft_bepan1_extracted.txt` (DOCX) dan `draft_bepan1_pdf_extracted.txt` (PDF) untuk ruang lingkup BAB 1:

- **Tidak ditemukan perbaikan material di BAB 1 antara DOCX dan PDF.** Kedua versi menggunakan narasi yang hampir identik untuk seluruh paragraf BAB 1, termasuk klaim 5 node, single-step, variabel target, dan aplikasi mobile.
- **Klaim aplikasi mobile sudah ada di DOCX dan tetap dipertahankan di PDF.** Oleh karena itu, ketidaksesuaian ini bukanlah regresi akibat konversi PDF, melainkan masalah yang belum terselesaikan sejak versi DOCX.
- Perbaikan signifikan dari DOCX ke PDF terjadi di luar BAB 1, khususnya di BAB 3 (misalnya: konsistensi 5 node dan 6 skenario evaluasi), sehingga tidak masuk dalam ruang lingkup verifikasi ini.

---

## Kesimpulan

BAB 1 PDF telah mengalami peningkatan konsistensi pada narasi inti metodologi dibandingkan dengan beberapa masalah yang pernah muncul di BAB 3 versi DOCX. Namun, **BAB 1 masih mengandung klaim aplikasi mobile yang tidak diimplementasikan dan bertentangan dengan batasan masalah [119]**. Perbaikan wajib dilakukan dengan salah satu dari dua pendekatan: (1) menghapus narasi aplikasi mobile dari rumusan masalah, tujuan, dan manfaat; atau (2) mengimplementasikan minimal prototipe aplikasi/mobile dashboard dan memperbarui batasan masalah.

Hingga perbaikan tersebut dilakukan, BAB 1 tidak dapat diberi status **APPROVED** secara teknis.

---

## A.2 BAB 2: Tinjauan Pustaka & Landasan Teori
# Laporan Verifikasi BAB 2 (Tinjauan Pustaka & Landasan Teori) terhadap Implementasi Kode

**Lingkup verifikasi:**
- File PDF: `/media/DiskE/SKRIPSI/Skripsi_Bevan/.understand-anything/tmp/draft_bepan1_pdf_extracted.txt`, BAB 2 dari `[121] TINJAUAN PUSTAKA` hingga sebelum `[268] METODE PENELITIAN`.
- File DOCX (pembanding): `/media/DiskE/SKRIPSI/Skripsi_Bevan/.understand-anything/tmp/draft_bepan1_extracted.txt`.
- Kode implementasi:
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py`
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/retrieval/base.py`
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/models/gnn.py`
- Dokumen metode: `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/METODE_PENELITIAN_PROYEK.md`

**Verdict:** **NEEDS_FIXES**

Ditemukan inkonsistensi mendasar antara narasi BAB 2 dalam PDF dengan kontrak dan implementasi kode yang aktif. Temuan paling kritis adalah klaim topologi graf *fully-connected* sedangkan kode secara eksplisit menggunakan topologi *star*.

---

## 1. Topologi Graf: Klaim Fully-Connected vs Implementasi Star

- **Lokasi di PDF:** Subbab 2.2.5.1 "Representasi Graph untuk Data Elevasi dengan Lima Node", baris ~1309–1310.
- **Klaim eksak PDF:**
  > "Struktur graph fully-connected memungkinkan informasi mengalir secara efisien antar kelima node tersebut."
- **Bukti kode yang kontradiktif:**
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 16:
    ```python
    GRAPH_TOPOLOGY = "star"
    ```
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 74–83 mendefinisikan hanya 8 edge terarah antara `MAIN` dan keempat node sekelilingnya:
    ```python
    STAR_EDGES = [
        (MAIN_NODE_NAME, "UP"),
        ("UP", MAIN_NODE_NAME),
        (MAIN_NODE_NAME, "DOWN"),
        ("DOWN", MAIN_NODE_NAME),
        (MAIN_NODE_NAME, "LEFT"),
        ("LEFT", MAIN_NODE_NAME),
        (MAIN_NODE_NAME, "RIGHT"),
        ("RIGHT", MAIN_NODE_NAME),
    ]
    ```
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 161–172, fungsi `build_star_edge_index()` hanya menggunakan `STAR_EDGES`.
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/models/gnn.py` baris 5 dan docstring modul: `"5-node star graph (MAIN, UP, DOWN, LEFT, RIGHT)"`.
- **Analisis:**
  Topologi *fully-connected* untuk 5 node berarti setiap node terhubung ke setiap node lainnya (10 edge tak berarah, atau 20 edge terarah). Implementasi kode hanya memiliki 8 edge terarah berbentuk bintang dengan `MAIN` di pusat. Ini adalah kontradiksi arsitektural yang sangat signifikan karena mempengaruhi cara informasi spasial mengalir dalam model.
- **Severity:** **FATAL**

---

## 2. Indeks Risiko Hipotermia: Klaim Aplikasi vs Tidak Ada Implementasi

- **Lokasi di PDF:**
  - Subbab 2.2.6.3 "Relevansi Nowcasting Probabilistik sebagai Mitigasi Risiko", baris ~1404–1405.
  - Subbab 2.3.1.3 "Celah pada Aplikasi Mitigasi Risiko Pendakian", baris ~1473–1475.
- **Klaim eksak PDF:**
  > "Output model berupa distribusi probabilitas pada node MAIN dapat diterjemahkan menjadi level risiko hipotermia berbasis multi-faktor (hujan, angin, dan kelembapan)."
  > <br><br>
  > "Belum banyak penelitian yang menghubungkan output distribusi probabilistik model dengan indeks risiko hipotermia berbasis multi-faktor (presipitasi, angin, dan kelembapan) untuk mendukung pengambilan keputusan cepat di lapangan."
- **Bukti kode yang kontradiktif:**
  - Pencarian di seluruh repositori untuk `hipotermia`, `hypothermia`, `risk_index`, atau `RiskIndex` tidak menemukan implementasi apa pun di dalam `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/` maupun `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/`.
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 86–90 menunjukkan output model hanya tiga variabel meteorologis:
    ```python
    FINAL_TARGET_COLS = [
        "precipitation",
        "wind_speed_10m",
        "relative_humidity_2m",
    ]
    ```
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/METODE_PENELITIAN_PROYEK.md` hanya membahas evaluasi prediksi cuaca (RMSE, MAE, CRPS, POD, FAR, CSI); tidak ada formulasi, threshold, atau modul untuk menghitung indeks risiko hipotermia.
- **Analisis:**
  BAB 2 menyatakan bahwa output model dapat diterjemahkan menjadi level risiko hipotermia multi-faktor, dan bahwa kesenjangan penelitian ini perlu diisi. Namun, implementasi yang ada hanya menghasilkan prediksi variabel cuaca. Tidak ada modul yang menerjemahkan prediksi tersebut menjadi skor/level risiko hipotermia.
- **Severity:** **MAJOR**

---

## 3. Metrik Retrieval: Klaim Euclidean atau Cosine vs Implementasi Hanya L2

- **Lokasi di PDF:** Subbab 2.2.4 "Retrieval-Based Historical Analogs", baris ~1212–1213.
- **Klaim eksak PDF:**
  > "Teknik retrieval memanfaatkan kemiripan fitur antara kondisi atmosfer saat ini dengan kejadian historis menggunakan metrik kesamaan seperti Euclidean distance atau cosine similarity pada ruang fitur yang telah direduksi."
- **Bukti kode:**
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/retrieval/base.py` baris 12–13:
    ```python
    # L2 Distance Index
    self.index = faiss.IndexFlatL2(embedding_dim)
    ```
  - Tidak ada `IndexFlatIP` (inner product / cosine), normalisasi L2, atau penggunaan metrik cosine lainnya di kode retrieval.
- **Analisis:**
  Klaim menyebutkan dua metrik (Euclidean distance **atau** cosine similarity), yang mengaburkan bahwa implementasi aktif hanya menggunakan L2/Euclidean. Secara teknis Euclidean memang benar, tetapi menyebutkan cosine tanpa implementasi mengurangi akurasi deskripsi metodologi.
- **Severity:** **MINOR**

---

## 4. Atribut Edge: Tidak Ada Klaim Eksplisit di BAB 2, tetapi Perlu Diperhatikan

- **Lokasi di PDF:** Tidak ditemukan klaim spesifik tentang atribut edge di BAB 2 PDF.
- **Bukti kode:**
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 186–207, fungsi `build_star_edge_attr()` menghitung jarak Euclidean lat/lon antar node. Karena grid ERA5 0.25° membuat jarak MAIN–tetangga selalu 0.25°, atribut edge bernilai konstan untuk semua edge.
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/METODE_PENELITIAN_PROYEK.md` secara transparan menyatakan: "`edge_attr` bersifat non-informatif karena bernilai identik untuk semua edge".
- **Analisis:**
  BAB 2 tidak mengklaim atribut edge informatif, sehingga tidak ada inkonsistensi langsung. Namun, perlu diwaspadai agar BAB 3 atau BAB 4 tidak mengklaim bahwa model memanfaatkan jarak geografis/elevasi sebagai edge weight yang bermakna.
- **Severity:** **INFORMATIF / belum menjadi temuan**

---

## 5. Jumlah Node: Klaim 5 Node vs Implementasi 5 Node

- **Lokasi di PDF:** Subbab 2.2.5 dan 2.3 (berulang kali menyebutkan lima node: MAIN, UP, DOWN, LEFT, RIGHT).
- **Bukti kode:**
  - `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py` baris 20–59 mendefinisikan tepat 5 node, dan baris 62:
    ```python
    NUM_NODES = len(NODE_DEFINITIONS)
    ```
- **Analisis:** Konsisten. Tidak ada temuan.
- **Severity:** **OK**

---

## Perbandingan DOCX → PDF: Apakah Ada Perbaikan?

| Aspek | Klaim di DOCX (baris) | Klaim di PDF (baris) | Evaluasi Perubahan |
|---|---|---|---|
| Topologi graf | "Struktur graf yang digunakan bersifat fully-connected dengan node MAIN sebagai pusat perhatian." (~375) | "Struktur graph fully-connected memungkinkan informasi mengalir secara efisien antar kelima node tersebut." (~1309–1310) | **Tidak ada perbaikan**; klaim keliru dipertahankan. |
| Metrik retrieval | "menggunakan metrik kesamaan seperti Euclidean distance atau cosine similarity" (~283) | "menggunakan metrik kesamaan seperti Euclidean distance atau cosine similarity" (~1212–1213) | **Tidak ada perbaikan**; tetap ambigu terhadap implementasi L2 murni. |
| Indeks risiko hipotermia | Mengklaim output dapat diterjemahkan menjadi level risiko hipotermia multi-faktor (~251, ~264) | Mengklaim output dapat diterjemahkan menjadi level risiko hipotermia multi-faktor (~1404–1405, ~1473–1475) | **Tidak ada perbaikan**; klaim aplikasi tetap tidak di-backup kode. |
| Jumlah node | Lima node MAIN, UP, DOWN, LEFT, RIGHT | Lima node MAIN, UP, DOWN, LEFT, RIGHT | Konsisten, tidak ada masalah. |

**Kesimpulan perbandingan:** PDF tidak memperbaiki kesalahan faktual utama yang ada di DOCX. Perubahan yang terjadi hanya penambahan narasi, kutipan, dan pemecahan subbab, bukan koreksi terhadap implementasi.

---

## Rekomendasi Perbaikan

1. **Koreksi topologi graf di BAB 2:**
   - Hapus atau ubah frasa "fully-connected" di subbab 2.2.5.1 menjadi deskripsi topologi *star* (bintang) terarah dengan `MAIN` sebagai pusat, sesuai `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/config.py`.
   - Jika memang ingin menggunakan fully-connected, kode harus diubah secara signifikan; namun berdasarkan dokumen metode, desain yang disengaja adalah star.

2. **Tentukan nasib klaim indeks risiko hipotermia:**
   - Jika indeks risiko hipotermia memang menjadi bagian dari penelitian, tambahkan implementasi di `/media/DiskE/SKRIPSI/Skripsi_Bevan/src/` (misal `risk_index.py`) dan jelaskan formulasi/threshold-nya di BAB 3.
   - Jika tidak diimplementasikan, kurangi klaim di BAB 2 dan BAB 3 agar tidak menyiratkan bahwa sistem menghasilkan level risiko hipotermia secara langsung.

3. **Perjelas metrik retrieval:**
   - Di BAB 2, sebutkan secara spesifik bahwa implementasi menggunakan `faiss.IndexFlatL2` (Euclidean/L2 distance), dan hapus "cosine similarity" kecuali metrik tersebut juga benar-benar diimplementasikan.

4. **Dokumentasikan atribut edge dengan hati-hati:**
   - Jika BAB 3/4 membahas atribut edge, pastikan menyebutkan bahwa `edge_attr` adalah jarak Euclidean lat/lon yang bernilai konstan 0.25° untuk semua edge sehingga bersifat non-informatif, sesuai `/media/DiskE/SKRIPSI/Skripsi_Bevan/docs/METODE_PENELITIAN_PROYEK.md`.

---

## A.3 BAB 3: Metode Penelitian
# Verifikasi BAB 3 (METODE PENELITIAN) terhadap Implementasi Kode

**Tanggal Audit:** 2026-07-09  
**Lingkup:** BAB 3 PDF (baris 1505-2399 pada `draft_bepan1_pdf_extracted.txt`, setara paragraf [268]-[479] pada DOCX) dibandingkan dengan kode aktual di `src/`, `run_eval_final.py`, dan artefak evaluasi.

## Ringkasan Eksekutif

**Verdict:** NEEDS_FIXES

BAB 3 PDF telah memperbaiki sejumlah kesalahan dari versi DOCX (terutama jumlah node), namun masih mengandung ketidakkonsistenan metodologis yang signifikan terhadap implementasi nyata. Ketidakkonsistenan paling kritis adalah klaim topologi graf **fully-connected**, padahal kode secara eksplisif menggunakan **star graph** (bintang) dengan 8 edge terarah. Selain itu, terdapat penyederhanaan/penghilangan detail penting seperti faktor pembobotan loss 5.0/10.0 dan laju subsampling evaluasi `eval_step=11`.

---

## Peningkatan DOCX → PDF

### 1. Jumlah Node: 3 → 5
- **DOCX [495]:** "Wilayah kajian direpresentasikan oleh tiga titik lokasi (node) pada elevasi berbeda, yaitu wilayah hilir (kaki gunung), wilayah lereng tengah, dan wilayah puncak."
- **PDF BAB 3.3.1 / 3.3.4.3 / 3.3.9:** Secara konsisten menyebutkan **lima node**: MAIN, UP, DOWN, LEFT, RIGHT.
- **Bukti Kode:** `src/config.py` baris 17-58 mendefinisikan `NODE_DEFINITIONS` dengan 5 node; `NUM_NODES = len(NODE_DEFINITIONS) = 5`.
- **Status:** Perbaikan positif, konsisten dengan kode.

---

## Ketidakkonsistenan

### Issue #1 — Topologi Graf Salah: Klaim Fully-Connected vs. Star Graph (MAJOR)

- **Lokasi PDF:** BAB 3.3.4.3 "Representasi Graf Spasial" (sekitar baris 2023 pada `draft_bepan1_pdf_extracted.txt`)
- **Klaim PDF:** "Struktur graf yang digunakan bersifat **fully-connected** dengan node MAIN sebagai pusat perhatian."
- **Bukti Kode:**
  - `src/config.py` baris 11: `GRAPH_TOPOLOGY = "star"`
  - `src/config.py` baris 48-60: `STAR_EDGES` berisi tepat 8 edge terarah: `(MAIN,UP), (UP,MAIN), (MAIN,DOWN), (DOWN,MAIN), (MAIN,LEFT), (LEFT,MAIN), (MAIN,RIGHT), (RIGHT,MAIN)`.
  - `src/config.py` baris 62: `STAR_EDGE_COUNT = len(STAR_EDGES) = 8`.
  - `src/train.py` baris 348-351 memvalidasi: `edge_index.shape[1] != STAR_EDGE_COUNT` akan memunculkan error.
  - `src/config.py` baris 138-141 membangun `edge_index` hanya dari `STAR_EDGES`, bukan kombinasi semua pasangan node.
- **Analisis:** Jika graf benar-benar *fully-connected* (5 node), jumlah edge terarah adalah 5×4 = 20, atau 10 jika tidak terarah. Kode hanya memiliki 8 edge terarah berbentuk bintang (star) dua arah antara MAIN dan masing-masing node sekitar. Ini bukan fully-connected.
- **Severitas:** MAJOR — klaim metodologi fundamental yang secara langsung memengaruhi interpretasi arsitektur GNN dan message-passing.

### Issue #2 — Optimizer: Adam vs. AdamW (MINOR)

- **Lokasi PDF:** BAB 3.3.6 "Prosedur Pelatihan Model" (sekitar baris 2158 pada `draft_bepan1_pdf_extracted.txt`)
- **Klaim PDF:** "Optimasi parameter dilakukan menggunakan algoritma **Adam** dengan mekanisme backpropagation."
- **Bukti Kode:**
  - `src/train.py` baris 416: `optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)`
  - `src/train_baseline.py` baris 155: `optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_weight_decay=1e-4)`
- **Analisis:** Meskipun AdamW merupakan varian Adam, secara teknis optimizer yang tercatat di kode adalah `AdamW`, bukan `Adam`. Penulisan metode sebaiknya mencerminkan implementasi aktual.
- **Severitas:** MINOR — perbedaan terminologi, tidak mengubah hasil secara prinsipil.

### Issue #3 — Faktor Pembobotan Loss Presipitasi 5.0/10.0 Tidak Diungkapkan (MINOR → MAJOR tergantung interpretasi)

- **Lokasi PDF:** BAB 3.1, 3.3.6 (berulang menyebutkan "weighted denoising loss" untuk meningkatkan sensitivitas terhadap kejadian ekstrem)
- **Klaim PDF:** "menggunakan **weighted denoising loss** untuk meningkatkan sensitivitas model terhadap kejadian presipitasi ekstrem" namun tidak menyebutkan nilai bobot apa pun.
- **Bukti Kode:**
  - `src/models/diffusion.py` baris 178-185:
    ```python
    weights[target_reference.abs() > 1.0] = 5.0
    weights[target_reference.abs() > 3.0] = 10.0
    ```
- **Analisis:** PDF mengaburkan parameter kritis yang menentukan seberapa agresif model memfokuskan pelatihan pada event ekstrem. Pembaca tidak dapat mereplikasi metode tanpa membuka kode.
- **Severitas:** MINOR sebagai ketidakkonsistenan/penghilangan detail; dapat menjadi MAJOR jika dari sudut replikasi ilmiah karena hyperparameter kunci tidak didokumentasikan.

### Issue #4 — Subsampling Evaluasi: `eval_step=11` Tidak Dideklarasikan (MAJOR)

- **Lokasi PDF:** BAB 3.3.8 "Metode Evaluasi Model"
- **Klaim PDF:** "Evaluasi dilakukan dengan pendekatan **one-step rolling forecast** pada data pengujian untuk mensimulasikan kondisi operasional nyata. Selain itu, dilakukan **subsampling temporal** untuk mengurangi pengaruh autokorelasi yang tinggi pada data meteorologis."
- **Bukti Kode & Artefak:**
  - `run_eval_final.py` baris 449-451: `--eval-step` memiliki **default `1`** (tiap jam, tanpa subsampling).
  - `result_test/EVALUATION_REPORT.md` metadata menunjukkan: `eval_step: 11`, `samples_per_scenario: 3189`.
  - Dengan ~35.000 jam data uji, `eval_step=11` berarti hanya mengevaluasi setiap jam ke-11, bukan setiap jam.
- **Analisis:** PDF mengakui adanya subsampling tetapi tidak menyebutkan laju `eval_step=11`. Lebih penting lagi, default kode adalah `eval_step=1` (rolling penuh), sementara artefak hasil yang menjadi dasar laporan skripsi menggunakan `eval_step=11`. Hal ini menciptakan jurang antara protokol yang dideskripsikan, default kode, dan artefak evaluasi aktual.
- **Severitas:** MAJOR — mempengaruhi jumlah sampel evaluasi, distribusi temporal sampel, dan generalisasi metrik yang dilaporkan.

---

## Aspek yang Konsisten dengan Kode

### A. Jumlah Node = 5
PDF BAB 3 menyebutkan konsisten lima node (MAIN, UP, DOWN, LEFT, RIGHT). Kode `src/config.py` memvalidasi 5 node di seluruh pipeline (ingest, loader, train, inference, eval).

### B. Jumlah Skenario = 6
PDF BAB 3.3.9 menyebutkan enam skenario eksperimen. Kode `run_eval_final.py` baris 464-501 menjalankan tepat 6 skenario: persistence, mlp_baseline, diff_only, diff_retrieval, diff_gnn, full_model.

### C. Deskripsi MLP Baseline
PDF BAB 3.3.9 menyebutkan "Model deterministik berbasis neural network yang hanya menggunakan fitur historis dari node MAIN."
- `src/models/mlp_baseline.py` adalah model deterministik MLP.
- `src/train_baseline.py` baris 54-58 memfilter `df[df["node"] == MAIN_NODE_NAME]` dan hanya menggunakan fitur MAIN.
- `run_eval_final.py` baris 168 komentar: "MLP baseline is a DETERMINISTIC regressor: single eval() forward pass."

### D. Tidak Ada Klaim Hybrid Persistence di BAB 3
PDF BAB 3 tidak menyertakan skenario "hybrid persistence" di antara baseline maupun skenario eksperimen. Hal ini konsisten dengan:
- `run_eval_final.py` yang hanya menjalankan 6 skenario tanpa hybrid.
- `docs/METODE_PENELITIAN_PROYEK.md` yang menyatakan evaluasi menggunakan 6 skenario.
- Catatan `github/skills/thesis-sentinel/SKILL.md`: "Hybrid persistence has been REMOVED from thesis scope."

### E. Variabel Target
PDF menyebutkan prediksi probabilistik untuk curah hujan, kecepatan angin, dan kelembapan relatif. Kode `src/config.py` baris 33-37 `FINAL_TARGET_COLS = ["precipitation", "wind_speed_10m", "relative_humidity_2m"]` sesuai.

### F. Transformasi Logaritmik dan Invers
PDF menyebutkan transformasi logaritmik pada presipitasi dan inversnya saat inferensi. Kode menggunakan `log1p`/`expm1` (`src/train.py` baris 85-86, 436; `run_eval_final.py` baris 198-199).

### G. Normalisasi Standard Scaling
PDF menyebutkan Standard Scaling. Kode menggunakan `(x - mean) / (std + epsilon)` pada `src/train.py` baris 88-104 dan diterapkan pada inference/evaluasi.

### H. Basis Retrieval Hanya dari Data Pelatihan
PDF menyatakan basis retrieval hanya dibangun dari data pelatihan. Kode `src/train.py` baris 395-397 memfilter `main_train = train_df[train_df["node"] == MAIN_NODE_NAME]` dan membangun `RetrievalDatabase` dari data tersebut; validation/test tidak ikut membangun basis.

---

## Rekomendasi Perbaikan untuk BAB 3

1. **Perbaiki topologi graf:** Ganti "fully-connected" menjadi **"star graph (bintang)"** dengan penjelasan bahwa hanya MAIN yang terhubung dua arah dengan UP, DOWN, LEFT, RIGHT (8 edge terarah).
2. **Tambahkan detail weighted loss:** Cantumkan faktor pembobotan `5.0` untuk target absolut > 1.0 dan `10.0` untuk target absolut > 3.0.
3. **Perjelas protokol evaluasi:** Cantumkan bahwa evaluasi aktual menggunakan `eval_step=11` (subsample setiap 11 jam) dan jelaskan alasan pemilihan angka tersebut (coprime terhadap 24 jam, mengurangi autokorelasi, menghemat komputasi).
4. **Konsistensi optimizer:** Ubah "Adam" menjadi "AdamW" agar sesuai kode.

---

*Laporan ini disusun tanpa melakukan perubahan pada berkas sumber apa pun.*

---

## A.4 Cross-Check Reports

- Completeness cross-check: see `.kimchi/docs/all_in_one_crosscheck_completeness.md`
- Consistency cross-check: see `.kimchi/docs/all_in_one_crosscheck_consistency.md`
- Technical cross-check: partially completed by subagent
