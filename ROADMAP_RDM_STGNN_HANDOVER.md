# Handover ML: RDM + STGNN

> **Scope:** hanya machine-learning pipeline `core/`.  
> **Sumber kebenaran:** implementasi kode, test, checkpoint, artifact lokal.  
> **Aturan baca:** label **Fakta kode** berarti terlihat di kode; **Batas implementasi** berarti kode belum menjamin hal tersebut; **Tidak dibuktikan** berarti jangan diklaim tanpa eksperimen/artifact yang cocok.

Dokumen ini berdiri sendiri untuk manusia pemula atau AI lain. Tujuannya: memahami, memodifikasi, melatih ulang, mengevaluasi, serta menghandover sistem tanpa menyamakan asumsi dengan fakta.

---

# 1. Mulai dari nol

## 1.1 Problem yang diselesaikan

Model memprediksi **distribusi** cuaca `MAIN` satu timestep ke depan dari:

1. enam snapshot kondisi cuaca lima node;
2. keadaan terbaru node `MAIN`;
3. tiga analog outcome historis dengan feature `MAIN` yang dekat menurut jarak L2.

Satu sampel output:

```text
[precipitation mm/jam, wind_speed_10m m/s, relative_humidity_2m %]
```

Karena diffusion sampling mulai dari noise random, model menghasilkan banyak sampel. Median, p10, p90, CRPS, atau peluang event adalah ringkasan ensemble; bukan satu prediksi deterministik.

## 1.2 Dua algoritma inti

| Nama | Tugas | Output |
|---|---|---|
| **STGNN** — Spatio-Temporal Graph Neural Network | Merangkum pola antar-lokasi dan antar-waktu. | `graph_emb [B,G]` |
| **RDM** — Retrieval-Augmented Diffusion Model | Membuat sampel target dengan kondisi current context, historical analog, dan `graph_emb`. | `samples [M,3]` |

Mental model:

```text
6 jam x 5 node x 9 feature
          |
          v
 STGNN: GAT spasial, lalu attention temporal
          |
          +------------------------------ graph_emb

feature MAIN jam terakhir ---- context -----+
                                             v
FAISS historical outcomes ---- retrieved --> conditional diffusion --> M sampel t+1
```

RDM di proyek ini **bukan** random diffusion model lain. Ia adalah *Retrieval-Augmented Diffusion Model*.

## 1.3 Tiga lapisan konsep

```text
Data:       angka cuaca aktual per node dan timestamp.
Tensor:     array angka berbentuk tetap untuk model.
Model:      fungsi berparameter yang memetakan tensor ke prediksi noise/sampel.
```

Contoh: suhu `20.0` adalah data; suhu setelah normalisasi di `x[t,node,feature]` adalah tensor; GAT/diffusion yang memakai tensor itu adalah model.

---

# 2. Kamus zero-to-hero

| Istilah | Arti sederhana | Arti tepat di proyek |
|---|---|---|
| Feature | Informasi masukan. | 9 variabel cuaca/topografi per node. |
| Target | Nilai yang ingin diprediksi. | 3 variabel `MAIN` pada jam berikutnya. |
| Node | Satu lokasi dalam graph. | `MAIN`, `UP`, `DOWN`, `LEFT`, `RIGHT`. |
| Edge | Hubungan antar-node. | Delapan edge terarah bintang. |
| Graph | Node + edge pada satu waktu. | 5 node cuaca pada satu timestamp. |
| Sequence/window | Deret graph masa lalu. | Default 6 snapshot sebelum target. |
| Batch (`B`) | Banyak sample diproses bersamaan. | DataLoader menggabungkan graph dengan PyG `Batch`. |
| Embedding | Vektor laten hasil transformasi model. | `graph_emb`, bukan feature mentah. |
| GNN | Jaringan yang bertukar pesan antar-node graph. | GAT dua lapis. |
| GAT | Graph Attention Network. | Belajar bobot pesan tetangga dari feature/edge. |
| Attention | Mekanisme bobot relevansi. | `MultiheadAttention` antar-timestep. |
| Causal mask | Larangan melihat waktu sesudah posisi query. | Mask segitiga atas temporal attention. |
| Pooling | Merangkum banyak node menjadi satu vektor. | `global_mean_pool` untuk setiap graph. |
| FAISS | Library nearest-neighbor. | `IndexFlatL2`, pencarian exact L2. |
| Retrieval key | Vektor yang dicari kemiripannya. | feature `MAIN` pada waktu `tau`. |
| Retrieval value | Informasi yang dikembalikan neighbor. | target `MAIN` pada `tau+1`. |
| L2 | Jarak kuadrat Euclidean. | Makin kecil, analog makin dekat di ruang feature normalized. |
| Diffusion | Belajar membalik proses target ditambah noise. | Denoiser MLP memprediksi noise. |
| DDPM | Scheduler diffusion standar. | Training memakai 1.000 timestep. |
| DDIM | Sampler reverse lebih cepat. | Helper inference memakai 20 langkah. |
| Denoiser | Model yang memprediksi noise. | `ConditionalDiffusionModel`. |
| Conditioning | Informasi tambahan yang mengarahkan sampling. | context + retrieval + graph embedding. |
| Ensemble (`M`) | Banyak sampel hasil model. | Misalnya 50 sampel `[50,3]`. |
| Normalisasi | Memusatkan/menyamakan skala nilai. | `(x-mean)/(std+1e-5)`. |
| `log1p` / `expm1` | Transform log stabil dan inversenya. | Hanya channel precipitation. |
| Wet head | Klasifier hujan/tidak hujan tambahan. | `BCEWithLogitsLoss`, jika aktif. |
| CRPS | Skor probabilistik ensemble. | Makin rendah biasanya lebih baik. |
| CSI/POD/FAR | Skor event. | Success, detection, false alarm hujan. |
| Leakage | Informasi masa depan masuk ke training/prediksi. | Retrieval training mencoba mencegahnya dengan strict-past. |
| Checkpoint | File parameter dan konfigurasi model. | `models/diffusion_chkpt.pth` relatif dari `core/`. |

---

# 3. Kontrak kanonis

**Fakta kode:** definisi berada di `core/src/config.py:15-223`.

| Kontrak | Nilai |
|---|---|
| Topologi | `star` |
| Node order wajib | `[MAIN, UP, DOWN, LEFT, RIGHT]` |
| Jumlah node | 5 |
| Target policy | `main_node_only` |
| Context policy | `main_node_context` |
| Sequence default | 6 |
| Target horizon | satu row/timestep sesudah window |
| Target order | precipitation, wind_speed_10m, relative_humidity_2m |
| Feature order | 9 feature pada bagian 3.2 |
| Edge explicit | 8 directed star edges |
| Cap physical precipitation | 60 mm/jam |

Mengubah salah satu contract ini dapat mengubah arti index tensor, statistik, retrieval, state checkpoint, dan evaluasi. Jangan sekadar patch dimensi. Buat migrasi, retrain, lalu evaluasi ulang.

## 3.1 Node dan topologi

| Index | Node | Role | Lat | Lon | Elevasi config |
|---:|---|---|---:|---:|---:|
| 0 | `MAIN` | main | -6.75 | 107.00 | 1529.0 |
| 1 | `UP` | surrounding | -6.50 | 107.00 | 162.0 |
| 2 | `DOWN` | surrounding | -7.00 | 107.00 | 0.0 |
| 3 | `LEFT` | surrounding | -6.75 | 106.75 | 823.0 |
| 4 | `RIGHT` | surrounding | -6.75 | 107.25 | 288.0 |

```text
      UP
       |
LEFT - MAIN - RIGHT
       |
     DOWN
```

Edge explicit:

```text
MAIN -> UP     UP -> MAIN
MAIN -> DOWN   DOWN -> MAIN
MAIN -> LEFT   LEFT -> MAIN
MAIN -> RIGHT  RIGHT -> MAIN
```

`edge_index` adalah tensor `[2,8]`; sumber `core/src/config.py:74-84,161-177`.

## 3.2 Feature dan target

Feature order `[F=9]`:

```text
temperature_2m
relative_humidity_2m
dewpoint_2m
surface_pressure
wind_speed_10m
wind_direction_10m
cloud_cover
precipitation_lag1
elevation
```

Target order `[T=3]`:

```text
0 precipitation
1 wind_speed_10m
2 relative_humidity_2m
```

**Batas implementasi:** banyak operasi mengasumsikan precipitation berada di index `0`: transform `log1p`, `expm1`, clipping, wet label. Mengganti urutan target tanpa mengaudit setiap lokasi akan salah secara diam-diam. Bukti: `core/src/data/temporal_loader.py:161-174`, `core/src/train.py:395,433-439,573-576`, `core/src/inference.py:271-289`.

## 3.3 Edge attribute dan self-loop

`edge_attr [8,1]` memuat jarak Euclidean lat/lon. Semua spoke berjarak `0.25` derajat dari `MAIN`; nilainya konstan, sehingga tidak memberi variasi jarak yang informatif. Sumber: `core/src/config.py:180-204`.

**Batas implementasi:** `GATConv` dipanggil dengan default PyG; default tersebut dapat menambah self-loop internal. Jadi delapan edge input bukan satu-satunya edge yang dipakai convolution. Jangan mendeskripsikan sistem seolah hanya delapan pesan antar-node tanpa self-message. Kode GAT: `core/src/models/gnn.py:72-90`.

---

# 4. Data sampai tensor

## 4.1 Ingestion

`core/src/data/ingest.py:149-250` meminta weather hourly untuk lima koordinat. Sebelum fetch bulk, `_validate_grid_identity` menolak collision grid active nodes dan memeriksa pusat grid `MAIN` (`core/src/data/ingest.py:79-127`).

`precipitation_lag1` dibuat per node dengan `groupby(node).shift(1).fillna(0.0)` (`core/src/data/ingest.py:206-215`). Row disortir berdasarkan `date`, lalu canonical node order.

**Fakta kode:** feature `elevation` yang ditulis ke data datang dari panggilan layanan elevasi live (`core/src/data/ingest.py:37-49,159,201-204`), bukan langsung dari konstanta elevasi config. Tidak ada equality check antara keduanya.

## 4.2 Split

```text
train: date <= 2018-12-31
val:   2018-12-31 < date <= 2021-12-31
test:  date > 2021-12-31
```

Sumber: `core/src/train.py:49-60`.

**Batas implementasi:** train, val, test dibangun menjadi dataset terpisah. Enam row pertama setiap split tidak mendapat history dari split sebelumnya; sample valid dimulai pada index `seq_len`. Sumber: `core/src/data/temporal_loader.py:149,190-205`.

## 4.3 Normalisasi

Untuk precipitation target:

```text
p_log = log1p(p)
p_norm = (p_log - t_mean[0]) / (t_std[0] + 1e-5)
```

Untuk target lain dan feature:

```text
z = (value - mean) / (std + 1e-5)
```

`target` stats diambil hanya dari `MAIN` training. Feature stats mula-mula dari `MAIN` training. Jika `std` feature `MAIN` kurang dari `1e-6`, only feature itu diganti mean/std dari semua node training. Ini mencegah elevation konstan `MAIN` menghasilkan nilai sangat besar di node lain. Sumber: `core/src/train.py:63-113`.

**Batas implementasi:**

- target standard deviation nol tidak memiliki fallback khusus; hanya pembagi `+1e-5`;
- validasi schema menolak kolom hilang atau kolom **seluruhnya** NaN, bukan partial NaN atau `Inf`;
- tidak ada validation boundary input inference yang memeriksa finite/physical ranges.

Sumber: `core/src/config.py:125-143`, `core/src/data/temporal_loader.py:161-174`, `core/src/inference.py:223-227`.

## 4.4 Window sample

Untuk target index `t`:

```text
graph sequence = t-6, t-5, t-4, t-3, t-2, t-1
target         = MAIN target di t
context        = MAIN feature di t-1
```

Sumber: `core/src/data/temporal_loader.py:193-209`.

**Batas implementasi:** loader mengurutkan unique timestamps lalu memperlakukan indeks yang bertetangga sebagai waktu bertetangga. Tidak ada cek cadence satu jam atau gap. “6 timestep” hanya sama dengan “6 jam kontigu” bila data memang lengkap dan hourly. Sumber: `core/src/data/temporal_loader.py:108-149,193-205`.

## 4.5 Shape tensor

`B=batch`, `S=6`, `N=5`, `F=9`, `T=3`, `K=3`, `G=64` default.

| Objek | Shape | Makna |
|---|---|---|
| Snapshot raw | `[N,F]` | 5 node pada satu timestamp. |
| Sample graph sequence | list `S` | tiap graph `x=[N,F]`. |
| Batch satu timestep | `x=[B*N,F]` | PyG menggabungkan node seluruh sample. |
| PyG batch map | `[B*N]` | index graph bagi setiap node. |
| Target | `[B,T]` | target `MAIN` pada `t`. |
| Context | `[B,F]` | `MAIN` pada `t-1`. |
| Retrieval | `[B,K,T]` | K outcome historical. |
| Flatten retrieval | `[B,K*T]` | masuk MLP retrieval. |
| Graph embedding | `[B,G]` | hasil STGNN. |
| Noisy target / predicted noise | `[B,T]` | state diffusion. |
| Ensemble output | `[M,T]` | M sample setelah postprocess. |

Sumber: `core/src/data/temporal_loader.py:134-235`, `core/src/models/gnn.py:136-161`, `core/src/models/diffusion.py:100-159`.

---

# 5. STGNN: graph ruang lalu waktu

Kode: `core/src/models/gnn.py`.

## 5.1 Intuisi

Pada satu jam, kondisi tiap lokasi dapat memengaruhi ringkasan kondisi wilayah. GAT mengirim pesan pada graph. Sesudah itu, enam ringkasan jam diurutkan dan attention memilih pola masa lalu yang berguna bagi timestamp terakhir.

```text
pada tiap timestep:
  [5,9] --GAT--> embedding tiap node --mean pool--> [H]

untuk 6 timestep:
  [B,6,H] --causal temporal attention--> [B,H] --Linear--> [B,G]
```

## 5.2 GAT spasial

`SpatialGNN` (`core/src/models/gnn.py:64-96`):

```text
GATConv(F, H, heads=4, concat=True, edge_dim=1)
ReLU
GATConv(4H, H, heads=1, concat=False, edge_dim=1)
global_mean_pool
```

Secara konsep, node `i` menerima penjumlahan pesan node tetangga `j` dengan bobot attention yang dipelajari. Bobot dipengaruhi transform feature node dan dapat menerima edge attribute.

**Fakta kode:** output setelah `global_mean_pool` adalah satu vektor per graph, bukan prediksi `MAIN` dan bukan 5 node embedding yang dikembalikan.

**Tidak dibuktikan:** tidak ada bukti dari arsitektur saja bahwa attention learned bersifat meteorologis, kausal, atau lebih baik daripada baseline. Bukti harus datang dari artifact evaluasi yang cocok.

## 5.3 Temporal attention

`TemporalAttention` (`core/src/models/gnn.py:14-62`):

1. menerima `[B,S,H]`;
2. tambah positional embedding yang dilatih;
3. buat causal mask: posisi `i` dilarang melihat `j>i`;
4. `MultiheadAttention(x,x,x)`;
5. residual + dropout + LayerNorm;
6. return `x[:, -1, :]`.

Positional embedding dibutuhkan karena attention sendiri tidak tahu apakah sebuah vektor berasal dari `t-6` atau `t-1`.

## 5.4 STGNN lengkap

`SpatioTemporalGNN.forward` menjalankan `SpatialGNN` untuk setiap graph dalam sequence, stack menjadi `[B,S,H]`, lalu temporal attention dan output projection (`core/src/models/gnn.py:99-161`).

Training default:

```text
train hidden_dim argument = 128
STGNN internal H          = 64   # hidden_dim // 2
graph_dim G               = 64
GAT heads                 = 4
temporal heads            = 4
```

Sumber: `core/src/train.py:475-486`.

**Batas implementasi:** saat load inference, `hidden_dim=config[hidden_dim]//2`, GAT heads=4, attention heads=4 dibuat hardcoded; jumlah heads tidak disimpan sebagai contract checkpoint. Jika arsitektur ini berubah, loader juga harus dimigrasikan. Sumber: `core/src/inference.py:139-146`.

---

# 6. Retrieval: analog historical

Kode: `core/src/retrieval/base.py`, precompute di `core/src/train.py:116-194,428-473`.

## 6.1 Database

```text
index: faiss.IndexFlatL2(F=9)
key[tau]:   normalized MAIN feature pada tau
value[tau]: normalized MAIN target pada tau+1
```

Contoh: current weather dengan feature mirip jam 09:00 historis mengambil outcome jam 10:00 historis. K=3 memberi `[3,3]` per query.

`IndexFlatL2` adalah exact L2. Tidak ada retrieval encoder/metric-learning terlatih.

## 6.2 Strict-past

Untuk target sample `t`, context berada di `t-1`. Key database `j` punya value `j+1`. Training filter mempertahankan:

```text
j < t - 1
```

Jadi value `j+1 < t`; target saat ini tidak masuk retrieval. Implementasi mengaktifkan `strict_past=True` dan `exclude_self=True`. Sumber: `core/src/train.py:449-461`.

## 6.3 Batas exactness dan fallback

`_build_precomputed_retrieval` hanya mencari paling banyak `max(K, 8K)` kandidat L2 terdekat, lalu memfilter strict-past. Jika semua kandidat tersebut tersaring, ia mengisi nol; ini tidak membuktikan tidak ada neighbor valid yang lebih jauh. Sumber: `core/src/train.py:138-194`.

Zero fallback tersebut hanya ada dalam helper precompute. `RetrievalDatabase.query` sendiri:

- menolak DB kosong;
- menolak `K > ntotal`;
- tidak melakukan temporal filtering;
- tidak memberi zero fallback.

Sumber: `core/src/retrieval/base.py:30-73`.

## 6.4 Rekonstruksi DB inference

`load_model_and_stats` membaca `checkpoint.config[data_path]`, memfilter `date <= train_end` dan node `MAIN`, kemudian membangun key/value lagi (`core/src/inference.py:175-209`).

**Batas implementasi:** tidak ada hash dataset, version schema, validasi order/cadence, atau audit provenance sebelum DB direbuild. Bila `data_path` tidak ada, fungsi tetap return DB kosong; kegagalan baru terjadi saat query. Jangan menyebut DB inference “terjamin identik” dengan DB training hanya karena path sama.

---

# 7. RDM: conditional diffusion

Kode: `core/src/models/diffusion.py`.

## 7.1 Kondisi

Denoiser menerima:

```text
x         noisy target             [B,3]
t         diffusion timestep        [B]
context   normalized MAIN feature   [B,9]
retrieved historical outcomes       [B,3,3] atau [B,9]
graph_emb STGNN output              [B,64]
```

Encoding actual:

```text
t_emb    = time_mlp(t)
cond     = cond_mlp(context)
cond    += retrieval_mlp(flatten(retrieved))  # jika diberi
cond    += graph_mlp(graph_emb)               # jika diberi
emb      = t_emb + cond
```

Konstruktor/encoding: `core/src/models/diffusion.py:43-120`.

Tiga kondisi digabung **additively**, bukan concat global. Masing-masing MLP memproyeksikan ke `hidden_dim` yang sama.

## 7.2 Denoiser MLP

```text
h1       = SiLU(Linear(x)) + emb
h2       = SiLU(Linear(h1))
hmid     = SiLU(Linear(h2))
hup      = concat(hmid, h2)
noisehat = Linear(SiLU(Linear(hup)))
```

`noisehat [B,3]` memperkirakan noise, bukan target langsung. Kode: `core/src/models/diffusion.py:91-159`.

## 7.3 Training DDPM

```text
noise epsilon ~ N(0,I)
t ~ Uniform integer 0..999
x_t = scheduler.add_noise(target, epsilon, t)
epsilon_hat = denoiser(x_t, t, context, retrieved, graph_emb)
```

Scheduler: `DDPMScheduler(num_train_timesteps=1000, clip_sample=False)` (`core/src/models/diffusion.py:170-175`).

Loss utama:

```text
error = (epsilon_hat - epsilon)^2
weight = 1, bila |target_norm| <= 1
weight = 5, bila |target_norm| > 1
weight = 10, bila |target_norm| > 3
loss_noise = mean(weight * error)
```

Sumber: `core/src/models/diffusion.py:176-186`.

## 7.4 Wet head

Jika aktif:

```text
wet_logit = wet_head(condition embedding)
wet_label = precipitation_mm >= threshold_mm
loss = loss_noise + wet_loss_weight * BCEWithLogitsLoss(wet_logit, wet_label)
```

Kode: `core/src/models/diffusion.py:84-89,122-127`; training: `core/src/train.py:393-425,568-579`.

Wet head bukan target diffusion keempat; ia adalah loss klasifikasi tambahan dari kondisi yang sama.

## 7.5 Conditioning dropout

Saat training, retrieval dan graph embedding dapat dinolkan terpisah untuk tiap sample dengan probability `cond_dropout` default 0.15 (`core/src/train.py:542-551`). Ini membuat model kadang belajar tanpa condition tertentu.

**Tidak dibuktikan:** ini bukan proof bahwa ablation inference adalah contribution causal bersih. Lihat bagian evaluasi.

## 7.6 DDIM inference

`sample_fast` memakai `DDIMScheduler` dan default public helper memakai 20 langkah (`core/src/models/diffusion.py:233-270`; `core/src/inference.py:262-269`).

```text
x mulai dari Gaussian noise [M,3]
untuk tiap DDIM timestep:
  epsilon_hat = denoiser(x, t, conditions)
  epsilon_hat di-clamp [-10,10]
  x = DDIM.step(...).prev_sample
  x = nan_to_num(x, nan=0)
```

Ada method `RainForecaster.sample` dengan reverse DDPM penuh (`core/src/models/diffusion.py:188-231`), tetapi helper `run_inference_real` memakai DDIM. Jangan menyebut reverse DDPM sebagai jalur inference utama tanpa menyatakan perbedaan ini.

---

# 8. Training joint

Kode utama: `core/src/train.py:255-855`.

```text
parquet
  -> split temporal
  -> stats
  -> TemporalGraphDataset train/val
  -> FAISS retrieval train-only
  -> STGNN + diffusion
  -> joint backward + AdamW
  -> validation + wet threshold calibration
  -> save best checkpoint
```

Parameter default penting:

```text
batch_size=512, epochs=20
hidden_dim=128, graph_dim=64, K=3
lr=1e-3, AdamW weight_decay=1e-4
grad_clip_norm=1.0
ReduceLROnPlateau
early_stop_patience=12
seed=1
```

Sumber: `core/src/train.py:255-281,497-507`.

`graph_emb` dan diffusion parameters masuk dalam satu optimizer; tidak ada loss STGNN mandiri (`core/src/train.py:497-498,542-566`). Gradient diffusion melatih STGNN melalui `graph_emb`.

## 8.1 Reproducibility

Python, NumPy, Torch, CUDA RNG diset dengan seed (`core/src/train.py:308-317`).

**Batas implementasi:** code mengizinkan TF32 CUDA dan tidak memaksa deterministic algorithms/CuDNN deterministic. Seed sama tidak menjamin bitwise-identical result di semua perangkat/software. Sumber: `core/src/train.py:317-322`.

## 8.2 Checkpoint

Checkpoint terbaik validation disimpan sebagai:

```text
diffusion_state
st_gnn_state
stats
config: dims, K, seq_len, columns, split, loss metadata,
        rain metadata, node/topology metadata
```

Write: `core/src/train.py:727-788`.

Loader memeriksa key metadata required, `graph_topology == star`, `target_node_policy == main_node_only`, lalu state-dict keys dengan pengecualian tertentu (`core/src/inference.py:102-172`).

**Batas implementasi:** loader tidak membuktikan kesetaraan checkpoint terhadap current `NODE_NAMES`, coordinates, feature/target order, edge list, seq length, stats shape, dataset provenance, atau versi dependency. Konfigurasi checkpoint adalah sumber rekonstruksi utama; validasi penuh harus ditambahkan bila diperlukan.

---

# 9. Inference contract yang benar

Kode: `core/src/inference.py:46-301`.

## 9.1 Input aman

Kontrak graph yang benar untuk satu request:

```text
features_norm: [S,N,F] = [seq_len, 5, 9]
```

dengan canonical node order dan stats checkpoint.

**Batas implementasi:** input 2D dipromosikan menjadi `[1,S,F]`, tetapi graph builder kemudian menuntut axis node `[S,N,F]`; 2D bukan graph input yang valid secara contract. Batch input tidak didukung: kode mengambil `features_norm[0]` sebelum membuat graph. Bukti: `core/src/inference.py:229-260`, graph validation `:46-73`.

Sequence pendek dipad dengan snapshot terakhir; sequence panjang diambil `seq_len` terakhir (`core/src/inference.py:237-244`). Ini convenience behavior, bukan ekuivalen observasi historis lengkap.

## 9.2 Langkah

1. Muat checkpoint + stats + retrieval DB.
2. Bentuk graph sequence dari metadata checkpoint.
3. STGNN menghasilkan `graph_emb`.
4. Ambil `context_last = MAIN` pada timestamp terakhir.
5. Query K retrieval values.
6. DDIM sample M ensemble.
7. Denormalisasi, inverse log precipitation, physical clipping.
8. Jika wet specialization aktif dan peluang wet di bawah threshold checkpoint, set precipitation semua sample menjadi nol.

Sumber: `core/src/inference.py:95-301`.

## 9.3 Meaning of return `raw`

Return `raw` **bukan** output diffusion mentah. Ia sudah:

```text
samples * t_std + t_mean
precipitation: expm1 + clamp [0,60]
wind: clamp >=0
humidity: clamp [0,100]
optional wet gate
```

Sumber: `core/src/inference.py:271-300`.

---

# 10. Evaluation dan arti hasil

## 10.1 Final evaluator

`core/scripts/run_eval_final.py` membandingkan:

```text
persistence
MLP baseline
diffusion only
diffusion + retrieval
diffusion + GNN
full diffusion + retrieval + GNN
```

Ia memakai indeks target yang sama secara nominal (`core/scripts/run_eval_final.py:122-130`). Metrik: MAE, RMSE, correlation, CRPS, CSI/POD/FAR/Brier (`core/scripts/run_eval_final.py:37-45`).

## 10.2 Arti “ablation” di sini

Semua scenario diffusion memuat **satu full-model checkpoint**. Retrieval atau graph diganti tensor nol saat inference (`core/scripts/run_eval_final.py:221-239`). Ini mengukur sensitivity model penuh terhadap zero condition, bukan perbandingan model yang dilatih ulang tanpa komponen. Jangan mengklaim causal contribution arsitektur hanya dari tabel ini.

## 10.3 Batas evaluator

| Batas | Bukti kode | Konsekuensi |
|---|---|---|
| Stats direcompute dari parquet sekarang | `core/scripts/run_eval_final.py:60-75` | Metric dapat berubah jika data berubah walau checkpoint sama. |
| `--seq-len` independent checkpoint | `core/scripts/run_eval_final.py:606-628`; `core/src/inference.py:52-59` | Mismatch dapat merusak indexing/semantik. |
| Per-node sequence align by row index | `core/scripts/run_eval_final.py:81-119` | Timestamp antar-node tidak di-join/divalidasi; missing/misaligned data dapat salah secara diam-diam. |
| Out-of-range spoke dibuat nol | `core/scripts/run_eval_final.py:110-115` | Zero bukan observasi meteorologis valid. |
| Wet gate berbeda public inference | evaluator melewati gate bila semua window kering `core/scripts/run_eval_final.py:268-281` | Hasil evaluator tidak persis sama dengan `run_inference_real`. |
| MLP provenance berbeda | MLP checkpoint/data di `core/src/train_baseline.py`, evaluator `:147-183` | Comparison aktif perlu audit data/stats provenance. |

## 10.4 Evaluator tambahan

| File | Fungsi |
|---|---|
| `core/scripts/eval_rain_robust.py` | Robustness hujan, actual-update one-step, pemilihan minggu dry/median/wet. |
| `core/scripts/plot_week_timeseries.py` | Plot series mingguan artifact. |
| `core/src/models/mlp_baseline.py` | Deterministic MLP baseline. |
| `core/src/train_baseline.py` | Training checkpoint MLP baseline. |

## 10.5 Hasil aktif

**Fakta artifact, bukan janji model:** report aktif menunjukkan persistence mengalahkan `full_model` untuk precipitation RMSE/MAE/CRPS; `diff_gnn` mengalahkan full model pada tiga metrik deterministik precipitation; retrieval memperburuk hasil aktif dibanding `diff_only`. Sumber: `core/results/result_test/EVALUATION_REPORT.md:27-54`.

**Batas provenance:** artifact tidak secara lengkap mengikat checkpoint hash, checksum dataset, git commit, dependency versions, device, dan waktu run. Jangan menggeneralisasi hasil ini sebagai kebenaran permanen atau cross-dataset.

---

# 11. Test coverage

| Test | Bukti yang diperiksa |
|---|---|
| `test_config_topology.py` | Topologi/config kanonis. |
| `test_loader_contract.py` | Canonical order, node missing. |
| `test_gnn_edge_attr.py` | Edge attribute pathway/shape. |
| `test_retrieval.py` | Shape/error/cache retrieval DB. |
| `test_inference_graphs.py` | Kontrak graph inference. |
| `test_inference_smoke.py` | Checkpoint, retrieval, finite output shape. |
| `test_checkpoint_contract.py` | Metadata checkpoint dan retrieval dimension. |

**Tidak dicakup saat ini:** partial NaN/Inf rejection, timestamp cadence, current-config/checkpoint equality, dataset provenance retrieval, filtered-neighbor exactness, alignment timestamp evaluator, wet-gate parity, deterministic reproduction.

---

# 12. Roadmap belajar

## Tahap 1 — data contract

Baca `core/src/config.py`, bagian 3-4 dokumen ini.

Lulus bila dapat menjawab:

- Mengapa node order adalah data contract, bukan kosmetik?
- Mana feature, target, horizon, index precipitation?
- Apa beda elevasi config dan elevasi feature ingestion?

## Tahap 2 — dataset/tensor

Baca `core/src/data/temporal_loader.py`.

Lulus bila dapat menggambar:

```text
row data -> [timestamps,5,9] -> window 6 graph -> target MAIN t
```

Dan menjelaskan gap timestamp tidak divalidasi.

## Tahap 3 — STGNN

Baca `core/src/models/gnn.py`.

Lulus bila dapat membedakan:

```text
GAT: hubungan lokasi pada satu timestep.
Attention: hubungan antar-timestep.
Pooling: 5 node menjadi 1 graph vector.
```

## Tahap 4 — retrieval

Baca `core/src/retrieval/base.py`, bagian retrieval `core/src/train.py`.

Lulus bila dapat menulis:

```text
feature[tau] -> target[tau+1]
j < t-1 untuk target t
```

Dan menjelaskan limit `8K` candidate.

## Tahap 5 — diffusion

Baca `core/src/models/diffusion.py`.

Lulus bila dapat membedakan target, noisy target, noise actual, predicted noise, DDPM, DDIM, wet head.

## Tahap 6 — joint training/checkpoint

Baca `core/src/train.py`.

Lulus bila dapat menunjukkan gradient path diffusion loss menuju STGNN serta apa yang checkpoint benar-benar validasi/tidak validasi.

## Tahap 7 — inference/evaluation

Baca `core/src/inference.py`, `run_eval_final.py`, `eval_rain_robust.py`.

Lulus bila dapat menjelaskan mengapa `raw` sudah postprocessed, dan mengapa ablation zero-condition bukan retrained ablation.

## Latihan aman

```bash
cd /media/DiskE/SKRIPSI/Skripsi_Bevan/core
python -m unittest discover -s tests -p 'test_*.py'
```

Jangan menjalankan data fetch atau training penuh hanya untuk belajar; keduanya dapat memakai network/waktu dan menulis artifact.

---

# 13. Handover AI ML-only

## 13.1 Prompt bootstrap

```text
Kerjakan hanya ML pipeline core dari repository Skripsi_Bevan. Baca ROADMAP_RDM_STGNN_HANDOVER.md dahulu.

Fakta utama: sistem memprediksi ensemble cuaca MAIN satu timestep ke depan. Full model adalah retrieval-augmented conditional diffusion dengan STGNN condition. Canonical node order [MAIN, UP, DOWN, LEFT, RIGHT], target MAIN-only, target order [precipitation, wind_speed_10m, relative_humidity_2m], sequence contract, feature order, stats, retrieval key/value, dan checkpoint metadata harus dipertahankan.

Sebelum edit, identifikasi tensor shape, leakage risk, checkpoint impact, retrain requirement, evaluator impact, dan test yang membuktikan perubahan. Bedakan fakta kode, batas implementasi, dan hasil experiment. Jangan mengklaim result full model terbaik atau checkpoint/data identik tanpa artifact provenance.

Jangan mengubah node/feature/target order, topology, sequence semantics, retrieval tau->tau+1, atau output postprocess tanpa migrasi, retrain, dan evaluasi ulang. Cantumkan path:line untuk klaim teknis.
```

## 13.2 Mandatory read set

```text
ROADMAP_RDM_STGNN_HANDOVER.md
core/src/config.py
core/src/data/temporal_loader.py
core/src/models/gnn.py
core/src/models/diffusion.py
core/src/retrieval/base.py
core/src/train.py
core/src/inference.py
core/scripts/run_eval_final.py
core/scripts/eval_rain_robust.py
core/tests/test_checkpoint_contract.py
core/tests/test_loader_contract.py
```

Tambahkan `core/src/train_baseline.py` dan `core/src/models/mlp_baseline.py` bila menyentuh perbandingan baseline.

## 13.3 Checklist perubahan

- [ ] Node order tetap canonical di setiap timestamp/tensor.
- [ ] Feature dan target order tersinkronisasi data, stats, model, checkpoint, inference, evaluator.
- [ ] Precipitation tetap index 0 atau semua asumsi index diperbarui.
- [ ] Timestamp cadence/gap dibuktikan atau limitation didokumentasikan.
- [ ] Stats hanya dari train sesuai contract; fallback variance dipahami.
- [ ] Retrieval key `tau` dan value `tau+1` tetap benar.
- [ ] Tidak ada retrieval future/self leakage.
- [ ] `retrieval_dim == num_targets * k_neighbors`.
- [ ] Graph sequence bentuk `[S,N,F]`; batch semantics eksplisit.
- [ ] Checkpoint loader, saver, evaluator, test dimigrasikan bersama bila contract berubah.
- [ ] Retrain dilakukan jika learned architecture/contract/bobot berubah.
- [ ] Evaluasi menyebut checkpoint, dataset, split, seed, K, seq_len, ensemble, evaluator.

---

# 14. Runtime dan command

Dependency tercantum di root `requirements.txt:1-15`: Torch, PyG, FAISS CPU, diffusers, pandas, parquet-capable environment, serta library ingestion. PyG/Torch/CUDA compatibility adalah constraint environment, bukan otomatis diselesaikan requirements generic.

Jalankan dari `core/`; banyak path dan import relatif mengasumsikan lokasi itu.

```bash
cd /media/DiskE/SKRIPSI/Skripsi_Bevan/core
python -m unittest discover -s tests -p 'test_*.py'
python src/train.py --help
python scripts/run_eval_final.py --help
python scripts/eval_rain_robust.py --help
```

Training default:

```bash
python src/train.py
```

**Batas operasional:** ingestion membentuk network client/cache saat import (`core/src/data/ingest.py:21-24`); fetch data dan training dapat menulis artifact. Jalankan dengan data/checkpoint provenance yang sengaja dipilih.

---

# 15. Referensi cepat

| Area | Path:line |
|---|---|
| Kontrak node/schema/edge | `core/src/config.py:15-223` |
| Ingestion/grid/lag | `core/src/data/ingest.py:79-250` |
| Window/normalization/batch | `core/src/data/temporal_loader.py:25-260` |
| Retrieval DB | `core/src/retrieval/base.py:5-73` |
| STGNN | `core/src/models/gnn.py:14-161` |
| Diffusion/DDPM/DDIM | `core/src/models/diffusion.py:22-270` |
| Train/retrieval/checkpoint | `core/src/train.py:49-855` |
| Load/inference/postprocess | `core/src/inference.py:46-301` |
| Final evaluation | `core/scripts/run_eval_final.py:1-628` |
| Rain robustness evaluation | `core/scripts/eval_rain_robust.py:1-373` |
| MLP baseline | `core/src/models/mlp_baseline.py:1-61` |
| Baseline training | `core/src/train_baseline.py:1-301` |

---

# 16. Definisi “paham”

Seorang pemula/AI memahami sistem bila mampu, tanpa menebak:

1. Menggambar data flow `parquet -> window -> STGNN/retrieval -> diffusion -> ensemble`.
2. Menyebut semua shape tensor utama serta canonical order.
3. Menjelaskan `feature[tau] -> target[tau+1]` dan strict-past `j<t-1`.
4. Membedakan DDPM training, DDIM inference, wet head, dan postprocessed `raw`.
5. Menyebut limitation penting: constant edge distance, no cadence check, partial NaN not rejected, checkpoint validation partial, approximate filtered retrieval, evaluator caveat.
6. Menentukan kapan edit mewajibkan retrain/evaluation ulang.
7. Menolak klaim performa yang tidak memiliki provenance artifact memadai.

Tidak ada dokumen yang dapat membuktikan “100% yakin” atas behavior yang tidak diuji atau provenance yang tidak tersimpan. Dokumen ini memisahkan bagian itu secara eksplisit agar handover tidak kehilangan konteks atau berubah menjadi asumsi.
