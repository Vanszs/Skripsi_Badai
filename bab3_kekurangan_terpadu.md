# Handover ML Bevan — Audit Bab III terhadap Kode

> **Status:** dokumen mandiri untuk handover ke manusia atau AI lain.
>
> **Tujuan:** mencatat arsitektur, alur data, rumus, dimensi tensor, konfigurasi, istilah, landasan teori yang aman, serta koreksi terhadap `bab3_kekurangan.md`.
>
> **Aturan source of truth:** jika uraian Bab III berbeda dengan kode aktif, kode aktif menjadi fakta implementasi. Dokumen ini tidak menganggap alasan desain sebagai fakta bila alasan tersebut tidak tertulis atau tidak dapat dibuktikan dari kode.

---

## 0. Source of truth dan batas audit

File inti yang dicocokkan:

1. `core/src/config.py` — kontrak node, fitur, target, topologi, edge attribute.
2. `core/src/data/temporal_loader.py` — sliding window, DataLoader, normalisasi.
3. `core/src/models/gnn.py` — GAT spasial, pooling, temporal attention.
4. `core/src/models/diffusion.py` — embedding, conditioning, MLP denoiser, scheduler.
5. `core/src/train.py` — statistik training, FAISS retrieval, noising, loss, optimizer, scheduler.
6. `core/src/train_baseline.py` — dataset dan training baseline MLP.
7. `core/src/retrieval/base.py` — wrapper FAISS.
8. `core/src/models/mlp_baseline.py` — arsitektur baseline.

Dokumen pembanding:

- `bab3_kekurangan.md`
- `docs/DRAFT_PRASKRIPSI_BEVAN_V4.md`

### Hierarki kepastian

- **Fakta kode:** dapat ditunjukkan langsung dengan path dan baris.
- **Konsekuensi matematis:** hasil langsung dari bentuk tensor atau operasi kode.
- **Interpretasi teori:** alasan umum dari literatur; bukan bukti bahwa penulis kode memilih parameter karena alasan tersebut.
- **Tidak boleh diklaim:** alasan yang tidak ditulis di kode dan tidak memiliki referensi yang benar-benar diverifikasi.

---

# 1. Ringkasan satu halaman

Model utama adalah **Retrieval-Augmented Conditional Diffusion Model dengan Spatio-Temporal Graph Neural Network (STGNN)**.

Alurnya:

```text
Data cuaca 5 node × 9 fitur × 6 waktu
        ↓
Normalisasi berbasis statistik training
        ↓
6 graph snapshot; tiap graph memiliki 5 node dan 9 fitur
        ↓
GAT1: 9 → 64 per head, 4 head, concat → 256
        ↓
GAT2: 256 → 64, 1 head
        ↓
Global Mean Pooling atas 5 node → 1 embedding per waktu, 64 dimensi
        ↓
Stack 6 waktu → [B, 6, 64]
        ↓
Temporal Multi-Head Self-Attention, 4 head, causal mask
        ↓
Ambil representasi waktu terakhir → [B, 64]
        ↓
Output projection 64 → graph_dim=64
        ↓
Graph MLP 64 → 128 → 128

Context MAIN pada waktu terakhir: [B, 9]
        ↓
Context MLP 9 → 128 → 128

FAISS analog: 3 tetangga × 3 target → [B, 3, 3] → flatten [B, 9]
        ↓
Retrieval MLP 9 → 128 → 128

Graph embedding + context embedding + retrieval embedding
        ↓ element-wise addition
Condition embedding [B, 128]

Timestep diffusion t ∈ {0,...,999}
        ↓ sinusoidal embedding 128 → time MLP 128 → 256 → 128

Condition [B,128] + time embedding [B,128]
        ↓
Target jam berikutnya [B,3] diberi Gaussian noise → noisy target x_t [B,3]
        ↓
Denoiser MLP:
3 → 128 → 256 → 256
              ↘ concat dengan h2 → 512 → 128 → 3
        ↓
Predicted noise ε̂ [B,3]
        ↓
Weighted noise MSE + optional wet/dry BCE
```

**Poin penting:** target model utama hanya node `MAIN`, tiga variabel. Graph memakai semua lima node untuk membangun kondisi prediksi, bukan untuk menghasilkan lima target.

---

# 2. Kamus istilah dari nol

| Istilah | Arti mudah |
|---|---|
| Node | Satu stasiun/lokasi cuaca. |
| Edge | Hubungan arah antar-node. |
| Directed graph | Hubungan `A→B` berbeda dari `B→A`. |
| Star topology | MAIN berada di pusat; empat node lain terhubung langsung ke MAIN. |
| Feature | Angka masukan yang menggambarkan kondisi node. |
| Target | Nilai yang ingin diprediksi. |
| Tensor | Array angka yang dapat memiliki banyak dimensi. |
| Batch `B` | Banyak sampel diproses bersama. |
| Timestep `T`/`seq_len` | Banyak waktu historis dalam satu sampel; di sini 6. |
| Hidden dimension | Panjang vektor representasi internal model. |
| Embedding | Vektor hasil transformasi yang dipakai sebagai representasi. |
| Linear layer | Transformasi `y = xW^T+b`. |
| Activation | Fungsi non-linear setelah linear layer. |
| ReLU | `max(0,x)`. Nilai negatif menjadi nol. |
| SiLU | `x·sigmoid(x)`. Nilai positif dan sebagian nilai negatif dilewatkan secara halus. |
| GELU | `x·Φ(x)`, dengan `Φ` CDF normal standar. |
| Attention | Mekanisme memberi bobot berbeda pada sumber informasi. |
| Head | Jalur attention independen. |
| Softmax | Mengubah skor menjadi bobot relatif yang jumlahnya 1 pada sumbu tertentu. |
| Pooling | Meringkas banyak vektor menjadi lebih sedikit; `global_mean_pool` menghitung rata-rata node. |
| MLP | Multi-Layer Perceptron; rangkaian linear layer dan aktivasi. |
| FAISS | Library pencarian nearest-neighbor berkecepatan tinggi. |
| L2 distance | Jarak kuadrat Euclidean antar-vektor. |
| Diffusion | Proses menambah noise lalu belajar menghapus noise. |
| DDPM | Scheduler/noising training diffusion. |
| DDIM | Scheduler sampling cepat saat inferensi. |
| Denoiser | Jaringan yang memprediksi noise pada data noisy. |
| Skip connection | Jalur yang membawa representasi layer sebelumnya ke layer lebih akhir. |
| Concatenation | Menempelkan vektor pada sumbu fitur; bukan menjumlahkannya. |
| Element-wise addition | Menjumlahkan indeks yang sama: `a[i]+b[i]`. |
| Backpropagation | Mengalirkan gradien loss kembali untuk memperbarui parameter. |

---

# 3. Kontrak data dan indexing waktu

## 3.1 Node

Urutan node wajib:

```text
index 0: MAIN
index 1: UP
index 2: DOWN
index 3: LEFT
index 4: RIGHT
```

Didefinisikan di `core/src/config.py:22-69`.

Topologi eksplisit:

```text
MAIN → UP       UP → MAIN
MAIN → DOWN     DOWN → MAIN
MAIN → LEFT     LEFT → MAIN
MAIN → RIGHT    RIGHT → MAIN
```

Total edge eksplisit: `8`, bentuk `edge_index=[2,8]` (`config.py:74-84,161-177`).

`GATConv` PyTorch Geometric secara default menambahkan self-loop internal karena `add_self_loops=True` tidak ditulis eksplisit (`gnn.py:73-75`). Jadi self-loop bukan bagian dari `STAR_EDGES` di `config.py`; ia adalah perilaku dependency.

## 3.2 Fitur dan target

Sembilan fitur (`config.py:97-107`):

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

Tiga target (`config.py:86-90`):

```text
precipitation
wind_speed_10m
relative_humidity_2m
```

Target diambil hanya dari node `MAIN` (`temporal_loader.py:202-205`).

## 3.3 Makna waktu yang tepat

Untuk satu indeks target `t`:

```text
input graph: t-6, t-5, t-4, t-3, t-2, t-1
context:     MAIN pada t-1
target:      MAIN pada t
```

Implementasi:

- `valid_indices = seq_len ... num_timestamps-1` (`temporal_loader.py:149`).
- graph window `t-seq_len ... t-1` (`temporal_loader.py:193-199`).
- target `targets[t, MAIN, :]` (`temporal_loader.py:202-204`).
- context `features[t-1, MAIN, :]` (`temporal_loader.py:203-205`).

Jangan menulis input sebagai `t-5...t` jika simbol `t` berarti target. Itu membuat diagram tampak memakai masa depan.

## 3.4 Bentuk array paling awal

Sebelum batching:

```text
feature_data: [T, 5, 9]
target_data:  [T, 5, 3]
```

Untuk satu sampel:

```text
6 graph snapshot
setiap snapshot: x=[5,9]
```

Setelah collate PyG:

```text
setiap timestep graph batch:
Batch.x         = [B×5, 9]
Batch.edge_index= [2, B×8] sebelum self-loop internal
Batch.batch     = [B×5]
```

Model tidak menerima satu tensor mentah `[B,6,5,9]` langsung ke GAT. DataLoader memecah dimensi waktu menjadi daftar berisi 6 objek PyG `Batch`; STGNN memproses satu objek per waktu lalu men-stack hasilnya.

---

# 4. Normalisasi dan transformasi target

## 4.1 Statistik hanya dari data training

`compute_stats_from_training()` memakai node `MAIN` untuk statistik target dan fitur (`train.py:63-90`). Ini mencegah statistik validasi/test bocor ke training.

Precipitation ditransformasi dahulu:

\[
y'_{rain}=\log(1+y_{rain})
\]

Lalu z-score:

\[
z=\frac{v-\mu}{\sigma+10^{-5}}
\]

Implementasi denominator aktual adalah `std + 1e-5`, bukan `1e-6` (`temporal_loader.py:161-174`, `train.py:82-85`).

## 4.2 Fallback std fitur nol

Masalah: `elevation` konstan di MAIN, sehingga `c_std≈0` jika statistik hanya dihitung dari MAIN. Jika fitur tetangga dibagi `0+1e-5`, nilainya dapat menjadi sangat besar.

Kode (`train.py:91-104`):

1. Hitung `c_std` dari MAIN.
2. Cari fitur dengan `c_std < 1e-6`.
3. Untuk fitur degenerat itu saja, hitung mean/std dari seluruh node training.
4. Jika all-node std tetap degenerat, pakai `1.0`.

Ini **bukan** berarti semua fitur memakai statistik all-node. Hanya fitur degenerat yang memakai fallback all-node.

## 4.3 Apakah input GAT sudah normalisasi?

Pada pipeline training utama: **ya**. `train.py:353-370` mengirim `stats` ke `TemporalGraphDataset`; loader kemudian memakai `features_norm` (`temporal_loader.py:145-148,198-199`).

Namun secara class-level, jika `stats=None`, loader menggunakan fitur mentah. Jadi pernyataan aman:

> Dalam konfigurasi training utama, fitur `[B,6,5,9]` yang masuk GAT sudah dinormalisasi. Class loader sendiri mendukung mode tanpa statistik.

## 4.4 Inverse transform

Saat inferensi:

- precipitation: inverse z-score, lalu `expm1`, clamp `[0,60]` mm/jam.
- wind: clamp minimum `0`.
- humidity: clamp `[0,100]`.

Rujukan `config.py:92-95` dan `core/src/inference.py:271-289`.

---

# 5. Spatial GNN: GAT1, GAT2, pooling

## 5.1 Input satu waktu

```text
x = [B×5, 9]
edge_index = [2, B×8]
edge_attr = [B×8, 1]
```

`edge_attr` adalah jarak Euclidean latitude/longitude. Pada grid canonical semua jarak bernilai `0.25`, sehingga konstan dan tidak informatif sebagai pembeda utama (`config.py:180-190`). Sinyal spasial utama datang dari topologi dan fitur node.

## 5.2 GAT layer 1

Instansiasi training:

```python
GATConv(
    in_channels=9,
    out_channels=64,
    heads=4,
    concat=True,
    dropout=0.1,
    edge_dim=1,
)
```

Rujukan `gnn.py:69-74`, instansiasi `train.py:479-486` memakai `hidden_dim=128//2=64`.

Setiap head menghasilkan 64 angka. Karena `concat=True`:

\[
\mathbf{h}^{(1)}_i
=\mathbin{\|}_{k=1}^{4}\mathbf{h}^{(1,k)}_i
\in\mathbb{R}^{4\times64}=\mathbb{R}^{256}
\]

Operasi message passing per head secara konseptual:

\[
\mathbf{h}^{(1,k)}_i
=\sum_{j\in\mathcal{N}(i)}
\alpha^{(k)}_{ij}W^{(k)}\mathbf{x}_j
\]

Kode kemudian memakai ReLU (`gnn.py:85-87`):

```python
h = self.conv1(...)
h = self.relu(h)
```

## 5.3 Arah attention dan self-loop

`STAR_EDGES` menyimpan pasangan `source,target`. Untuk output node target, GAT mengagregasi edge masuk ke target.

Dengan self-loop default PyG:

- MAIN menerima self-loop + empat tetangga: biasanya 5 incoming relation.
- Tiap tetangga menerima self-loop + MAIN: biasanya 2 incoming relation.

Softmax attention dinormalisasi atas incoming relation node tujuan. Jangan menulis bahwa `MAIN→UP` dinormalisasi atas lima node; itu membingungkan arah. Nilai `α(MAIN,UP)` dan `α(UP,MAIN)` juga tidak harus sama karena arah, fitur, parameter, dan normalizer berbeda.

Self-loop penting secara mekanis karena mempertahankan pesan dari node itu sendiri. Tetapi kalimat “self-loop wajib menurut desain kita” harus diubah menjadi:

> Self-loop aktif sebagai default `GATConv`; ia memungkinkan fitur node sendiri ikut dalam agregasi. Kode konfigurasi tidak menambahkan self-loop secara manual.

## 5.4 GAT layer 2

```python
GATConv(
    in_channels=256,
    out_channels=64,
    heads=1,
    concat=False,
    dropout=0.1,
    edge_dim=1,
)
```

Output setiap node:

```text
[B×5, 64]
```

Tidak ada ReLU tambahan setelah GAT2 (`gnn.py:89-90`).

## 5.5 Global mean pooling

Untuk setiap graph:

\[
\mathbf{g}_\tau=\frac{1}{5}\sum_{i=1}^{5}\mathbf{h}_{i,\tau}
\in\mathbb{R}^{64}
\]

Kode `global_mean_pool` ada di `gnn.py:92-95`.

Makna penting: lima node **tidak berubah menjadi MAIN'**. Kelima node tetap memiliki representasi masing-masing setelah GAT; baru kemudian kelimanya dirata-ratakan menjadi satu graph embedding untuk waktu tersebut.

---

# 6. Temporal attention

## 6.1 Input

Enam graph embedding ditumpuk:

```text
[B, 6, 64]
```

Rujukan `gnn.py:143-155`.

## 6.2 Modul aktual

```python
nn.MultiheadAttention(
    embed_dim=64,
    num_heads=4,
    dropout=0.1,
    batch_first=True,
)
```

Ada juga:

- learnable positional embedding `[1,max_len,64]`, diinisialisasi normal `std=0.02`;
- causal upper-triangular mask;
- residual `x + dropout(attn_out)`;
- `LayerNorm`;
- output hanya `x[:,-1,:]`.

Rujukan `gnn.py:14-61`.

## 6.3 Makna attention

Temporal attention tidak menghitung attention langsung atas lima node. Spatial GAT sudah memproses node dahulu; global pooling menghasilkan satu vektor per jam. Temporal attention memilih kombinasi informasi antar-enam waktu.

Penjelasan awam:

> Setiap jam sudah diringkas menjadi satu vektor 64 angka. Temporal attention mempelajari bagian waktu mana yang paling membantu representasi waktu terakhir, dengan larangan melihat masa depan.

Lebih presisi daripada mengatakan “attention score dikalikan fitur node MAIN”. Yang dikalikan adalah representasi temporal hasil spatial GNN dan pooling.

## 6.4 Bukan sekadar bobot lalu rata-rata

`nn.MultiheadAttention` memakai query, key, value yang dipelajari melalui proyeksi internal. Outputnya bukan hanya `softmax(score) × embedding` sederhana, walaupun intuisi weighted aggregation benar. Kode mengambil representasi timestep terakhir setelah residual dan LayerNorm.

## 6.5 Output STGNN

```text
TemporalAttention output: [B,64]
output_proj: Linear(64,64)
graph_emb: [B,64]
```

Training utama memanggil:

```python
SpatioTemporalGNN(
    node_features=9,
    hidden_dim=64,
    output_dim=64,
    num_gat_heads=4,
    num_attn_heads=4,
    seq_len=6,
)
```

Rujukan `train.py:479-486`.

---

# 7. FAISS retrieval

## 7.1 Bagaimana FAISS memperoleh data saat training awal?

FAISS tidak membutuhkan embedding hasil STGNN. FAISS dibangun dari **feature vector yang sudah tersedia** sebelum model dilatih.

Kode `train.py:428-446`:

```text
train_features_norm[τ]  = fitur MAIN waktu τ, sudah dinormalisasi
train_targets_norm[τ+1] = target MAIN waktu τ+1, sudah ditransformasi/dinormalisasi
```

Database:

```text
key   = train_features_norm[:-1]  → dimensi 9
value = train_targets_norm[1:]    → dimensi 3
```

Jadi pada awal training:

- tidak ada ketergantungan pada embedding neural yang belum belajar;
- query adalah context MAIN berdimensi 9;
- FAISS mencari key historis berdimensi 9;
- value yang dikembalikan adalah tiga target masa depan, bukan embedding graph.

## 7.2 Index dan jarak

`core/src/retrieval/base.py:9-12`:

```python
faiss.IndexFlatL2(9)
```

Jarak kuadrat L2:

\[
d(\mathbf q,\mathbf k_i)
=\sum_{f=1}^{9}(q_f-k_{i,f})^2
\]

Normalisasi FAISS bukan normalisasi khusus FAISS. Key dan query sudah memakai statistik fitur yang sama sebelum masuk index/search. `IndexFlatL2` hanya menghitung jarak L2; ia tidak melakukan z-score.

## 7.3 Output retrieval

FAISS mengembalikan:

```text
[B, 3, 3]
```

Artinya 3 tetangga, tiap tetangga memiliki 3 target. Retrieval MLP me-*flatten*:

```text
[B, 3, 3] → [B, 9]
```

Lalu:

```text
9 → 128 → 128
```

## 7.4 Leakage protection

Training memakai:

- `strict_past=True`;
- `exclude_self=True`;
- key masa depan disaring;
- jika tetangga kurang, kandidat terakhir diulang;
- jika tidak ada kandidat, dipakai nol.

Rujukan `train.py:153-194,449-470`.

Validasi pada kode ini memakai `strict_past=False`, sehingga jangan mengklaim filtering strict-past berlaku identik untuk semua split.

---

# 8. Conditioning embedding

## 8.1 Tiga sumber kondisi

### Context

```text
MAIN pada t-1: [B,9]
Context MLP: 9 → 128 → 128
Aktivasi: SiLU setelah linear pertama
```

### Retrieval

```text
FAISS values: [B,3,3]
flatten: [B,9]
Retrieval MLP: 9 → 128 → 128
Aktivasi: SiLU setelah linear pertama
```

### Graph

```text
STGNN graph_emb: [B,64]
Graph MLP: 64 → 128 → 128
Aktivasi: SiLU setelah linear pertama
```

Rujukan `diffusion.py:64-83`.

## 8.2 Penjumlahan element-wise

Kode:

```python
cond_emb = context_emb
cond_emb = cond_emb + r_emb
cond_emb = cond_emb + g_emb
```

Semua bentuk `[B,128]`, sehingga:

\[
\mathbf c
=\mathbf c_{context}+\mathbf c_{retrieval}+\mathbf c_{graph}
\in\mathbb R^{128}
\]

Ya, penjumlahan dilakukan indeks demi indeks:

```text
cond[0] = context[0] + retrieval[0] + graph[0]
cond[1] = context[1] + retrieval[1] + graph[1]
...
```

Makna asli tidak “hilang” secara matematis karena setiap encoder belajar memetakan sumbernya ke ruang bersama. Namun penjumlahan memang tidak menjaga label sumber secara eksplisit seperti concatenation. Ini trade-off desain: ukuran tetap 128 dan operasi sederhana, tetapi sumber tidak dapat dipisahkan langsung dari vektor gabungan.

Jangan menyebut `cond_emb` sebagai “bias” secara ketat. Ia adalah embedding kondisi aditif.

## 8.3 Time embedding

Timestep diffusion adalah integer acak:

```text
t ∈ {0,...,999}
```

Sinusoidal embedding membuat identitas kontinu untuk setiap nilai `t`:

\[
\omega_i=\exp\left(-i\frac{\log(10000)}{H/2-1}\right)
\]

\[
e(t)= [\sin(t\omega_0),...,\sin(t\omega_{H/2-1}),
\cos(t\omega_0),...,\cos(t\omega_{H/2-1})]
\]

Kode `diffusion.py:8-20`.

Untuk `hidden_dim=128`:

```text
sinusoidal embedding: [B,128]
time MLP: 128 → 256 → 128
aktivasi: GELU
```

Sinusoidal embedding **bukan noise Gaussian**. Ia hanya memberi model identitas/posisi timestep. Gaussian noise tetap dibuat terpisah dengan `torch.randn_like(targets)`.

Kemudian:

\[
\mathbf e = \mathbf t_{emb}+\mathbf c
\]

Keduanya `[B,128]`, dijumlahkan indeks demi indeks.

---

# 9. Diffusion training dan denoiser

## 9.1 Target yang diberi noise

Target yang masuk noising adalah target yang sudah:

1. precipitation `log1p`;
2. dinormalisasi z-score;
3. diambil hanya untuk MAIN.

Kode:

```python
noise = torch.randn_like(targets)
timesteps = torch.randint(0, 1000, (...))
noisy_target = scheduler.add_noise(targets, noise, timesteps)
```

Bentuk:

```text
noise:       [B,3]
timesteps:   [B]
target y0:   [B,3]
noisy x_t:   [B,3]
```

Rumus:

\[
\mathbf x_t
=\sqrt{\bar\alpha_t}\mathbf y_0
+\sqrt{1-\bar\alpha_t}\boldsymbol\epsilon,
\quad \boldsymbol\epsilon\sim\mathcal N(0,I)
\]

Rujukan `train.py:552-554`.

Yang diberi noise adalah target `[B,3]`, bukan final condition embedding. Condition dipakai untuk membantu denoiser memprediksi noise.

## 9.2 Denoiser aktual

Dengan `hidden_dim=128`:

```text
x_t:    [B,3]
down1:  3 → 128 + SiLU
h1:     down1(x_t) + time_emb + cond_emb → [B,128]
down2:  128 → 256 + SiLU → h2 [B,256]
mid:    256 → 256 + SiLU → h_mid [B,256]
concat: h_mid || h2 → [B,512]
up1:    512 → 128 + SiLU → [B,128]
out:    128 → 3 → predicted noise ε̂ [B,3]
```

Rujukan `diffusion.py:91-98,141-154`.

Rumus:

\[
\mathbf h_1=\operatorname{SiLU}(W_1\mathbf x_t+b_1)+\mathbf e
\]

\[
\mathbf h_2=\operatorname{SiLU}(W_2\mathbf h_1+b_2)
\]

\[
\mathbf h_m=\operatorname{SiLU}(W_m\mathbf h_2+b_m)
\]

\[
\mathbf h_c=[\mathbf h_m\Vert\mathbf h_2]
\]

\[
\hat{\boldsymbol\epsilon}=W_o\operatorname{SiLU}(W_u\mathbf h_c+b_u)+b_o
\]

**Skip connection:** concatenation, bukan addition, bukan Conv1D, bukan MaxPooling.

## 9.3 Mengapa hidden dimension diperbesar 128→256?

Fakta: kode memang memakai 256 (`diffusion.py:93-97`).

Penjelasan teori yang aman:

> Dimensi diperbesar untuk menyediakan ruang representasi internal lebih luas sebelum diproyeksikan kembali ke output 3 dimensi. Ini adalah keputusan kapasitas arsitektur, bukan hukum wajib dan bukan bukti bahwa 256 selalu optimal.

Jangan menulis bahwa kode memilih 256 karena Cover’s theorem, hardware, atau bukti eksperimen tertentu kecuali ada eksperimen/referensi yang benar-benar mendukung klaim itu. Cover’s theorem membahas pemetaan ke ruang berdimensi tinggi secara umum; ia tidak menentukan angka 256 untuk model ini.

## 9.4 SiLU versus ReLU versus GELU

### ReLU

\[
\operatorname{ReLU}(x)=\max(0,x)
\]

Bahasa mudah: angka negatif dipotong menjadi nol. Kode memakai ReLU setelah GAT1 dan baseline MLP.

### SiLU

\[
\operatorname{SiLU}(x)=x\cdot\sigma(x)
\]

Bahasa mudah: angka dilewatkan melalui gerbang sigmoid yang halus. Angka positif biasanya lewat kuat; angka negatif tidak langsung dipotong kasar menjadi nol, hanya diredam.

Kode memakai SiLU pada context MLP, retrieval MLP, graph MLP, wet head, dan denoiser.

### GELU

\[
\operatorname{GELU}(x)=x\Phi(x)
\]

Bahasa mudah: input dilewatkan berdasarkan peluang halus yang bergantung pada besar nilainya. Kode memakai GELU khusus pada time MLP.

Pernyataan aman untuk sidang:

> ReLU dipakai pada GAT1 dan baseline. SiLU dipakai pada encoder kondisi dan denoiser agar transformasi non-linear berlangsung halus. GELU dipakai pada time MLP sesuai implementasi. Kode tidak membuktikan bahwa satu aktivasi selalu lebih baik; pemilihan tersebut adalah konfigurasi arsitektur yang perlu divalidasi melalui eksperimen.

---

# 10. Objective dan training controls

## 10.1 Weighted noise MSE

Error per elemen:

\[
e=(\hat\epsilon-\epsilon)^2
\]

Bobot:

```text
1  jika |target| ≤ 1
5  jika |target| > 1
10 jika |target| > 3
```

Loss:

\[
L_{noise}=\operatorname{mean}(w\odot(\hat\epsilon-\epsilon)^2)
\]

Rujukan `diffusion.py:176-186`.

## 10.2 Wet/dry auxiliary head

Model juga memiliki `wet_head`:

```text
condition [B,128] → 128 → 1 logit
aktivasi hidden: SiLU
probability: sigmoid(logit)
```

Jika rain specialization aktif:

\[
L=L_{noise}+0.7L_{wet}
\]

`L_wet` adalah `BCEWithLogitsLoss` dengan `pos_weight` dari data training (`train.py:400-420,572-578`). Threshold probabilitas dikalibrasi pada validation memakai CSI (`train.py:691-703`).

## 10.3 Optimizer dan kontrol numerik

Model utama:

```text
optimizer: AdamW
learning rate: 1e-3
a weight decay: 1e-4
gradient clipping: norm 1.0 default
non-finite batch: dilewati
```

Scheduler learning rate:

```text
ReduceLROnPlateau
factor=0.5
patience=3
threshold=1e-4
min_lr=1e-6
```

Early stopping default patience `12`.

Ini berbeda dari **diffusion noise scheduler**. Ada dua scheduler:

1. scheduler diffusion: mengatur beta/alpha noise;
2. scheduler optimizer: mengatur learning rate.

---

# 11. DDPM training dan DDIM inference

## 11.1 Training scheduler

Kode membuat:

```python
DDPMScheduler(num_train_timesteps=1000, clip_sample=False)
```

Yang dapat dipastikan dari source lokal:

- 1000 timestep training;
- `clip_sample=False`.

Beta schedule detail berasal dari default versi `diffusers` yang terpasang, bukan argumen eksplisit kode ini. Jangan menulis angka beta linear tertentu sebagai konfigurasi eksplisit tanpa mengunci versi library atau memeriksa konfigurasi runtime.

## 11.2 Sampling

`sample()` memakai scheduler DDPM dan mulai dari:

```text
x ~ N(0,I), shape [num_samples,3]
```

`sample_fast()` membuat `DDIMScheduler(num_train_timesteps=1000, clip_sample=False)` dan default:

```text
num_inference_steps=50
```

`20` bukan default fungsi `sample_fast()` pada source yang diaudit. Jika caller tertentu mengirim 20, tulis sebagai konfigurasi caller tersebut, bukan default sistem.

Sampling fast juga:

- memakai AMP di CUDA;
- clamp `noise_pred` ke `[-10,10]`;
- mengganti NaN state menjadi `0`.

Rujukan `diffusion.py:233-270`.

Penjelasan aman tentang DDIM:

> DDIM mengurangi jumlah langkah sampling dengan memetakan sejumlah langkah inferensi ke scheduler 1000 timestep. Jalur dapat deterministik terhadap initial state dan konfigurasi, tetapi initial state tetap diambil dari `torch.randn`; determinisme total membutuhkan pengaturan seed.

---

# 12. Baseline MLP

Baseline hanya menggunakan node MAIN. Tidak menggunakan graph, GAT, temporal attention, FAISS, atau diffusion.

## 12.1 Input 54

Window:

```text
6 waktu × 9 fitur = 54 angka
```

Kode:

```python
x = features[idx : idx + seq_len].flatten()
y = targets[idx + seq_len]
```

Bentuk:

```text
satu sample x: [54]
batch x:       [B,54]
target:        [B,3]
```

`flatten first` berarti array temporal `[6,9]` dibentangkan menjadi satu vektor `[54]`; bukan membuat `[54,54]`.

## 12.2 Arsitektur baseline

```text
Linear(54,128)
ReLU
Dropout(0.2)
Linear(128,128)
ReLU
Dropout(0.2)
Linear(128,3)
```

Rujukan `core/src/models/mlp_baseline.py:43-50`.

Training:

```text
loss: MSE
optimizer: AdamW
scheduler: CosineAnnealingLR
```

---

# 13. Audit koreksi terhadap `bab3_kekurangan.md`

## Koreksi wajib

1. **Normalisasi:** gunakan `1e-5` sebagai epsilon implementasi; fallback all-node hanya fitur degenerat.
2. **Input GAT:** pada training pipeline memang sudah normalized; class loader bisa raw jika `stats=None`.
3. **GAT1:** 9→64 per head ×4→256; ini default training, bukan invariant semua constructor.
4. **Self-loop:** tidak ada di `STAR_EDGES`; muncul dari default PyG `GATConv`.
5. **Softmax:** normalisasi atas incoming edge untuk node tujuan; jangan menyebut arah MAIN→UP seolah denominator selalu lima.
6. **Edge attribute:** diteruskan ke GAT, tetapi semua canonical distance `0.25`; konstan.
7. **Retrieval value:** tiga target normalized waktu berikutnya, bukan historical features.
8. **FAISS L2:** berlaku pada query/key sembilan fitur; bukan pada tensor retrieval hasil flatten.
9. **GELU:** memang kode memakai GELU; klaim “pasti lebih cocok karena Gaussian” terlalu kuat.
10. **Condition:** `cond_emb` adalah penjumlahan embedding, bukan bias khusus.
11. **Target noising:** target sudah `log1p`/z-score, bukan target fisik mentah.
12. **Denoiser:** pure MLP; tidak ada Conv1D atau MaxPooling.
13. **Skip:** `torch.cat([h_mid,h2])`, hasil 512; bukan residual addition.
14. **DDIM:** default `sample_fast` adalah 50 langkah, bukan 20.
15. **DDPM beta:** detail beta default library tidak eksplisit di kode lokal.
16. **Wet head:** model memiliki auxiliary wet/dry head dan BCE tambahan.
17. **Physical clamps:** inverse inference memiliki batas precipitation, wind, humidity.
18. **Baseline:** input 54 dan arsitektur memiliki dua ReLU serta dua Dropout.
19. **Window:** model utama memakai graph `t-6...t-1`, context `t-1`, target `t`.
20. **Statistik:** target dan fitur berasal dari MAIN training; all-node fallback hanya fitur std degenerat.

## Klaim lama yang harus dihapus atau dilemahkan

- “4 head adalah standar yang pasti” → ubah menjadi konfigurasi yang dipakai dan didukung prinsip multi-head attention; optimalitas perlu eksperimen.
- “64 adalah information bottleneck yang terbukti optimal” → ubah menjadi kapasitas yang dipilih pada implementasi.
- “128 dipilih karena mencegah hilangnya variabilitas” → tidak terbukti dari kode.
- “256 dipilih karena Cover’s theorem” → tidak boleh ditulis sebagai alasan resmi tanpa bukti desain/eksperimen.
- “SiLU mencegah NaN” → tidak dapat disimpulkan dari aktivasi saja.
- “GELU lebih presisi karena noise Gaussian” → interpretasi spekulatif.
- “DDIM selalu deterministik” → harus dikaitkan dengan initial random state dan seed.
- “FAISS menghasilkan embedding” → salah; FAISS menghasilkan indeks tetangga/value retrieval.

---

# 14. Landasan teori yang aman untuk Bab III

## 14.1 GAT

Landasan umum yang relevan:

- Graph Attention Network memperbarui node dengan weighted neighborhood aggregation.
- Multi-head attention menyediakan beberapa proyeksi perhatian dan dapat digabung dengan concatenation.
- Self-loop mempertahankan kontribusi node sendiri.

Rujukan dasar GAT memang lebih lama dari 2020. Jika aturan skripsi melarang referensi sebelum 2020, jangan memalsukan tahun atau menyebut paper 2021 yang tidak diverifikasi. Gunakan paper modern yang benar-benar tersedia di daftar pustaka, atau nyatakan teori sebagai definisi metode dan pisahkan dari klaim kebaruan.

## 14.2 Attention temporal

Landasan aman:

- self-attention menghitung interaksi antarposisi;
- positional embedding menjaga informasi urutan;
- causal mask mencegah akses masa depan;
- residual dan LayerNorm membantu stabilitas transformasi.

Khusus kode ini, positional embedding bukan sinusoidal melainkan parameter trainable (`gnn.py:31-32`). Sinusoidal hanya dipakai pada timestep diffusion (`diffusion.py:8-20`).

## 14.3 Standardization

Landasan aman:

- z-score menyamakan skala numerik fitur;
- statistik seharusnya dihitung dari training split;
- `log1p` membantu meredam skew precipitation;
- epsilon mencegah pembagian nol;
- fallback all-node adalah mitigasi numerik spesifik implementasi.

Jangan mengklaim kode “mengikuti WeatherBench” kecuali referensi dan kesamaan prosedur benar-benar dimasukkan serta diverifikasi.

## 14.4 Diffusion

Landasan aman:

- forward process menambahkan Gaussian noise bertahap;
- model belajar memprediksi noise;
- reverse process mengurangi noise;
- DDPM dan DDIM adalah pilihan scheduler berbeda.

## 14.5 Dimensi 64, 128, 256

Jawaban sidang paling aman:

> Angka 64, 128, dan 256 adalah hyperparameter kapasitas representasi pada implementasi. Dimensi 64 cukup untuk graph embedding yang berasal dari lima node, 128 dipakai sebagai ruang bersama agar context, retrieval, graph, dan timestep dapat dijumlahkan, sedangkan 256 memberi ruang kerja lebih besar pada hidden layer denoiser sebelum dikembalikan ke tiga target. Angka tersebut bukan konsekuensi matematis wajib; pemilihannya harus dibuktikan melalui konfigurasi dan, bila ditanyakan optimalitasnya, melalui ablation/validasi eksperimen.

Ini lebih akurat daripada mengklaim angka tersebut “standar” atau ditentukan oleh satu theorem.

---

# 15. Handover singkat untuk AI lain

Gunakan konteks berikut sebagai prompt awal:

```text
Saya bekerja pada /media/DiskE/SKRIPSI/Skripsi_Bevan.
Source of truth ML:
- core/src/config.py
- core/src/data/temporal_loader.py
- core/src/models/gnn.py
- core/src/models/diffusion.py
- core/src/train.py
- core/src/train_baseline.py
- core/src/retrieval/base.py
- core/src/models/mlp_baseline.py

Kontrak model utama:
- graph directed star: MAIN, UP, DOWN, LEFT, RIGHT
- 5 node, 9 fitur/node, sequence 6 waktu
- input graph untuk target t adalah t-6..t-1
- context MAIN pada t-1
- target MAIN pada t, 3 variabel
- fitur training dinormalisasi z-score; precipitation memakai log1p
- MAIN-only stats; all-node fallback hanya feature std < 1e-6; epsilon 1e-5
- GAT1: 9→64/head, 4 heads, concat=True →256, ReLU
- GAT2: 256→64, 1 head, concat=False
- global mean pool 5 node →64 per waktu
- temporal attention: MultiheadAttention 64 dim, 4 heads, causal mask, learnable positional embedding, residual+LayerNorm, ambil timestep terakhir
- STGNN output [B,64]
- graph/context/retrieval encoders menjadi [B,128] dan dijumlahkan element-wise
- FAISS IndexFlatL2 pada key fitur MAIN normalized dim 9; value adalah target MAIN normalized waktu berikutnya; k=3; output [B,3,3]→flatten [B,9]
- timestep t integer 0..999 memiliki sinusoidal embedding lalu time MLP 128→256→128 GELU
- target normalized [B,3] diberi Gaussian noise menjadi x_t
- denoiser: 3→128→256→256; concat mid+h2→512; 512→128→3; SiLU kecuali output linear
- loss weighted noise MSE, optional wet/dry BCE
- training scheduler DDPM 1000 timestep; fast inference DDIM default 50 step
- baseline: MAIN-only, 6×9=54 input, Linear54→128→128→3 dengan ReLU/Dropout

Saat menjawab, pisahkan fakta kode, konsekuensi matematika, dan interpretasi teori. Jangan mengarang alasan pemilihan hyperparameter. Jangan menyebut DDIM default 20; source default 50.
```

---

# 16. Checklist final sebelum Bab III disahkan

- [ ] Semua path kode di atas masih sesuai branch yang dipakai.
- [ ] Dimensi GAT1 ditulis `64 per head × 4 = 256`.
- [ ] GAT2 ditulis `256→64`, bukan `256→256`.
- [ ] Temporal attention dibedakan dari spatial attention.
- [ ] Temporal attention mengambil representasi terakhir, bukan mean waktu sederhana.
- [ ] Input GAT disebut normalized hanya dalam konteks training dengan `stats`.
- [ ] `std + 1e-5` ditulis tepat.
- [ ] Fallback all-node dibatasi fitur degenerat.
- [ ] FAISS query/key dibedakan dari retrieval value.
- [ ] Retrieval value ditulis target waktu berikutnya berdimensi 3.
- [ ] `cond_emb` ditulis penjumlahan element-wise tiga encoder.
- [ ] Time embedding dibedakan dari Gaussian noise.
- [ ] Target yang diberi noise disebut normalized target.
- [ ] Denoiser disebut pure MLP.
- [ ] Skip connection disebut concatenation 512.
- [ ] Wet/dry head tidak dilupakan.
- [ ] DDPM training dan DDIM inference tidak tertukar.
- [ ] Default DDIM ditulis 50, bukan 20.
- [ ] Baseline `6×9=54` dan dropout dicantumkan.
- [ ] Referensi teori tidak dipakai untuk mengklaim alasan kode tanpa bukti.

---

## Kesimpulan

Dokumen ini mencakup kekurangan rumus, konfigurasi layer, flow data, indexing waktu, dimensi tensor, normalisasi, FAISS, temporal attention, diffusion, denoiser, loss, scheduler, baseline, dan koreksi klaim lama. Semua fakta implementasi utama dicocokkan langsung dengan source yang disebut pada bagian 0. Klaim tentang optimalitas angka 64/128/256, pemilihan aktivasi, atau superioritas metode tetap harus disebut sebagai justifikasi desain atau hipotesis eksperimen, bukan fakta yang dibuktikan oleh kode.
