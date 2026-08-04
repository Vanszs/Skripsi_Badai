# Revisi V5 — Audit Bab II, Aturan Rumus, dan Handover Implementasi

## Tujuan dan batas dokumen

Dokumen ini mandiri. Basis auditnya adalah `bab3_kekurangan_terpadu.md`, tetapi fakta implementasi dicocokkan kembali dengan source code aktif. Dokumen ini berfokus pada enam kebutuhan:

1. menentukan apakah alur GNN/spatio-temporal sudah ada di draft Bab II;
2. memisahkan teori yang layak ditulis di Bab II dari implementasi yang harus ditulis di Bab III;
3. mengklasifikasikan seluruh komponen model dan status kebutuhan rumusnya;
4. menyediakan rumus inti dengan notasi yang konsisten dengan kode;
5. memberi rekomendasi paragraf untuk setiap section;
6. menjaga sitasi akademik tetap nyata, relevan, dan dapat diverifikasi.

Aturan utama: **kode aktif adalah sumber fakta implementasi; literatur adalah sumber teori; hasil eksperimen adalah satu-satunya dasar klaim performa proyek.** Tidak ada PCA atau reduced retrieval space pada implementasi. Tidak ada rumus fundamental baru yang terbukti; kebaruan berada pada komposisi dan penerapan sistem untuk data stasiun Gunung Gede–Pangrango.

---

## 1. Jawaban langsung: apakah flow GNN/spatio-temporal sudah ada di Bab II?

### 1.1 Status pada draft v4

**Sudah ada secara konseptual, tetapi belum cukup sebagai uraian teknis yang dapat diturunkan ulang.** Draft v4 telah memuat:

- judul dan tujuan yang menyebut *Spatio-Temporal Graph Neural Network*;
- konsep node, edge, dan topologi lima node;
- alasan umum penggunaan representasi graph untuk relasi spasial non-grid;
- section `2.2.5 Spatio-Temporal Graph Conditioning pada Representasi Data Elevasi`;
- pernyataan bahwa graph conditioning digabungkan dengan diffusion model;
- rumus diffusion pada section `2.2.3`, tetapi bukan rumus GAT dan temporal attention yang sesuai implementasi.

Bagian yang relevan dapat ditemukan di `docs/DRAFT_PRASKRIPSI_BEVAN_V4.md:1735-1755` dan bagian konteks diffusion di `docs/DRAFT_PRASKRIPSI_BEVAN_V4.md:1397-1441`.

### 1.2 Gap detail yang harus ditutup di Bab II

Bab II perlu menambahkan landasan teori dan notasi berikut, bukan menyalin seluruh source code:

1. **Definisi graph berarah.** Nyatakan `G=(V,E)` dengan lima node `V={MAIN,UP,DOWN,LEFT,RIGHT}` dan delapan edge eksplisit dua arah pada star topology.
2. **Pesan GAT.** Jelaskan GAT standar Veličković et al. (2018), arXiv:1710.10903, dengan skor attention berbasis fitur node, normalisasi softmax pada tetangga masuk, dan agregasi multi-head. Jangan atribusikan formula edge-feature kepada paper GAT asli. Pada implementasi PyG, `edge_dim=1` menambahkan jalur edge feature sesuai API `GATConv`; itu harus ditulis sebagai detail implementasi/dependency, bukan sebagai isi formula GAT standar.
3. **Arah agregasi.** Untuk node tujuan `i`, koefisien dinormalisasi atas `j∈N(i)`. Jangan menyatakan edge `MAIN→UP` dinormalisasi atas lima relasi; edge itu masuk ke node `UP`, yang menerima self-loop dan edge dari `MAIN`.
4. **Self-loop.** Bab II boleh menjelaskan fungsi self-loop secara umum. Bab III harus menyatakan bahwa self-loop tidak tercantum dalam `STAR_EDGES`; ia muncul dari default `GATConv` (`add_self_loops=True`), bukan dari desain edge eksplisit proyek.
5. **Pooling.** Jelaskan `global_mean_pool` sebagai pemetaan node-level menjadi graph-level embedding.
6. **Dimensi waktu.** Tulis enam snapshot graph historis `t-6,...,t-1`, bukan `t-5,...,t`, ketika `t` berarti waktu target.
7. **Temporal attention.** Jelaskan bahwa implementasi memakai `nn.MultiheadAttention`, positional embedding trainable, causal upper-triangular mask, residual plus `LayerNorm`, lalu mengambil representasi timestep terakhir. Ini bukan satu scalar weight per jam dan bukan LSTM.
8. **Urutan spatio-temporal.** Tegaskan bahwa source menjalankan spatial GAT pada setiap snapshot secara berurutan, melakukan pooling pada setiap snapshot, men-stack hasilnya, lalu menjalankan temporal attention. Spatial GAT dan temporal attention bukan operasi simultan pada tensor yang sama.
9. **Interface conditioning.** Jelaskan bahwa context `MAIN` pada `t-1`, retrieval, dan graph masing-masing diproyeksikan ke `[B,128]`, lalu dijumlahkan element-wise. Time embedding kemudian dijumlahkan ke condition sebelum masuk ke denoiser.
10. **Batas target.** Graph memproses semua node pada enam waktu, tetapi diffusion hanya men-noise dan memprediksi target normalized `MAIN` berukuran `[B,3]`.
11. **Normalisasi.** Statistik non-degenerate berasal dari `MAIN` pada training split. Hanya fitur dengan statistik degenerate yang memakai fallback statistik all-node; denominator tetap memakai epsilon `1e-5`.
12. **Retrieval dan wet gate.** Database training dibangun dari `MAIN` pada training split dengan key fitur `[9]` dan value target langkah berikutnya `[3]`. Training memakai strict-past dan exclude-self; validation memakai database training dengan `strict_past=False`. Wet/dry gate hanya aktif bila checkpoint/config mengaktifkannya.
13. **Novelty yang proporsional.** GAT, Transformer attention, diffusion, FAISS, AdamW, GELU, SiLU, dropout, dan DDIM adalah komponen berbasis literatur/standar. Kontribusi proyek adalah integrasi retrieval + STGNN + conditional diffusion untuk skenario stasiun lokal, yang tetap harus dibuktikan melalui eksperimen ablation dan baseline.

### 1.3 Flow aktual yang perlu digambar di Bab II dan dijelaskan rinci di Bab III

```text
features [T, 5, 9]
  -> normalisasi training; precipitation target memakai log1p
  -> enam graph: t-6,...,t-1
  -> untuk setiap snapshot: GATConv 9 -> 4 x 64, concat -> 256
  -> ReLU -> GATConv 256 -> 64
  -> untuk setiap snapshot: global_mean_pool atas 5 node
  -> stack enam pooled output -> [B, 6, 64]
  -> setelah seluruh snapshot diproses: MultiheadAttention 4 head + causal mask + trainable positional embedding
  -> ambil timestep terakhir -> graph_emb [B,64]
  -> graph_mlp -> [B,128]

context MAIN pada t-1 [B,9] -> cond_mlp -> [B,128]
FAISS training database: MAIN key [9] pada τ -> MAIN next-step target [3] pada τ+1, K=3
  -> retrieval [B,3,3] -> flatten [B,9] -> retrieval_mlp -> [B,128]

a = context_emb + retrieval_emb + graph_emb  # masing-masing [B,128]
t_emb = time_mlp(timestep) [B,128]
condition = a + t_emb
noise epsilon [B,3], timestep [B]
target normalized MAIN pada t [B,3] -> add_noise -> noisy_target [B,3]
noisy_target + condition
  -> denoiser -> predicted noise [B,3]
  -> weighted noise MSE; optional wet/dry BCE
```

Rujukan source: `core/src/models/gnn.py:14-161`, `core/src/models/diffusion.py:8-159`, `core/src/data/temporal_loader.py:161-209`, `core/src/train.py:428-565`.

---

## 2. Aturan penulisan rumus Bab II versus Bab III

### 2.1 Bab II: rumus teori dan definisi umum

Bab II menjawab **apa konsepnya dan bagaimana formulasi umum bekerja**. Rumus yang ditulis harus:

- memakai notasi abstrak yang mudah dibaca;
- menyebut sumber primer tepat di dekat rumus;
- tidak memasukkan path file, batch shape, nilai default, atau detail helper;
- tidak menyatakan hyperparameter proyek sebagai konsekuensi teori;
- membedakan rumus standar dari hipotesis alasan desain.

Rumus Bab II yang layak: GAT attention, multi-head aggregation, mean pooling, scaled dot-product attention, forward diffusion, objective noise prediction, L2 distance, GELU/SiLU, dropout, dan AdamW secara ringkas bila memang dipakai sebagai landasan.

### 2.2 Bab III: rumus implementasi dan instansiasi proyek

Bab III menjawab **bagaimana konsep tersebut diwujudkan dalam sistem ini**. Setiap rumus harus disertai:

- path source dan approximate line range;
- definisi variabel sesuai nama kode;
- dimensi tensor;
- nilai konfigurasi yang benar-benar ada di kode;
- batasan atau perilaku dependency yang relevan;
- penanda apakah operasi itu standar, konfigurasi, adaptasi, atau integrasi.

Contoh: Bab II menulis `α_ij` pada GAT secara umum. Bab III menulis bahwa `GATConv` pertama menerima 9 fitur, `heads=4`, `concat=True`, `edge_dim=1`; layer kedua menerima 256 dan menghasilkan 64. Bab III tidak boleh menulis alasan “empat head pasti paling stabil” tanpa eksperimen.

### 2.3 Larangan umum

- Jangan mengulang rumus teori panjang di Bab III tanpa mengaitkannya ke tensor kode.
- Jangan memasukkan semua operasi `Linear` sebagai rumus baru; cukup tulis transformasi dan dimensi.
- Jangan menyebut `IndexFlatL2` sebagai jarak Euclidean biasa tanpa kata **kuadrat**.
- Jangan menulis PCA, reduksi dimensi retrieval, encoder retrieval, atau metric learning; semuanya tidak ada.
- Jangan menyebut temporal attention sebagai “bobot satu scalar untuk tiap jam”.
- Jangan menyebut diffusion men-noise graph atau seluruh fitur; yang diberi noise hanya target `MAIN` `[B,3]`.
- Jangan menyebut DDIM sepenuhnya deterministik tanpa syarat: sampling dimulai dari `torch.randn`; reproducibility memerlukan seed dan kondisi eksekusi yang sama.
- Jangan menyatakan beta schedule linear/start/end sebagai konfigurasi proyek; kode hanya mengunci `num_train_timesteps=1000` dan `clip_sample=False`, sedangkan detail lain berasal dari default dependency versi tertentu.
- Jangan menyatakan semua flow retrieval/evaluation sudah terverifikasi dari `train.py`; test/evaluation retrieval flow yang tidak tampak di `core/src/train.py` harus dicari dan diverifikasi dari source/artifact terkait, bukan diasumsikan.
- Jangan menyatakan semua sitasi modern sudah terverifikasi; metadata final daftar pustaka tetap perlu dicek pada publisher/arXiv.

---

## 3. Klasifikasi lengkap komponen dan status rumus

| Komponen | Klasifikasi | Fakta proyek / lokasi | Status rumus | Cara tulis |
|---|---|---|---|---|
| Graph berarah | Konfigurasi proyek | 5 node; 8 edge eksplisit star dua arah; `core/src/config.py:74-84,161-177` | Mention + definisi `G=(V,E)` | Tulis tabel node/edge dan gambar topology; rumus sederhana cukup |
| Fitur node | Konfigurasi proyek | 9 fitur; `core/src/config.py:97-107` | Mention | Daftar fitur dan satuan bila tersedia |
| Target | Konfigurasi proyek | `precipitation`, `wind_speed_10m`, `relative_humidity_2m`; hanya `MAIN`; `[B,3]` | Mention + shape | Tegaskan target bukan lima node |
| Sliding window | Konfigurasi proyek | graph `t-6...t-1`, context `t-1`, target `t`; `core/src/data/temporal_loader.py:149,193-205` | Rumus indexing opsional, wajib di Bab III | Tulis persamaan indeks dan diagram waktu |
| Normalisasi z-score | Standar/adopsi + konfigurasi | `(x-mean)/(std+1e-5)`; statistik non-degenerate dari `MAIN` training; fitur degenerate saja fallback all-node; `temporal_loader.py:161-174` | Rumus wajib di Bab III; teori singkat | Nyatakan epsilon `1e-5`; bedakan stats training, fallback degenerate, dan klaim generalisasi |
| `log1p` precipitation | Adaptasi | target hujan ditransformasi sebelum normalisasi; `temporal_loader.py:162-164` | Rumus singkat | Jelaskan inverse `expm1` bila dibahas |
| GAT message passing | Standar/adopsi; edge feature = implementasi PyG | GAT standar memakai fitur node; PyG `GATConv(edge_dim=1)` menerima edge_attr; `core/src/models/gnn.py:64-96` | Rumus inti wajib Bab II; formula edge feature hanya Bab III bila API didokumentasikan | Sitasi Veličković et al. (2018) hanya untuk GAT standar; jangan atribusikan edge-feature formula ke paper asli |
| GAT layer 1 | Konfigurasi proyek | `9 -> 64`, 4 heads, concat -> 256, `edge_dim=1` | Mention dimensi; bukan rumus baru | Jangan klaim optimalitas 4 head |
| ReLU | Standar/adopsi | Setelah GAT1; `gnn.py:76,85-87` | Rumus optional | `ReLU(x)=max(0,x)` cukup sekali |
| GAT layer 2 | Konfigurasi proyek | `256 -> 64`, 1 head, `concat=False`; `gnn.py:73-90` | Mention | Jelaskan output node-level |
| Self-loop GAT | Perilaku dependency | default `GATConv`, bukan `STAR_EDGES`; `gnn.py:73-74` | Mention + caveat | Jangan mengklaim dirancang manual |
| Edge attribute | Konfigurasi proyek; bukan teori GAT standar | jarak lat/lon `[E,1]`; canonical grid menghasilkan tepat `0.25` pada setiap edge; `config.py:180-205` | Mention; formula jarak optional, formula edge-feature bukan sitasi GAT asli | Tegaskan `0.25` konstan sehingga bukan sinyal jarak informatif; sinyal utama berasal dari topology dan node features |
| Global mean pooling | Standar/adopsi | `global_mean_pool`; `gnn.py:92-95` | Rumus singkat wajib bila menjelaskan graph embedding | `h_graph = 1/|V| Σ h_i` |
| Spatial processing per waktu | Integrasi | GAT dijalankan pada enam graph; `gnn.py:143-155` | Flow/mention | Tidak perlu formula baru |
| Temporal `MultiheadAttention` | Standar/adopsi | PyTorch `nn.MultiheadAttention`, 4 heads; `gnn.py:14-61` | Rumus attention wajib Bab II; detail mask Bab III | Sitasi Vaswani et al. (2017) |
| Positional embedding | Konfigurasi proyek | trainable, normal init; `gnn.py:31-32,45-46` | Mention | Jangan menyebut sinusoidal |
| Causal mask | Konfigurasi proyek | upper triangular bool diagonal 1; `gnn.py:48-55` | Mention + notasi mask opsional | Jelaskan tidak melihat future timestep |
| Last timestep selection | Integrasi | `x[:, -1, :]`; `gnn.py:60-61` | Mention | Tegaskan bukan weighted sum semua jam |
| Graph output projection | Konfigurasi proyek | `Linear(64,64)`; `gnn.py:133-161` | Mention | Tidak perlu rumus mandiri |
| Context MLP | Konfigurasi proyek | 9 -> 128 -> 128; `diffusion.py:71-76` | Mention | Jangan menyebut “bias” untuk `cond_emb` |
| Retrieval MLP | Konfigurasi proyek | retrieval 3x3 flatten -> 9 -> 128 -> 128; `diffusion.py:64-69` | Mention | Tidak ada retrieval encoder neural |
| Graph MLP | Konfigurasi proyek | 64 -> 128 -> 128; `diffusion.py:78-83` | Mention | Jelaskan penyamaan dimensi |
| Additive conditioning | Integrasi | `cond_emb = cond_emb + r_emb + g_emb`; `diffusion.py:100-120` | Rumus inti wajib | Bukan concatenation global |
| FAISS index | Standar/adopsi | `faiss.IndexFlatL2(9)` exact search; `retrieval/base.py:5-73` | Rumus squared L2 wajib | Tulis `d^2`, bukan `d` Euclidean |
| Retrieval key/value | Konfigurasi proyek | database hanya dari training split; key normalized MAIN `[9]` pada `τ`; value normalized MAIN next-step target `[3]` pada `τ+1`; `train.py:428-445` | Rumus indexing/mention | Jelaskan `K=3`; bukan seluruh data historis; tidak ada PCA |
| Strict-past filter | Adaptasi | training `j<t-1`, `strict_past=True`, `exclude_self=True`; validation memakai training database dengan `strict_past=False`; `train.py:449-470` | Pseudocode/mention wajib | Jelaskan fallback zero/repeat candidate secara hati-hati; jangan generalisasi strict-past ke semua split |
| Retrieval fallback | Integrasi/implementasi | helper mencari kandidat terbatas hingga `max(K,8K)`; fallback dapat berupa zero/repeat candidate sesuai helper aktif; `ROADMAP_RDM_STGNN_HANDOVER.md:360-369` | Caveat, tanpa rumus baru | Nyatakan fallback tidak membuktikan tidak ada neighbor valid; test/evaluation flow yang tidak tampak di `train.py` perlu diverifikasi |
| Sinusoidal timestep embedding | Standar/adopsi | `diffusion.py:8-20` | Rumus inti wajib | Bedakan dari Gaussian noise |
| Time MLP | Konfigurasi proyek | 128 -> 256 -> 128, GELU; `diffusion.py:57-62` | Mention | Jangan memberi rationale unsupported |
| DDPM noising | Standar/adopsi | scheduler `num_train_timesteps=1000`; `train.py:552-554`, `diffusion.py:170-175` | Rumus inti wajib Bab II | Beta detail tidak boleh di-overclaim |
| Noised target | Konfigurasi proyek | hanya normalized target `MAIN` `[B,3]`; `train.py:552-554` | Shape + mention wajib | Bukan seluruh graph |
| Denoiser MLP | Konfigurasi proyek | 3 -> 128 -> 256 -> 256; `diffusion.py:91-98,141-155` | Persamaan forward ringkas | `h_up=concat(h_mid,h_2)`, bukan residual addition |
| Timestep/condition injection | Integrasi | `emb=t_emb+cond_emb`, lalu `h1=down1(x)+emb`; `diffusion.py:141-148` | Rumus wajib Bab III | Jelaskan additive injection |
| Weighted noise MSE | Adaptasi | bobot 1/5/10 berdasarkan `abs(target_norm)`; `diffusion.py:176-186` | Rumus wajib jika loss dibahas | Ini loss proyek, bukan DDPM standar murni |
| Wet/dry head | Adaptasi | auxiliary BCE dari condition embedding; `diffusion.py:84-89,122-127`, `train.py:568-579` | Rumus BCE optional/wajib bila aktif | Bukan target diffusion keempat |
| Conditioning dropout | Adaptasi | graph/retrieval dapat dinolkan terpisah; `train.py:542-551` | Mention | Klaim causal ablation perlu eksperimen |
| AdamW | Standar/adopsi | optimizer `torch.optim.AdamW`; `train.py:497-498` | Mention atau rumus update singkat | Sitasi Loshchilov & Hutter (2019) |
| DDIM sampling | Standar/adopsi | `sample_fast`, default aktif 50 langkah; `diffusion.py:233-270` | Rumus reverse umum optional | Initial state random; clamp dan `nan_to_num` |
| Dropout | Standar/adopsi | GAT/attention dropout; `gnn.py:23-27,73-74` | Mention | Sitasi Srivastava et al. (2014) bila dibahas |
| GELU | Standar/adopsi | timestep MLP; `diffusion.py:60` | Rumus optional | Sitasi Hendrycks & Gimpel (2016) |
| SiLU | Standar/adopsi | retrieval/context/graph/denoiser; `diffusion.py:67,74,81,92-97` | Rumus optional | Sitasi Elfwing et al. (2017) |
| Validation/performance | Project-specific | metric/artifact eksperimen | Tidak ada rumus teori pengganti hasil | Hanya klaim dari tabel eksperimen yang reproducible |

---

## 4. Rumus inti wajib dengan notasi sesuai kode

### 4.1 Notasi data dan waktu

Gunakan:

- `B`: batch size;
- `T_s=6`: panjang sequence;
- `N=5`: jumlah node;
- `F=9`: fitur node/context;
- `D=3`: target;
- `t`: waktu target;
- `X_{t-k}`: fitur graph pada waktu historis;
- `y_t∈R^3`: target normalized node `MAIN` pada waktu `t`;
- `c_{t-1}∈R^9`: context normalized `MAIN` pada `t-1`.

Window aktual:

\[
\mathcal{X}_t=(X_{t-6},X_{t-5},X_{t-4},X_{t-3},X_{t-2},X_{t-1}),\quad
c_t=X_{t-1}[MAIN,:],\quad y_t=Y_t[MAIN,:].
\]

Normalisasi:

\[
z=(x-\mu)/(\sigma+10^{-5}).
\]

Secara default, statistik non-degenerate dihitung dari node `MAIN` pada training split. Bila fitur tertentu memiliki statistik degenerate, fitur tersebut saja memakai fallback statistik all-node; fallback tidak berlaku otomatis untuk semua fitur. Epsilon numerik yang dipakai adalah `1e-5`. Detail harus dicocokkan dengan pembentukan `stats` di `core/src/train.py` dan penerapannya di `core/src/data/temporal_loader.py:161-174`.

Untuk precipitation sebelum normalisasi:

\[
x'_{rain}=\log(1+x_{rain}).
\]

Implementasi: `core/src/data/temporal_loader.py:161-174,193-205`.

### 4.2 GAT

#### 4.2.1 GAT standar menurut Veličković et al. (2018)

Untuk node tujuan `i`, GAT standar memproyeksikan fitur node dan menghitung skor attention dari pasangan fitur node:

\[
\mathbf{h}'_i=\mathbf{W}\mathbf{h}_i,
\]

\[
e_{ij}=\operatorname{LeakyReLU}\left(\mathbf{a}^{\mathsf T}
[\mathbf{W}\mathbf{h}_i\,\Vert\,\mathbf{W}\mathbf{h}_j]\right),
\]

\[
\alpha_{ij}=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal{N}(i)}\exp(e_{ik})},
\qquad
\mathbf{h}'_i=\sigma\left(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}\mathbf{W}\mathbf{h}_j\right).
\]

`N(i)` adalah sumber yang mengirim pesan ke node tujuan `i`, termasuk self-loop bila dependency menambahkannya. Untuk head `m`:

\[
\mathbf{h}'_i=\mathbin{\Vert}_{m=1}^{H}
\sigma\left(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}^{(m)}\mathbf{W}^{(m)}\mathbf{h}_j\right).
\]

Rumus ini adalah GAT standar dari Veličković et al. (2018), arXiv:1710.10903. Paper tersebut tidak menjadi atribusi untuk formula yang memasukkan edge feature.

#### 4.2.2 Edge feature pada implementasi PyG

Source memakai `GATConv(..., edge_dim=1)`. Secara implementasi, PyG menerima `edge_attr` dan memasukkannya ke mekanisme attention sesuai API `GATConv`; detail ini adalah perilaku library/dependency, bukan formula GAT asli. Pada proyek, `edge_attr` dibentuk sebagai jarak Euclidean pada koordinat lat/lon di `core/src/config.py:180-205`. Canonical grid menghasilkan nilai tepat `0.25` untuk setiap edge, sehingga edge attribute tersebut konstan dan **bukan sinyal jarak informatif**. Sinyal spasial utama berasal dari topology `edge_index` dan fitur node. Formula API edge feature hanya boleh ditulis di Bab III setelah versi PyG/API yang dipakai dikunci.

Pada kode: GAT pertama menerima `F=9`, `hidden_dim=64`, `H=4`, `concat=True`, sehingga output 256; GAT kedua menerima 256 dan menghasilkan 64 dengan satu head dan `concat=False`. Rujukan implementasi: `core/src/models/gnn.py:69-90`. Karena graph berarah, `MAIN→UP` dan `UP→MAIN` dapat memiliki koefisien berbeda.

### 4.3 Global mean pooling

Untuk graph `b` dengan node `V_b`:

\[
\mathbf{g}_b=\frac{1}{|V_b|}\sum_{i\in V_b}\mathbf{h}_{b,i}.
\]

Kode: `global_mean_pool` pada `core/src/models/gnn.py:92-95`. Dengan lima node, setiap snapshot menjadi satu vektor 64 dimensi.

### 4.4 Temporal multi-head attention

Untuk sequence graph embedding `H∈R^{B×6×64}` dan positional embedding trainable `P`:

\[
\widetilde{H}=H+P.
\]

Untuk satu head Transformer:

\[
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M\right)V,
\]

\[
\operatorname{MHA}(H)=\mathbin{\Vert}_{m=1}^{M}
\operatorname{Attn}(\widetilde{H}W_Q^{(m)},
\widetilde{H}W_K^{(m)},
\widetilde{H}W_V^{(m)})W_O.
\]

Mask causal `M` memberi nilai `-∞` pada posisi future (`j>i`). Kode kemudian memakai residual, dropout, dan LayerNorm:

\[
H_{out}=\operatorname{LayerNorm}(\widetilde{H}+\operatorname{Dropout}(\operatorname{MHA}(\widetilde{H}))),
\qquad h_{graph}=H_{out}[:, -1,:].
\]

Implementasi: `core/src/models/gnn.py:14-61,126-161`. Rujukan teori: Vaswani et al. (2017), arXiv:1706.03762.

### 4.5 Retrieval squared L2

Query `q∈R^9` dibandingkan dengan key historis `k_j∈R^9` menggunakan `IndexFlatL2`:

\[
d^2(q,k_j)=\sum_{f=1}^{9}(q_f-k_{j,f})^2.
\]

`IndexFlatL2` mengembalikan **squared L2 distance** dan melakukan exact flat search. Database yang tampak pada source training dibangun dari **training split saja**, bukan seluruh data historis proyek. Key/value proyek:

\[
k_\tau=\operatorname{norm}(x_{\tau}^{MAIN})\in\mathbb{R}^{9},
\qquad
v_\tau=\operatorname{norm}(y_{\tau+1}^{MAIN})\in\mathbb{R}^{3}.
\]

Untuk training, filter mempertahankan `j<t-1` dan `exclude_self=True`; validation memakai database training yang sama dengan `strict_past=False`. `K=3` value dikembalikan sebagai `[B,3,3]` dan di-flatten menjadi `[B,9]`. Helper dapat memakai fallback zero atau repeat candidate sesuai jalur aktif; fallback terbatas tidak membuktikan tidak ada neighbor valid yang lebih jauh. Tidak ada PCA, reduced retrieval space, learned embedding, atau metric learning. Lokasi: `core/src/retrieval/base.py:5-73`, `core/src/train.py:428-470`.

Test/evaluation retrieval flow yang tidak tampak di `core/src/train.py` harus diverifikasi dari source atau artifact terkait. Dokumen ini tidak mengasumsikan flow test tersebut. Lokasi training yang terlihat: `core/src/train.py:428-470`.

### 4.6 Additive conditioning

Context `MAIN` pada `t-1`, retrieval, dan graph masing-masing diproyeksikan ke `[B,128]`:

\[
 c_{emb}=f_c(c_{t-1}),\qquad
 r_{emb}=f_r(\operatorname{flatten}(r)),\qquad
 g_{emb}=f_g(g),
\]

\[
 c=c_{emb}+r_{emb}+g_{emb}.
\]

Dengan `c_{t-1}∈R^9`, retrieval `r∈R^{3×3}` yang di-flatten menjadi `R^9`, dan graph embedding `g∈R^{64}`, ketiga output tersebut masing-masing berbentuk `[B,128]`. Time embedding juga menghasilkan `[B,128]`, lalu dijumlahkan ke condition:

\[
 e=t_{emb}+c,
\qquad
 \hat\epsilon=\epsilon_\theta(x_s,t,c_{t-1},r,g),
\]

\[
 h_1=\operatorname{SiLU}(W_1x_s+b_1)+e.
\]

Jadi rumus noise prediction harus memuat context secara eksplisit, meskipun context telah diringkas ke `c`. Kode: `core/src/models/diffusion.py:43-159`; conditioning builder `:100-120`, injection `:141-155`. `c` adalah conditioning embedding, bukan “bias” khusus.

### 4.7 Sinusoidal timestep embedding

Untuk dimensi embedding `d=128`, `i=0,...,d/2-1`:

\[
\omega_i=\exp\left(-\frac{\log(10000)i}{d/2-1}\right),
\]

\[
\operatorname{emb}(τ)=
[\sin(τω_0),...,\sin(τω_{d/2-1}),
\cos(τω_0),...,\cos(τω_{d/2-1})].
\]

Ini adalah embedding untuk diffusion timestep, bukan noise Gaussian. Kode: `core/src/models/diffusion.py:8-20`.

### 4.8 Forward diffusion dan loss

Secara teori DDPM:

\[
q(x_s|x_{s-1})=\mathcal{N}(x_s;\sqrt{1-\beta_s}x_{s-1},\beta_s I),
\]

atau bentuk closed form:

\[
x_s=\sqrt{\bar\alpha_s}x_0+\sqrt{1-\bar\alpha_s}\epsilon,
\qquad \epsilon\sim\mathcal{N}(0,I).
\]

Pada training proyek, `x_0=targets` sudah berupa target normalized `MAIN` `[B,3]`:

```python
noise = torch.randn_like(targets)
timesteps = torch.randint(0, 1000, (targets.shape[0],))
noisy_target = scheduler.add_noise(targets, noise, timesteps)
```

Lokasi: `core/src/train.py:552-554`. Scheduler: `DDPMScheduler(num_train_timesteps=1000, clip_sample=False)` di `core/src/models/diffusion.py:170-175`. Jangan menetapkan angka beta lain sebagai fakta proyek tanpa mengunci versi `diffusers`.

Denoiser menerima `x_s` yang berasal hanya dari target normalized `MAIN` `[B,3]`, bukan dari graph atau seluruh fitur. Denoiser aktual memiliki jalur `3 -> 128 -> 256 -> 256`, menggabungkan `h_mid` dan `h_2` melalui concatenation menjadi 512, lalu `512 -> 128 -> 3`; `h_1` menerima additive injection `e=t_emb+c`.

\[
\hat\epsilon=\epsilon_\theta(x_s,s,c_{t-1},r,g),
\qquad
L_{noise}=\frac{1}{BD}\sum_{b=1}^{B}\sum_{d=1}^{D}
 w_{bd}(\hat\epsilon_{bd}-\epsilon_{bd})^2.
\]

Bobot aktif:

\[
w_{bd}=\begin{cases}
10,&|y_{bd}|>3,\\
5,&1<|y_{bd}|\le3,\\
1,&|y_{bd}|\le1.
\end{cases}
\]

Kode: `core/src/models/diffusion.py:176-186`. Ini adaptasi loss proyek, bukan objective DDPM standar tanpa modifikasi.

### 4.9 Wet/dry auxiliary loss

Jika checkpoint/config mengaktifkan rain specialization, wet head menghasilkan satu logit dari conditioning embedding:

\[
L=L_{noise}+0.7\operatorname{BCEWithLogitsLoss}(\ell_{wet},y_{wet}),
\]

\[
y_{wet}=\mathbf{1}[precipitation_{mm}\ge 0.1].
\]

Threshold `0.1` dan weight `0.7` hanya berlaku ketika gate/checkpoint/config terkait aktif; jangan menulis wet/dry loss sebagai jalur unconditional. Wet head bukan target diffusion keempat. Lokasi model: `core/src/models/diffusion.py:84-89,122-127`; training dan gate: `core/src/train.py:393-425,568-579`.

### 4.10 DDIM sampling

Jalur inferensi cepat memakai `DDIMScheduler`; helper `sample_fast` memiliki default `num_inference_steps=50` di `core/src/models/diffusion.py:233-270`. Namun, caller aktif di `core/src/inference.py` tampak dapat meneruskan `20` langkah (`core/src/inference.py:262-269`, bila source aktif pada checkout ini masih demikian). Bab III harus mengunci dan melaporkan angka yang benar-benar dipakai caller, bukan hanya default helper; perbedaan 50 versus 20 wajib dicek sebelum klaim latency atau reproduksibilitas.

Sampling dimulai dari `x∼N(0,I)` pada shape `[M,3]`, menjalankan denoising DDIM, clamp predicted noise ke `[-10,10]`, lalu `nan_to_num`. DDIM dapat disebut deterministic **bersyarat pada initial noise, seed, dan konfigurasi eksekusi yang sama**, bukan deterministic secara absolut.

---

## 5. Rekomendasi paragraf per section Bab II

### 2.1 Data cuaca dan representasi spasial

Mulai dari masalah prakiraan satu jam berikutnya pada lima lokasi. Definisikan node sebagai stasiun/lokasi, fitur sebagai sembilan variabel input, dan target sebagai tiga variabel pada `MAIN`. Jelaskan bahwa graph adalah representasi relasi antar lokasi, bukan bukti bahwa atmosfer hanya memiliki delapan hubungan fisik.

### 2.2.1 Graph dan directed star topology

Definisikan `G=(V,E)`. Tampilkan lima node dan delapan edge eksplisit. Jelaskan perbedaan edge berarah `i→j` dan `j→i`. Hindari klaim bahwa topologi star adalah konfigurasi universal; ini adalah abstraksi proyek yang perlu dievaluasi melalui eksperimen.

### 2.2.2 Graph Attention Network

Perkenalkan message passing dan attention coefficient berdasarkan Veličković et al. (2018). Tulis tiga rumus inti: transformasi, softmax attention, dan agregasi multi-head. Tutup dengan kalimat bahwa Bab III menginstansiasikan GAT dua layer melalui PyG `GATConv`; Bab II tidak perlu memuat nama kelas.

### 2.2.3 Temporal attention

Jelaskan mengapa embedding graph dari beberapa snapshot membentuk sequence. Gunakan formulasi scaled dot-product attention dari Vaswani et al. (2017). Jelaskan causal mask sebagai pembatas akses future timestep. Jangan menggambarkan mekanisme ini sebagai satu bobot scalar per jam; output aktual adalah representasi sequence yang kemudian diambil pada posisi terakhir.

### 2.2.4 Pooling dan graph-level conditioning

Jelaskan kebutuhan mengubah representasi node menjadi satu embedding graph. Tulis mean pooling. Hubungkan embedding graph dengan conditional model, tetapi simpan dimensi 64/128 dan path source untuk Bab III.

### 2.2.5 Spatio-temporal graph conditioning

Gabungkan spatial GAT per snapshot dengan temporal attention antar-snapshot. Nyatakan bahwa “spatio-temporal” di sini adalah komposisi dua tahap: spatial graph processing lalu temporal sequence attention. Jangan menyatakan bahwa kode memakai satu operator STGNN end-to-end yang berbeda dari dua tahap tersebut.

### 2.2.6 Retrieval analog

Perkenalkan nearest-neighbor retrieval sebagai sumber analog historis. Tulis squared L2 dan jelaskan key/value time shift. Nyatakan bahwa implementasi memakai exact `IndexFlatL2`, bukan learned retrieval encoder. Rujukan analog ensemble dapat memakai Delle Monache et al. (2013), tetapi jangan mengklaim performa proyek mengikuti hasil paper tersebut.

### 2.2.7 Conditional diffusion

Jelaskan forward noising, reverse denoising, timestep embedding, dan noise prediction berdasarkan Ho et al. (2020). Untuk diffusion generatif cuaca, gunakan Asperti et al. (2024, DOI: 10.1007/s10489-024-06048-y) secara hati-hati sebagai related work, bukan bukti bahwa arsitektur proyek identik.

### 2.2.8 DDIM dan probabilistic output

Jelaskan DDIM dari Song et al. (2021), arXiv:2010.02502, sebagai alternatif sampling. Bedakan teori DDIM dari konfigurasi `sample_fast` di Bab III. Tekankan bahwa random initial state membuat output probabilistik.

### 2.2.9 Aktivasi, dropout, dan optimizer

Tulis singkat: GELU (Hendrycks & Gimpel, 2016), SiLU (Elfwing et al., 2017), dropout (Srivastava et al., 2014), dan AdamW (Loshchilov & Hutter, 2019). Jangan memberikan rationale proyek yang tidak diuji, misalnya “SiLU mencegah NaN” atau “GELU pasti cocok karena noise Gaussian”.

### 2.2.10 Related work dan posisi penelitian

Gunakan related work untuk membedakan domain dan skala, bukan untuk meminjam klaim performa. Ravuri et al. (2021, Nature, DOI: 10.1038/s41586-021-03854-z) dan Zhang et al. (2023, NowcastNet, DOI: 10.1038/s41586-023-06184-4) terutama relevan sebagai pembanding generative/radar nowcasting berskala berbeda. Liu et al. (2024), *Retrieval-Augmented Diffusion Models for Time Series Forecasting*, arXiv:2410.18712, relevan sebagai pembanding retrieval-augmented diffusion time series, bukan bukti bahwa implementasi lokal sama.

### 2.2.11 Sintesis gap dan kontribusi

Tutup Bab II dengan gap yang terukur: belum diketahui apakah kombinasi graph conditioning, retrieval, dan diffusion membantu pada data lima node lokal. Nyatakan kontribusi sebagai integrasi/application-driven. Hindari “novel formula” atau klaim superioritas sebelum ablation, baseline, confidence evaluation, dan leakage audit selesai.

---

## 6. Rekomendasi struktur paragraf Bab III

### 3.1 Data dan kontrak tensor

Paragraf pertama mendefinisikan lima node, sembilan fitur, tiga target. Paragraf kedua menjelaskan urutan node dan edge. Paragraf ketiga memberi shape `[T,5,9]`, target `[T,5,3]`, lalu window aktual.

### 3.2 Preprocessing

Jelaskan bahwa statistik non-degenerate berasal dari `MAIN` pada training split. Fitur degenerate saja memakai fallback all-node; jangan menulis fallback all-node sebagai aturan seluruh fitur. Precipitation memakai `log1p`, lalu z-score dengan epsilon `1e-5`. Nyatakan transformasi dilakukan sebelum graph diproses. Jangan menyatakan standardisasi otomatis menjaga distribusi spasial tetap setara; fallback std kecil hanya mitigasi numerik.

### 3.3 STGNN

Uraikan alur enam graph secara berurutan: untuk setiap snapshot, GAT1, ReLU, GAT2, lalu pooling; setelah keenam output di-stack, temporal `MultiheadAttention` causal dijalankan dan representasi timestep terakhir diambil. Jangan menulis spatial GAT dan temporal attention sebagai operasi simultan. Cantumkan dimensi dan path `core/src/models/gnn.py:14-161` serta konfigurasi pemanggilan `core/src/train.py:479-486`. Pisahkan rumus GAT standar dari detail PyG `edge_dim=1`; canonical `edge_attr=0.25` konstan dan bukan sinyal jarak informatif.

### 3.4 Retrieval

Uraikan bahwa database dibangun dari training split `MAIN` saja: key fitur normalized `[9]` pada `τ`, value normalized next-step target `[3]` pada `τ+1`, exact `IndexFlatL2`, dan `K=3`. Tulis squared L2, flatten, strict-past + exclude-self pada training, validation memakai training database dengan `strict_past=False`, serta fallback zero/repeat candidate secara hati-hati. Tegaskan tidak ada PCA. Test/evaluation retrieval flow yang tidak tampak di `core/src/train.py` harus diverifikasi, bukan diasumsikan. Cantumkan `core/src/retrieval/base.py:5-73` dan `core/src/train.py:428-470`.

### 3.5 Conditional diffusion

Uraikan context MAIN pada `t-1`, retrieval, dan graph yang masing-masing diproyeksikan ke `[B,128]`, dijumlah element-wise, lalu dijumlah lagi dengan time embedding `[B,128]`. Detail denoiser wajib: input target normalized MAIN `[B,3]` saja yang diberi noise; jalur `3 -> 128 -> 256 -> 256`, concatenation skip `h_mid || h2`, lalu output `[B,3]` predicted noise. Cantumkan `core/src/models/diffusion.py:43-159` dan `core/src/train.py:552-565`.

### 3.6 Objective dan training

Tulis weighted noise MSE. Tambahkan optional `BCEWithLogitsLoss` hanya jika checkpoint/config mengaktifkan rain gate, dengan threshold `0.1` dan weight `0.7` sesuai source aktif. Sertakan conditioning dropout, joint backward, dan AdamW. Pisahkan formula loss dari alasan pemilihan bobot; bobot harus disebut sebagai konfigurasi adaptasi, bukan optimum teoretis.

### 3.7 Sampling dan evaluasi

Bedakan reverse DDPM penuh dari helper DDIM cepat. Helper `sample_fast` default 50 langkah, tetapi caller `core/src/inference.py` harus dicek karena dapat aktif memakai 20. Kunci angka caller pada laporan final. Laporkan preprocessing inverse, seed, jumlah sample, dan metrik sebagai prosedur eksperimen. Klaim peningkatan hanya boleh muncul dari tabel hasil dan confidence interval/ulang eksperimen yang tersedia.

### 3.8 Batasan reproduksibilitas

Sebutkan dependency default scheduler, hardcoded head count pada inference bila relevan, random initial noise, metadata checkpoint, dan aturan retrieval validation. Bagian ini meningkatkan kejujuran metodologis dan mencegah pembaca menganggap seluruh detail sudah dikunci.

---

## 7. Temuan draft V4 yang wajib dikoreksi

| Temuan pada draft V4 | Koreksi wajib | Dasar pengecekan |
|---|---|---|
| Retrieval space disebut direduksi/PCA | Hapus klaim PCA, reduced retrieval space, atau embedding retrieval terlatih. Key tetap normalized MAIN `[9]`. | `core/src/train.py:428-445`; `core/src/retrieval/base.py:5-73` |
| `IndexFlatL2` ditulis sebagai sqrt/L2 biasa | Tulis squared L2: `d²(q,k)=Σ(q_f-k_f)²`. Jangan menambahkan akar kuadrat. | FAISS `IndexFlatL2`; `core/src/retrieval/base.py:12,50` |
| Window memakai indeks yang ambigu/salah | Dengan `t` sebagai target, graph input adalah `t-6,...,t-1`, context `t-1`, target `t`; jangan menulis `t-5,...,t`. | `core/src/data/temporal_loader.py:149,193-205` |
| `[27]` dipakai untuk mendukung diffusion | Jangan gunakan `[27]` draft untuk diffusion bila daftar pustaka `[27]` adalah paper IoT outdoor safety. Gunakan Ho et al. (2020) arXiv:2006.11239 untuk DDPM, Song et al. (2021) arXiv:2010.02502 untuk DDIM, dan Asperti sebagai related work. | `docs/DRAFT_PRASKRIPSI_BEVAN_V4.md:1416-1441`; daftar pustaka draft perlu dicek |
| Performance retrieval/diffusion di-overclaim | Sitasi paper tidak membuktikan peningkatan pada dataset lokal. Laporkan hanya hasil baseline, ablation, leakage audit, dan pengulangan eksperimen proyek. | Hasil eksperimen lokal; bukan teori |
| GAT digambarkan hanya mengolah MAIN atau langsung menghasilkan target MAIN | GAT memproses semua node pada setiap snapshot; pooling menghasilkan graph embedding; diffusion kemudian memprediksi target MAIN `[B,3]` saja. | `core/src/models/gnn.py:143-161`; `core/src/data/temporal_loader.py:193-205` |
| GAT standar dicampur dengan edge-feature formula PyG | Pisahkan formula GAT standar Veličković dari detail `GATConv(edge_dim=1)`. Canonical `edge_attr` bernilai konstan `0.25`, bukan jarak informatif. | `core/src/config.py:180-205`; `core/src/models/gnn.py:73-74` |

## 8. Sitasi yang aman digunakan

### 8.1 Sumber primer/standar untuk teori

Sumber berikut adalah paper nyata yang dipakai untuk teori standar; sitasi tetap harus dicocokkan dengan metadata final daftar pustaka:

1. Veličković, P. et al. (2018). *Graph Attention Networks*. arXiv:1710.10903. — teori GAT standar.
2. Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762. — teori scaled dot-product/multi-head attention.
3. Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239. — teori DDPM.
4. Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. arXiv:2010.02502. — teori DDIM.
5. Delle Monache et al. (2013). *Probabilistic weather prediction based on an analog ensemble*. Monthly Weather Review. DOI: `10.1175/MWR-D-12-00281.1`. — analog ensemble, bukan bukti performa proyek.
6. Johnson, J., Douze, M., & Jégou, H. *Billion-scale similarity search with GPUs*. arXiv:1702.08734. — kutip versi arXiv; jangan mengisi detail IEEE yang belum dikunci.
7. Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. arXiv:1606.08415.
8. Elfwing, S., Uchibe, E., & Doya, K. (2017). *Sigmoid-weighted linear units for neural network function approximation in reinforcement learning*. arXiv:1702.03118.
9. Srivastava, N. et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. arXiv:1207.0580.
10. Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. arXiv:1711.05101.

### 8.2 Related work, bukan dasar teori implementasi

1. Asperti, A., Merizzi, F., Paparella, A., Pedrazzi, G., Angelinelli, M., & Colamonaco, S. *Precipitation nowcasting with generative diffusion models*. Applied Intelligence. DOI: `10.1007/s10489-024-06048-y`. Tahun online/issue, volume, dan halaman **perlu cek metadata final daftar pustaka**.
2. Ravuri et al. (2021). *Skilful precipitation nowcasting using deep generative models of radar*. Nature. DOI: `10.1038/s41586-021-03854-z`.
3. Zhang et al. (2023). *Skilful nowcasting of extreme precipitation with NowcastNet*. Nature. DOI: `10.1038/s41586-023-06184-4`.
4. Liu, J., Yang, L., Li, H., & Hong, S. (2024). *Retrieval-Augmented Diffusion Models for Time Series Forecasting*. arXiv:2410.18712. Mark as comparison, not as proof of local performance.

Paper modern pada daftar di atas disebut hanya sejauh identitas dan DOI/arXiv yang diberikan; jangan menyatakan seluruh metadata modern sudah diverifikasi sebelum pemeriksaan publisher/arXiv final.

WRF-GNN/orographic paper DOI `10.1007/s00500-025-10635-7` tidak dipakai sebagai sitasi utama di sini karena detail judul/penulis dan relevansi tepatnya belum diperlukan untuk menjawab gap proyek. Jika dimasukkan kemudian, verifikasi metadata penerbit dan nyatakan secara hati-hati sebagai related work, bukan dasar validasi arsitektur lokal.

### 8.3 Sitasi yang harus dihapus atau dikarantina

- “Multistream GAT (IEEE, 2021)” bila metadata lengkap tidak tersedia.
- Cover’s Theorem sebagai alasan angka hidden dimension 256.
- Li et al. (2020) bila detail paper tidak diverifikasi.
- Candido, Singh, & Delle Monache (2020) sebagai bukti retrieval meningkatkan performa proyek.
- WeatherBench sebagai dasar implementasi; kode hanya menunjukkan normalisasi training split.
- Klaim bahwa diffusion selalu lebih tajam, GAT pasti menangkap propagasi orografis, atau retrieval meningkatkan akurasi tanpa hasil eksperimen lokal.

---

## 9. Citation reliability: fakta, teori, dan klaim yang harus diuji

### Status sitasi

- **Teori standar dengan identitas arXiv/DOI yang diberikan:** Veličković et al. (2018) untuk GAT standar; Vaswani et al. (2017) untuk attention; Ho et al. (2020) untuk DDPM; Song et al. (2021) untuk DDIM; GELU, SiLU, dropout, AdamW; FAISS arXiv:1702.08734; dan analog ensemble Delle Monache et al. (2013).
- **Related work:** Asperti et al. dengan DOI `10.1007/s10489-024-06048-y`, Ravuri et al. (2021), Zhang et al. (2023), dan RATD Liu et al. (2024) arXiv:2410.18712. Related work tidak membuktikan performa atau identitas arsitektur proyek.
- **Batas kepastian:** Jangan menyatakan semua sitasi modern sudah diverifikasi. Tahun issue, volume, halaman, metadata penulis, dan format daftar pustaka final **perlu cek metadata final daftar pustaka** pada publisher/arXiv.
- **Larangan:** `[27]` pada draft V4 tidak boleh dipakai untuk mendukung diffusion bila entri `[27]` adalah paper IoT outdoor safety; gunakan Ho/Song untuk teori diffusion dan Asperti hanya sebagai related work.

### Fakta implementasi yang harus disandarkan ke kode

- Topologi, fitur, target, dan edge attribute: `core/src/config.py:74-107,161-205`.
- Window, normalisasi, target MAIN: `core/src/data/temporal_loader.py:149-205`.
- GAT dan temporal attention: `core/src/models/gnn.py:14-161`.
- Diffusion denoiser dan scheduler: `core/src/models/diffusion.py:8-175`.
- Loss, retrieval, noising, model construction: `core/src/train.py:428-565`.
- FAISS wrapper: `core/src/retrieval/base.py:5-73`.
- Ringkasan handover: `bab3_kekurangan_terpadu.md:1-1112`, `ROADMAP_RDM_STGNN_HANDOVER.md:320-479`.

### Klaim yang memerlukan eksperimen, bukan sitasi

- STGNN meningkatkan performa dibanding MAIN-only baseline.
- Retrieval meningkatkan skill atau calibration.
- Graph edge attribute membantu dibanding topology-only.
- Empat head lebih baik dari jumlah head lain.
- Hidden dimension 64/128/256 optimal.
- Wet/dry auxiliary head meningkatkan CSI/POD/FAR.
- DDIM 50 langkah menjaga akurasi dengan latency lebih rendah.
- Model lebih baik pada hujan ekstrem atau risiko hipotermia.

---

## 10. Checklist final sebelum Bab II/Bab III dikunci

- [ ] Bab II menyatakan GNN/STGNN sudah ada secara konseptual, lalu menutup gap rumus GAT, pooling, dan temporal attention.
- [ ] Bab II memakai Veličković et al. (2018) untuk GAT dan Vaswani et al. (2017) untuk attention.
- [ ] Bab III menyebut flow aktual enam graph `t-6...t-1`, context `t-1`, target `t`.
- [ ] Semua target diffusion ditulis sebagai `MAIN [B,3]`; tidak ada klaim semua node diberi noise.
- [ ] GAT disebut memproses semua node pada setiap snapshot.
- [ ] Temporal attention disebut `MultiheadAttention` dengan causal mask dan last timestep.
- [ ] Self-loop disebut sebagai default `GATConv`, bukan edge manual.
- [ ] `IndexFlatL2` disebut mengembalikan squared L2.
- [ ] Tidak ada PCA/reduced retrieval space/learned retrieval embedding.
- [ ] Strict-past hanya diklaim untuk training sesuai source; validation dibedakan.
- [ ] DDIM default aktif ditulis 50 langkah bila merujuk `sample_fast`; angka lain diberi konteks caller.
- [ ] Beta schedule tidak diberi angka yang tidak eksplisit di kode.
- [ ] `cond_emb` disebut additive conditioning embedding, bukan bias.
- [ ] Skip connection denoiser ditulis concatenation `h_mid || h2`, bukan residual addition.
- [ ] Hyperparameter tidak diberi rationale teoritis yang tidak diuji.
- [ ] Tidak ada klaim fundamental novel formula.
- [ ] Klaim performa dipindahkan ke bab eksperimen dan disertai baseline/ablation.
- [ ] Sitasi tidak memakai metadata yang belum diverifikasi.
- [ ] Path dan line range sudah dicocokkan dengan source aktif.
- [ ] Daftar pustaka final memeriksa ulang DOI, judul, penulis, dan tahun terhadap publisher/arXiv.

---

## 11. Handover prompt

> Audit dan revisi Bab II–III berdasarkan `revisi_v5.md`.
>
> Gunakan kode aktif sebagai sumber fakta: `core/src/config.py`, `core/src/data/temporal_loader.py`, `core/src/models/gnn.py`, `core/src/models/diffusion.py`, `core/src/retrieval/base.py`, dan `core/src/train.py`. Pertahankan notasi: lima node, sembilan fitur, tiga target MAIN, sequence `t-6...t-1`, context `t-1`, target `t`, GAT dua layer, temporal `MultiheadAttention` causal dengan last timestep, FAISS `IndexFlatL2` squared L2, additive conditioning, dan noising hanya pada `[B,3]` target. Pisahkan teori Bab II dari implementasi Bab III. Jangan menambahkan PCA, reduced retrieval space, learned retrieval encoder, atau alasan hyperparameter yang tidak diuji. Gunakan hanya sitasi terverifikasi dalam section citation reliability. Nyatakan kontribusi sebagai komposisional/application-driven sampai eksperimen membuktikan lebih dari itu. Setelah edit, jalankan pemeriksaan konsistensi path, shape, arah edge, scheduler, dan daftar sitasi; laporkan klaim yang masih membutuhkan eksperimen.

---

## 12. Kesimpulan

Draft v4 telah memiliki gagasan GNN/spatio-temporal, tetapi belum mendokumentasikan mekanisme aktual dengan cukup presisi. Revisi yang aman bukan menambah klaim kebaruan, melainkan menambahkan rumus teori yang tepat di Bab II, memindahkan konfigurasi dan tensor shape ke Bab III, mengoreksi arah agregasi GAT, memperjelas temporal attention, dan membatasi klaim pada apa yang dibuktikan oleh kode atau eksperimen.

**Status kontribusi:** integrasi retrieval historis, STGNN, dan conditional diffusion untuk prakiraan multi-output pada lima node lokal. **Status novelty formula:** tidak ada formula fundamental baru yang terbukti. **Status performa:** belum boleh disimpulkan tanpa hasil eksperimen yang membandingkan baseline, ablation, leakage control, dan sampling configuration.

---

*Dokumen dibuat sebagai file baru; file sumber existing tidak dimodifikasi.*
