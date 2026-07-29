# Dokumentasi Detail Arsitektur Model Per Layer (End-to-End)

Dokumen ini mendokumentasikan secara rinci arsitektur model **Retrieval-Augmented Diffusion dengan Spatio-Temporal Graph Conditioning** per layer, lengkap dengan dimensi input/output, fungsi aktivasi, dan alur data dari hulu ke hilir.

---

## 1. Diagram Alur Sistem End-to-End

```
                      [ INPUT DATA ]
               Sequence L=6 jam, 5 Node Graf
                            │
         ┌──────────────────┴──────────────────┐
         ▼ (Data Spasio-Temporal)              ▼ (Fitur Kueri t)
    [ MODUL STGNN ]                     [ FAISS RETRIEVAL ]
  GAT & Temporal Attn                  Cari k=3 Analog Terdekat
         │                                     │
    Embedding Graf (64-d)               Vektor Target Hist (96-d)
         │                                     │
         │                                     ▼
         │                               [ RETRIEVAL MLP ]
         │                              Linear (96 -> 64-d)
         ▼                                     │
    [ FUSION ] <─── Sum (Tambah Elemen) ───────┘
  Embedding Gabung (64-d)
         │
         ▼
    [ U-NET MLP DENOISER ] <─── Noised Target (3-d) + Time (64-d)
  Denoising (DDPM/DDIM)
         │
         ▼
  [ OUTPUT ENSEMBLE ]
 30 Realisasi Prediksi (3-d)
```

---

## 2. Rincian Layer per Modul

### 2.1 Modul 1: Spatio-Temporal GNN (STGNN)
Bertugas memproses representasi spasial 5 stasiun cuaca dan ketergantungan temporalnya.
* **Input**: Sequence graf sepanjang $L=6$ jam. Setiap timestep memiliki fitur node $\mathbf{X}_t \in \mathbb{R}^{5 \times 9}$ (5 node, 9 input features) dan matriks ketetanggaan graf bintang $E \in \mathbb{R}^{2 \times 8}$.

#### A. Spatial GNN (SpatialGNN)
Dipanggil terpisah sebanyak 6 kali (1 kali per timestep):
1. **Layer 1: Graph Attention Network (GATConv)**
   * **Input**: Node features $[5, 9]$, Edge index $[2, 8]$, Edge attributes $[8, 1]$.
   * **Bobot**: $4$ attention heads, output dimension per head = $64$.
   * **Output**: $[5, 256]$ (karena concatenation: $64 \times 4$).
2. **Aktivasi**: **ReLU** $\rightarrow$ Output $[5, 256]$.
3. **Layer 2: GATConv (Final Spatial)**
   * **Input**: $[5, 256]$, Edge index $[2, 8]$, Edge attributes $[8, 1]$.
   * **Bobot**: $1$ attention head, output dimension = $64$ (tanpa concat).
   * **Output**: $[5, 64]$.
4. **Global Mean Pooling (`global_mean_pool`)**:
   * Merata-rata fitur dari kelima node menjadi representasi graf tunggal.
   * **Output Timestep $\tau$**: $[1, 64]$.

#### B. Temporal Attention (TemporalAttention)
Menggabungkan output spasial dari 6 timestep ($L=6$):
1. **Input**: Stack 6 timestep $\rightarrow$ Tensor dimension $[Batch, 6, 64]$.
2. **Positional Encoding Addition**:
   * Menambahkan tensor parameter posisi terpelajar: $\mathbf{P} \in \mathbb{R}^{1 \times 6 \times 64}$.
3. **Layer Self-Attention (MultiheadAttention)**:
   * **Parameter**: `embed_dim` = $64$, `num_heads` = $4$, `dropout` = $0.1$.
   * **Causal Masking**: Matriks segitiga atas biner diaktifkan untuk mencegah kebocoran informasi masa depan ($t+1 \rightarrow t$).
   * **Output**: $[Batch, 6, 64]$.
4. **Layer Normalization & Residual**:
   * $x = \text{LayerNorm}(x_{\text{input}} + \text{Dropout}(\text{Attn}(x)))$.
5. **Last Step Slice**:
   * Mengambil timestep terakhir ($L=6$) sebagai representasi rangkuman: $x[:, -1, :] \rightarrow$ Dimension $[Batch, 64]$.

#### C. Output Projection
* **Layer**: `nn.Linear(64, 64)`.
* **Output Embedding Graf ($h_t^G$)**: Dimension $[Batch, 64]$.

---

### 2.2 Modul 2: FAISS Retrieval L2
Mencari kemiripan pola cuaca saat ini dengan database training untuk mengatasi keterbatasan data ekstrem.
* **Input Query**: Fitur cuaca saat ini pada node MAIN ($1 \times 9$).
* **Proses FAISS (`IndexFlatL2`)**:
  * Melakukan pencarian tetangga terdekat menggunakan jarak Euclidean L2.
  * Menghasilkan $k=3$ analog historis terdekat.
* **Retrieval Output ($R_t$)**:
  * Mengambil nilai target asli ($3$ variabel: curah hujan, angin, kelembapan) pada $t+1$ dari ke-3 hari analog.
  * Digabungkan (flat) $\rightarrow$ Dimension $[Batch, 96]$ ($3 \text{ analog} \times 3 \text{ variabel} \times 30 \text{ batch?}$ tidak, ukuran dimensi flat $= k \times \text{num\_targets} = 3 \times 3 = 9$?? Tunggu, di kode `retrieval_dim = num_targets * k_neighbors` $= 3 \times 32$?? Tidak, di `diffusion.py` `retrieval_dim=32` sebagai default, tetapi di `train.py` diset ke `num_targets * k_neighbors` yaitu $3 \times 3 = 9$).
* **Retrieval MLP**:
  * **Layer 1**: `nn.Linear(9, 64)` $\rightarrow$ Aktivasi **SiLU** $\rightarrow$ **Layer 2**: `nn.Linear(64, 64)`.
  * **Output Embedding Retrieval ($R_{\text{emb}}$)**: Dimension $[Batch, 64]$.

---

### 2.3 Modul 3: Conditioning Fusion
Menggabungkan pemandu spasio-temporal dan historis:
* **Current Weather MLP (`cond_mlp`)**:
  * **Input**: Fitur cuaca saat ini dari stasiun MAIN $[Batch, 64]$.
  * **Layer**: `Linear(64, 64) -> SiLU -> Linear(64, 64)` $\rightarrow$ Output $[Batch, 64]$.
* **STGNN Projection (`graph_mlp`)**:
  * **Input**: Embedding STGNN $h_t^G$ $[Batch, 64]$.
  * **Layer**: `Linear(64, 64) -> SiLU -> Linear(64, 64)` $\rightarrow$ Output $[Batch, 64]$.
* **Fusion Equation (Additive Conditioning)**:
  $$\mathbf{c}_t = \text{cond\_mlp}(\text{context}) + \text{retrieval\_mlp}(R_t) + \text{graph\_mlp}(h_t^G)$$
  * **Output Conditioning Vector ($\mathbf{c}_t$)**: Dimension $[Batch, 64]$.

---

### 2.4 Modul 4: Conditional Diffusion Model
Pembangkit target probabilistik lewat proses *reverse denoising*.
* **Time Step Embedding MLP (`time_mlp`)**:
  * **Input**: Integer step noise $t_{\text{diff}} \in [1, 1000] \rightarrow$ Dimension $[Batch, 1]$.
  * **Sinusoidal Position Projection**: Mengodekan nilai skalar ke dimensi $64$.
  * **MLP Layers**: `Linear(64, 128) -> GELU -> Linear(128, 64)`.
  * **Output Time Embedding ($t_{\text{emb}}$)**: Dimension $[Batch, 64]$.
* **Total Conditioning (Embedding)**:
  $$\mathbf{emb} = t_{\text{emb}} + \mathbf{c}_t \quad (\text{Dimension } [Batch, 64])$$

#### A. U-Net-like MLP Denoiser
1. **Downsampling Block 1 (`down1`)**:
   * **Input**: Noised target $x_t$ $[Batch, 3]$.
   * **Layers**: `nn.Linear(3, 64) -> SiLU`.
   * **Koneksi**: Hasil di-tambah dengan $\mathbf{emb}$ (Additive conditioning: $h_1 = \text{down1}(x_t) + \mathbf{emb}$).
   * **Output**: $[Batch, 64]$.
2. **Downsampling Block 2 (`down2`)**:
   * **Input**: $[Batch, 64]$.
   * **Layers**: `nn.Linear(64, 128) -> SiLU`.
   * **Output**: $[Batch, 128]$ (disimpan untuk skip connection).
3. **Bottleneck Middle Block (`mid`)**:
   * **Input**: $[Batch, 128]$.
   * **Layers**: `nn.Linear(128, 128) -> SiLU`.
   * **Output**: $[Batch, 128]$.
4. **Upsampling Block 1 (`up1` dengan Skip Connection)**:
   * **Input**: Concatenate dari output `mid` dan `down2` $\rightarrow$ Dimension $[Batch, 256]$ ($128 + 128$).
   * **Layers**: `nn.Linear(256, 64) -> SiLU`.
   * **Output**: $[Batch, 64]$.
5. **Output Projection Layer (`out`)**:
   * **Input**: $[Batch, 64]$.
   * **Layers**: `nn.Linear(64, 3)`.
   * **Output (Prediksi Noise $\hat{\boldsymbol{\epsilon}}_\theta$)**: Dimension $[Batch, 3]$ (untuk target: presipitasi, kecepatan angin, kelembapan).

#### B. Auxiliary Head (`wet_head`)
Mencegah bias curah hujan berlebih pada kondisi kering (zero-inflation):
* **Input**: Conditioning vector $\mathbf{c}_t$ $[Batch, 64]$.
* **Layers**: `Linear(64, 64) -> SiLU -> Linear(64, 1)` $\rightarrow$ Output Logit Wet/Dry $[Batch, 1]$.

---

## 3. Ringkasan Dimensi & Aktivasi Tiap Layer

| Modul | Nama Layer / Operasi | Dimensi Input | Dimensi Output | Aktivasi |
| :--- | :--- | :--- | :--- | :--- |
| **STGNN** | GAT Layer 1 (`conv1`) | $[5, 9]$ | $[5, 256]$ | ReLU |
| **STGNN** | GAT Layer 2 (`conv2`) | $[5, 256]$ | $[5, 64]$ | None |
| **STGNN** | Global Mean Pool | $[5, 64]$ | $[1, 64]$ | None |
| **STGNN** | Temporal Self-Attention | $[B, 6, 64]$ | $[B, 64]$ | None (LayerNorm) |
| **STGNN** | Output Projection | $[B, 64]$ | $[B, 64]$ | None |
| **FAISS** | Retrieval MLP Layer 1 | $[B, 9]$ | $[B, 64]$ | SiLU |
| **FAISS** | Retrieval MLP Layer 2 | $[B, 64]$ | $[B, 64]$ | None |
| **Fusion** | Weather context MLP | $[B, 64]$ | $[B, 64]$ | SiLU |
| **Fusion** | Graph projection MLP | $[B, 64]$ | $[B, 64]$ | SiLU |
| **Diffusion**| Sinusoidal Position Embed | $[B, 1]$ | $[B, 64]$ | None |
| **Diffusion**| Time Embedding MLP | $[B, 64]$ | $[B, 64]$ | GELU |
| **Diffusion**| Downsampling Block 1 | $[B, 3]$ | $[B, 64]$ | SiLU |
| **Diffusion**| Downsampling Block 2 | $[B, 64]$ | $[B, 128]$ | SiLU |
| **Diffusion**| Bottleneck Mid Block | $[B, 128]$ | $[B, 128]$ | SiLU |
| **Diffusion**| Upsampling Block (Skip Cat) | $[B, 256]$ | $[B, 64]$ | SiLU |
| **Diffusion**| Output Projection (`out`) | $[B, 64]$ | $[B, 3]$ | None |
| **Aux Head** | Wet/Dry Head (`wet_head`) | $[B, 64]$ | $[B, 1]$ | SiLU |
