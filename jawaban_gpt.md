# Jawaban Verifikasi Pipeline Model: GAT, Pooling, Temporal Attention, dan Diffusion

Dokumen ini memuat jawaban verifikasi langkah-demi-langkah mengenai integrasi Spatio-Temporal Graph Neural Network (STGNN) dan Retrieval-Augmented Diffusion Model (RDM) berdasarkan kode aktual pada direktori `core/`.

---

### 1. Bentuk tensor input sebelum masuk GAT
Untuk satu sampel graf pada satu timestep, bentuk tensor fiturnya adalah `[N, F]`, dengan jumlah node $N = 5$ dan jumlah fitur $F = 9$. Pada batch training, tensor digabungkan menjadi `x` berukuran `[B * 5, 9]` ($B = \text{Batch Size}$) dengan indeks hubungan `edge_index` berukuran `[2, B * 8]`.

### 2. Efek Z-Score
Z-Score hanya melakukan transformasi/standardisasi nilai fitur (mean = 0, std = 1) tanpa mengubah jumlah kolom fitur. Jumlah fitur tetap $F = 9$.

### 3. Waktu pelaksanaan transformasi linear (W)
Transformasi linear ($W$) dilakukan **sebelum** perhitungan skor atensi. Fitur setiap node diproyeksikan terlebih dahulu menggunakan matriks bobot latih $W$ sebelum masuk ke fungsi perhitungan koefisien atensi.

### 4. Dimensi embedding stasiun setelah transformasi linear
* **Pada GAT layer 1 (`conv1`):** Dimensi proyeksi per head adalah $64$. Karena menggunakan $4$ head dengan opsi penggabungan aktif (`concat=True`), dimensi output stasiun menjadi $64 \times 4 = 256$ dimensi.
* **Pada GAT layer 2 (`conv2`):** Dimensi proyeksi adalah $64$ dengan $1$ head (`concat=False`), sehingga dimensinya kembali menjadi $64$ dimensi.

### 5. Perhitungan atensi $\mathbf{a}^T [W\mathbf{h}_i \,||\, W\mathbf{h}_j]$
1. Fitur node target $i$ dan tetangga $j$ diproyeksikan linear menjadi $\mathbf{z}_i = W\mathbf{h}_i$ dan $\mathbf{z}_j = W\mathbf{h}_j$.
2. Kedua vektor digabungkan secara horizontal (concatenation) $[\mathbf{z}_i \,||\, \mathbf{z}_j]$.
3. Hasil gabungan dikalikan dengan transpose vektor parameter atensi latih $\mathbf{a}^T$ untuk menghasilkan nilai skalar tunggal.
4. Nilai skalar tersebut diaktifkan menggunakan fungsi `LeakyReLU`.
5. Hasilnya dinormalisasi menggunakan fungsi `Softmax` terhadap seluruh tetangga dari node target $i$ untuk menghasilkan skor atensi final ($\alpha_{ij}$).

### 6. Pengali atensi
Skor atensi $\alpha_{ij}$ (skalar) dikalikan dengan **embedding terproyeksi** tetangga hasil transformasi linear ($W\mathbf{h}_j$ berukuran 64 dimensi per head), bukan fitur 9 dimensi mentah.

### 7. Pembaruan embedding node MAIN
1. Ambil embedding terproyeksi ($\mathbf{z}_j$) dari stasiun tetangga: `MAIN` (self-loop), `UP`, `DOWN`, `LEFT`, dan `RIGHT`.
2. Kalikan masing-masing $\mathbf{z}_j$ dengan skor atensinya terhadap `MAIN` ($\alpha_{\text{MAIN}, j}$).
3. Jumlahkan kelima vektor 64-dimensi terbobot tersebut secara elemen-demi-elemen:
   $$\mathbf{h}'_{\text{MAIN}} = (\alpha_{\text{MAIN, MAIN}} \cdot \mathbf{z}_{\text{MAIN}}) + (\alpha_{\text{MAIN, UP}} \cdot \mathbf{z}_{\text{UP}}) + \dots + (\alpha_{\text{MAIN, RIGHT}} \cdot \mathbf{z}_{\text{RIGHT}})$$

### 8. Jumlah embedding setelah pembaruan stasiun
Konvolusi graf bekerja pada level stasiun (*node-to-node*). Setiap stasiun memperbarui representasi dirinya sendiri secara paralel tanpa mengurangi jumlah entitas. Output dari layer 2 GAT tetap berupa matriks berukuran `[5, 64]`.

### 9. Peran Global Mean Pool
Diffusion model membutuhkan ringkasan representasi wilayah (graf) berupa satu vektor tunggal. `global_mean_pool` merata-ratakan kelima vektor embedding stasiun tersebut pada sumbu stasiun:
$$\mathbf{h}_{\text{graph}} = \frac{1}{5} \sum_{n=0}^{4} \mathbf{h}_n$$
Proses ini menghasilkan **1 vektor graf (graph embedding)** berukuran `[64]` untuk timestep tersebut.

### 10. Pembentukan graph embedding per jam
Melalui pemrosesan independen matriks fitur `[5, 9]` dari masing-masing 6 jam menggunakan modul SpatialGNN (2 layer GAT + Global Mean Pool), model menghasilkan 6 vektor graf berukuran `[64]`.

### 11. Pemrosesan 6 graph embedding oleh Temporal Attention
1. Keenam vektor ditumpuk menjadi tensor `[B, 6, 64]`.
2. Ditambahkan *positional embedding* latih berukuran `[1, 6, 64]` secara aditif.
3. Diterapkan *causal mask* segitiga atas agar timestep tidak dapat memperhatikan masa depan.
4. Diproses oleh self-attention (`MultiheadAttention`).
5. Diambil representasi timestep terakhir (indeks `-1`) yang telah menyerap informasi jam-jam sebelumnya.

### 12. Output Temporal Attention tetap 64 dimensi
Karena dimensi internal `MultiheadAttention` (`embed_dim`) dikonfigurasi sebesar 64, dan proyeksi akhir `output_proj` memetakan dimensi tersebut kembali ke dimensi output target (`output_dim = 64`).

### 13. Alasan diffusion model menerima embedding graph
Karena target prediksi diffusion difokuskan hanya untuk memprediksi stasiun `MAIN` pada jam ke-7. Informasi spasio-temporal wilayah secara global diwakili oleh `graph_emb`, sedangkan kondisi lokal stasiun target pada jam terakhir diwakili langsung oleh *raw context* `MAIN` (`[9]`).

### 14. Tahap pembaruan parameter W dan vektor atensi a
Pembaruan terjadi setelah kalkulasi fungsi loss pada keluaran model diffusion (jam ke-7 stasiun `MAIN`) dijalankan menggunakan `loss.backward()`. Gradien kesalahan mengalir mundur melewati denoiser, temporal attention, pooling, hingga ke GAT, kemudian optimizer memperbarui nilai parameter matriks $W$ dan vektor atensi $\mathbf{a}$ lewat panggilan `optimizer.step()`.
