# Dokumentasi Metode Penelitian Proyek  
## Nowcasting probabilistik presipitasi untuk mitigasi resiko pendaki di gunung gede-pangrango dengan menggunakan retrieval augemented difussion model dengan spatio temporal graph conditioning

Dokumen ini menjelaskan metode penelitian proyek secara sistematis, mulai dari objek penelitian, sumber data, desain pipeline, metode analisis, hingga prosedur validasi dan evaluasi. Tujuan utama sistem adalah menghasilkan prediksi nowcasting jangka pendek untuk titik utama (MAIN) pada wilayah studi Gunung Gede-Pangrango dengan memanfaatkan konteks spasial dari node sekeliling serta memori historis berbasis retrieval.

### 1. Objek dan Lingkup Proyek
Objek penelitian pada proyek ini adalah sistem prediksi cuaca jam-ke-jam (hourly nowcasting) untuk tiga variabel target: curah hujan (`precipitation`), kecepatan angin 10 m (`wind_speed_10m`), dan kelembapan relatif 2 m (`relative_humidity_2m`). Fokus prediksi ditetapkan secara tegas pada satu node target utama, yaitu `MAIN` dengan koordinat grid kanonik `(-6.75, 107.00)`. Empat node lain (`UP`, `DOWN`, `LEFT`, `RIGHT`) digunakan sebagai konteks spasial untuk memperkaya representasi dinamika lokal.

Lingkup proyek dibatasi pada arsitektur aktif 5-node dengan topologi graf bintang (star topology), di mana hanya koneksi dua arah antara `MAIN` dan setiap node sekeliling yang diizinkan. Konfigurasi ini dipilih untuk menjaga interpretabilitas spasial sekaligus menekan kompleksitas graf yang tidak perlu. Kebijakan target aktif adalah `main_node_only`, sehingga tidak ada perataan (averaging) antar node untuk label target.

Dalam konteks penggunaan, sistem ini ditujukan untuk nowcasting operasional berbasis pembaruan data aktual terbaru. Artinya, prediksi pada waktu \(t+1\) selalu dibuat menggunakan jendela observasi hingga waktu \(t\). Saat observasi baru tersedia, jendela input diperbarui dan prediksi berikutnya dihitung ulang. Dengan desain ini, sistem merepresentasikan skenario operasional real-time, bukan simulasi recursive multi-horizon tanpa pembaruan observasi.

Batasan penelitian meliputi: (1) sumber data tunggal dari model reanalisis ERA5 melalui Open-Meteo, (2) resolusi temporal hourly, (3) area studi yang direpresentasikan oleh lima titik grid tetap, dan (4) evaluasi utama pada node `MAIN`.

### 2. Jenis dan Sumber Data
Data yang digunakan merupakan data sekunder time series multivariat dengan referensi spasial tetap. Data diperoleh dari Open-Meteo Archive API dengan parameter model wajib `models=era5`. Pemilihan ERA5 dilakukan untuk menjaga konsistensi historis jangka panjang, stabilitas ketersediaan data, dan keseragaman variabel meteorologis antar titik.

Dataset aktif proyek tersimpan pada:
- `data/raw/pangrango_era5_5node_2005_2025.parquet`
- laporan validasi grid: `data/raw/pangrango_era5_5node_grid_validation.json`

Karakteristik data aktif:
- periode waktu: `2004-12-31 17:00:00+00:00` sampai `2025-12-31 16:00:00+00:00`;
- frekuensi: hourly;
- jumlah timestamp unik: `184.080`;
- jumlah node per timestamp: tepat `5` (tanpa missing/duplikasi);
- total baris: `920.400`;
- total kolom: `21`.

Klasifikasi variabel:
1. Variabel target:
   - `precipitation`
   - `wind_speed_10m`
   - `relative_humidity_2m`
2. Variabel fitur utama:
   - `temperature_2m`
   - `relative_humidity_2m`
   - `dewpoint_2m`
   - `surface_pressure`
   - `wind_speed_10m`
   - `wind_direction_10m`
   - `cloud_cover`
   - `precipitation_lag1`
   - `elevation`

Data spasial dikunci pada lima node kanonik berikut:
- `MAIN`: `(-6.75, 107.00)`
- `UP`: `(-6.50, 107.00)`
- `DOWN`: `(-7.00, 107.00)`
- `LEFT`: `(-6.75, 106.75)`
- `RIGHT`: `(-6.75, 107.25)`

**Catatan keterbatasan node dan elevasi.** Elevasi yang digunakan berasal dari metadata Open-Meteo/ERA5 grid 0.25°, bukan pengukuran topografi exact. Konsekuensinya, elevasi MAIN terekam sebagai 1529 m, yaitu rata-rata sel grid yang lebih rendah dari elevasi puncak Gede–Pangrango sekitar 3000 m. Node DOWN (-7.00°, 107.00°) terekapi Open-Meteo sebagai 0 m, meskipun koordinat tersebut secara geografis berada di wilayah pesisir daratan. Perbedaan rezim cuaca antar node tetap dapat dipelajari model karena fitur cuaca tiap node berbeda, sedangkan nilai elevasi disimpan sebagai node feature untuk membedakan karakteristik lokasi.

Setiap node divalidasi terhadap pusat grid yang benar-benar dikembalikan API (grid-center strict). Jika dua node jatuh pada grid center yang sama (collision), proses ingestion dihentikan (fail-fast). Pendekatan ini digunakan untuk menghindari bias analisis yang seolah-olah menganggap dua titik berbeda padahal identik secara grid meteorologis.

### 3. Tahapan Proyek (End-to-End Pipeline)
Pipeline disusun sebagai rangkaian tahap berurutan agar dapat direplikasi dari data mentah hingga hasil evaluasi.

#### 3.1 Pengumpulan Data
1. Sistem memanggil Open-Meteo Archive API untuk 5 koordinat node kanonik.
2. Parameter wajib `models=era5` diterapkan pada setiap request.
3. Sistem mengambil variabel hourly meteorologis serta metadata elevasi.
4. Sebelum data disimpan, sistem memvalidasi identitas grid center tiap node.
5. Data seluruh node digabung, lalu diurutkan ketat menurut waktu dan urutan node kanonik `[MAIN, UP, DOWN, LEFT, RIGHT]`.

Alasan metodologis: urutan node yang konsisten diperlukan agar tensor spasio-temporal yang masuk ke model tidak mengalami drift indeks antar tahap (ingest, loader, train, inferensi, evaluasi).

#### 3.2 Preprocessing Data
Preprocessing dilakukan dengan prinsip "validasi struktural sebelum pelatihan".

Langkah operasional:
1. Harmonisasi nama kolom legacy ke kontrak kolom aktif.
2. Pemeriksaan skema wajib fitur dan target; jika ada kolom hilang maka proses dihentikan.
3. Pemeriksaan nilai fitur agar tidak seluruhnya NaN.
4. Pembuatan fitur lag:  
   \[
   \text{precipitation\_lag1}_t = \text{precipitation}_{t-1}
   \]
   dengan nilai awal tiap node diisi \(0.0\).
5. Pemisahan data temporal kronologis:
   - train: `date <= 2018-12-31`
   - validation: `2018-12-31 < date <= 2021-12-31`
   - test: `date > 2021-12-31`

Alasan metodologis: split kronologis dipilih untuk mencegah leakage lintas waktu. Model tidak boleh melihat pola masa depan saat mempelajari parameter.

#### 3.3 Pemilihan Metode dan Alasan
Model utama dipilih sebagai gabungan tiga komponen:

**Catatan keterbatasan graph conditioning.** Pada konfigurasi 5-node star dengan grid ERA5 0.25°, jarak euclidean antar MAIN dan setiap node tetangga adalah konstan 0.25°. Oleh karena itu, atribut edge (`edge_attr`) yang digunakan pada `GATConv(edge_dim=1)` bersifat non-informatif karena bernilai identik untuk semua edge. Spatial conditioning berasal dari (i) topologi graf bintang yang memperkenalkan hubungan MAIN–tetangga, dan (ii) fitur node yang berbeda antar lokasi. Eksperimen dengan atribut edge tambahan (misal, perbedaan elevasi) telah dicoba tetapi menyebabkan training divergen, sehingga dipertahankan pendekatan edge-attribute konstan yang didokumentasikan secara transparan.
1. **Spatio-Temporal GNN** untuk menangkap ketergantungan antar node dan antar waktu.
2. **Retrieval k-NN berbasis FAISS** untuk mengambil analog historis dari kondisi saat ini.
3. **Conditional Diffusion** untuk menghasilkan prediksi probabilistik multi-target.

Alasan pemilihan:
- persistence dan MLP one-step sangat kuat pada horizon pendek karena autocorrelation tinggi; karena itu model utama harus memanfaatkan informasi tambahan (spasial + retrieval + probabilistik), bukan sekadar hubungan lag lokal.
- diffusion memberi keluaran ensemble, sehingga ketidakpastian dapat diukur (CRPS, Brier, reliabilitas), bukan hanya prediksi titik.

Baseline pembanding:
- **Persistence**: \(\hat{y}_{t+1}=y_t\).
- **MLP baseline**: input lag MAIN-only (\(L \times F\) fitur diratakan), output 3 target pada \(t+1\).


**Catatan evaluasi MLP baseline.** MLP adalah model deterministik satu-langkah (one-step point forecast) yang memetakan fitur lag node MAIN ke tiga target pada waktu \(t+1\). Berbeda dengan diffusion yang secara alami menghasilkan ensemble probabilistik, MLP tidak dirancang untuk memberikan rentang ketidakpastian. Oleh karena itu, pada evaluasi MLP dijalankan dalam mode `eval()` dengan satu kali forward pass tanpa MC-dropout; CRPS untuk MLP berarti mengukur ketidakpastian sebatas error absolut (MAE). Pendekatan ini memastikan perbandingan antar baseline tetap fair.


#### 3.4 Implementasi Sistem/Program
Implementasi terdiri dari modul terpisah:
- `src/data/ingest.py`: pengambilan data dan validasi grid;
- `src/data/temporal_loader.py`: pembentukan urutan graf temporal + fail-fast contract checks;
- `src/models/gnn.py`: encoder spasio-temporal (GAT + temporal attention);
- `src/models/diffusion.py`: model diffusion kondisional + wet/dry auxiliary head;
- `src/train.py`: training model utama;
- `src/train_baseline.py`: training baseline MLP;
- `src/inference.py`: pemuatan checkpoint dan inferensi;
- `run_eval_final.py`: evaluasi 6 skenario dan pembuatan laporan.

Prinsip implementasi yang dijaga:
1. **Contract consistency**: metadata node, topologi, dan kebijakan target wajib ada di checkpoint.
2. **Fail-fast**: jika urutan node salah, jumlah node per timestamp tidak tepat, atau ada nama node tidak dikenal, proses berhenti.
3. **No silent drop**: data tidak boleh diam-diam dibuang saat mismatch.

#### 3.5 Proses Analisis/Prediksi
Proses prediksi one-step nowcasting berjalan sebagai berikut:
1. Bentuk jendela konteks sepanjang `seq_len=6` jam terakhir.
2. Ubah jendela menjadi urutan graf 5-node (star graph tetap).
3. Hitung embedding graf dari STGNN.
4. Ambil context vektor node MAIN pada langkah terakhir.
5. Query retrieval ke basis train-only dengan k-nearest neighbors.
6. Jalankan sampling diffusion (DDIM accelerated sampling).
7. Ubah hasil ke skala asli (denormalisasi, `expm1` untuk precipitation).
8. Ambil median ensemble sebagai prediksi titik.
9. Jika rain specialization aktif, probabilitas wet/dry dipakai sebagai gate untuk memutuskan apakah prediksi hujan dipertahankan atau diset nol.

#### 3.6 Evaluasi dan Validasi Hasil
Evaluasi dilakukan pada enam skenario:
1. Persistence
2. MLP baseline
3. Diffusion only
4. Diffusion + Retrieval
5. Diffusion + GNN
6. Full model (Diffusion + Retrieval + GNN)

Output evaluasi dihasilkan dalam bentuk:
- metrik per skenario/variabel (`result_test/*/metrics.json`);
- ringkasan komparasi (`result_test/comparison/comparison_summary.csv`);
- laporan markdown (`result_test/EVALUATION_REPORT.md`);
- visualisasi (bar chart, scatter, reliability diagram, time series sample, dan ablation plot).

### 4. Metode Analisis Data
Metode analisis utama adalah kombinasi pembelajaran representasi spasio-temporal dan generative probabilistic forecasting.

#### 4.1 Formulasi Masalah
Untuk setiap waktu \(t\), model memetakan jendela observasi:
\[
X_t = \{x_{\tau,n}\ |\ \tau=t-L+1,\ldots,t,\ n \in \mathcal{N}\}
\]
dengan \(\mathcal{N}=\{\text{MAIN, UP, DOWN, LEFT, RIGHT}\}\), menjadi distribusi target MAIN:
\[
p(y_{t+1}^{\text{MAIN}} \mid X_t)
\]
di mana \(y\) berisi tiga variabel target.

#### 4.2 Normalisasi dan Transformasi
Statistik normalisasi dihitung hanya dari data train node MAIN:
\[
\tilde{x}=\frac{x-\mu_c}{\sigma_c+\epsilon},\quad
\tilde{y}=\frac{g(y)-\mu_t}{\sigma_t+\epsilon}
\]
dengan \(g(y)\) untuk precipitation adalah \(\log(1+y)\), dan untuk variabel lain identitas.

Alasan: distribusi curah hujan sangat skewed dan zero-inflated; transformasi log mengurangi dominasi outlier besar pada proses optimasi.

#### 4.3 Komponen Spasial-Temporal (STGNN)
Pada tiap timestep, node features diproses menggunakan Graph Attention Network (GAT) dengan
`edge_attr` konstan yang dialirkan ke `GATConv(edge_dim=1)`. Pada grid ERA5 0.25°, jarak
lat/lon antar MAIN dan setiap node tetangga identik (0.25°), sehingga `edge_attr` bersifat
non-informatif; spatial conditioning utama berasal dari topologi graf bintang dan perbedaan
fitur node antar lokasi. Hasil representasi per waktu kemudian diagregasi oleh temporal
self-attention kausal, sehingga informasi masa depan tidak bocor ke masa lalu.

Secara konseptual:
1. Spatial encoding per \(\tau\):
   \[
   h_\tau = \text{GAT}(G_\tau)
   \]
2. Temporal aggregation:
   \[
   h_t^{G} = \text{TemporalAttention}(h_{t-L+1},\ldots,h_t)
   \]

Output \(h_t^{G}\) menjadi kondisi tambahan untuk model diffusion.

#### 4.4 Retrieval-Augmented Memory
Basis data analog historis dibangun dari data train node MAIN saja, dengan pasangan
**(kunci = fitur MAIN pada waktu \(\tau\), nilai = outcome target MAIN pada waktu \(\tau+1\), ternormalisasi)**.
Vektor context MAIN terbaru \(c_t\) di-query ke basis ini:
\[
R_t = \text{kNN}(c_t,\mathcal{D}_{train},k)
\]
sehingga yang dikembalikan adalah **analog outcome jam-berikut** (bukan fitur), berdimensi
\(k \times \text{num\_targets}\). Inilah makna "retrieval-augmented" yang sebenarnya: model
dikondisikan oleh hasil historis dari situasi serupa. Pada training precompute retrieval,
pencarian dibatasi strict-past dan self-neighbor dikeluarkan untuk mencegah leakage temporal
(kunci pada posisi \(j=\tau\) hanya dipakai jika \(\tau+1 < t\)).

#### 4.5 Conditional Diffusion Forecasting
Model diffusion mempelajari prediksi noise pada target ternormalisasi. Skema dasar:
\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
\]
Model memprediksi:
\[
\hat{\epsilon}_\theta = f_\theta(x_t, t, c_t, R_t, h_t^G)
\]
Loss utama (weighted noise MSE):
\[
\mathcal{L}_{noise} = \mathbb{E}\left[w(x_0)\|\epsilon-\hat{\epsilon}_\theta\|_2^2\right]
\]
dengan bobot error meningkat pada target bernilai absolut besar.

Untuk menangani zero-inflation pada hujan, ditambahkan wet/dry auxiliary head:
\[
\mathcal{L}_{wet} = \text{BCEWithLogits}(z_{wet}, y_{wet})
\]
dan total loss:
\[
\mathcal{L}_{total}=\mathcal{L}_{noise}+\lambda_{wet}\mathcal{L}_{wet}
\]

Saat inferensi, model menghasilkan \(S\) sampel:
\[
\{y_{t+1}^{(s)}\}_{s=1}^{S}
\]
Prediksi titik ditetapkan sebagai median ensemble:
\[
\hat{y}_{t+1} = \text{median}_s\left(y_{t+1}^{(s)}\right)
\]
Jika probabilitas wet \(<\tau_{wet}\), komponen precipitation diset \(0\) sesuai kebijakan rain gate.

**Conditioning dropout (robust ablation).** Agar studi ablation bermakna, saat training
komponen kondisi graph (\(h_t^G\)) dan retrieval (\(R_t\)) di-nol-kan secara acak per-sampel
dengan probabilitas \(p=0.15\) (independen). Tanpa ini, mematikan salah satu kondisi saat
evaluasi (mengisinya dengan nol) menghasilkan input out-of-distribution sehingga prediksi noise
menjadi liar. Dengan dropout ini, skenario ablation (`diff_only`, `diff_retrieval`, `diff_gnn`)
tetap berada dalam distribusi yang dikenal model, sehingga kontribusi tiap komponen dapat
dibandingkan secara adil.

**Batas fisik precipitation.** Pada denormalisasi, hasil precipitation di-clamp ke rentang
\([0, 60]\) mm/jam. Batas atas 60 mm/jam adalah pengaman numerik (maksimum dataset \(\approx 21.5\)
mm/jam), jauh di atas nilai nyata sehingga tidak mendistorsi prediksi sehat, namun mencegah
ledakan akibat \(\exp(\cdot)\) pada input ekstrem.

#### 4.6 Cara Output Digunakan
Output model digunakan dalam dua mode:
1. **Point forecast** (median) untuk metrik deterministik (RMSE, MAE, korelasi).
2. **Ensemble forecast** untuk metrik probabilistik (CRPS, Brier, POD, FAR, CSI).

Dengan demikian, model tidak hanya dinilai dari ketepatan angka tunggal, tetapi juga dari kualitas distribusi prediksi dan kemampuan deteksi event hujan.

#
**Catatan protokol evaluasi.** Evaluasi model pada data test dilakukan dengan protokol one-step hourly: setiap sampel memprediksi satu jam ke depan (`t+1`) berdasarkan 6 jam observasi sebelumnya (`t-6` hingga `t-1`). Untuk mengurangi beban komputasi pada eksperimen utama, evaluasi dijalankan dengan `eval_step=11` yang memberikan cakupan diurnal seragam (coprime terhadap 24 jam) dan menghasilkan sekitar 3.200 sampel dari ~35.000 jam test. Kode mendukung evaluasi hourly penuh (`eval_step=1`) untuk angka definitif yang dapat dijalankan pada tahap akhir penelitian.


## 5. Validasi dan Evaluasi
Validasi proyek dilakukan pada dua lapisan: validasi struktural pipeline dan evaluasi performa prediksi.

## 5.1 Validasi Struktural
Validasi struktural mencakup:
1. verifikasi node order dan kelengkapan node per timestamp;
2. verifikasi topologi graf star dengan 8 edge terarah;
3. verifikasi metadata checkpoint (node names, roles, coordinates, policy, topology);
4. verifikasi bahwa retrieval dan normalisasi hanya menggunakan data train;
5. pengujian unit otomatis (`7/7` pass).

Tujuan lapisan ini adalah memastikan model yang dievaluasi benar-benar merepresentasikan desain metodologis yang dideklarasikan.

## 5.2 Metrik Evaluasi
Metrik deterministik:
1. Root Mean Square Error (RMSE):
   \[
   \text{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i-y_i)^2}
   \]
2. Mean Absolute Error (MAE):
   \[
   \text{MAE}=\frac{1}{N}\sum_{i=1}^{N}|\hat{y}_i-y_i|
   \]
3. Pearson Correlation:
   \[
   r=\frac{\sum(\hat{y}_i-\bar{\hat{y}})(y_i-\bar{y})}
   {\sqrt{\sum(\hat{y}_i-\bar{\hat{y}})^2}\sqrt{\sum(y_i-\bar{y})^2}}
   \]

Metrik probabilistik:
1. CRPS:
   \[
   \text{CRPS}= \mathbb{E}|X-y|-\frac{1}{2}\mathbb{E}|X-X'|
   \]
2. Brier Score untuk event hujan melebihi threshold.
3. POD, FAR, CSI pada threshold precipitation \(\{2,5,10\}\) mm.

Interpretasi umum:
- RMSE/MAE lebih kecil menandakan error lebih rendah;
- korelasi lebih tinggi menandakan kesesuaian pola temporal lebih baik;
- CRPS/Brier lebih kecil lebih baik;
- POD tinggi, FAR rendah, CSI tinggi menunjukkan deteksi event lebih seimbang.

## 5.3 Kriteria Keberhasilan Sistem
Kriteria keberhasilan didefinisikan pada tiga tingkat:
1. **Keberhasilan teknis pipeline**: seluruh kontrak data-graf-target valid, artefak utama terbentuk, dan pipeline berjalan end-to-end tanpa mismatch struktural.
2. **Keberhasilan komparatif model**: model utama dinilai terhadap baseline pada metrik deterministik dan probabilistik, bukan hanya satu metrik tunggal.
3. **Keberhasilan operasional nowcasting**: evaluasi dilakukan dengan protokol one-step rolling menggunakan observasi terbaru, sehingga hasil relevan untuk deployment real-time.

Secara metodologis, proyek ini tidak mengasumsikan satu model selalu unggul di semua variabel. Keberhasilan dianalisis per target (precipitation, wind, humidity) dan per jenis metrik (nilai titik vs event/probabilistik), karena karakter statistik tiap variabel berbeda.

---

Dokumentasi ini disusun dari implementasi aktif proyek pada direktori `src/`, `run_eval_final.py`, dan artefak validasi di `docs/` serta `result_test/`. Jika konfigurasi eksperimen berubah (misalnya batas split, nilai `seq_len`, atau threshold rain gate), bagian metode harus diperbarui agar tetap konsisten dengan implementasi terbaru.

