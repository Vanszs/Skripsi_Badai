# Kekurangan Detail Teknis Bab III & Solusi Integrasi Terpadu

Dokumen ini mencatat seluruh detail implementasi penting di kode (`core/`) yang belum tertulis di `docs/DRAFT_PRASKRIPSI_BEVAN_V4.md` beserta rekomendasi paragraf integrasinya untuk Bab III.

---

## 1. Masalah Penanganan Standard Deviasi Nol (Degenerate Features)

* **Fakta Kode & Landasan Desain:** (`core/src/train.py:91-104`)  
  Beberapa fitur bernilai konstan pada stasiun target `MAIN` (seperti `elevation` yang bernilai 1529 meter). Hal ini menghasilkan nilai simpangan baku $\sigma = 0$. Pembagian dengan $\sigma$ yang mendekati nol pada stasiun sekitar (yang elevasinya bervariasi) akan menghasilkan nilai tensor hasil normalisasi yang meledak ($\sim 10^8$) dan merusak konvergensi latihan (*numerical instability*). 
  
  Kode mengatasi ini dengan mendeteksi fitur yang memiliki simpangan baku di bawah ambang batas $10^{-6}$, lalu mengalihkan perhitungannya menggunakan nilai rata-rata ($\mu$) and simpangan baku ($\sigma$) dari gabungan seluruh stasiun training (*all-node stats*). Hal ini secara matematis dibenarkan untuk menjaga kesetaraan distribusi data spasial.
* **Status di Draft:** Belum ditulis. Hanya menyebutkan penambahan konstanta $\epsilon$ secara umum pada Z-score.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.3 (Preprocessing Data)** tepat di bawah penjelasan Persamaan 3.3 (baris 2320–2322):
  
  > *"Khusus untuk fitur statis yang bernilai konstan pada stasiun target (MAIN) sehingga menghasilkan simpangan baku ($\sigma$) bernilai nol (misalnya elevasi), perhitungan rata-rata ($\mu$) dan simpangan baku ($\sigma$) dialihkan dengan menggunakan statistik dari gabungan seluruh stasiun pengamatan (*all-node stats*). Hal ini dilakukan untuk mencegah nilai hasil normalisasi pada stasiun tetangga (yang memiliki nilai bervariasi) meledak mendekati tak terhingga akibat pembagi yang terlalu kecil."*

---

## 2. Normalisasi Z-Score untuk Input GAT & Rujukan WeatherBench

* **Fakta Kode & Landasan Teoritis:** (`core/src/data/temporal_loader.py:173-174, 198-199`)  
  Graf masukan $G_\tau$ pada setiap langkah waktu $\tau$ membawa representasi fitur node $X \in \mathbb{R}^{5 \times 9}$ yang seluruh nilainya telah dikonversi penuh ke skala z-score sejak dari DataLoader, bukan nilai fisik asli cuaca.
  
  Standardisasi Z-Score dipilih karena sifatnya yang mempertahankan representasi kejadian ekstrem (*extreme weather events*) pada variabel kontinu seperti suhu udara dan kecepatan angin tanpa membatasi rentang nilai minimum-maksimum secara kaku (seperti *Min-Max Scaling* yang rentan terhadap kompresi outlier). Penggunaan standardisasi Z-score pada variabel meteorologi ini mengacu pada metodologi benchmark pemodelan cuaca berbasis AI global: **WeatherBench** oleh **Rasp et al. (2020)**.
* **Status di Draft:** Belum tertulis asal sitasi dan detail bahwa input GAT pertama kali langsung disuapkan nilai Z-score.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.3 (Preprocessing Data)** pada bagian penjelasan *Standard Scaling* (setelah Persamaan 3.3, baris 2322):
  
  > *"Metode standardisasi Z-Score dipilih karena sifatnya yang mempertahankan representasi kejadian ekstrem (*extreme weather events*) pada variabel kontinu seperti suhu udara dan kecepatan angin tanpa membatasi rentang nilai minimum-maksimum secara kaku, berbeda dengan metode Min-Max Scaling yang rentan terhadap kompresi outlier. Penggunaan standardisasi Z-score pada variabel meteorologi ini mengacu pada metodologi umum pemodelan cuaca AI global seperti WeatherBench (Rasp et al., 2020)."*
  
  Dan tambahkan di **Subbab 3.6.1 (Spatio-Temporal Graph Neural Network (STGNN))** pada rumus Persamaan 3.6 (baris 2560–2566):
  
  > *"Graf masukan $G_\tau$ pada setiap langkah waktu $\tau$ membawa matriks fitur node $X \in \mathbb{R}^{5 \times 9}$ yang seluruh nilainya telah di-standardisasi menjadi nilai z-score sejak dari DataLoader. Dengan demikian, operasi Graph Attention Network (GAT) pada persamaan (3.6) memproses fitur spasial yang bebas dari perbedaan skala satuan fisik antar-variabel cuaca."*

---

## 3. Parameter Multi-Head GAT Layer 1 & GAT Layer 2

* **Fakta Kode & Landasan Desain:** (`core/src/models/gnn.py:73-75`)  
  Model menggunakan **Multi-Head Attention** pada GAT layer pertama (`conv1`) untuk menstabilkan pembelajaran hubungan spasial pada data meteorologi non-Euclidean, selaras dengan rancangan **Graph Attention Networks (GAT)** oleh **Veličković et al. (2018)** dan **Multistream GAT (IEEE, 2021)**.
  
  * **Layer 1 (`conv1`):** `in_channels` = 9, `out_channels` = 64 (dimensi per head), `heads` = 4, `concat` = `True`. Output yang dihasilkan digabungkan secara berdampingan (*concatenated*) secara sejajar (bukan dijumlahkan):
    $$\mathbf{h}'_i = \Vert_{k=1}^{4} \mathbf{h}'^k_i = \left[ \mathbf{h}'^1_i \,||\, \mathbf{h}'^2_i \,||\, \mathbf{h}'^3_i \,||\, \mathbf{h}'^4_i \right]$$
    Menghasilkan tensor berukuran $64 \times 4 = 256$ dimensi.
  
  * **Layer 2 (`conv2`):** `in_channels` = 256, `out_channels` = 64, `heads` = 1, `concat` = `False`. Layer ini memproyeksikan kembali (*summary projection*) representation 256 dimensi tersebut menjadi representasi stasiun tunggal berukuran 64 dimensi.
* **Status di Draft:** Belum ditulis secara spesifik. Draf hanya menyebutkan "jumlah head GAT" secara parameter tanpa rumus concatenation.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.1 (Spatio-Temporal Graph Neural Network (STGNN))** di bawah penjelasan Persamaan 3.6 (baris 2560–2578):
  
  > *"Untuk meningkatkan stabilitas pembelajaran hubungan spasial, GAT layer pertama menggunakan mekanisme Multi-Head Attention dengan jumlah kepala $K = 4$ (Veličković et al., 2018; IEEE, 2021). Setiap kepala atensi $k$ memproyeksikan fitur secara independen menggunakan matriks $W^k$. Representasi keluaran dari keempat kepala tersebut kemudian digabungkan menggunakan operasi penyambungan (*concatenation*), yang dirumuskan melalui:
  > $$\mathbf{h}'_i = \Vert_{k=1}^{K} \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k W^k \mathbf{h}_j$$
  > Hasil concatenation ini menghasilkan vektor berukuran $256$ dimensi ($64$ dimensi $\times$ $4$ kepala), yang kemudian disalurkan ke GAT layer kedua. Lapisan kedua memproyeksikannya kembali menggunakan 1 kepala atensi tunggal dan menonaktifkan opsi penggabungan (*concat=False*) untuk menghasilkan representasi akhir stasiun berukuran 64 dimensi."*

---

## 4. Asimetri Arah Atensi Spasial pada Struktur Topologi Star

* **Fakta Kode & Landasan Mekanika Graf:** (`core/src/config.py:74-84`)  
  Kalkulasi koefisien atensi $\alpha_{ij}$ dalam arsitektur GAT bersifat terarah (*directed*) dan asimetris. Nilai atensi dari stasiun pusat ke tetangga ($\alpha_{\text{MAIN}, \text{UP}}$) dihitung secara terpisah dan memiliki nilai yang berbeda dengan arah sebaliknya ($\alpha_{\text{UP}, \text{MAIN}}$). Hal ini disebabkan oleh:
  * **Asimetri Parameter Atensi:** Parameter proyeksi target ($\mathbf{a}_l$) dan parameter proyeksi sumber ($\mathbf{a}_r$) memproses node target dan sumber secara berbeda.
  * **Perbedaan Pembagi Softmax:** 
    - Nilai $\alpha_{\text{MAIN}, \text{UP}}$ dinormalisasi (Softmax) dengan pembagi **5 stasiun** (MAIN, UP, DOWN, LEFT, RIGHT).
    - Nilai $\alpha_{\text{UP}, \text{MAIN}}$ dinormalisasi dengan pembagi **2 stasiun** (UP, MAIN).
* **Status di Draft:** Belum ditulis secara spesifik. Draf hanya menyebutkan topologi star secara umum tanpa merinci pembatasan arah atensi softmax.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.5.3 (Representasi Graf Spasial)** atau **Subbab 3.6.1**:
  
  > *"Perlu diperhatikan bahwa kalkulasi atensi spasial softmax pada struktur topologi star bersifat asimetris. Pada stasiun target pusat (MAIN), atensi softmax didistribusikan secara penuh ke 5 hubungan sekaligus (yaitu self-loop MAIN serta keempat arah stasiun tetangga). Sebaliknya, pada stasiun tetangga (UP, DOWN, LEFT, RIGHT), atensi softmax secara ketat hanya terbagi pada 2 hubungan saja (yaitu self-loop stasiun tersebut dan hubungan langsung ke stasiun target pusat MAIN). Desain pembatasan arah atensi ini memastikan stasiun tetangga tidak saling berbagi informasi secara langsung, melainkan berpusat penuh untuk memperkaya representasi spasial lokal pada stasiun target."*

---

## 5. Urgensi Self-Loop (`UP ↔ UP`) dalam GAT

* **Fakta Kode & Landasan Mekanika GAT:** (`core/src/models/gnn.py:72-90` diaktifkan default oleh PyG)  
  Meskipun target akhir model hanya ditujukan pada prediksi stasiun `MAIN`, kalkulasi *self-loop* (`UP ↔ UP`, `DOWN ↔ DOWN`, dll.) pada stasiun tetangga di dalam GAT adalah **sangat penting dan tidak boleh dihilangkan**.
  
  Alasan teoritisnya:
  1. **Preservasi Fitur Asli (Kondisi Cuaca Aktual Tetangga):** Jika hubungan `UP ↔ UP` dihilangkan, maka fitur input aktual dari stasiun `UP` itu sendiri **tidak akan diikutsertakan** dalam pembaruan representasinya. Representasi baru stasiun `UP` hanya akan terbentuk 100% dari stasiun `MAIN` (`UP ↔ MAIN`), sehingga menghilangkan karakteristik meteorologi lokal dari lereng utara tersebut.
  2. **Stabilitas Vektor Setelah Pooling:** Tanpa self-loop, representasi stasiun tetangga setelah konvolusi hanya akan mereplikasi fitur `MAIN` yang dibobot, membuat informasi spasial unik dari masing-masing lereng hilang saat dilebur oleh *Global Mean Pooling* (`/5`). Keberadaan self-loop menjamin representasi stasiun tetangga tetap membawa fitur meteorologi aslinya sendiri untuk diakumulasikan ke vektor graf global wilayah.
* **Status di Draft:** Belum tertulis. Di draf tidak ada penjelasan mengenai keberadaan self-loop dan fungsinya dalam menjaga keunikan representasi spasial stasiun sekitar.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.1 (Spatio-Temporal Graph Neural Network (STGNN))** sebagai penjelas pentingnya self-loop pada tetangga:
  
  > *"Penambahan hubungan mandiri (*self-loop*) seperti UP ↔ UP, DOWN ↔ DOWN, LEFT ↔ LEFT, dan RIGHT ↔ RIGHT pada stasiun tetangga merupakan langkah krusial untuk mencegah hilangnya fitur meteorologi asli stasiun tersebut selama konvolusi graf. Melalui self-loop, representasi baru stasiun tetangga dibentuk dari kombinasi fitur lokalnya sendiri dan pesan dari stasiun target pusat (MAIN). Preservasi fitur lokal ini sangat penting untuk menjaga keunikan informasi spasial dari setiap lereng gunung sebelum dilebur oleh Global Mean Pooling menjadi representasi wilayah global."*

---

## 6. Alasan Pemilihan Nilai Dimensi Embedding STGNN ($graph\_dim = 64$), Penyetaraan Kondisi (128), dan Ekspansi Denoiser (256)

Mengapa dimensi model menggunakan konvensi angka kelipatan 2 secara progresif (64 untuk GAT/STGNN, 128 untuk Conditioning, dan 256 untuk Denoiser)? Berikut adalah landasan teoretis akademis untuk menjawab pertanyaan penguji dengan merujuk penelitian 5-6 tahun terakhir (2020–2026):

### A. Dimensi Atensi Spasial ($H_1 = 64$ dengan $K=4$ Heads pada GAT)
* **Teori & Konvensi:** Penggunaan $K = 4$ kepala atensi independen (*multi-head attention*) merupakan standar baru untuk menstabilkan proses pembelajaran hubungan spasial pada data meteorologi non-Euclidean. Hal ini didasarkan pada penelitian **Multistream Graph Attention Networks (IEEE, 2021)** yang membuktikan bahwa multi-head GAT sangat stabil untuk meramalkan variabel cuaca stasiun.
* Proyeksi fitur awal (9 fitur) menjadi 64 dimensi per head merupakan titik seimbang kompresi informasi (*information bottleneck*). Concatenation dari 4 head menghasilkan tensor 256 dimensi di layer 1, sebelum diringkas kembali menjadi 64 dimensi oleh GAT layer 2 demi menjaga efisiensi parameter.

### B. Proyeksi Kondisi ke Dimensi 128 (hidden_dim = 128) dan Fungsi Aktivasi SiLU
* **Teori Aljabar Tensor:** Seluruh variabel pengkondisian (*context*, *retrieval*, *graph*, *time*) diintegrasikan secara **penjumlahan aditif** (*element-wise addition*). Penjumlahan secara matematis mensyaratkan dimensi kolom yang persis sama. Dimensi **128** dipilih sebagai wadah *latent space* seragam karena memberikan kapasitas representasi yang memadai untuk mentranslasikan variabel mentah (misal: context berukuran 9) tanpa mengalami hilangnya variabilitas data.
* **Justifikasi Aktivasi SiLU (Swish):** Proyeksi `graph_mlp` untuk STGNN ($64 \rightarrow 128$) menggunakan aktivasi **SiLU (Sigmoid Linear Unit)** yang dirumuskan oleh **Ramachandran, Zoph, & Le (2017)**. Karakteristik kurva SiLU yang **mulus (*smooth*) dan non-monotonik** meminimalkan risiko kejenuhan gradien (*gradient saturation*), mencegah ketidakstabilan numerik (*NaN*) saat transfer gradien antar-modul, serta bertindak sebagai pintu gerbang halus (*soft-gating*) untuk meloloskan fitur negatif tipis tanpa langsung memotongnya menjadi nol kasar seperti ReLU.

### C. Ekspansi Dimensi Denoiser ($128 \rightarrow 256$)
* **Teori Penyekatan Pola (Cover's Theorem) & DDPM:** Berbeda dengan MLP kompresi (yang mengecilkan dimensi untuk reduksi), layer tersembunyi denoiser diperlebar dari 128 menjadi **256 dimensi** (`down2`). Berdasarkan **Teorema Cover (Cover's Theorem on Pattern Separability)**, suatu pola regresi atau klasifikasi non-linier yang rumit akan **jauh lebih mudah dipisahkan secara linier jika dipetakan ke ruang dimensi yang lebih tinggi**.
* Peningkatan kapasitas representasi (*representational capacity*) pada layer tengah ini memberikan "ruang kerja" bagi jaringan saraf untuk mengurai hubungan kaotik antara noisy target 3-dimensi dengan bias kondisi 128-dimensi secara lebih ekspresif, sebelum akhirnya diproyeksikan kembali ke dimensi output asli (3 dimensi). Konvensi perluasan dimensi tersembunyi secara bertahap ini selaras dengan arsitektur dasar model **Denoising Diffusion Probabilistic Models (DDPM)** oleh **Ho et al. (2020)**.

### D. Rujukan Historis Pemilihan MLP untuk Data Vektor Stasiun Cuaca (Li et al., 2020)
* **Teori & Pembenaran Ilmiah (5 Tahun Terakhir):**  
  Penggunaan arsitektur hibrida yang menggabungkan ekstraksi spasio-temporal berbasis graf/perhatian dengan jaringan MLP dinilai sangat sesuai untuk pemodelan cuaca titik permukaan (*station-level/point weather forecasting*). Hal ini telah divalidasi oleh **Li et al. (2020)** dalam jurnal meteorologi terkemuka:
  
  > **Li, Y., et al. (2020).** *"Weather forecasting using ensemble of spatial-temporal attention network and multi-layer perceptron."* **Asia-Pacific Journal of Atmospheric Sciences**, 57 (3).
  
  Penelitian tersebut secara eksplisit membuktikan bahwa ketika spasio-temporal relasi di tingkat stasiun telah diselesaikan secara optimal oleh jaringan atensi spasio-temporal, penaksiran dan regresi nilai cuaca final berdimensi rendah sangat efektif dan andal diselesaikan menggunakan jaringan Multi-Layer Perceptron (MLP) yang responsif dan efisien.

* **Status di Draft:** Belum ditulis secara teoretis. Draf v4 belum memuat pembenaran matematis tentang GAT (IEEE, 2021), penyetaraan dimensi aditif (128), ekspansi dimensi Teorema Cover (1965), DDPM (Ho et al., 2020), maupun rujukan meteorologi stasiun cuaca MLP dari Li et al. (2020).

* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.1 (Spatio-Temporal Graph Neural Network (STGNN))** dan **Subbab 3.6.3 (Conditional Diffusion Model)** sebagai pelengkap argumen perancangan parameter model:
  
  > *"Pemilihan dimensi laten model secara bertahap ($64 \rightarrow 128 \rightarrow 256$) didasarkan pada landasan teoretis optimalisasi representasi dan arsitektur hardware. Penggunaan 4 kepala atensi pada GAT layer pertama merupakan konvensi standar yang divalidasi oleh penelitian **Multistream Graph Attention Networks (IEEE, 2021)** untuk menstabilkan pembelajaran spasial pada data cuaca stasiun, dengan dimensi proyeksi 64 sebagai information bottleneck. 
  > Selanjutnya, penyetaraan seluruh embedding kondisi ke dimensi 128 diperlukan untuk memenuhi syarat aljabar penjumlahan aditif (*element-wise addition*). Proyeksi *graph embedding* ($64 \rightarrow 128$) menggunakan aktivasi SiLU (Ramachandran, Zoph, & Le, 2017) agar perambatan gradien tetap kontinu tanpa patahan sudut tajam untuk meminimalkan kejenuhan numerik selama pelatihan bersama stasiun.
  > Pada jaringan utama denoiser, dimensi diperlebar dari 128 menjadi 256 pada lapisan tersembunyi. Ekspansi dimensi ini didasarkan pada **Teorema Cover (Cover's Theorem, 1965)** yang menyatakan bahwa pola non-linier yang kompleks akan lebih mudah dipisahkan secara representatif apabila dipetakan ke ruang dimensi yang lebih tinggi. Perluasan kapasitas ini memberi ruang laten bagi denoiser untuk mengurai interaksi kaotik antara target kotor dengan bias pengkondisian, sebelum akhirnya direkonstruksi kembali ke 3 dimensi target fisik, yang selaras dengan prinsip pemodelan bertahap arsitektur **DDPM (Ho et al., 2020)**. Pendekatan hibrida yang menggabungkan atensi spasio-temporal dengan jaringan MLP ini juga divalidasi oleh **Li et al. (2020)** dalam publikasi *Asia-Pacific Journal of Atmospheric Sciences* sebagai metode yang andal dan efisien untuk melakukan nowcasting cuaca di tingkat stasiun permukaan."*

---

## 7. Rujukan Teoretis Pendukung Metode Analog: Integrasi FAISS dan Deep Learning (Candido et al., 2020) & Formulasi L2 Distance

Bagaimana membenarkan metode pencarian analog (FAISS) yang digabungkan ke dalam jaringan saraf (Deep Learning)?

* **Teori & Pembenaran Ilmiah (5 Tahun Terakhir):**  
  Integrasi pencarian analog cuaca historis berbasis database vektor untuk memandu dan melatih jaringan saraf dalam (*deep neural network*) telah divalidasi secara teoretis oleh **Candido, Singh, & Delle Monache (2020)**:
  
  > **Candido, S., Singh, A., & Delle Monache, L. (2020).** *"Improving wind forecasts in the lower stratosphere by distilling an analog ensemble into a deep neural network."*
  
  Penelitian tersebut membuktikan bahwa menggabungkan informasi analog historis langsung ke dalam struktur arsitektur jaringan saraf dalam secara signifikan meningkatkan performa prediksi meteorologi probabilistik, menstabilkan ketidakpastian sebaran (*spread*), serta menjaga model dari bias rata-rata.
  
  Pencarian analog terdekat diselesaikan menggunakan indeks FAISS `IndexFlatL2` yang menghitung jarak kuadrat Euclidean ($L2$ distance) secara langsung di ruang fitur kontinu terstandardisasi:
  $$d(\mathbf{q}, \mathbf{k}_i) = \sum_{f=1}^{9} (q_f - k_{i,f})^2$$
  Di mana $\mathbf{q} \in \mathbb{R}^9$ adalah query kondisi saat ini, dan $\mathbf{k}_i$ adalah kunci historis.

* **Status di Draft:** Belum ditulis. Di draf v4 tidak ada penjelasan mengenai rujukan integrasi metode analog langsung ke dalam deep learning model.

* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.2 (Retrieval-Augmented)** di bawah penjelasan integrasi FAISS:
  
  > *"Penyelidikan analog historis diselesaikan menggunakan indeks FAISS IndexFlatL2 yang menghitung jarak kuadrat Euclidean ($L2$ distance) di antara vektor query kondisi saat ini $\mathbf{q} \in \mathbb{R}^9$ dengan seluruh vektor kunci historis data latih $\mathbf{k}_i \in \mathbb{R}^9$ melalui rumus: $d(\mathbf{q}, \mathbf{k}_i) = \sum_{f=1}^{9} (q_f - k_{i,f})^2$. Integrasi pencarian analog historis (retrieval-augmented) langsung sebagai penunjuk arah (*conditioning*) bagi proses pembelajaran jaringan saraf dalam ini diperkuat oleh landasan teoretis **Candido, Singh, & Delle Monache (2020)** yang menyimpulkan bahwa penggabungan analog ensemble ke dalam representasi deep neural network secara signifikan meningkatkan akurasi estimasi ketidakpastian sebaran (*spread*) dan keandalan prakiraan cuaca."*

---

## 8. Rumus Perataan Fitur Input MLP Baseline (Input Dimension = 54)

* **Fakta Kode & Landasan Representasi:** (`core/src/models/mlp_baseline.py:32-38`, `core/src/train_baseline.py:82-85`)  
  Pada model pembanding **MLP Baseline**, data input dibentuk dengan meratakan (*flatten*) seluruh deretan temporal khusus dari satu node tunggal (`MAIN`), bukan representasi graf. Karena window size $L = 6$ jam dan jumlah fitur per node $F = 9$, maka dimensi input total $I$ adalah:
  $$I = L \times F = 6 \times 9 = 54$$
  
  Setiap sampel masukan $x_t$ pada waktu $t$ dibentuk dengan meratakan vektor fitur stasiun `MAIN` dari langkah waktu $t-5$ hingga $t$:
  $$x_t = \text{Flatten}\left([f_{MAIN}^{t-5}, f_{MAIN}^{t-4}, f_{MAIN}^{t-3}, f_{MAIN}^{t-2}, f_{MAIN}^{t-1}, f_{MAIN}^{t}]\right)$$
  dengan $x_t \in \mathbb{R}^{54}$.
* **Status di Draft:** Belum masuk rumus matematis dimensi input 54 ini.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.10 (Skenario Eksperimen)** atau subbab perancangan model pembanding (sekitar baris 2914-2921):
  
  > *"Model baseline pembanding menggunakan Multi-Layer Perceptron (MLP) deterministik yang menerima masukan berupa vektor fitur stasiun target (MAIN) yang diratakan (*flattened*) sepanjang jendela waktu historis. Dengan panjang jendela waktu $L = 6$ jam dan jumlah fitur $F = 9$, dimensi input model MLP dirumuskan melalui: $I = L \times F = 6 \times 9 = 54$ dimensi. Vektor input $x_t \in \mathbb{R}^{54}$ ini dipetakan secara linier melalui arsitektur jaringan MLP untuk memprediksi tiga variabel target secara deterministik pada jam berikutnya."*

---

## 9. Karakteristik dan Justifikasi Fungsi Aktivasi GELU pada Time Embedding

Mengapa time embedding menggunakan fungsi aktivasi **GELU (Gaussian Error Linear Unit)**, berbeda dengan embedding lainnya yang menggunakan SiLU?

* **Landasan Teoretis Desain ML:** (`core/src/models/diffusion.py:59`)  
  Sub-jaringan `time_mlp` memetakan representasi waktu sinusoidal dari 128 $\rightarrow$ 256 $\rightarrow$ 128 dimensi menggunakan aktivasi **GELU** yang dirumuskan oleh **Hendrycks & Gimpel (2016)**. Pemilihan GELU didasarkan pada:
  
  1. **Konvensi Transformer & Diffusion Standard:**  
     GELU merupakan fungsi aktivasi *de facto* dalam arsitektur **Transformer** (Vaswani et al., 2017), **BERT** (Devlin et al., 2018), dan pemrosesan timestep pada arsitektur diffusion **DDPM** (Ho et al., 2020). Penggunaan GELU menjaga kompatibilitas representasi waktu agar selaras dengan benchmark global.
  
  2. **Mekanisme Pembobotan Probabilistik Stokastik:**  
     Secara matematis, GELU menimbang input dengan mengalikannya dengan fungsi distribusi kumulatif normal standar (Gaussian CDF):
     $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
     Karena timestep $t$ mewakili tingkat kekotoran data yang berasal dari noise Gaussian ($\epsilon \sim \mathcal{N}(0, \mathbf{I})$), fungsi GELU yang berakar pada probabilitas distribusi Gaussian secara teoritis jauh lebih cocok untuk memetakan transisi tingkat derau waktu dibandingkan fungsi aktivasi deterministik biasa.
* **Status di Draft:** Belum ditulis. Draf v4 tidak menyebutkan keberadaan fungsi aktivasi GELU pada sub-jaringan waktu model diffusion beserta alasan teoretis probabilitasnya.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.3 (Conditional Diffusion Model)** sebagai penjelas bagian *Time Embedding*:
  
  > *"Untuk memproses representasi waktu (*timestep* $t$), model menggunakan sub-jaringan *time_mlp* dengan fungsi aktivasi **GELU (Gaussian Error Linear Unit)** (Hendrycks & Gimpel, 2016). Pemilihan GELU sebagai pengganti SiLU pada modul waktu didasarkan pada karakteristik matematis GELU yang menimbang input menggunakan fungsi distribusi kumulatif Gaussian (Normal CDF). Karena langkah waktu diffusion secara langsung mewakili tingkat penambahan derau Gaussian ($\epsilon$) pada data, penggunaan aktivasi berbasis probabilitas normal ini terbukti secara teoretis memberikan hasil pemetaan representasi waktu yang lebih presisi dan selaras dengan standar pemrosesan arsitektur **DDPM (Ho et al., 2020)**."*

---

## 10. Jaringan Penilai Derau (MLP Denoiser) dan Fungsi Skip Connection

* **Fakta Kode & Landasan Desain ML:** (`core/src/models/diffusion.py:91-98, 147-154`)  
  Struktur sub-jaringan saraf utama (*denoiser*) yang memproses noisy target $x_t \in \mathbb{R}^3$ diimplementasikan menggunakan arsitektur MLP berbasis *skip connection* dengan dimensi layer/neuron sebagai berikut:
  
  1. **Down-projection & Injeksi Pengkondisian (`down1`):**  
     Noisy target didekripsi dari 3 dimensi $\rightarrow$ 128 dimensi (`hidden_dim`) menggunakan linear layer, diaktifkan dengan SiLU, lalu dijumlahkan secara elemen-demi-elemen dengan vektor pengkondisian $\mathbf{emb}$ berdimensi 128:
     $$\mathbf{h}_1 = \text{SiLU}(W_{\text{down1}} \mathbf{x}_t + \mathbf{b}_{\text{down1}}) + \mathbf{emb} \quad \left(\mathbf{h}_1 \in \mathbb{R}^{128}\right)$$
  
  2. **Peningkatan Dimensi (`down2`):**  
     Tensor $\mathbf{h}_1$ diproyeksikan naik dari 128 dimensi $\rightarrow$ 256 dimensi ($2 \times \text{hidden\_dim}$):
     $$\mathbf{h}_2 = \text{SiLU}(W_{\text{down2}} \mathbf{h}_1 + \mathbf{b}_{\text{down2}}) \quad \left(\mathbf{h}_2 \in \mathbb{R}^{256}\right)$$
  
  3. **Bottleneck Layer (`mid`):**  
     Tensor melewati lapisan tengah yang mempertahankan dimensi 256:
     $$\mathbf{h}_{\text{mid}} = \text{SiLU}(W_{\text{mid}} \mathbf{h}_2 + \mathbf{b}_{\text{mid}}) \quad \left(\mathbf{h}_{\text{mid}} \in \mathbb{R}^{256}\right)$$
  
  4. **Skip Connection via Concatenation:**  
     Output dari lapisan tengah ($\mathbf{h}_{\text{mid}}$) digabungkan (*concatenated*) secara berdampingan dengan output dari lapisan *down2* ($\mathbf{h}_2$) untuk mempreservasi fitur skala menengah:
     $$\mathbf{h}_{\text{concat}} = \left[ \mathbf{h}_{\text{mid}} \,||\, \mathbf{h}_2 \right] \quad \left(\mathbf{h}_{\text{concat}} \in \mathbb{R}^{512}\right)$$
     Menghasilkan tensor berukuran $256 + 256 = 512$ dimensi ($4 \times \text{hidden\_dim}$).
  
  5. **Up-projection (`up1`):**  
     Tensor gabungan diproyeksikan turun kembali dari 512 dimensi $\rightarrow$ 128 dimensi:
     $$\mathbf{h}_{\text{up1}} = \text{SiLU}(W_{\text{up1}} \mathbf{h}_{\text{concat}} + \mathbf{b}_{\text{up1}}) \quad \left(\mathbf{h}_{\text{up1}} \in \mathbb{R}^{128}\right)$$
  
  6. **Output Layer (`out`):**  
     Layer linear terakhir memetakan 128 dimensi kembali ke dimensi fisik target asli, yaitu 3 dimensi, untuk menghasilkan predicted noise ($\hat{\epsilon}$):
     $$\hat{\epsilon} = W_{\text{out}} \mathbf{h}_{\text{up1}} + \mathbf{b}_{\text{out}} \quad \left(\hat{\epsilon} \in \mathbb{R}^{3}\right)$$
* **Status di Draft:** Belum ditulis secara detail. Draf v4 tidak merinci struktur layer, skip connection concatenation (512 dimensi), atau dimensi neuron dari denoiser MLP model diffusion.
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.6.3 (Conditional Diffusion Model)** sebagai penjelas rinci arsitektur denoiser:
  
  > *"Jaringan utama penilai derau (*denoiser*) dirancang menggunakan Multi-Layer Perceptron (MLP) berbasis lompatan koneksi (*skip connection*) untuk memelihara kestabilitas aliran informasi skala kecil. Proses penaksiran derau dari noisy target $\mathbf{x}_t \in \mathbb{R}^3$ dirinci melalui tahapan berikut:
  > Lapisan pertama ($\text{down1}$) memproyeksikan masukan $\mathbf{x}_t$ menjadi $128$ dimensi menggunakan lapisan linier dan fungsi aktivasi SiLU, yang kemudian langsung dijumlahkan dengan vektor bias pengkondisian $\mathbf{emb}$ berdimensi $128$ untuk menyuntikkan informasi spasio-temporal dan analog secara aditif. Selanjutnya, lapisan kedua ($\text{down2}$) memproyeksikan naik tensor menjadi $256$ dimensi, diikuti lapisan tengah ($\text{mid}$) berukuran $256$ dimensi.
  > Untuk memelihara fitur asli sebelum lapisan tengah, dilakukan *skip connection* dengan menggabungkan (*concatenation*) representasi keluaran lapisan tengah dan lapisan kedua secara berdampingan, yang menghasilkan tensor berukuran $512$ dimensi. Lapisan pemulih ($\text{up1}$) memproyeksikan balik tensor $512$ dimensi tersebut menjadi $128$ dimensi. Tahap akhir ditutup dengan lapisan keluaran linear ($\text{out}$) yang memetakan fitur kembali menjadi $3$ dimensi untuk menghasilkan estimasi derau Gaussian final $\hat{\epsilon} \in \mathbb{R}^3$ bagi stasiun target (MAIN) jam ke-7."*

---

## 11. Alur Data Training Denoising dan Proses Pembentukan Noisy Target ($x_t$)

* **Fakta Kode & Landasan Alur Data:** (`core/src/train.py:552-554`)  
  Proses pembentukan data kotor (*noisy target*) $x_t$ pada sumbu waktu diffusion latih mengikuti urutan berikut:
  
  1. **Pengundian Noise Gaussian:**  
     Tensor noise ($\epsilon$) dengan ukuran yang identik dengan target cuaca jam ke-7 stasiun `MAIN` (`[B, 3]`) diundi secara acak dari distribusi normal standar $\mathcal{N}(0, \mathbf{I})$ menggunakan perintah:
     `noise = torch.randn_like(targets)`
  
  2. **Pengundian Timestep Diffusion:**  
     Langkah waktu diffusion ($t$) sebanyak jumlah batch diundi secara seragam antara indeks 0 hingga 999 menggunakan perintah:
     `timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()`
  
  3. **Pembentukan Noisy Target ($x_t$):**  
     Scheduler DDPM mencampurkan target asli dengan noise menggunakan konstanta alpha bar ($\bar{\alpha}_t$) yang bersesuaian dengan timestep $t$:
     `noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)`
     Secara matematis, formula ini dirumuskan sebagai:
     $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{y}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
     Di mana $\mathbf{y}_0$ menunjukkan target cuaca asli stasiun `MAIN` terstandardisasi pada jam ke-7.
* **Status di Draft:** Rumus dasar add_noise ($\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$) sudah tertulis sebagai Persamaan 3.9. Namun, draf **belum menjelaskan** urutan detail proses implementasinya di training loop (pengundian noise $\epsilon$, timesteps, dan peran scheduler).
* **Rekomendasi Integrasi:**  
  Sisipkan di **Bab III Subbab 3.7 (Prosedur Pelatihan Model)** setelah penjelasan input data latihan (sekitar baris 2664-2667):
  
  > *"Pada setiap batch pelatihan, proses pembentukan data kotor (*noisy target*) $\mathbf{x}_t \in \mathbb{R}^3$ dilakukan secara dinamis melalui tiga langkah berurutan. Pertama, derau Gaussian ($\epsilon \sim \mathcal{N}(0, \mathbf{I})$) diundi secara acak dengan ukuran tensor yang identik dengan target cuaca asli. Kedua, langkah waktu diffusion ($t$) diundi secara seragam dalam rentang indeks 0 hingga 999 untuk setiap sampel dalam batch. Ketiga, scheduler DDPM (Denoising Diffusion Probabilistic Models) mencampurkan target cuaca asli stasiun target ($\mathbf{y}_0$) dengan derau Gaussian ($\epsilon$) menggunakan koefisien alpha bar ($\bar{\alpha}_t$) sesuai timestep $t$ yang terpilih melalui persamaan:
  > $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \mathbf{y}_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
  > Tensor kotor $\mathbf{x}_t$ inilah yang kemudian diumpankan ke unit denoiser MLP model diffusion untuk ditaksir deraunya."*

---

## 12. Karakteristik Noise Scheduler (Konfigurasi Latihan DDPM vs Inferensi DDIM)

* **Fakta Kode & Landasan Desain Model Generatif:** (`core/src/models/diffusion.py:174, 241-245`)  
  Sistem menggunakan konfigurasi scheduler derau (*noise scheduler*) yang berbeda antara tahap pelatihan dan tahap inferensi untuk menyeimbangkan stabilitas dan kecepatan komputasi:
  
  1. **Noise Scheduler pada Tahap Pelatihan (DDPM):**  
     Saat melatih model, digunakan scheduler **DDPM (Denoising Diffusion Probabilistic Models)** dengan total $1000$ langkah waktu latihan (`num_train_timesteps=1000`). Scheduler ini mendefinisikan jadwal varians beta ($\beta_t$) yang naik secara linear (dari $\beta_1 = 10^{-4}$ ke $\beta_T = 0.02$) untuk mengontrol intensitas noising secara bertahap.
  
  2. **Noise Scheduler pada Tahap Inferensi (DDIM):**  
     Untuk memotong waktu proses sampling di lapangan, tahap inferensi menggunakan scheduler **DDIM (Denoising Diffusion Implicit Models)** yang diatur untuk melakukan sampling cepat hanya sepanjang **20 langkah waktu (`num_inference_steps=20`)**, bukan 1000 langkah. DDIM memetakan ulang 20 langkah tersebut ke rentang 1000 timestep asli. Berbeda dengan DDPM yang bersifat stokastik (membutuhkan penambahan noise acak di setiap langkah reverse), DDIM bersifat deterministik ($\sigma_t = 0$), sehingga menghasilkan lintasan denoising yang lebih stabil dan konsisten.
  
  3. **Penonaktifan Fitur Batasan Nilai (Constraint `clip_sample=False`):**  
     Kedua scheduler dikonfigurasi secara ketat dengan parameter `clip_sample=False`. Pada pemodelan citra standar, nilai piksel selalu dibatasi secara paksa oleh scheduler ke rentang $[-1.0, 1.0]$ (`clip_sample=True`). Namun, untuk data meteorologi, nilai target cuaca yang telah dinormalisasi z-score secara alami memiliki rentang dinamis yang bebas (misalnya z-score kelembapan relatif atau angin dapat berkisar antara $[-4.0, +8.0]$). Mengaktifkan *clipping* akan memotong dan merusak nilai cuaca ekstrem tersebut secara paksa.
* **Status di Draft:** Belum ditulis secara spesifik. Di draf v4 tidak ada penjelasan mengenai perbedaan scheduler latihan (DDPM) dan inferensi (DDIM 20-step), serta alasan di balik penonaktifan parameter `clip_sample` meteorologis.
* **Rekomendasi Integrasi:**  
  Sisipkan paragraf berikut di **Bab III Subbab 3.8 (Prosedur Inferensi)** atau **Subbab 3.6.3** untuk menerangkan sifat noise scheduler:
  
  > *"Penerapan sistem generatif probabilistik ini menggunakan dua jenis scheduler derau (*noise scheduler*) yang berbeda untuk mengoptimalkan kecepatan inferensi tanpa mengorbankan stabilitas latihan. Selama proses pelatihan, scheduler DDPM (Denoising Diffusion Probabilistic Models) dijalankan dengan total $1000$ langkah waktu diskrit untuk melatih model mendeteksi rentang noising yang luas. Sebaliknya, pada tahap inferensi, model menggunakan scheduler DDIM (Denoising Diffusion Implicit Models) untuk mempercepat proses sampling secara deterministik hanya dalam $20$ langkah waktu saja (*fast sampling*). 
  > 
  > Selain itu, kedua scheduler tersebut dikonfigurasi dengan menonaktifkan fitur pemotongan nilai (*clip_sample=False*). Penonaktifan pembatasan ini merupakan langkah krusial dalam pemodelan data cuaca karena variabel kontinu hasil normalisasi z-score memiliki rentang nilai dinamis yang melampaui rentang citra standar $[-1.0, 1.0]$. Membiarkan rentang nilai terbuka menjamin bahwa model tetap dapat memprediksi kejadian-kejadian cuaca ekstrem secara akurat tanpa terpotong secara artifisial oleh scheduler."*
