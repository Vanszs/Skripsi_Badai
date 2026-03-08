Daftar Diagram
Diagram Diagram Alur Penelitian
Menjelaskan alur keseluruhan penelitian mulai dari pengambilan data ERA5, tahap pra-pemrosesan, pembentukan fitur dan graf spasio-temporal, pelatihan model, proses inferensi, hingga tahap evaluasi model.

Diagram Skema Pembagian Dataset Temporal
Menjelaskan pembagian dataset berdasarkan waktu menjadi data pelatihan (2005–2018), validasi (2019–2021), dan pengujian (2022–2025) untuk menghindari kebocoran informasi temporal.

Diagram Struktur Graf Spasio-Temporal Antar Node
Menjelaskan bagaimana tiga lokasi elevasi direpresentasikan sebagai node dalam graf fully-connected, serta bagaimana informasi meteorologis antar node dipertukarkan melalui mekanisme Graph Attention.

Diagram Mekanisme Graph Attention Network (GAT)
Menjelaskan proses attention pada Graph Attention Network yang digunakan untuk mempelajari tingkat pengaruh antar node berdasarkan kesamaan fitur meteorologis.

Diagram Mekanisme Retrieval Historical Analogs
Menjelaskan proses pencarian kondisi meteorologis historis yang paling mirip menggunakan indeks FAISS dan bagaimana hasil retrieval digunakan sebagai konteks tambahan dalam model.

Diagram Arsitektur Conditional Diffusion Model
Menjelaskan struktur utama model probabilistik yang digunakan, termasuk proses forward diffusion, reverse denoising, serta integrasi informasi dari graf spasio-temporal dan modul retrieval.

Diagram Proses Pembangkitan Ensemble pada Tahap Inferensi
Menjelaskan bagaimana model diffusion menghasilkan beberapa sampel prediksi melalui proses sampling noise dan bagaimana sampel tersebut membentuk distribusi probabilistik prediksi.

Diagram Kerangka Evaluasi dan Analisis Trade-Off Model
Menjelaskan hubungan antara evaluasi deterministik, probabilistik, serta analisis threshold dan spike events dalam menilai trade-off antara akurasi rata-rata dan sensitivitas terhadap kejadian ekstrem.




Daftar Tabel
Tabel Lokasi Node Penelitian
Menunjukkan koordinat geografis dan elevasi dari setiap node yang digunakan dalam penelitian.


Tabel Variabel Meteorologis yang Digunakan
Menunjukkan daftar variabel meteorologis yang digunakan dalam penelitian beserta satuannya.

Tabel Fitur Input dan Variabel Target Model
Menunjukkan pembagian antara fitur input (conditioning features) dan variabel target yang diprediksi oleh model.

Tabel Konfigurasi Pra-pemrosesan Data
Menunjukkan metode transformasi dan normalisasi yang diterapkan pada data sebelum digunakan dalam pelatihan model.


Tabel Konfigurasi Arsitektur Model
Menunjukkan parameter utama pada arsitektur model seperti jumlah layer GAT, jumlah attention heads, dimensi embedding, dan parameter diffusion.

Tabel Hyperparameter Pelatihan Model
Menunjukkan parameter pelatihan yang digunakan selama proses training.

Tabel 3.7 – Parameter Hybrid Persistence

Isi tabel:
Menunjukkan bobot kombinasi antara prediksi model dan nilai lag yang digunakan dalam skema hybrid persistence untuk masing-masing variabel.

Tabel Metrik Evaluasi Model
Menunjukkan metrik evaluasi yang digunakan dalam penelitian serta tujuan penggunaannya.


Tabel Ambang Intensitas untuk Evaluasi Threshold
Menunjukkan kategori intensitas hujan yang digunakan dalam evaluasi berbasis ambang untuk analisis deteksi kejadian ekstrem.