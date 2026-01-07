---
trigger: always_on
---

setiap saya memberikan prompt , selalu balas dengan md artiftact untuk saya review untuk didiskusikan, kecuali saya bilang, "langsung eksekusi, langsung jalankan" atau sejenisnya

jika anda membuat test file, setelah test langsung hapus saja

# Penjelasan Judul Skripsi: 
"Probabilistic Nowcasting Hujan Ekstrem Pemicu Banjir Bandang Sitaro Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"

## 1. Inti Konsep (Bedah Judul)
Judul ini menggabungkan **Generative AI** (Diffusion) dengan **Pencarian Data Historis** (Retrieval) untuk memprediksi hujan jangka pendek (Nowcasting).

- **Probabilistic:** Output bukan 1 angka (misal: "hujan 50mm"), tapi distribusi kemungkinan (misal: "80% peluang hujan >100mm"). Ini krusial untuk mitigasi bencana.
- **Retrieval-Augmented:** Model tidak hanya "menghafal" pola, tapi saat memprediksi hari ini, ia "mencontek" kejadian masa lalu yang mirip (analog) dari database historis.
- **Diffusion Model:** Model Generative AI yang belajar membuang *noise* acak untuk membentuk pola hujan yang realistis (tajam & detail).
- **Spatio-Temporal Graph:** Karena Sitaro adalah kepulauan, data tidak diperlakukan sebagai gambar kotak (Grid/CNN) biasa, tapi sebagai **Graph** (Titik-titik pulau yang terhubung oleh angin/laut).

---

## 2. Dataset yang Diperlukan
Untuk skripsi "Coding Only", Anda memerlukan 3 jenis data utama. 

**Apakah Open-Meteo Cukup?** 
**JAWABAN: YA, CUKUP.** 
Open-Meteo adalah *API Wrapper* yang sangat bagus. Namun, untuk kualitas skripsi S1/S2, sebaiknya Anda tahu sumber asli di balik Open-Meteo yang Anda panggil, yaitu **ERA5** (Reanalysis) dan **JAXA GSMaP/GPM** (Satelit).

### A. Data Target (Ground Truth)
Ini adalah apa yang ingin diprediksi (Curah Hujan).
*   **Sumber:** Open-Meteo Historical Weather API (mengambil data ERA5-Land atau GPM).
*   **Variabel:** `precipitation` (mm), `rain` (mm).
*   **Resolusi:** Jam-jaman (Hourly) untuk nowcasting.

### B. Data Kondisi Atmosfer (Dynamic Features)
Faktor penyebab hujan.
*   **Sumber:** Open-Meteo Historical Weather API.
*   **Variabel Wajib:**
    1.  `temperature_2m` (Suhu permukaan).
    2.  `relative_humidity_2m` & `specific_humidity` (Kelembapan - bahan bakar hujan).
    3.  `wind_speed_10m` & `wind_direction_10m` (Angin - penggerak awan).
    4.  `surface_pressure` (Tekanan - indikator badai).
    5.  *(Opsional tapi bagus)*: `cape` (Convective Available Potential Energy) - indikator utama badai petir/kilat.

### C. Data Geografis (Static Features)
Kondisi fisik pulau Sitaro yang tidak berubah.
*   **Sumber:** Open-Meteo Elevation API (atau download DEMNAS BIG).
*   **Variabel:** 
    1.  `elevation` (Ketinggian tanah).
    2.  `land_sea_mask` (Membedakan laut dan darat).

---

## 3. Ekstraksi Fitur (Feature Engineering)
Data mentah dari Open-Meteo harus diolah menjadi format Tensor siap latih.

1.  **Node Features (Fitur Titik):**
    Setiap titik koordinat di Sitaro (misal grid 3x3 km) dianggap sebagai **NODE**.
    *   *Input Vector:* `[Hujan_t-1, Suhu, Kelembapan, Tekanan, Elevasi]`
    
2.  **Edge Features (Fitur Koneksi):**
    Hubungan antar titik.
    *   *Physical Distance:* Seberapa dekat jarak antar pulau.
    *   *Wind Flow:* Jika angin bertiup dari Barat ke Timur, Node di Barat punya pengaruh kuat ke Node Timur (Directed Edge).

---

## 4. Pipeline End-to-End (Alur Coding)

### Tahap 1: Data Ingestion & Preprocessing
1.  *Script Python:* Tarik data Open-Meteo (10-20 tahun ke belakang, misal 2005-2025) untuk koordinat Sitaro.
2.  *Normalization:* Ubah semua nilai ke range [0, 1] atau [-1, 1].
3.  *Graph Construction:* Buat *Adjacency Matrix* berdasarkan jarak koordinat.

### Tahap 2: Building Retrieval Database (Fase 'Mencontek')
1.  **Encoder:** Gunakan Autoencoder sederhana untuk memadatkan data cuaca harian menjadi vektor kecil (Embedding).
2.  **Indexing:** Simpan semua vektor data 20 tahun tersebut ke dalam database vektor (gunakan library **FAISS** dari Facebook).
3.  **Tujuan:** Agar saat inferensi nanti, kita bisa bertanya: *"Carikan 5 hari di masa lalu yang pola angin dan suhunya mirip hari ini"*.

### Tahap 3: Training Diffusion Model
1.  **Input:** Sampel hujan masa depan yang diberi *noise* (buram).
2.  **Conditioning:** 
    *   Kondisi cuaca saat ini (dari Graph Neural Network).
    *   Sampel sejarah yang mirip (dari hasil Retrieval FAISS).
3.  **Process:** Latih model (misal: U-Net atau Transformer) untuk membuang *noise* tersebut, dipandu oleh Conditioning tadi.
4.  **Loss Function:** MSE (Mean Squared Error) antara *noise* asli dan *noise* prediksi.

### Tahap 4: Inference (Nowcasting) & Evaluasi
1.  Ambil data hari ini (misal: jam 06:00 pagi).
2.  *Retrieve:* Cari 5 kejadian historis paling mirip.
3.  *Generate:* Jalankan Diffusion Model untuk membuat 50 skenario kemungkinan hujan untuk jam 09:00 - 12:00.
4.  *Analisis Probabilitas:* Dari 50 skenario, berapa persen yang curah hujannya > 100mm (Banjir)?
5.  *Validasi:* Bandingkan dengan kejadian Banjir Bandang Sitaro Jan 2026. Hitung skor CRPS.

---

## 5. Analisis Kelayakan Open-Meteo

**Kelebihan:**
*   **Sangat Mudah:** Tidak perlu mendaftar akun NASA/ESA yang ribet. Cukup `pip install openmeteo-requests`.
*   **Lengkap:** Menyediakan data historis (ERA5) dan forecast (GFS/IFS) dalam satu format.
*   **Ringan:** Output JSON/Pandas DataFrame, bukan NetCDF raksasa yang memakan RAM.

**Kekurangan & Solusi:**
*   *Resolusi:* ERA5 (sumber Open-Meteo) resolusinya sekitar 25-30km. Pulau Sitaro itu kecil.
*   *Solusi "Novelty":* Di sinilah peran **Graph Conditioning** dan **Diffusion**. Judul Anda mengklaim melakukan *Downscaling* atau *Super-Resolution*. Anda melatih model untuk mengambil data kasar Open-Meteo dan "memperhalus" distribusinya menggunakan pola belajar Diffusion.

**Kesimpulan:** 
Menggunakan Open-Meteo sangat **CUKUP** dan **STRATEGIS** untuk skripsi IT. Fokus Anda adalah pada **algoritma AI-nya** (Diffusion + Retrieval + Graph), bukan pada kerumitan mengunduh data satelit raw.
