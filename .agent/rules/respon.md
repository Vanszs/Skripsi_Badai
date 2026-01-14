---
trigger: always_on
---


setiap saya memberikan prompt , selalu balas dengan md artiftact untuk saya review untuk didiskusikan, kecuali saya bilang, "langsung eksekusi, langsung jalankan" atau sejenisnya

jika anda membuat test file, setelah test langsung hapus saja

# Penjelasan Judul Skripsi: 
"Nowcasting Probabilistik Hujan Lebat untuk Keselamatan Pendaki di Gunung Gede–Pangrango Menggunakan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"

## 1. Inti Konsep (Bedah Judul)
Judul ini menggabungkan **Generative AI** (Diffusion) dengan **Pencarian Data Historis** (Retrieval) untuk memprediksi hujan jangka pendek (Nowcasting) secara **probabilistik** pada kawasan pegunungan yang kritis untuk pendaki.

- **Probabilistic:** Output bukan 1 angka tunggal (misal: "0.8 mm"), tapi distribusi kemungkinan (misal: "30% peluang hujan >2 mm, 10% peluang >5 mm"). Ini krusial untuk **keputusan naik/turun gunung** dan mitigasi hipotermia / longsor.
- **Retrieval-Augmented:** Model tidak hanya "menghafal" pola, tapi saat memprediksi kondisi hari ini, ia **mencari kembali (retrieve)** kejadian masa lalu yang polanya mirip (analog cuaca) dari database historis.
- **Diffusion Model:** Model Generative AI yang belajar menghilangkan *noise* acak untuk membentuk pola hujan yang realistis (tajam & konsisten dengan dinamika atmosfer).
- **Spatio-Temporal Graph:** Karena kawasan Gede–Pangrango terdiri dari **puncak, lereng, dan kaki gunung**, data diperlakukan sebagai **Graph** (titik-titik lokasi yang saling terhubung oleh aliran angin dan orografi), bukan grid kotak biasa.

Contoh node:
- Node 0: Puncak Gede–Pangrango (ketinggian ~2.950 m).
- Node 1: Pos pendakian (Cibodas / Gunung Putri, ketinggian ~1.300–1.500 m).
- Node 2: Hilir awal DAS (sekitar Cianjur / Puncak Pass).

---

## 2. Dataset yang Diperlukan

Untuk skripsi **“Coding Only”** dengan studi kasus Gede–Pangrango, Anda memerlukan 3 jenis data utama.

**Apakah Open-Meteo Cukup?**  
**JAWABAN: YA, CUKUP.**  
Open-Meteo adalah *API wrapper* yang sangat praktis. Di balik API ini terdapat sumber data utama seperti **ERA5(-Land)** (reanalysis) dan model numerik global. Untuk kualitas skripsi S1/S2, sebaiknya Anda paham bahwa:

- Anda mengambil data dari Open-Meteo,
- Tapi **secara ilmiah** merujuk ke ERA5 (dan/atau GFS/IFS) sebagai model dasar.

### A. Data Target (Ground Truth Operasional)
Ini adalah variabel yang ingin diprediksi (curah hujan jam-an).

* **Sumber:** Open-Meteo Historical Weather API (ERA5/ERA5-Land).
* **Variabel utama:**  
  - `precipitation` (mm) atau `rain` (mm) per jam.
* **Resolusi temporal:**  
  - **Hourly** (jam-jaman) untuk nowcasting 0–6 jam ke depan.
* **Lokasi:**  
  - Titik-titik di sekitar Gede–Pangrango (puncak, lereng, kaki gunung).

Jika memungkinkan, untuk **validasi tambahan**, bisa membandingkan dengan:
- Data BMKG terdekat (mis. Citeko Bogor / stasiun hujan Cianjur) sebagai “observasi lapangan”.

### B. Data Kondisi Atmosfer (Dynamic Features)
Faktor penyebab hujan di kawasan pegunungan.

* **Sumber:** Open-Meteo Historical Weather API.
* **Variabel wajib (per node):**
  1. `temperature_2m` — Suhu permukaan (penting untuk hipotermia & level freezing).
  2. `relative_humidity_2m` dan/atau `dewpoint_2m` — Kelembapan; bahan bakar awan dan kabut.
  3. `wind_speed_10m` & `wind_direction_10m` — Angin permukaan; aliran lembah–gunung dan adveksi uap air.
  4. `surface_pressure` — Tekanan; indikasi adanya sistem tekanan rendah / peralihan cuaca.
  5. `cloud_cover` — Tutupan awan; indikasi awan tebal / kabut.
  6. *(Opsional tapi sangat menarik)*: `cape` (Convective Available Potential Energy) — indikator potensi badai petir / hujan konvektif di lereng.

### C. Data Geografis (Static Features)
Kondisi fisik Gede–Pangrango yang tidak berubah dari waktu ke waktu.

* **Sumber:**
  - Open-Meteo Elevation API,  
  - atau DEMNAS BIG (Digital Elevation Model Nasional) jika ingin lebih detail.
* **Variabel:**
  1. `elevation` — Ketinggian (penting untuk hujan orografis & suhu).
  2. `land_sea_mask` — Tetap 1 (darat) untuk semua node di gunung, tapi tetap bisa disimpan untuk konsistensi pipeline.
  3. *(Opsional)* slope/kemiringan lereng, aspek lereng (dari DEMNAS) → menjelaskan perbedaan hujan antara sisi barat–timur.

---

## 3. Ekstraksi Fitur (Feature Engineering)

Data mentah dari Open-Meteo perlu diolah menjadi **tensor** siap latih dalam bentuk **sequence of graphs**.

1. **Node Features (Fitur per Lokasi / Node)**  
   Setiap node (puncak, pos pendakian, hilir) memiliki vektor fitur pada tiap jam \(t\):

   *Contoh input vector per node pada waktu t:*
   ```text
   [precipitation_{t-1}, precipitation_{t-3}, temperature_2m,
    relative_humidity_2m, surface_pressure,
    wind_speed_10m, wind_direction_10m,
    cloud_cover, elevation]
   ```

   - `precipitation_{t-1}, {t-3}` → fitur lag autoregresif (remembering recent rain).
   - `elevation` → fitur statis; sama di setiap timestep tapi beda antar node.

2. **Edge Features (Fitur Koneksi Antar Node)**  
   Hubungan antara puncak–lereng–hilir:

   - *Distance-based:* Jarak horizontal antar node (km).
   - *Orographic flow:* Arah angin relatif terhadap garis lembah–puncak.  
     Misalnya, jika angin bertiup dari kaki ke puncak, edge “hilir → puncak” bisa diberi bobot lebih tinggi.

   Dalam implementasi sederhana, Anda boleh memakai:
   - Graph fully-connected (semua node saling terhubung),
   - Edge weight = 1 / jarak atau hanya 1 (unweighted) jika ingin minimal.

---

## 4. Pipeline End-to-End (Alur Coding)

### Tahap 1: Data Ingestion & Preprocessing
1. **Script Python:**  
   - Tarik data Open-Meteo (2005–2025) untuk 3 koordinat (puncak–lereng–hilir).
2. **Cleaning & Transform:**
   - Isi missing value (jika ada),
   - Lakukan **log1p transform** pada curah hujan (untuk heavy-tail),
   - Normalisasi semua fitur dengan z-score.
3. **Graph Construction (per timestep):**
   - Buat graph dengan 3 node dan 6 edge (fully-connected),
   - Simpan sebagai objek `torch_geometric.data.Data` untuk tiap jam.

### Tahap 2: Building Retrieval Database (Fase “Mencontek”)
1. **Context Vector:**  
   Untuk setiap timestep \(t\), buat vektor konteks: mis. gabungan fitur node di waktu t-1 (atau agregat).
2. **Indexing dengan FAISS:**
   - Normalisasi context vector → bangun index vektor (FAISS `IndexFlatL2`).
   - Simpan seluruh 20 tahun data dalam index.
3. **Tujuan:**  
   Saat inferensi, model bisa bertanya:
   > “Carikan 3–5 jam/kejadian di masa lalu yang kondisi suhunya, anginnya, dan kelembapannya mirip dengan saat ini di Gede–Pangrango.”

### Tahap 3: Training Diffusion Model
1. **Input:**  
   - Target: curah hujan jam depan (atau beberapa jam ke depan) yang ditransform ke ruang log-norm.
   - Lalu ditambahkan *noise* sesuai skedul diffusion (DDPM/DDIM).
2. **Conditioning:**  
   - Embedding spatio-temporal dari GNN (menangkap pola hujan antara puncak–lereng–hilir).
   - Embedding retrieval (analog historis dari FAISS).
   - Embedding waktu (time-of-day, bulan/musim).
3. **Proses:**  
   Latih U-Net / MLP time-series untuk **memprediksi noise** yang ditambahkan.
4. **Loss Function:**  
   - MSE antara noise sejati vs noise yang diprediksi.
   - Bisa diperkuat dengan bobot lebih besar untuk sampel hujan lebat (>5 atau >10 mm).

### Tahap 4: Inference (Nowcasting) & Evaluasi
1. **Ambil kondisi saat ini** (misal jam 06:00 di Gunung Gede–Pangrango).
2. **Retrieve:**  
   Cari 3–5 analog historis paling mirip dari 20 tahun data.
3. **Sampling Diffusion:**  
   Jalankan proses sampling diffusion untuk menghasilkan **N=50 skenario** curah hujan jam 07:00–12:00 untuk semua node.
4. **Analisis Probabilitas:**
   - Hitung probabilitas hujan lebat di puncak:
     - \(P(R > 2~\text{mm/jam})\), \(P(R > 5~\text{mm/jam})\).
   - Terjemahkan ke *risk level* untuk pendaki (contoh: **Merah** jika \(P>0.7\)).
5. **Validasi:**  
   - Bandingkan dengan kejadian hujan lebat faktual (data BMKG / laporan SAR / berita kejadian pendaki terjebak badai).
   - Hitung metrik:
     - RMSE/MAE untuk nilai median,
     - **CRPS** untuk distribusi probabilistik,
     - **Brier Score, ROC-AUC, F1** untuk event \(R>5\) atau \(R>10\) mm/jam.

---

## 5. Analisis Kelayakan Open-Meteo untuk Gede–Pangrango

**Kelebihan:**
- **Akses sangat mudah:** Cukup pakai `openmeteo-requests`, tanpa akun rumit.
- **Konsisten 20 tahun:** Cocok untuk membuat basis analog historis besar.
- **Cakupan spasial:** ERA5 mencakup puncak dan sekitarnya, walau grid cukup kasar.

**Keterbatasan:**
- **Resolusi spasial (~25–30 km):**  
  Detil lokal di lereng tertentu (misalnya beda satu lembah) tidak tertangkap sepenuhnya.
- **Under-estimate puncak konvektif:**  
  Puncak hujan 20–30 mm/jam di lapangan bisa “turun” di ERA5 menjadi 5–15 mm/jam.

**Solusi “Novelty” dalam judul:**
- **Spatio-Temporal Graph Conditioning:**  
  Memanfaatkan variasi antara puncak–lereng–hilir untuk *memperbaiki pola spasial* hujan orografis.
- **Diffusion Model:**  
  Melatih model untuk menghasilkan distribusi hujan yang:
  - Lebih “tajam” saat kondisi sangat lembab/instabil,
  - Tetap konsisten dengan statistik jangka panjang ERA5.
- **Retrieval-Augmentation:**  
  Menggunakan analog historis untuk membantu model mengenali pola-pola spesifik Gede–Pangrango (misalnya kombinasi angin baratan + kelembapan tinggi yang sering memicu badai sore).

**Kesimpulan:**  
Menggunakan Open-Meteo (ERA5) **sangat layak** dan **strategis** untuk skripsi dengan judul ini. Fokus utama skripsi Anda bukan pada akuisisi data observasi yang rumit, tetapi pada:
