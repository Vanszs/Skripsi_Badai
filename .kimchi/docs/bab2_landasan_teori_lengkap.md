# BAB 2 — Landasan Teori (Penjelasan Lengkap)

> Penjelasan ini disusun berdasarkan `Draft Bepan1.docx` dan disederhanakan agar mudah dipahami untuk sidang.

---

## 2.1 Konsep Nowcasting dan Skala Waktu Prediksi Atmosfer

### 2.1.1 Definisi dan Ruang Lingkup Nowcasting

**Apa itu nowcasting?**
Nowcasting adalah prakiraan cuaca untuk jangka waktu sangat pendek, biasanya 0–6 jam ke depan.

**Definisi menurut WMO:**
Prakiraan cuaca lokal pada skala mesoscale atau lebih kecil, mencakup kondisi saat ini hingga sekitar 6 jam ke depan.

**Tiga aspek utama nowcasting:**
1. **Horizon waktu pendek**: 0–6 jam.
2. **Resolusi spasial tinggi**: fokus pada area lokal.
3. **Detail lokal**: memperhatikan kondisi spesifik tempat, bukan skala besar.

**Dalam penelitian ini:**
- Prediksi fokus pada **1 jam ke depan** (one-step).
- Input: **6 jam observasi terakhir**.
- Sesuai definisi WMO, tapi dijalankan secara berulang setiap jam.

**Kenapa penting?**
Pendakian gunung butuh peringatan dini jangka pendek. Cuaca harian terlalu umum, sementara cuaca di pegunungan bisa berubah drastis dalam 1 jam.

---

### 2.1.2 Karakteristik Dinamika pada Skala Waktu Per Jam di Wilayah Pegunungan

**Skala waktu yang relevan:**
- **0–6 jam** = skala mesoscale dan convective scale.
- Pada skala ini, fenomena konvektif sangat sensitif terhadap kondisi lokal.

**Apa itu convective initiation?**
Pemicu awal pembentukan awan konvektif. Proses ini sangat dipengaruhi oleh:
- Kelembapan udara
- Suhu
- Dinamika angin lokal

**Lifetime sistem konvektif:**
- Sel konvektif tunggal: 30–60 menit.
- Sistem konveksi mesoskal: 3–6 jam atau lebih.

**Mengapa sulit diprediksi?**
Karena atmosfer bersifat **nonlinier** dan **kaotis**. Kesalahan kecil pada kondisi awal bisa tumbuh besar dengan cepat.

**Relevansi di Gunung Gede–Pangrango:**
- Gradien elevasi curam menyebabkan perbedaan cuaca ekstrem dalam jarak pendek.
- Lintasan pendakian dari hilir ke puncak melewati berbagai zona mikroklimat.
- Prediksi per jam menjadi sangat penting untuk keselamatan.

---

### 2.1.3 Keterbatasan Pendekatan Deterministik dan Relevansi Pendekatan Probabilistik

**Apa itu pendekatan deterministik?**
Model yang menghasilkan **satu nilai prediksi tunggal**, misalnya:
- Numerical Weather Prediction (NWP)
- Regresi machine learning

**Keterbatasannya:**
1. **Over-smoothing**: hasil prediksi terlalu halus, kehilangan detail ekstrem.
2. **Under-prediction**: cenderung memprediksi nilai rata-rata, sehingga meleset pada hujan deras.
3. **Tidak merepresentasikan ketidakpastian**: tidak memberi tahu seberapa yakin prediksinya.
4. **Update lambat**: NWP biasanya diperbarui beberapa jam sekali.

**Mengapa regresi ML under-predict?**
Karena fungsi kerugiannya biasanya **Mean Squared Error (MSE)**. MSE menghukum error besar, sehingga model "takut" memprediksi nilai ekstrem dan lebih memilih nilai aman di tengah.

**Relevansi pendekatan probabilistik:**
- Model probabilistik menghasilkan **distribusi kemungkinan**, bukan satu angka.
- Diffusion model termasuk pendekatan probabilistik yang mampu menghasilkan ensemble prediksi.
- Cocok untuk pengambilan keputusan berbasis risiko.

**Kenapa penting untuk penelitian?**
Ini adalah fondasi kenapa kamu tidak pakai regresi biasa, melainkan **diffusion model**.

---

## 2.2 Dinamika Presipitasi Orografis di Pegunungan Tropis

### 2.2.0 Pengantar Presipitasi Orografis

**Apa itu presipitasi orografis?**
Presipitasi (curah hujan) yang terbentuk karena pengaruh topografi pegunungan.

**Mekanisme sederhana:**
Angin lembab → bertemu pegunungan → terpaksa naik → mendingin → mengembun → hujan.

**Mengapa di pegunungan tropis?**
- Kelembapan udara tinggi.
- Atmosfer tidak stabil.
- Hujan bisa sangat intens dan lokal.

**Di Gunung Gede–Pangrango:**
Gradien elevasi tajam dari dataran rendah ke puncak menciptakan perbedaan kondisi cuaca ekstrem dalam jarak pendek.

---

### 2.2.1 Mekanisme Orographic Lifting dan Convective Initiation

#### 2.2.1.1 Angkat Paksa (Orographic Lifting)

**Apa itu?**
Ketika angin lembab bertemu dengan lereng gunung, udara tidak bisa menembus, jadi terpaksa naik ke atas.

**Analogi sederhana:**
Seperti kamu meniupkan uap ke kaca es — uap akan mengembun. Begitu juga udara lembab yang naik ke atas pegunungan.

**Apa yang terjadi saat naik?**
- Tekanan udara berkurang karena ketinggian meningkat.
- Udara mengembang dan mendingin.
- Jika mencapai titik embun, uap air berubah menjadi tetesan air → awan.

---

#### 2.2.1.2 Pendinginan Adiabatik dan Kondensasi

**Apa itu pendinginan adiabatik?**
Pendinginan yang terjadi karena udara mengembang saat naik, tanpa bertukar panas dengan lingkungan.

**Prosesnya:**
1. Udara naik.
2. Tekanan menurun.
3. Volume udara membesar.
4. Suhu turun.
5. Uap air jenuh → mengembun.
6. Awan terbentuk.

**Kondensasi:**
Perubahan uap air menjadi tetesan air cair. Inilah awal pembentukan awan.

---

#### 2.2.1.3 Convective Initiation (Pemicu Konvektif)

**Apa itu convective initiation?**
Proses awal terbentuknya awan konvektif yang bisa menghasilkan hujan deras.

**Kapan terjadi?**
Ketika udara yang naik mencapai **Level of Free Convection (LFC)** — ketinggian di mana udara menjadi cukup panas dan tidak stabil sehingga terus naik sendiri tanpa dipaksa.

**Ibaratkan:**
Seperti balon yang dilepaskan. Jika balon sudah cukup ringan, dia akan terus naik sendiri.

**Kenapa sulit diprediksi?**
Karena bergantung pada kondisi lokal yang sangat dinamis: kelembapan, suhu, dan angin.

---

#### 2.2.1.4 Peran Cross-Slope Wind

**Apa itu cross-slope wind?**
Angin yang bertiup melintang terhadap lereng gunung.

**Pengaruhnya:**
- Semakin kuat angin, semakin dalam penetrasi gangguan ke lereng.
- Meningkatkan lifting dan konvergensi.
- Bisa meningkatkan curah hujan hingga 20–30% per m/s peningkatan kecepatan angin.

**Kenapa penting?**
Angin dari arah tertentu bisa membawa hujan lebih deras ke titik MAIN, sehingga mempengaruhi risiko hipotermia.

---

#### 2.2.1.5 Forcing Mekanis dan Termal

**Forcing mekanis:**
Angkat paksa karena angin bertabrakan dengan pegunungan.

**Forcing termal:**
Pemanasan radiasi lereng di siang hari membuat udara di lereng menjadi lebih panas dan naik.

**Kombinasi keduanya:**
- Siang: forcing termal dominan.
- Sore/malam: forcing mekanis + lepasnya ketidakstabilan siang.
- Hasilnya: variabilitas spasial tinggi, hujan bisa terkonsentrasi di lereng tertentu.

**Kenapa penting untuk model?**
Model harus bisa menangkap kombinasi dua forcing ini, makanya digunakan representasi spasial multi-node dan temporal.

---

### 2.2.2 Variabilitas Spasial dan Temporal Presipitasi Orografis

#### 2.2.2.1 Variabilitas Spasial

**Apa itu variabilitas spasial?**
Perbedaan curah hujan antar lokasi dalam area yang berdekatan.

**Fenomena utama:**
- **Windward slope**: lereng yang menghadap angin → hujan deras.
- **Leeward slope**: lereng yang terlindung → kering (rain shadow).
- **Zona tengah lereng**: biasanya curah hujan maksimum.
- **Puncak**: bisa lebih kering karena udara sudah kehilangan kelembapan.

**Di Gunung Gede–Pangrango:**
- Lereng selatan dan barat sering menjadi windward terhadap aliran monsun.
- Rain shadow terjadi di sisi lindung.

**Kenapa penting?**
Makanya model butuh 5 node (MAIN, UP, DOWN, LEFT, RIGHT) untuk menangkap variasi dari berbagai arah.

---

#### 2.2.2.2 Variabilitas Temporal

**Apa itu variabilitas temporal?**
Perubahan curah hujan seiring waktu.

**Siklus diurnal:**
- Puncak hujan biasanya sore hingga malam.
- Penyebab: pemanasan siang → ketidakstabilan → konveksi sore.

**Gangguan mekanis:**
- Jika angin lintas lereng sangat kuat, siklus diurnal bisa tergantikan.
- Hujan bisa terjadi kapan saja dalam skala per jam.

**Kenapa penting?**
Nowcasting per jam sangat relevan karena hujan bisa muncul tiba-tiba di luar pola siang-malam.

---

#### 2.2.2.3 Distribusi Heavy-Tailed

**Apa itu heavy-tailed?**
Distribusi di mana kejadian ekstrem jarang terjadi, tapi dampaknya besar.

**Ciri presipitasi orografis:**
- Banyak jam tanpa hujan (nilai nol).
- Sedikit jam dengan hujan sangat deras.
- Tidak mengikuti distribusi normal (bell curve).

**Kenapa penting?**
- Model deterministik cenderung under-predict ekstrem.
- Diffusion model + weighted loss + retrieval membantu menangani heavy-tail.

---

### 2.2.3 Relevansi dengan Nowcasting Probabilistik dan Mitigasi Risiko

#### 2.2.3.1 Keterbatasan Model Deterministik untuk Presipitasi Orografis

Model deterministik sering gagal karena:
- Cuaca orografis sangat lokal dan cepat berubah.
- Pola hujan tidak rata dan tidak mengikuti distribusi normal.
- Smoothing efek menghilangkan detail ekstrem.

---

#### 2.2.3.2 Keunggulan Pendekatan Probabilistik

Model probabilistik mampu:
- Menghasilkan banyak kemungkinan skenario (ensemble).
- Memberikan informasi ketidakpastian.
- Lebih sensitif terhadap kejadian ekstrem.

---

#### 2.2.3.3 Peran Spatio-Temporal Graph Conditioning

Graph conditioning membantu model:
- Menangkap hubungan antar elevasi (hilir, lereng, puncak).
- Mempelajari propagasi presipitasi dari berbagai arah.
- Meningkatkan prediksi di titik MAIN melalui message passing.

---

#### 2.2.3.4 Hubungan dengan Mitigasi Risiko Hipotermia

Hujan orografis yang tiba-tiba + angin + kelembapan tinggi = risiko hipotermia.
Prediksi probabilistik memberi lead time bagi pendaki untuk:
- Mencari tempat berteduh.
- Mengganti pakaian basah.
- Memutuskan turun sebelum kondisi memburuk.

---

## 2.3 Model Generatif Probabilistik dengan Diffusion Models

### 2.3.1 Prinsip Kerja dan Keunggulan Diffusion Models untuk Nowcasting

#### 2.3.1.1 Dua Proses Diffusion

**Forward diffusion:**
- Data asli ditambahkan noise secara bertahap.
- Setelah banyak langkah, data menjadi noise murni.

**Reverse diffusion:**
- Model belajar menghilangkan noise secara bertahap.
- Dari noise murni, model merekonstruksi data yang realistis.

---

#### 2.3.1.2 Keunggulan Diffusion Model

- **Anti mode collapse**: tidak terjebak pada satu pola seperti GAN.
- **Sampel bervariasi**: bisa menghasilkan banyak prediksi realistis.
- **Berkualitas tinggi**: detail prediksi lebih tajam.
- **Mudah dikondisikan**: bisa menerima input observasi terkini.

---

#### 2.3.1.3 Diffusion untuk Nowcasting

- Atmosfer bersifat stokastik → cocok dengan sifat probabilistik diffusion.
- Bisa menghasilkan ensemble prediksi.
- Bisa dikondisikan pada fitur cuaca, retrieval, dan graph.

---

### 2.3.2 Keterbatasan Model Deterministik dan Relevansi Pendekatan Generatif

**Ulangi dari 2.1.3:**
Model deterministik cenderung rata-rata dan kurang sensitif terhadap ekstrem.

**Generatif lebih baik karena:**
- Memodelkan distribusi penuh, bukan hanya nilai tengah.
- Bisa menghasilkan skenario ekstrem yang plausible.
- Lebih fleksibel untuk pola kompleks di pegunungan tropis.

---

### 2.3.3 Relevansi dengan Retrieval-Augmented Diffusion Model

#### 2.3.3.1 Masalah Pure Diffusion

Diffusion model standar kesulitan dengan rare events karena data ekstrem langka.

#### 2.3.3.2 Solusi Retrieval

- Cari kondisi historis yang mirip dengan kondisi saat ini.
- Gunakan hasil historis tersebut sebagai conditioning.
- Model jadi punya "memori" kejadian serupa di masa lalu.

#### 2.3.3.3 Keunggulan Kombinasi

- Lebih sensitif terhadap ekstrem.
- Ensemble lebih terkalibrasi.
- Interpretabilitas lebih tinggi karena bisa melihat analog mana yang berpengaruh.

---

## 2.4 Retrieval-Based Historical Analogs

### 2.4.1 Prinsip Kerja Retrieval-Based Historical Analogs

#### 2.4.1.1 Konsep Dasar

- Simpan semua data historis dalam database.
- Saat ada kondisi saat ini, cari kondisi masa lalu yang paling mirip.
- Ambil beberapa tetangga terdekat (k-NN).

#### 2.4.1.2 Implementasi dengan FAISS

- FAISS = Facebook AI Similarity Search.
- Digunakan untuk pencarian k-NN yang cepat pada data besar.
- Metric yang dipakai: Euclidean L2 (bukan cosine).

#### 2.4.1.3 Cara Penggunaan dalam Model

- Kunci pencarian: fitur node MAIN saat ini.
- Yang diambil: **target outcome MAIN pada τ+1** dari data train.
- Hasil retrieval digunakan sebagai conditioning pada diffusion model.

**Catatan penting:**
Di draft tertulis "Euclidean atau cosine", padahal kode hanya Euclidean L2. Ini perlu diperbaiki.

---

### 2.4.2 Keunggulan Retrieval-Augmented dibandingkan Pure Generative Models

#### 2.4.2.1 Mengatasi Data Scarcity

Kejadian ekstrem langka. Dengan retrieval, model bisa belajar dari contoh historis yang serupa.

#### 2.4.2.2 Meningkatkan Kalibrasi

Ensemble prediksi lebih sesuai dengan distribusi nyata.

#### 2.4.2.3 Interpretabilitas

Kita bisa tahu kejadian masa lalu mana yang paling mirip dengan kondisi saat ini.

---

## 2.5 Spatio-Temporal Graph Conditioning pada Representasi Data Elevasi

### 2.5.1 Representasi Graph untuk Data Elevasi dengan Lima Node

#### 2.5.1.1 Lima Node

- **MAIN**: target prediksi utama.
- **UP**: node di utara MAIN.
- **DOWN**: node di selatan MAIN.
- **LEFT**: node di barat MAIN.
- **RIGHT**: node di timur MAIN.

Setiap node punya fitur meteorologis sendiri.

#### 2.5.1.2 Graph Attention Network (GAT)

- GAT adalah jenis GNN yang menggunakan mekanisme attention.
- Model belajar seberapa besar pengaruh setiap node tetangga terhadap MAIN.
- Bobot attention bersifat adaptif berdasarkan fitur node.

#### 2.5.1.3 Catatan Penting tentang Topologi

Draft mengklaim "fully-connected", tetapi kode sebenarnya pakai **star topology**.

**Star topology:**
- MAIN ↔ UP
- MAIN ↔ DOWN
- MAIN ↔ LEFT
- MAIN ↔ RIGHT
- Tetangga tidak saling terhubung.

Ini adalah salah satu anomali yang harus diperbaiki di draft.

---

### 2.5.2 Keunggulan Spatio-Temporal Graph dibandingkan Pendekatan Konvensional

#### 2.5.2.1 Keterbatasan CNN/LSTM/Transformer Murni

- CNN: asumsi hubungan spasial grid tetap.
- LSTM: hanya menangkap temporal, tidak eksplisit spasial.
- Transformer: mahal dan kurang efektif untuk graph non-Euclidean.

#### 2.5.2.2 Keunggulan Graph

- Message passing antar node secara eksplisit.
- Attention mechanism belajar bobot hubungan.
- Lebih cocok untuk hubungan multidirectional di pegunungan.

#### 2.5.2.3 Integrasi dengan Diffusion

Graph conditioning memberikan informasi spasial-temporal ke diffusion model, sehingga prediksi lebih konsisten secara spasial.

---

## 2.6 Risiko Hipotermia dan Mitigasi Pendakian

### 2.6.1 Mekanisme Hipotermia di Lingkungan Pegunungan Tropis

#### 2.6.1.1 Definisi Hipotermia

Suhu inti tubuh turun di bawah 35°C.

#### 2.6.1.2 Hipotermia Basah-Dingin

Di pegunungan tropis, suhu udara mungkin tidak sangat dingin, tetapi kombinasi hujan + angin + kelembapan tinggi mempercepat kehilangan panas.

#### 2.6.1.3 Tiga Mekanisme Kehilangan Panas

1. **Konduksi**: panas hilang ke pakaian dan kulit yang basah.
2. **Konveksi**: angin mempercepat kehilangan panas.
3. **Evaporasi**: penguapan air dari pakaian basah membuang panas.

#### 2.6.1.4 Gejala dan Bahaya

- Menggigil, kelelahan, penurunan koordinasi motorik.
- Bisa berkembang cepat menjadi kondisi mengancam nyawa.
- Sering diabaikan karena dikira kelelahan biasa.

---

### 2.6.2 Faktor Meteorologis Pemicu Risiko Hipotermia

#### 2.6.2.1 Tiga Faktor Utama

1. **Curah hujan tinggi**: membasahi pakaian.
2. **Kecepatan angin tinggi**: memperbesar wind chill.
3. **Kelembapan relatif tinggi**: menghambat penguapan keringat dan memperburuk sensasi dingin.

#### 2.6.2.2 Peran Node Sekitar

Kondisi di UP, DOWN, LEFT, RIGHT bisa memengaruhi kondisi di MAIN. Misalnya angin dari arah tertentu bisa membawa hujan lebih deras.

#### 2.6.2.3 Hubungan dengan Output Model

Model memprediksi 3 variabel target: precipitation, wind_speed_10m, relative_humidity_2m. Ketiganya berkaitan langsung dengan risiko hipotermia.

**Catatan:**
Suhu tidak dijadikan target. Ini perlu dijustifikasi lebih kuat karena suhu adalah faktor paling langsung terkait hipotermia.

---

## 2.7 Analisis Celah Penelitian

### 2.7.1 Celah pada Representasi Spasial dan Temporal

Pendekatan konvensional seperti CNN/LSTM belum optimal untuk ketergantungan spasial non-Euclidean di pegunungan. Representasi 5 node dengan MAIN sebagai target belum banyak dieksplorasi.

### 2.7.2 Celah pada Penanganan Kejadian Ekstrem dan Ketidakpastian

Banyak diffusion model kesulitan dengan distribusi heavy-tailed presipitasi ekstrem di lingkungan tropis. Penggunaan retrieval untuk nowcasting orografis masih jarang.

### 2.7.3 Celah pada Aplikasi Mitigasi Risiko Pendakian

Belum banyak penelitian yang menghubungkan output probabilistik model dengan indeks risiko hipotermia berbasis multi-faktor.

### 2.7.4 Kontribusi Penelitian Ini

Mengusulkan Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning menggunakan 5 node untuk nowcasting presipitasi di Gunung Gede–Pangrango, dengan orientasi mitigasi risiko hipotermia.

---

## Ringkasan Alur Logika

```
Nowcasting diperlukan karena cuaca pegunungan berubah cepat (2.1)
    ↓
Presipitasi orografis menyebabkan variabilitas tinggi dan heavy-tail (2.2)
    ↓
Model deterministik kurang mumpuni → Diffusion model (2.3)
    ↓
Diffusion butuh bantuan untuk rare events → Retrieval (2.4)
    ↓
Butuh representasi spasial antar elevasi → Graph (2.5)
    ↓
Tujuannya untuk mitigasi risiko hipotermia pendaki (2.6)
    ↓
Kombinasi ini mengisi celah penelitian yang ada (2.7)
```
