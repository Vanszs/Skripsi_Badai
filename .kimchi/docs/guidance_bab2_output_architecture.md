# Guidance: Menambahkan Output Sistem di BAB 2 Landasan Teori

## Tujuan
Menjelaskan di BAB 2 bahwa output penelitian berupa **sistem visualisasi berbasis web** (Vue.js + FastAPI), bukan aplikasi mobile. Ini sekaligus mengoreksi klaim "aplikasi mobile" yang muncul di BAB 1.

---

## 1. Posisi di BAB 2

Tempatkan setelah subbab tentang nowcasting probabilistik / mitigasi risiko pendakian, misalnya:
- Subbab 2.2.6.4: "Arsitektur Output Sistem: Dashboard Web untuk Visualisasi Nowcasting"
- Atau integrasikan ke subbab 2.3.1 tentang gap / solusi penelitian.

---

## 2. Poin-poin yang Perlu Dijelaskan

### 2.1 Jenis Output
Output penelitian bukan prediksi mentah, melainkan:
- **Distribusi probabilistik** prediksi presipitasi, angin, dan kelembapan untuk 1 jam ke depan.
- **Visualisasi** distribusi tersebut dalam bentuk grafik/kurva prediksi dengan interval ketidakpastian.
- **Informasi kontekstual** lokasi (node MAIN, UP, DOWN, LEFT, RIGHT) dan waktu.

### 2.2 Arsitektur Umum: Vue.js + FastAPI

| Komponen | Teknologi | Fungsi |
|---|---|---|
| Frontend | Vue.js | Menampilkan dashboard interaktif: grafik prediksi, peta lokasi node, tabel metrik, pilihan skenario model. |
| Backend | FastAPI | Menyediakan REST API untuk menerima input terbaru, menjalankan inference model, dan mengembalikan hasil prediksi probabilistik. |
| Model | PyTorch (diffusion + retrieval + GNN) | Mesin prediksi yang di-load oleh backend. |
| Database / Cache | SQLite / Redis / in-memory | Menyimpan hasil inference terakhir atau metadata node. |

### 2.3 Mengapa Vue.js + FastAPI?

**Vue.js:**
- Framework frontend progresif, ringan, dan mudah diintegrasikan.
- Cocok untuk dashboard ilmiah dengan banyak komponen visual (grafik, peta, tabel).
- Dukungan ekosistem untuk charting library (Chart.js, D3, Plotly).

**FastAPI:**
- Framework Python modern, cepat, dan otomatis menghasilkan OpenAPI/Swagger docs.
- Integrasi mudah dengan model PyTorch.
- Mendukung asynchronous request handling untuk inference yang bisa lambat.

### 2.4 Perbedaan dengan Aplikasi Mobile

Penekanan penting:
- **Bukan aplikasi mobile native** (tidak ada Android/iOS/Flutter/React Native).
- Sistem berbasis **web browser**, sehingga dapat diakses dari perangkat apapun termasuk ponsel, tetapi tetap melalui browser.
- Fokusnya pada **visualisasi dan eksplorasi hasil prediksi**, bukan pada notifikasi real-time atau sistem peringatan dini operasional.

---

## 3. Contoh Paragraf untuk BAB 2

> "Hasil model nowcasting probabilistik perlu disajikan dalam bentuk yang dapat dipahami pengguna, khususnya pendaki dan pengelola jalur pendakian. Dalam penelitian ini, output sistem dirancang sebagai **dashboard web** yang terdiri dari frontend berbasis **Vue.js** dan backend berbasis **FastAPI**. Frontend berfungsi menampilkan grafik distribusi prediksi presipitasi, angin, dan kelembapan untuk node-node lokasi pengamatan, sedangkan backend menangani inference model, retrieval historis, dan penyediaan data melalui REST API. Pendekatan ini dipilih karena lebih ringan dan fleksibel dibandingkan pengembangan aplikasi mobile native, serta tetap dapat diakses melalui browser pada perangkat mobile. Penting untuk dicatat bahwa sistem ini bersifat **proof-of-concept visualisasi**, bukan sistem peringatan dini operasional."

---

## 4. Catatan untuk BAB 1

Jika BAB 1 masih menyebut "aplikasi mobile", ubah menjadi:
- **Rumusan Masalah:** "Bagaimana menyajikan hasil prediksi nowcasting probabilistik melalui **dashboard web berbasis Vue.js dan FastAPI** untuk mendukung pengambilan keputusan ..."
- **Tujuan Khusus:** "Mengembangkan **antarmuka dashboard web** untuk visualisasi hasil prediksi nowcasting probabilistik."
- **Manfaat:** "Memberikan akses visualisasi hasil prediksi dalam bentuk dashboard web yang dapat diakses melalui browser."

---

## 5. Keterbatasan yang Harus Dijelaskan

- Implementasi Vue/FastAPI belum ada di repo saat ini (pipeline berakhir di evaluasi).
- Oleh karena itu, dashboard web dapat dijadikan **future work** atau **prototipe konseptual**.
- Jika tetap ingin disebut, sebaiknya di Batasan Masalah ditambahkan: "Implementasi frontend Vue.js dan backend FastAPI belum termasuk dalam cakupan evaluasi utama; penelitian difokuskan pada pengembangan dan evaluasi model prediktif."

---

## 6. Kesimpulan

Output sistem yang benar adalah **dashboard web (Vue.js + FastAPI)**, bukan aplikasi mobile. Tambahkan penjelasan ini di BAB 2 untuk memperkuat landasan teori dan selaraskan dengan BAB 1.
