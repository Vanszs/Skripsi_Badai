# Solusi di Penelitian Kamu — Penjelasan Ulang

## Masalah yang Dihadapi

Data curah hujan di pegunungan tropis punya tiga masalah utama:

1. **Banyak nilai nol** → model sulit belajar pola hujan.
2. **Sedikit nilai ekstrem** → model regresi cenderung mengabaikan kejadian berbahaya.
3. **Distribusi tidak normal** → metode statistik standar kurang cocok.

Untuk mengatasinya, penelitian ini menggunakan 4 teknik utama:

---

## 1. Log Transform pada Presipitasi

### Masalah

Curah hujan memiliki rentang yang sangat lebar:
- Minimum: 0 mm/jam
- Maksimum: bisa 20–50 mm/jam atau lebih
- Banyak nilai nol.

Kalau langsung diproses, model akan sulit karena jarak antar nilai tidak seimbang.

### Solusi: Log Transform

Mengubah nilai presipitasi menjadi skala logaritmik.

Rumus:
```
transformed = log(1 + precipitation)
```

Atau ditulis:
```
log1p(x) = ln(1 + x)
```

### Contoh

| Presipitasi Asli (mm/jam) | Setelah Log Transform |
|---|---|
| 0 | 0 |
| 1 | 0.69 |
| 2 | 1.10 |
| 5 | 1.79 |
| 10 | 2.40 |
| 20 | 3.04 |

### Apa yang Terjadi?

- Nilai kecil tetap kecil.
- Nilai besar "ditekan" menjadi tidak terlalu besar.
- Rentang data menjadi lebih seimbang.

### Analogi

Seperti mengubah skala uang dari rupiah ke jutaan rupiah.
- Rp 1.000.000 → 1 juta
- Rp 10.000.000 → 10 juta
- Rp 100.000.000 → 100 juta

Dengan satuan jutaan, angka-angka lebih mudah dibandingkan dan diproses.

### Di Kode

```python
# Di src/data/temporal_loader.py dan src/train.py
precip_transformed = np.log1p(precipitation)
# Saat inferensi, dikembalikan ke skala asli
precip_original = np.expm1(precip_transformed)
```

---

## 2. Weighted Noise Loss

### Masalah

Diffusion model dilatih untuk memprediksi noise.
Loss standar adalah MSE (Mean Squared Error):
```
MSE = rata-rata dari (noise_pred - noise)^2
```

MSE menganggap semua sampel sama pentingnya. Akibatnya:
- Sampel dengan hujan nol (banyak) mendominasi.
- Sampel dengan hujan ekstrem (sedikit) kurang dipelajari.

### Solusi: Weighted Noise Loss

Memberi bobot lebih besar pada sampel yang nilai targetnya ekstrem.

Aturan di kode kamu:
```
Jika |target| > 1 standar deviasi → bobot 5x
Jika |target| > 3 standar deviasi → bobut 10x
```

### Contoh

| Sampel | Presipitasi | Status | Bobot |
|---|---|---|---|
| A | 0 mm/jam | Normal | 1x |
| B | 2 mm/jam | Sedikit besar | 5x |
| C | 15 mm/jam | Ekstrem | 10x |

### Apa yang Terjadi?

- Model "diperhatikan lebih" saat hujan ekstrem.
- Model tidak lagi mengabaikan kejadian langka.
- Prediksi menjadi lebih sensitif terhadap hujan deras.

### Analogi

Seperti ujian sekolah:
- Tanpa bobot: semua soal nilainya sama.
- Dengan bobot: soal-soal sulit diberi nilai lebih besar.
- Siswa jadi lebih serius belajar soal sulit.

### Di Kode

```python
# src/models/diffusion.py
error = (noise_pred - noise) ** 2
weights = torch.ones_like(error)
weights[target_reference.abs() > 1.0] = 5.0
weights[target_reference.abs() > 3.0] = 10.0
loss = (error * weights).mean()
```

---

## 3. Retrieval-Augmented Historical Analogs

### Masalah

Kejadian hujan ekstrem sangat langka dalam data pelatihan.
Model mungkin hanya melihat contoh ekstrem beberapa kali.

### Solusi: Retrieval

Sebelum memprediksi, model mencari kejadian serupa di masa lalu.

### Prosesnya

```
Kondisi saat ini (MAIN pada waktu t)
        ↓
Cari k kondisi paling mirip di data train
        ↓
Ambil outcome target mereka pada t+1
        ↓
Gunakan sebagai conditioning untuk diffusion model
```

### Contoh

Hari ini:
- Suhu: 22°C
- Kelembapan: 90%
- Angin: 10 m/s dari barat
- Presipitasi jam lalu: 5 mm

Model mencari di data train:
- 5 kondisi paling mirip dengan hari ini.
- Misalnya tahun 2015, 2018, 2020, 2022, 2023.
- Lihat berapa hujan yang terjadi 1 jam setelah kondisi tersebut.
- Rata-rata outcome tersebut jadi "petunjuk" untuk prediksi hari ini.

### Analogi

Seperti dokter yang menangani penyakit langka:
- Dokter belum pernah melihat kasus persis sama.
- Dokter mencari rekam medis pasien dengan gejala serupa.
- Dari pengalaman pasien-pasien itu, dokter memperkirakan kemungkinan hasil.

### Keunggulan

- Model punya "memori" kejadian ekstrem.
- Tidak perlu menunggu melihat banyak contoh ekstrem lagi.
- Prediksi lebih baik untuk rare events.

### Di Kode

```python
# src/retrieval/base.py
# Menggunakan FAISS IndexFlatL2 untuk k-NN
index = faiss.IndexFlatL2(dimension)
index.add(train_features)
distances, indices = index.search(query, k)
retrieved_outcomes = train_targets[indices]
```

---

## 4. Diffusion Model

### Masalah

Model deterministik hanya menghasilkan **satu prediksi**.
Padahal atmosfer bersifat tidak pasti — banyak kemungkinan bisa terjadi.

### Solusi: Diffusion Model

Diffusion model menghasilkan **banyak kemungkinan prediksi** (ensemble).

### Proses Singkat

**Forward Process (pelatihan):**
```
Data asli → ditambah noise sedikit demi sedikit → menjadi noise murni
```

**Reverse Process (prediksi):**
```
Noise murni → model hapus noise sedikit demi sedikit → menghasilkan data realistis
```

### Contoh Ensemble

Untuk satu kondisi input, model menghasilkan 30 sampel:

| Sampel | Prediksi Hujan (mm/jam) |
|---|---|
| 1 | 2.1 |
| 2 | 3.5 |
| 3 | 0.8 |
| ... | ... |
| 30 | 12.4 |

Dari ensemble ini, bisa dihitung:
- **Median**: prediksi titik.
- **Spread**: seberapa tidak pasti.
- **Probabilitas hujan > 10 mm**: berapa persen sampel melebihi threshold.

### Analogi

Seperti memprediksi hasil pertandingan bola:
- Model deterministik: "Skor akan 2-1."
- Model probabilistik: "Kemungkinan skor: 1-0 (20%), 2-1 (30%), 1-1 (25%), 0-2 (15%), dst."

Pendekatan probabilistik memberi gambaran risiko yang lebih lengkap.

### Di Kode

```python
# src/models/diffusion.py
# Sampling reverse diffusion
x = torch.randn((num_samples, num_targets))
for t in scheduler.timesteps:
    noise_pred = model(x, t, context, retrieved, graph_emb)
    x = scheduler.step(noise_pred, t, x).prev_sample
```

---

## Hubungan Keempat Teknik

```
Data hujan banyak nol + sedikit ekstrem
        ↓
Log transform: perkecil rentang nilai
        ↓
Weighted loss: perhatikan sampel ekstrem
        ↓
Retrieval: ingat kejadian serupa di masa lalu
        ↓
Diffusion: hasilkan banyak kemungkinan prediksi
        ↓
Nowcasting probabilistik yang lebih baik untuk hujan ekstrem
```

---

## Kesimpulan

| Teknik | Fungsi Utama |
|---|---|
| **Log transform** | Meratakan rentang nilai presipitasi |
| **Weighted loss** | Membuat model lebih fokus pada kejadian ekstrem |
| **Retrieval** | Memberikan memori historis untuk situasi serupa |
| **Diffusion** | Menghasilkan distribusi probabilitas, bukan satu angka |

Keempatnya bekerja sama untuk mengatasi karakteristik heavy-tailed pada presipitasi orografis di pegunungan tropis.
