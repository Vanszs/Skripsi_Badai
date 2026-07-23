# Variabilitas Temporal dan Distribusi Heavy-Tailed

> Penjelasan lengkap untuk BAB 2.2.2

---

# 1. Variabilitas Temporal Presipitasi Orografis

## Definisi Sederhana

**Variabilitas temporal** = perubahan curah hujan seiring waktu.

Di satu lokasi, hujan tidak selalu sama setiap jam. Ada jam yang cerah, ada jam yang hujan deras, dan pola ini berubah-ubah sepanjang hari.

---

## Siklus Diurnal

### Apa itu Siklus Diurnal?

Pola hujan yang berulang setiap 24 jam. Biasanya:

| Waktu | Kondisi | Penjelasan |
|---|---|---|
| Pagi (06:00–10:00) | Relatif kering | Udara belum cukup panas |
| Siang (10:00–14:00) | Pemanasan maksimum | Lereng dipanaskan matahari |
| Sore (14:00–18:00) | **Puncak hujan** | Konveksi paling kuat |
| Malam (18:00–24:00) | Masih bisa hujan | Lepasnya ketidakstabilan siang |
| Dini hari (00:00–06:00) | Mulai menurun | Pendinginan permukaan |

### Kenapa Puncaknya Sore–Malam?

1. Siang hari: matahari memanaskan permukaan bumi dan lereng.
2. Udara di dekat permukaan menjadi panas.
3. Udara panas naik (konveksi).
4. Sore hari: konveksi mencapai puncaknya.
5. Uap air naik, mendingin, mengembun, menjadi awan.
6. Sore/malam: awan konvektif melepaskan hujan.

### Analogi Sederhana

Seperti merebus air:
- Api menyala terus sepanjang siang.
- Air mulai mendidih paling hebat di sore hari.
- Uap (awan/hujan) keluar paling banyak saat mendidih.

---

## Gangguan Mekanis Menggantikan Siklus Diurnal

### Kapan Terjadi?

Ketika angin lintas lereng sangat kuat, pola siang-malam bisa terganggu.

### Prosesnya

```
Angin lembab sangat kuat datang
        ↓
Menabrak lereng pegunungan
        ↓
Orographic lifting sangat kuat
        ↓
Hujan terjadi KAPAN SAJA, tidak harus sore
```

### Contoh

- Monsoon kuat datang pada pagi hari.
- Meskipun biasanya pagi kering, angin kuat bisa langsung memaksa hujan orografis.
- Hasil: hujan deras di pagi hari, meskipun siklus diurnal normalnya baru sore.

---

## Kenapa Variabilitas Temporal Penting?

### A. Nowcasting Per Jam Sangat Relevan

Karena hujan bisa muncul tiba-tiba kapan saja, prediksi per jam dibutuhkan untuk peringatan dini.

### B. Model Harus Tangkap Pola Jam-an

Dengan input 6 jam terakhir, model belajar:
- Apakah hujan sedang meningkat?
- Apakah siklus diurnal sedang berlangsung?
- Apakah ada gangguan mekanis yang menggantikan siklus?

### C. Heavy-Tailed Distribution

Perubahan curah hujan dari jam ke jam bisa sangat ekstrem: dari 0 mm/jam menjadi 20 mm/jam dalam satu jam.

---

## Ilustrasi Variabilitas Temporal

```
Curah Hujan (mm/jam)
   │
20 │                    ╱╲
   │                   ╱  ╲
15 │                  ╱    ╲
   │                 ╱      ╲
10 │                ╱        ╲
   │    ╱╲         ╱          ╲
 5 │   ╱  ╲       ╱            ╲    ╱╲
   │  ╱    ╲     ╱              ╲  ╱  ╲
 0 │_╱______╲___╱________________╲╱____╲____
   00  06  12  18  24  06  12  18  24
              ↑           ↑
          puncak sore   gangguan mekanis
                        (hujan pagi)
```

---

# 2. Distribusi Heavy-Tailed

## Definisi Sederhana

**Heavy-tailed distribution** = distribusi di mana nilai ekstrem jarang terjadi, tapi memiliki dampak sangat besar.

### Ciri Utama Presipitasi:

- **Banyak nilai nol** atau sangat kecil (tidak hujan).
- **Sedikit nilai sangat besar** (hujan deras).
- Bentuknya tidak seperti lonceng (tidak normal).

---

## Perbandingan dengan Distribusi Normal

### Distribusi Normal (Bell Curve)

```
     ╱╲
    ╱  ╲
   ╱    ╲
  ╱      ╲
 ╱        ╲
╱__________╲
```

- Nilai rata-rata paling sering.
- Nilai ekstrem sangat jarang.
- Contoh: tinggi badan manusia.

### Distribusi Presipitasi (Heavy-Tailed / Zero-Inflated)

```
│
│╲
│ ╲
│  ╲
│   ╲
│    ╲
│     ╲
│      ╲
│       ╲
│        ╲
│         ╲
│          ╲
│           ╲     ╱╲
│            ╲   ╱  ╲
│____________╲__╱____╲____
0          sedang    ekstrem
```

- Nilai nol sangat sering.
- Nilai sedikit banyak.
- Nilai ekstrem jarang tapi ada "ekor panjang" di kanan.

---

## Mengapa Presipitasi Heavy-Tailed?

### 1. Zero-Inflated

Sebagian besar jam dalam setahun tidak hujan.

### 2. Event Ekstrem Langka

Hujan deras (>10 mm/jam) hanya terjadi pada beberapa jam tertentu.

### 3. Dampak Besar

Satu jam hujan deras bisa menyebabkan:
- Banjir bandang.
- Tanah longsor.
- Risiko hipotermia mendadak.

---

## Quasi-Stationary System

### Apa itu?

Sistem hujan yang "bertahan" di satu lokasi cukup lama karena:
- Angin terus datang dari arah yang sama.
- Lereng terus memaksa udara naik.
- Hujan terus turun di lereng yang sama.

### Kenapa Penting?

- Hujan orografis bisa bertahan berjam-jam di lereng yang sama.
- Pendaki yang melewati lereng tersebut terpapar hujan lama.
- Risiko hipotermia meningkat karena pakaian basah terus-menerus.

---

## Kenapa Heavy-Tailed Penting untuk Model Kamu?

### A. Model Regresi Bias ke Nilai Rata-Rata

MSE loss menghukum error besar, sehingga model "takut" memprediksi nilai ekstrem. Akibatnya:
- Saat observasi = 0 mm/jam → prediksi = 0 (benar).
- Saat observasi = 20 mm/jam → prediksi = 5 mm/jam (salah besar).

### B. Solusi di Penelitian Kamu

1. **Log transform**: mengurangi skewness pada presipitasi.
2. **Weighted noise loss**: memberi bobot lebih besar pada sampel ekstrem.
3. **Retrieval-augmented**: mengingat kejadian ekstrem di masa lalu.
4. **Diffusion model**: menghasilkan ensemble yang bisa mencakup nilai ekstrem.

---

## Ilustrasi Heavy-Tailed dalam Data

```
Jam ke:  1   2   3   4   5   6   7   8   9   10
Hujan:   0   0   0   2   0   0   0   0  15   0
         ↑___________________↑       ↑
            80% tidak hujan        ekstrem langka
```

Dalam 10 jam, hanya 1 jam yang hujan ekstrem (15 mm/jam), tapi dampaknya besar.

---

# 3. Hubungan Variabilitas Spasial, Temporal, dan Heavy-Tailed

Ketiganya terkait erat:

```
Variabilitas Spasial
    ↓
Hujan tidak merata antar lereng
    ↓
Variabilitas Temporal
    ↓
Hujan berubah cepat per jam
    ↓
Heavy-Tailed Distribution
    ↓
Banyak tidak hujan, sedikit ekstrem
    ↓
Butuh Nowcasting Probabilistik + Graph + Retrieval
```

---

# 4. Kesimpulan

## Variabilitas Temporal

- Hujan di pegunungan mengikuti siklus diurnal (puncak sore).
- Bisa diganggu oleh forcing mekanis (angin kuat kapan saja).
- Nowcasting per jam penting karena perubahan cepat.

## Heavy-Tailed Distribution

- Banyak jam tidak hujan, sedikit jam hujan ekstrem.
- Model deterministik/regresi cenderung under-predict.
- Solusi: log transform, weighted loss, retrieval, diffusion.

---

# 5. Jawaban Sidang Singkat

> "Variabilitas temporal adalah perubahan curah hujan seiring waktu. Di pegunungan tropis, hujan biasanya memuncak pada sore hingga malam akibat siklus diurnal, tetapi angin lintas lereng yang kuat bisa menggantikan pola ini sehingga hujan terjadi kapan saja. Sementara itu, distribusi presipitasi bersifat heavy-tailed, artinya sebagian besar jam tidak hujan namun ada sedikit jam dengan hujan ekstrem yang berdampak besar. Kedua karakteristik ini membuat pendekatan deterministik kurang mumpuni, sehingga diperlukan nowcasting probabilistik dengan model diffusion, weighted loss, dan retrieval-augmented."
