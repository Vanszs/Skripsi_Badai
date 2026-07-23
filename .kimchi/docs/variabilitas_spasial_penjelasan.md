# Variabilitas Spasial Presipitasi Orografis (Penjelasan Lengkap)

## Definisi Sederhana

**Variabilitas spasial** = perbedaan kondisi cuaca (terutama curah hujan) antar tempat yang jaraknya tidak jauh.

Di dataran rendah, hujan bisa relatif merata dalam radius beberapa kilometer.
Di pegunungan, hujan bisa **sangat tidak merata**: satu lereng hujan deras, lereng di sebelahnya cerah.

---

## Mengapa Terjadi di Pegunungan?

Karena ada **topografi** (bentuk permukaan bumi):

1. **Angin lembab** datang dari satu arah.
2. Angin tersebut **menabrak lereng**.
3. Lereng yang menghadap angin mendapat hujan banyak.
4. Lereng yang membelakangi angin menjadi kering.

---

## Fenomena Utama

### 1. Windward Slope (Lereng Angin)

**Definisi:** Lereng yang **menghadap ke arah datangnya angin**.

**Apa yang terjadi?**
- Angin lembab menabrak lereng.
- Udara terpaksa naik.
- Naik → mendingin → mengembun.
- Hujan deras.

**Analogi:**
Seperti kamu meniup uap ke kaca dingin — uap menempel dan berembun di sisi yang kamu tiup.

---

### 2. Leeward Slope (Lereng Lindung / Rain Shadow)

**Definisi:** Lereng yang **terlindung dari angin**, berada di belakang puncak.

**Apa yang terjadi?**
- Angin sudah kehilangan kelembapan saat melewati puncak.
- Udara turun ke lereng belakang.
- Turun → memanas → uap air tidak mengembun.
- Kondisi menjadi kering.

**Analogi:**
Seperti kamu berdiri di belakang gedung tinggi saat hujan — kamu tidak kehujanan karena tertutup gedung.

---

### 3. Zona Tengah Lereng (Maximum Precipitation Zone)

**Definisi:** Area di sepanjang lereng angin di mana curah hujan mencapai nilai **maksimum**.

**Kenapa di tengah?**
- Di bagian bawah lereng: udara belum cukup naik, belum banyak embun.
- Di zona tengah: udara sudah naik cukup tinggi, banyak kondensasi.
- Di dekat puncak: udara sudah kehilangan banyak uap air, hujan mulai berkurang.

---

### 4. Puncak

**Apa yang terjadi?**
- Udara yang naik sudah melepaskan kelembapan di lereng angin.
- Saat melewati puncak, udara menjadi lebih kering.
- Hujan di puncak bisa lebih ringan dibanding zona tengah lereng.

**Catatan:** Ini tidak selalu benar untuk semua gunung, tapi merupakan pola umum.

---

## Ilustrasi Cross-Section Gunung

```
Angin lembab datang dari barat →

         Zona Tengah Lereng
              ↓
         (hujan maksimum)
              ↓
            /‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾\         Leeward
           /    ↑↑↑↑↑↑↑↑↑    \        (kering)
          /   ↑↑  HUJAN  ↑↑    \
         /  ↑↑   DERAS    ↑↑    \
        / ↑↑                ↑↑   \
       /↑↑                  ↑↑    \
      /↑   WINDWARD          ↑     \
     /↑   (lereng angin)      ↑      \
    /___________________________________\

Legenda:
↑↑↑ = naiknya udara lembab
\   = lereng gunung
```

Penjelasan gambar:
- Angin lembab dari barat menabrak sisi kiri gunung.
- Sisi kiri = windward → hujan deras.
- Puncak = udara mulai kering.
- Sisi kanan = leeward → kering (rain shadow).

---

## Contoh di Gunung Gede–Pangrango

### Arah Angin dan Lereng

- **Lereng selatan dan barat** sering menjadi **windward** terhadap aliran monsoon.
- Artinya: angin lembab dari selatan/barat menabrak lereng tersebut → hujan lebih deras.
- **Sisi timur dan utara** bisa menjadi **leeward** → lebih kering.

### Konsekuensi untuk Pendaki

- Jalur pendakian dari Cibodas (selatan) mungkin melewati lereng angin → sering hujan.
- Jalur yang berbeda di sisi lain gunung bisa lebih kering.
- Makanya penting memprediksi kondisi di titik spesifik, bukan hanya regional.

---

## Kenapa Penting untuk Model Kamu?

### Tanpa Representasi Spasial

Kalau model hanya melihat satu titik (MAIN), dia tidak tahu:
- Apakah angin sedang datang dari barat?
- Apakah lereng di sebelah barat sedang hujan?
- Apakah kondisi di sekitar MAIN akan segera berubah?

### Dengan 5 Node

Model melihat:
- MAIN = titik target.
- UP, DOWN, LEFT, RIGHT = konteks dari 4 arah.

Sehingga model bisa belajar:
- Jika LEFT dan DOWN hujan, berapa lama sampai MAIN terpengaruh?
- Jika angin dari barat, apa dampaknya ke MAIN?

---

## Kesimpulan

**Variabilitas spasial** adalah fenomena di mana curah hujan sangat berbeda antar lokasi dalam jarak pendek di pegunungan. Hal ini terjadi karena:

1. Angin lembab menabrak lereng → windward hujan.
2. Angin kehilangan kelembapan → leeward kering.
3. Zona tengah lereng biasanya hujan maksimum.
4. Puncak sering lebih kering.

**Relevansi dengan penelitian:** inilah alasan mengapa model kamu butuh **5 node** dan **graph spatio-temporal**, karena satu titik saja tidak cukup untuk merepresentasikan dinamika orografis.

---

## Jawaban Singkat untuk Sidang

> "Variabilitas spasial adalah perbedaan curah hujan antar lokasi dalam jarak pendek akibat topografi. Di pegunungan, lereng yang menghadap angin (windward) mendapat hujan deras, sementara lereng terlindung (leeward) mengalami rain shadow. Karena Gunung Gede–Pangrango memiliki gradien elevasi curam, kondisi cuaca bisa sangat berbeda antar arah, sehingga dibutuhkan representasi multi-node seperti MAIN, UP, DOWN, LEFT, RIGHT untuk menangkap variabilitas ini."
