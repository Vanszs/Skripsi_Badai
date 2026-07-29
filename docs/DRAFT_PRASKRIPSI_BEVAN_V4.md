# DRAFT PRA-SKRIPSI (v4)

PRA-SKRIPSI

NOWCASTING CUACA UNTUK MITIGASI
RISIKO PENDAKI DI GUNUNG GEDEPANGRANGO MENGGUNAKAN RETRIEVALAUGMENTED DIFFUSION MODEL DENGAN
SPATIO-TEMPORAL
GRAPH
NEURAL
NETWORK

BEVANTYO SATRIA PINANDHITA
NPM 22081010153

DOSEN PEMBIMBING
Dr. Faisal Muttaqin, S.Kom, M.T
Andreas Nugroho Sihananto, S.Kom., M.Kom.

KEMENTERIAN PENDIDIKAN, KEBUDAYAAN, RISET, DAN TEKNOLOGI
UNIVERSITAS PEMBANGUNAN NASIONAL VETERAN JAWA TIMUR
FAKULTAS ILMU KOMPUTER
PROGRAM STUDI INFORMATIKA
SURABAYA
2026

ABSTRAK
Nama Mahasiswa / NPM :
Judul Skripsi
:

Dosen Pembimbing

:

Bevantyo Satria Pinandhita / 22081010153
Nowcasting Cuaca Untuk Mitigasi Risiko Pendaki di
Gunung Gede–Pangrango Menggunakan RetrievalAugmented Diffusion Model Dengan SpatioTemporal Graph Neural Network
1. Dr. Faisal Muttaqin, S.Kom, M.T
2. Andreas Nugroho Sihananto, S.Kom., M.Kom.

[DRAFT]
Nowcasting cuaca jangka pendek pada skala waktu 0–6 jam memiliki peran
penting dalam mitigasi risiko bencana dan keselamatan aktivitas luar ruang,
terutama pada wilayah pegunungan dengan dinamika atmosfer yang cepat berubah.
Kawasan Gunung Gede–Pangrango merupakan salah satu lokasi pendakian dengan
intensitas kunjungan tinggi dan kondisi mikrocuaca yang kompleks, sehingga
perubahan presipitasi, angin, dan kelembapan dapat menimbulkan risiko
keselamatan bagi pendaki. Namun, pendekatan prakiraan cuaca deterministik
konvensional sering kali kurang mampu merepresentasikan ketidakpastian
atmosfer serta cenderung mengalami bias terhadap nilai rata-rata, yang
menyebabkan keterbatasan dalam mendeteksi kejadian presipitasi ekstrem.
Penelitian ini bertujuan mengembangkan model nowcasting presipitasi
probabilistik untuk mendukung mitigasi risiko pendakian di kawasan Gunung
Gede–Pangrango. Model yang diusulkan menggunakan pendekatan RetrievalAugmented Diffusion Model dengan Spatio-Temporal Graph Conditioning untuk
memodelkan hubungan spasial dan temporal antar lokasi pengamatan. Data cuaca
diperoleh dari dataset reanalysis ERA5 melalui Open-Meteo API dengan resolusi
waktu per jam. Model dilatih untuk menghasilkan distribusi probabilistik kondisi
presipitasi satu jam ke depan (𝑡 + 1), yang berada dalam rentang nowcasting 0–6
jam menurut WMO. Kinerja model dievaluasi menggunakan metrik statistik dan
probabilistik serta dibandingkan dengan pendekatan regresi deterministik berbasis
machine learning. Hasil penelitian diharapkan dapat memberikan pendekatan
nowcasting yang lebih adaptif dalam merepresentasikan ketidakpastian cuaca serta
meningkatkan sensitivitas deteksi terhadap kejadian presipitasi yang berpotensi
membahayakan aktivitas pendakian.
Kata kunci: Diffusion model, Mitigasi risiko pendakian, Nowcasting cuaca,
Presipitasi probabilistik, Spatio-temporal graph

ii

ABSTRACT
Nama Mahasiswa / NPM :
Judul Skripsi
:

Dosen Pembimbing

:

Bevantyo Satria Pinandhita / 22081010153
Nowcasting Cuaca Untuk Mitigasi Risiko Pendaki di
Gunung Gede–Pangrango Menggunakan RetrievalAugmented Diffusion Model Dengan SpatioTemporal Graph Neural Network
1. Dr. Faisal Muttaqin, S.Kom, M.T
2. Andreas Nugroho Sihananto, S.Kom., M.Kom.

[DRAFT]
Short-term weather nowcasting within a 0–6 hour horizon plays a crucial role in
disaster risk mitigation and outdoor activity safety, particularly in mountainous
regions where atmospheric conditions can change rapidly. The Gunung Gede–
Pangrango area is one of the most popular hiking destinations in Indonesia and
exhibits complex microclimate dynamics, where variations in precipitation, wind,
and humidity may pose significant risks to hikers. Conventional deterministic
weather forecasting approaches often fail to adequately represent atmospheric
uncertainty and tend to exhibit bias toward mean values, limiting their capability to
detect extreme precipitation events. This study aims to develop a probabilistic
precipitation nowcasting model to support risk mitigation for hiking activities in the
Gunung Gede–Pangrango region. The proposed approach employs a RetrievalAugmented Diffusion Model combined with Spatio-Temporal Graph Conditioning
to capture spatial and temporal dependencies among observation locations. Weather
data are obtained from the ERA5 reanalysis dataset through the Open-Meteo API
with hourly temporal resolution. The model is trained to generate probabilistic
distributions of precipitation conditions one hour ahead (t+1), within the WMO
nowcasting range of 0–6 hours. Model performance is evaluated using statistical
and probabilistic metrics and compared with deterministic machine learning
regression approaches. The expected outcome of this research is a more adaptive
nowcasting framework capable of representing atmospheric uncertainty while
improving sensitivity to precipitation events that may pose hazards for mountain
hiking activities.
Keywords: Diffusion model, hiking risk mitigation, probabilistic precipitation,
spatio-temporal graph, weather nowcasting

iii

DAFTAR ISI
DAFTAR ISI .......................................................................................................... iv
DAFTAR GAMBAR ............................................................................................. vi
DAFTAR TABEL ................................................................................................. vii
BAB I

PENDAHULUAN ...................................................................................1

1.1

Latar Belakang ............................................................................................. 1

1.2

Rumusan Masalah ........................................................................................ 5

1.3

Tujuan Penelitian.......................................................................................... 6

1.4

Manfaat Penelitian........................................................................................ 7

1.5

Batasan Masalah ........................................................................................... 7

BAB II

TINJAUAN PUSTAKA ........................................................................10

2.1

Penelitian Terdahulu .................................................................................. 10

2.2

Landasan Teori ........................................................................................... 20

2.2.1 Konsep Nowcasting dan Skala Waktu Prediksi Atmosfer ....................... 21
2.2.1.2 Definisi dan Ruang Lingkup Nowcasting ...................................... 21
2.2.1.3 Karakteristik Dinamika pada Skala Waktu Per Jam di Wilayah
Pegunungan .................................................................................... 22
2.2.1.4 Keterbatasan Pendekatan Deterministik dan Relevansi Pendekatan
Probabilistik.................................................................................... 23
2.2.2 Dinamika Presipitasi Orografis di Pegunungan Tropis ........................... 24
2.2.2.1 Mekanisme Orographic Lifting dan Convective Initiation ............ 25
2.2.2.2 Variabilitas Spasial dan Temporal Presipitasi Orografis................ 26
2.2.2.3 Relevansi dengan Nowcasting Probabilistik dan Mitigasi Risiko .. 27
2.2.3 Model Generatif Probabilistik dengan Diffusion Models ........................ 27
2.2.3.1 Prinsip Kerja dan Keunggulan Diffusion Models untuk Nowcasting
...................................................................................................... 29
2.2.3.2 Keterbatasan Model Deterministik dan Relevansi Pendekatan
Generatif ......................................................................................... 29
2.2.3.3

Relevansi dengan Retrieval-Augmented Diffusion Model .......... 30

2.2.4 Retrieval-Based Historical Analogs ......................................................... 30
2.2.4.1 Prinsip Kerja Retrieval-Based Historical Analogs ......................... 31
2.2.4.2 Keunggulan Retrieval-Augmented dibandingkan Pure Generative
Models ............................................................................................ 32
2.2.5 Spatio-Temporal Graph Conditioning pada Representasi Data Elevasi .. 32
2.2.5.1 Representasi Graph untuk Data Elevasi dengan Lima Node ............ 33
iv

2.2.5.2 Keunggulan Spatio-Temporal Graph dibandingkan Pendekatan
Konvensional .................................................................................................. 34
2.2.6 Risiko Hipotermia dan Mitigasi Pendakian ............................................. 35
2.2.6.1 Mekanisme Hipotermia di Lingkungan Pegunungan Tropis ............ 36
2.2.6.2 Faktor Meteorologis Pemicu Risiko Hipotermia ................................. 36
2.2.6.3 Relevansi Nowcasting Probabilistik sebagai Mitigasi Risiko .......... 37
2.2.7 Visualisasi Web Berbasis Vue.js dan FastAPI ........................................ 38
2.3

Analisis Celah Penelitian ....................................................................... 39
2.3.1.1 Celah pada Representasi Spasial dan Temporal ................................... 40
2.3.1.2 Celah pada Penanganan Kejadian Ekstrem dan Ketidakpastian ...... 40
2.3.1.3 Celah pada Aplikasi Mitigasi Risiko Pendakian .................................. 40

BAB III METODE PENELITIAN ......................................................................42
3.1

Tahapan Penelitian ...................................................................................................... 42

3.2

Pengambilan Data ....................................................................................................... 44

3.3

Preprocessing Data...................................................................................................... 47

3.4

Pembagian Dataset Temporal .................................................................................. 49

3.5

Pembentukan Fitur dan Representasi Graf ........................................................... 50

3.5.1 Pembentukan Fitur Input.......................................................................... 50
3.5.2 Pembentukan Jendela Waktu (Temporal Window) .................................. 51
3.5.3 Representasi Graf Spasial ........................................................................ 51
3.5.4 Representasi Spasio-Temporal................................................................. 52
3.6

Perancangan Arsitektur Model ................................................................................ 52

3.6.1 Spatio-Temporal Graph Neural Network (STGNN)................................ 52
3.6.2 Retrieval-Augmented ............................................................................... 53
3.6.3 Conditional Diffusion Model ................................................................... 53
3.6.4 Integrasi Antar Komponen Model ........................................................... 54
3.7

Prosedur Pelatihan Model ......................................................................................... 54

3.8

Prosedur Inferensi ....................................................................................................... 55

3.9

Metode Evaluasi Model ............................................................................................. 56

3.10 Skenario Eksperimen .................................................................................................. 58
3.11 Rancangan Visualisasi Hasil Prediksi Berbasis Web ....................................... 60
DAFTAR PUSTAKA ............................................................................................62

v

DAFTAR GAMBAR
Gambar 3.1 Alur Tahapan Penelitian ...................................................................42
Gambar 3.2 Diagram Proses Pengambilan Data ERA5 melalui Open-Meteo API ..
..........................................................................................................44
Gambar 3.3 Posisi Lima Node Pengamatan pada Wilayah Studi Gunung Gede–
Pangrango ..............................................................................................45
Gambar 3.4 Flowchart Preprocessing Data ..........................................................47
Gambar 3.5 Ilustrasi Pembagian Dataset Temporal .............................................49

vi

DAFTAR TABEL
Tabel 2.1 Tabel Penelitian Terdahulu ....................................................................18
Tabel 2.2 Perbandingan Skala Prakiraan Atmosfer disusun [7] [19]. ...................21
Tabel 2.3 Perbandingan Keluaran Deterministik dan Probabilistik pada
Nowcasting [4], [19], [27]. ....................................................................24
Tabel 2.4 Definisi Node pada Representasi Graph Lima Node [9], [12], [31]......34
Tabel 2.5 Faktor Meteorologis terkait Paparan Dingin Pendaki [28], [22] ...........36
Tabel 3.1 Koordinat Node .....................................................................................45
Tabel 3.2 Karakteristik Dataset Penelitian ............................................................46
Tabel 3.3 Daftar Fitur Input Model .......................................................................50
Tabel 3.4 Ringkasan Metrik Evaluasi....................................................................57
Tabel 3.5 Matriks Skenario Eksperimen dan Pengujian ........................................59

vii

BAB I
PENDAHULUAN

1.1

Latar Belakang
Hipotermia merupakan kondisi darurat ketika suhu inti tubuh turun secara

abnormal akibat kehilangan panas yang melampaui kemampuan tubuh untuk
memproduksinya. Pada aktivitas pendakian, risiko hipotermia meningkat ketika
pendaki terpapar suhu rendah, angin, hujan, pakaian basah, dan kelelahan dalam
durasi yang panjang [1], [2]. Urgensi masalah tersebut terlihat pada data
BASARNAS periode 2015–2018 yang dilaporkan Aminullah et al. [2], yaitu
hipotermia menjadi penyebab dominan kecelakaan pendakian sebesar 47%; pada
2018, tiga pendaki di Gunung Tampomas di dilaporkan meninggal akibat
hipotermia. Temuan ini menunjukkan bahwa keselamatan pendaki tidak hanya
ditentukan oleh kesiapan fisik dan perlengkapan, tetapi juga oleh ketersediaan
informasi kondisi cuaca yang tepat waktu untuk mendukung tindakan pencegahan
sebelum paparan risiko berkembang menjadi kondisi medis serius.
Nowcasting cuaca dalam jangka waktu yang pendek memiliki peran krusial
dalam mitigasi risiko keselamatan aktivitas luar ruang termasuk resiko terjadinya
hipotermia. Skala waktu hourly (per jam) lebih kritis dibandingkan prakiraan harian
karena mampu memberikan informasi yang lebih cepat mengenai perubahan
kondisi atmosfer yang berlangsung secara dinamis. Dalam konteks ini, informasi
mengenai presipitasi probabilistik, yaitu peluang terjadinya presipitasi dalam
rentang waktu tertentu, menjadi penting untuk memahami tingkat ketidakpastian
kondisi cuaca yang akan terjadi. Pendekatan probabilistik memungkinkan prediksi
tidak hanya berupa nilai tunggal, tetapi juga distribusi kemungkinan kejadian
presipitasi yang dapat berkaitan dengan potensi hujan intensitas tinggi maupun
perubahan kondisi atmosfer lain seperti peningkatan kecepatan angin. Informasi
tersebut sangat relevan untuk mendukung pengambilan keputusan yang lebih
adaptif dalam menghadapi risiko lain, seperti bencana banjir dan tanah longsor,
terutama pada aktivitas yang sangat bergantung pada kondisi cuaca jangka pendek.
Nowcasting memungkinkan deteksi dini dan respons cepat terhadap kondisi
cuaca ekstrem, sehingga dapat mengurangi kerugian jiwa dan materi serta
1

meningkatkan keselamatan masyarakat, terutama dalam kegiatan yang sangat
bergantung pada kondisi cuaca seperti pendakian gunung dan aktivitas outdoor
lainnya [3], [4]. Sistem peringatan dini yang mengandalkan nowcasting cuaca
jangka pendek dapat mengintegrasikan data radar, model numerik, dan kecerdasan
buatan untuk menghasilkan prediksi yang lebih tepat dan real-time, mendukung
pengambilan keputusan yang cepat dan efektif oleh otoritas dan masyarakat [4], [5].
Penelitian juga menunjukkan bahwa metode nowcasting berbasis deep learning
mampu memberikan prediksi hujan dan angin dengan resolusi spasial dan temporal
tinggi, yang sangat penting untuk keselamatan dan mitigasi risiko di area rawan [6],
[7], [8].
Kawasan Gunung Gede–Pangrango merupakan salah satu taman nasional
dengan intensitas aktivitas wisata dan pendakian yang tinggi, terutama di Resort
Cibodas dan kawasan Curug Cibeureum, sehingga memiliki tingkat paparan risiko
lingkungan dan keselamatan yang signifikan. Kajian daya dukung kawasan Curug
Cibeureum [9], menunjukkan bahwa angka kunjungan harian pada akhir pekan dan
musim liburan telah melampaui real carrying capacity (RCC) dan effective carrying
capacity (ECC), meskipun masih berada di bawah physical carrying capacity
(PCC), yang mengindikasikan potensi gangguan terhadap kualitas lingkungan dan
keselamatan pengunjung. Berbagai faktor biofisik seperti curah hujan tinggi,
kemiringan lereng yang curam, erodibilitas tanah, serta sensitivitas ekosistem dan
satwa liar diidentifikasi sebagai pembatas utama aktivitas wisata di kawasan ini,
dengan curah hujan muncul sebagai faktor koreksi dominan yang secara signifikan
menurunkan nilai RCC dan ECC hingga pada periode tertentu memaksa penutupan
penuh akses ke Curug Cibeureum demi keselamatan [9]. Temuan tersebut
menegaskan bahwa variabilitas cuaca jangka pendek tidak hanya memengaruhi
kenyamanan wisata, tetapi juga secara langsung menentukan keputusan manajemen
terkait pembatasan/pembukaan jalur, pengaturan jumlah pengunjung, dan mitigasi
risiko kecelakaan maupun bencana hidrometeorologis. Dalam konteks ini,
ketersediaan informasi cuaca per jam yang akurat dan adaptif menjadi krusial untuk
mendukung sistem peringatan dini dan pengambilan keputusan operasional di
kawasan pegunungan yang bercirikan dinamika atmosfer cepat dan heterogenitas
spasial tinggi [5]. Oleh karena itu, pengembangan pendekatan nowcasting cuaca
2

jangka pendek yang mampu menyajikan informasi probabilistik mengenai hujan,
angin, dan kelembapan pada skala lokal menjadi sangat relevan untuk melengkapi
strategi pengelolaan berbasis daya dukung dan meningkatkan keselamatan aktivitas
pendakian di Gunung Gede–Pangrango [6], [10].
Hujan dalam skala waktu per jam di wilayah pegunungan memiliki
karakteristik yang kompleks akibat interaksi dinamis antara hujan, angin, dan
kelembapan yang secara bersama-sama dapat memicu risiko hipotermia bagi
pendaki dan aktivitas luar ruang lainnya. Wilayah orografis seperti Gunung Gede–
Pangrango menunjukkan variabilitas spasial yang sangat tinggi dan dinamika cuaca
yang cepat berubah, sehingga prediksi cuaca di area ini menjadi sangat menantang
bagi model numerik konvensional. Penelitian di berbagai pegunungan tropis dan
subtropis menunjukkan bahwa curah hujan orografis sangat dipengaruhi oleh
mekanisme pengangkatan udara akibat topografi, yang memodulasi suhu dan
kelembapan di lapisan troposfer bawah, serta kecepatan dan arah angin yang
membawa uap air [11], [12]. Selain itu, variabilitas spasial dan temporal hujan di
pegunungan sangat dipengaruhi oleh siklus diurnal dan musiman, di mana puncak
hujan sering terjadi pada siang hingga sore hari, terkait dengan pemanasan
permukaan dan konveksi lokal [13]. Studi juga menemukan bahwa gradien curah
hujan terhadap elevasi dapat sangat bervariasi, dengan korelasi positif antara curah
hujan dan kelembapan relatif, serta korelasi negatif dengan kecepatan angin, yang
menunjukkan peran penting interaksi antara kelembapan dan angin dalam
pembentukan hujan orografis [14], [15]. Oleh karena itu, pemahaman mendalam
tentang interaksi hujan, angin, dan kelembapan di wilayah pegunungan sangat
penting untuk mengembangkan model nowcasting yang akurat dan andal guna
mitigasi risiko hipotermia dan bahaya cuaca ekstrem selama pendakian gunung.

Model deterministik dalam prakiraan cuaca sering gagal merepresentasikan
ketidakpastian yang melekat dalam sistem atmosfer, sehingga kurang mampu
menangkap variabilitas dan risiko cuaca ekstrem secara akurat. Model regresi
berbasis machine learning (ML) cenderung bias ke nilai rata-rata, yang
menyebabkan under-prediction pada kejadian ekstrem atau distribusi ekor berat
(heavy-tail), sehingga mengurangi keandalan prediksi untuk peristiwa cuaca yang
3

jarang namun berdampak besar. Selain itu, dataset gridded seperti ERA5 yang
sering digunakan untuk pelatihan model ML memiliki efek smoothing yang
ekstrem, sehingga melemahkan representasi kuantil atas dari variabel cuaca, yang
penting untuk memprediksi kejadian ekstrem dengan tepat. Studi menunjukkan
bahwa meskipun model ML seperti PanguWeather dan GraphCast menunjukkan
akurasi yang kompetitif dengan model numerik fisika, mereka masih kurang dalam
reproduksi fenomena mesoskal dan sub-sinoptik serta konsistensi fisik, yang
berdampak pada interpretasi dan keandalan prediksi mereka [16], [17]. Pendekatan
ML probabilistik terbaru seperti FuXi-ENS [18], mulai mengatasi keterbatasan ini
dengan menghasilkan ensemble prediksi yang lebih baik dalam merepresentasikan
ketidakpastian dan kejadian ekstrem, namun tantangan dalam resolusi spasial dan
representasi fisik masih ada [19]. Oleh karena itu, pengembangan model
nowcasting yang menggabungkan aspek probabilistik dan fisika atmosfer tetap
menjadi kebutuhan penting untuk meningkatkan akurasi dan keandalan prakiraan
cuaca jangka pendek, terutama di wilayah pegunungan dengan dinamika kompleks.

Meskipun berbagai pendekatan deep learning telah dikembangkan untuk
nowcasting curah hujan, sebagian besar studi masih berfokus pada prediksi
deterministik atau peringatan berbasis threshold tanpa mengintegrasikan secara
simultan model probabilistik generatif, pencarian analog historis berbasis retrieval,
dan representasi grafis spatio-temporal. Pendekatan deep generative models telah
menunjukkan kemampuan dalam menghasilkan prediksi probabilistik yang realistis
dan konsisten secara spasial dan temporal, namun penerapannya masih terbatas
pada data radar dan wilayah datar, belum banyak dieksplorasi untuk konteks
mikrocuaca di wilayah pegunungan dengan data reanalysis titik-lokasi [6], [7].
Studi lain menggunakan model deep learning seperti LSTM dan ConvLSTM untuk
prediksi curah hujan di pegunungan, tetapi masih menghadapi tantangan dalam
menangani variabilitas spasial tinggi dan dinamika cepat yang khas di wilayah
orografis [10], [20], [21]. Selain itu, integrasi data multi-sumber dan teknik
representasi grafis untuk menangkap hubungan spasial dan temporal secara
simultan masih jarang diterapkan dalam nowcasting curah hujan di pegunungan,
padahal hal ini penting untuk meningkatkan akurasi dan ketahanan model terhadap
4

ketidakpastian [22], [23]. Pendekatan hybrid yang menggabungkan model fisika
dan deep learning mulai menunjukkan hasil yang menjanjikan, tetapi belum
menggabungkan aspek probabilistik dan retrieval historis secara komprehensif [22].
Oleh karena itu, terdapat gap riset yang signifikan dalam mengembangkan model
nowcasting curah hujan yang mengintegrasikan probabilistic generative modeling,
retrieval-based historical analogs, dan spatio-temporal graph representation secara
bersamaan untuk konteks mikrocuaca pegunungan berbasis data reanalysis titiklokasi.

Penelitian ini mengevaluasi model nowcasting curah hujan yang
mengintegrasikan probabilistic generative modeling, retrieval-based historical
analogs, dan spatio-temporal graph representation secara simultan untuk konteks
mikrocuaca Gunung Gede–Pangrango. Studi ini berorientasi pada mitigasi risiko
pendakian dengan menelaah trade-off antara akurasi prediksi dan sensitivitas
terhadap kejadian ekstrem. Pendekatan ini relevan untuk menilai kegunaan prediksi
probabilistik sebagai informasi pendukung pengambilan keputusan sebelum dan
selama pendakian [24], [25], [26], [27].
1.2

Rumusan Masalah
Berdasarkan latar belakang yang telah diuraikan, permasalahan penelitian ini

dapat dirumuskan sebagai berikut:
1. Bagaimana kinerja pendekatan nowcasting probabilistik berbasis probabilistic
generative modeling dalam memprediksi dinamika hujan, angin, dan
kelembapan pada skala waktu per jam di wilayah pegunungan Gunung Gede–
Pangrango berbasis data reanalysis titik-lokasi?
2. Sejauh mana integrasi retrieval-based historical analogs dan spatio-temporal
graph representation mampu meningkatkan kemampuan model dalam
menangkap ketidakpastian serta dinamika spasial-temporal mikrocuaca di
wilayah pegunungan?
3. Bagaimana kemampuan model probabilistik dalam memberikan estimasi
ketidakpastian prakiraan cuaca multi-variabel melalui kerangka ensemble
nowcasting?
5

4. Bagaimana trade-off antara akurasi prediksi rata-rata dan sensitivitas terhadap
kejadian ekstrem pada konteks aplikasi mitigasi risiko pendakian di wilayah
pegunungan?
5. Bagaimana hasil prediksi nowcasting probabilistik dapat disiapkan sebagai
keluaran untuk sistem visualisasi berbasis web dalam mendukung pengambilan
keputusan aktivitas pendakian di wilayah pegunungan?
1.3

Tujuan Penelitian
Tujuan umum dari penelitian ini adalah mengevaluasi kemampuan

pendekatan nowcasting probabilistik berbasis probabilistic generative modeling
yang terintegrasi dengan retrieval-based historical analogs dan spatio-temporal
graph representation dalam memprediksi dinamika hujan, angin, dan kelembapan
pada skala waktu per jam di wilayah pegunungan Gunung Gede–Pangrango untuk
mendukung mitigasi risiko pendakian.
Untuk mencapai tujuan umum tersebut, penelitian ini memiliki tujuan khusus
sebagai berikut:
1. Menganalisis kinerja model nowcasting probabilistik berbasis probabilistic
generative modeling dalam memprediksi hujan, angin, dan kelembapan pada
skala waktu per jam di wilayah pegunungan berbasis data reanalysis titik-lokasi.
2. Mengevaluasi kontribusi integrasi retrieval-based historical analogs dan spatiotemporal graph representation terhadap kemampuan model dalam menangkap
ketidakpastian serta dinamika spasial-temporal mikrocuaca di wilayah
pegunungan.
3. Membandingkan kemampuan model probabilistik dengan pendekatan regresi
deterministik berbasis machine learning konvensional dalam mendeteksi
kejadian cuaca ekstrem, khususnya pada distribusi curah hujan yang bersifat
heavy-tailed.
4. Menganalisis trade-off antara akurasi prediksi rata-rata dan sensitivitas terhadap
kejadian ekstrem dalam konteks aplikasi mitigasi risiko pendakian di wilayah
pegunungan.
5. Menyajikan hasil prediksi nowcasting probabilistik sebagai artefak evaluasi
yang dapat menjadi dasar pengembangan sistem visualisasi berbasis web untuk
mitigasi risiko pendakian.
6

1.4

Manfaat Penelitian
Penelitian ini diharapkan dapat memberikan kontribusi dalam pengembangan

metode nowcasting cuaca serta penerapannya dalam mendukung mitigasi risiko
aktivitas pendakian di wilayah pegunungan. Manfaat yang diharapkan dari
penelitian ini adalah sebagai berikut:
1. Memberikan kontribusi dalam pengembangan dan evaluasi pendekatan
nowcasting probabilistik berbasis probabilistic generative modeling untuk
memodelkan dinamika hujan, angin, dan kelembapan pada skala waktu per jam
di wilayah pegunungan.
2. Menyediakan pemahaman yang lebih komprehensif mengenai efektivitas
integrasi retrieval-based historical analogs dan spatio-temporal graph
representation dalam menangkap ketidakpastian serta hubungan spasialtemporal mikrocuaca.
3. Memberikan gambaran empiris mengenai perbandingan kinerja antara model
probabilistik dan pendekatan regresi deterministik berbasis machine learning
dalam mendeteksi kejadian cuaca ekstrem, khususnya pada distribusi presipitasi
yang bersifat heavy-tailed.
4. Menyediakan analisis terkait trade-off antara akurasi prediksi rata-rata dan
sensitivitas terhadap kejadian ekstrem sebagai dasar dalam pengambilan
keputusan berbasis risiko pada aktivitas pendakian di wilayah pegunungan.
5. Menyediakan keluaran prediksi probabilistik yang dapat digunakan sebagai
dasar pengembangan sistem visualisasi berbasis web untuk penyampaian
informasi cuaca jangka pendek.
1.5

Batasan Masalah
Untuk menjaga fokus penelitian dan memastikan ketercapaian tujuan dalam

ruang lingkup skripsi, penelitian ini dibatasi oleh beberapa ketentuan sebagai
berikut:
1. Wilayah kajian dibatasi pada kawasan Gunung Gede–Pangrango yang
direpresentasikan oleh 5 titik lokasi (node) yang berbeda. Pembatasan ini
dilakukan untuk merepresentasikan gradasi orografis utama secara sederhana
namun relevan, sekaligus menjaga kompleksitas model dan ketersediaan data
agar tetap seimbang.
7

2. Model menggunakan jendela observasi 6 jam terakhir sebagai input untuk
memprediksi kondisi cuaca satu jam ke depan (single-step hourly nowcasting).
Pendekatan ini sesuai dengan definisi nowcasting menurut WMO, yaitu
prakiraan dengan rentang waktu 0–6 jam, di mana model dijalankan secara
iteratif pada setiap jam. Batasan ini dipilih karena perubahan cuaca yang berisiko
tinggi terhadap keselamatan pendakian umumnya terjadi dalam rentang waktu
singkat, sehingga informasi prediksi per jam lebih relevan dibandingkan
prakiraan harian.
3. Variabel cuaca yang diprediksi dibatasi pada curah hujan, kecepatan angin, dan
kelembapan relatif. Ketiga variabel tersebut dipilih karena relevan terhadap
bahaya cuaca selama pendakian. Suhu udara tetap digunakan sebagai fitur input,
tetapi tidak menjadi target agar fokus model tetap pada tiga variabel keluaran
dan kompleksitas pelatihan tidak bertambah.
4. Sumber data cuaca dibatasi pada dataset reanalysis ERA5 yang diperoleh melalui
Open-Meteo API dengan resolusi spasial grid dan resolusi temporal per jam.
Dataset ini dipilih karena ketersediaannya yang konsisten dalam jangka panjang
serta cakupan historis yang memadai untuk pelatihan model, meskipun memiliki
keterbatasan dalam merepresentasikan kejadian ekstrem berskala kecil.
5. Penelitian ini tidak menggunakan data radar cuaca, satelit resolusi tinggi,
maupun observasi stasiun hujan lokal sebagai sumber data utama. Pembatasan
ini dilakukan untuk memfokuskan penelitian pada evaluasi pendekatan
metodologis berbasis machine learning probabilistik terhadap data reanalysis
titik-lokasi, serta menghindari ketergantungan pada data yang tidak selalu
tersedia secara operasional di wilayah pegunungan.
6. Pendekatan pemodelan dibatasi pada evaluasi model probabilistik berbasis
probabilistic generative modeling yang terintegrasi dengan retrieval-based
historical analogs dan spatio-temporal graph representation. Penelitian ini tidak
bertujuan untuk melakukan eksplorasi menyeluruh terhadap seluruh arsitektur
deep learning yang tersedia, melainkan mengevaluasi pendekatan yang telah
ditentukan secara konseptual sesuai dengan tujuan penelitian.
7. Data presipitasi bersifat zero-inflated dan heavy-tailed sehingga kejadian
ekstrem langka. Penanganan melalui transformasi log1p, weighted denoising
8

loss, retrieval analog historis, dan ensemble diffusion tidak menjamin deteksi
kejadian ekstrem; keterbatasan ini dianalisis secara eksplisit pada metrik
berbasis ambang.
8. ERA5 merupakan data reanalysis historis dengan latensi sekitar 5–7 hari,
sehingga tidak digunakan sebagai sumber operasional real-time. Penerapan
operasional memerlukan penggantian sumber data dengan radar, stasiun cuaca
otomatis, atau satelit near real-time.

9

BAB II
TINJAUAN PUSTAKA
2.1

Penelitian Terdahulu
Penelitian-penelitian yang sudah dilakukan peneliti sebelumnya yang terkait

dengan penelitian yang akan dilakukan adalah sebagai berikut.
1. Penelitian “Training of AI-based Nowcasting Models for Rainfall Early Warning
Should Take into Account User Requirements” oleh Ayzel dan Heistermann
(2025) [28].
Penelitian

oleh

Ayzel

dan

Heistermann

mengkaji

keterbatasan

fundamental model nowcasting berbasis deep learning dalam mendukung sistem
peringatan dini, khususnya dalam memprediksi presipitasi ekstrem. Studi ini
berangkat dari observasi bahwa meskipun model deep learning mampu
menangkap pola kompleks dari data radar, performanya menurun signifikan
pada kejadian hujan lebat yang justru paling krusial secara operasional. Penulis
mengusulkan bahwa akar permasalahan bukan semata pada arsitektur model,
melainkan pada ketidaksesuaian antara tujuan pelatihan (training objective) dan
kebutuhan pengguna. Untuk menguji hipotesis ini, mereka mengubah formulasi
masalah dari regresi kontinu menjadi prediksi berbasis threshold (segmentation),
serta dari resolusi temporal tinggi (5 menit) menjadi akumulasi curah hujan per
jam yang lebih relevan dengan sistem peringatan. Hasil eksperimen
menunjukkan bahwa model berbasis threshold (RainNet2024-S) secara
konsisten mengungguli model regresi dalam mendeteksi kejadian hujan ekstrem,
terutama pada metrik CSI dan FSS, meskipun secara umum semua model masih
menunjukkan skill yang terbatas pada intensitas tinggi. Temuan ini menegaskan
bahwa penyederhanaan target prediksi yang selaras dengan kebutuhan pengguna
dapat meningkatkan performa operasional model tanpa harus meningkatkan
kompleksitas arsitektur. Namun demikian, penelitian ini masih terbatas pada
pendekatan deterministik berbasis threshold dan belum mengakomodasi
representasi distribusi probabilistik secara eksplisit, sehingga kemampuan untuk
mengkuantifikasi ketidakpastian dan mengevaluasi risiko tetap terbatas. Selain
itu, ketergantungan pada data radar resolusi tinggi membatasi generalisasi ke
konteks dengan keterbatasan data.
10

2. Penelitian “Precipitation Nowcasting with Generative Diffusion Models” oleh
Asperti et al. (2023) [29].
Penelitian oleh Asperti et al. mengusulkan pendekatan baru untuk
nowcasting curah hujan berbasis generative diffusion models, dengan argumen
bahwa pendekatan probabilistik lebih sesuai dibandingkan model deterministik
dalam menangani sifat stokastik atmosfer. Berbeda dengan model konvensional
yang dioptimalkan menggunakan mean squared error dan cenderung
menghasilkan prediksi yang “blur” akibat averaging, diffusion model secara
eksplisit memodelkan distribusi probabilitas curah hujan melalui proses
denoising bertahap dari noise menjadi kondisi atmosfer yang realistis. Kontribusi
utama penelitian ini adalah pengembangan Generative Ensemble Diffusion
(GED), yaitu pendekatan yang menghasilkan multiple skenario prediksi
(ensemble) dan menggabungkannya melalui rata-rata atau post-processing UNet untuk meningkatkan akurasi. Hasil eksperimen pada dataset ERA5
menunjukkan bahwa meskipun single diffusion model masih kalah dibandingkan
model U-Net konvensional, pendekatan ensemble GED secara konsisten
mengungguli semua baseline, terutama pada metrik MSE dan recall untuk
prediksi hingga 3 jam ke depan. Selain itu, hasil dari penelitian ini menunjukan
performa yang meningkat signifikan ketika menggunakan post-processing
dibanding sekadar averaging. Namun, penelitian ini memiliki beberapa
keterbatasan penting: ketergantungan pada banyak sampling (hingga 15
generasi) meningkatkan biaya komputasi dan latency, yang berpotensi
menghambat implementasi real-time; selain itu, evaluasi masih berfokus pada
MSE yang kurang sensitif terhadap kejadian ekstrem. Secara konseptual,
meskipun diffusion model menawarkan keunggulan dalam representasi
ketidakpastian, pendekatan ini masih belum sepenuhnya menyelesaikan tradeoff antara akurasi, efisiensi, dan interpretabilitas dalam sistem nowcasting
operasional.
3. Penelitian “On Some Limitations of Current Machine Learning Weather
Prediction Models” oleh Bonavita (2024) [16].
Penelitian oleh Bonavita (2024) secara kritis mengevaluasi keterbatasan
model prediksi cuaca berbasis machine learning (MLWP) seperti Pangu11

Weather, FourCastNet, dan GraphCast, dengan menyoroti bahwa klaim
superioritas terhadap model fisika tradisional perlu ditinjau ulang dari perspektif
konsistensi fisik. Meskipun model-model ini menunjukkan performa kompetitif
pada metrik deterministik dan memiliki efisiensi komputasi yang jauh lebih
tinggi, analisis spektral menunjukkan bahwa prediksi ML cenderung kehilangan
energi pada skala spasial kecil, menghasilkan output yang terlalu halus (oversmoothing) dan gagal merepresentasikan fenomena mesoskal seperti siklon
tropis secara akurat. Lebih jauh, evaluasi keseimbangan fisik mengungkap
inkonsistensi mendasar, termasuk pelemahan hubungan geostrofik antara angin
dan geopotensial, penurunan komponen divergen aliran, serta estimasi gerakan
vertikal yang secara signifikan lebih lemah hingga 40–50% lebih kecil dibanding
model fisika, yang berdampak langsung pada ketidakakuratan prediksi fenomena
cuaca aktif seperti presipitasi. Selain itu, seperti ditunjukkan dalam analisis
spektrum dan error growth, model ML tidak mampu mereproduksi dinamika
pertumbuhan error nonlinier (butterfly effect), sehingga lebih menyerupai
estimator rata-rata distribusi daripada simulator atmosfer yang sebenarnya.
Implikasi utamanya adalah bahwa performa tinggi pada metrik seperti MSE
dapat bersifat menyesatkan karena dipengaruhi oleh rendahnya variabilitas
prediksi. Secara konseptual, penelitian ini menegaskan bahwa Machine Learning
Weather Prediction (MLWP) saat ini lebih tepat diposisikan sebagai alat postprocessing daripada pengganti model fisika, serta menyoroti tantangan
fundamental dalam mencapai keseimbangan antara akurasi prediksi dan
konsistensi dinamika atmosfer.
4. Penelitian “Skilful Nowcasting of Extreme Precipitation with NowcastNet” oleh
Zhang et al (2024) [7].
Penelitian oleh Zhang et al. mengembangkan NowcastNet, sebuah model
nowcasting presipitasi ekstrem yang mengintegrasikan pendekatan berbasis
fisika dan deep learning dalam kerangka generatif bersyarat untuk mengatasi
keterbatasan metode sebelumnya yang cenderung menghasilkan prediksi blur,
kehilangan intensitas, atau error lokasi. Berbeda dengan pendekatan advection
tradisional yang hanya akurat hingga sekitar 1 jam dan model deep learning
murni yang sering melanggar prinsip fisika, NowcastNet menggabungkan dua
12

komponen utama yaitu evolution network berbasis hukum kontinuitas atmosfer
dan generative network berbasis latent stochastic sampling untuk menangkap
dinamika multiskala dari proses presipitasi. Model ini mampu menghasilkan
prediksi hingga 3 jam ke depan dengan resolusi tinggi serta mempertahankan
struktur mesoskal dan detail konvektif secara simultan. Evaluasi kuantitatif
menggunakan metrik CSI menunjukkan peningkatan signifikan terutama pada
intensitas hujan tinggi di atas 16 mm/jam, sementara analisis spektral
menunjukkan kemampuan mempertahankan variabilitas spasial yang lebih
realistis dibandingkan baseline seperti DGMR, PredRNN, dan pySTEPS. Selain
itu, evaluasi oleh 62 meteorolog profesional menunjukkan bahwa NowcastNet
dipilih sebagai model terbaik pada sekitar 67–76% kasus, menegaskan nilai
operasionalnya dalam konteks prakiraan cuaca ekstrem. Namun demikian,
model ini masih sangat bergantung pada data radar resolusi tinggi dan belum
secara eksplisit mengeksplorasi distribusi probabilitas untuk analisis risiko atau
aplikasi berbasis ketidakpastian, sehingga membuka ruang penelitian lanjutan
pada integrasi probabilistik dan adaptasi pada data dengan keterbatasan resolusi
seperti reanalysis.
5. Penelitian “A Hybrid Approach to Physical and Deep Learning Models for
Radar-Based Precipitation Nowcasting” Oleh Kim et al (2025) [22].
Penelitian oleh Kim et al. mengusulkan pendekatan hybrid nowcasting
presipitasi berbasis radar dengan menggabungkan model fisik dan deep learning
menggunakan algoritma boosting untuk meningkatkan akurasi prediksi curah
hujan jangka pendek. Model ini mengintegrasikan PySTEPS (semi-Lagrangian
berbasis adveksi) yang unggul dalam mempertahankan intensitas hujan, serta
RainNet (arsitektur U-Net berbasis CNN) yang efektif dalam menangkap pola
spasial hujan, kemudian mengombinasikan keduanya menggunakan LightGBM
sebagai boosting ensemble yang secara iteratif memperbaiki bias prediksi.
Pipeline penelitian dimulai dari prediksi masing-masing model, dilanjutkan
dengan proses blending berbasis fitur piksel hujan sebelum menghasilkan
prediksi akhir. Dataset radar Korea Selatan (2016–2021) dengan resolusi 1 km
dan interval 10 menit digunakan untuk pelatihan dan evaluasi. Hasil eksperimen
menunjukkan bahwa model hybrid secara konsisten mengungguli model tunggal
13

pada berbagai metrik seperti CSI, POD, FAR, RMSE, dan FSS hingga lead time
90 menit, dengan peningkatan CSI sekitar 10–13% pada ambang intensitas 0.1–
1 mm/jam. Hasil dari penelitian ini menunjukkan bahwa LightGBM mampu
mempertahankan struktur spasial dari RainNet sekaligus intensitas hujan dari
PySTEPS. Namun, performa menurun pada hujan intensitas tinggi (>5 mm/jam)
untuk lead time panjang, serta masih bergantung pada kualitas data radar resolusi
tinggi. Secara keseluruhan, pendekatan blending ini menunjukkan bahwa
integrasi model fisik dan data-driven memberikan trade-off optimal antara
akurasi spasial dan intensitas, serta menjadi arah menjanjikan untuk sistem
prediksi banjir berbasis nowcasting.
6. Penelitian “Predicting rainfall using machine learning, deep learning, and time
series models across an altitudinal gradient in the North-Western Himalayas”
oleh Wani et al (2024) [20].
Penelitian oleh Wani et al. mengkaji prediksi curah hujan di wilayah
Himalaya Barat Laut dengan membandingkan tiga pendekatan utama, yaitu
machine learning (ML), deep learning (DL), dan model time series,
menggunakan data meteorologi selama 40 tahun (1980–2021) dari enam stasiun
dengan variasi ketinggian. Variabel input meliputi suhu maksimum/minimum
dan faktor meteorologi lain, dengan proses preprocessing mencakup penanganan
missing value dan pembagian data 80:20 untuk training-testing. Model yang
diuji meliputi RF, SVR, ANN (ML); RNN, LSTM, GRU, Bi-LSTM (DL); serta
ARIMA dan TBATS (time series). Berdasarkan hasil evaluasi menggunakan
RMSE dan MAE, model DL secara konsisten menunjukkan performa terbaik,
dengan urutan akurasi tertinggi adalah Bi-LSTM, LSTM, RNN, Deep LSTM,
dan GRU, sementara ML berada di bawahnya dengan ANN sebagai yang terbaik
dan time series menjadi yang terendah. Penelitian ini juga menunjukan DL
mampu menangkap pola non-linear dan dependensi temporal yang kompleks.
Selain itu, studi ini menyoroti bahwa ketinggian (altitude) secara signifikan
mempengaruhi akurasi model, karena variasi topografi menyebabkan
heterogenitas pola curah hujan. Namun, penelitian ini tidak mengusulkan model
baru, melainkan menunjukkan bahwa pemilihan model yang tepat dan

14

ketersediaan data berkualitas tinggi menjadi faktor utama dalam meningkatkan
akurasi prediksi, terutama di wilayah dengan kompleksitas geografis tinggi.
7. Penelitian “The spatiotemporal variability in precipitation gradients based on
meteorological station observations in mountainous areas of Northwest China”
oleh Zhang et al (2023) [14].
Penelitian oleh Zhang et al menganalisis variabilitas spasio-temporal curah
hujan dan gradien presipitasi (precipitation gradient/PG) di wilayah pegunungan
Tianshan dan Qilian menggunakan data stasiun meteorologi periode 1961–2017.
Hasil pada penelitian ini menunjukkan bahwa curah hujan di kedua wilayah
mengalami tren peningkatan signifikan dalam beberapa dekade terakhir, dengan
pola distribusi yang sangat dipengaruhi oleh topografi. Di Tianshan, curah hujan
lebih tinggi di bagian utara dan barat, sedangkan di Qilian lebih tinggi di bagian
timur. Penelitian ini juga menemukan adanya ketinggian maksimum presipitasi
(maximum precipitation height) sekitar 1942,5 m di Tianshan dan sekitar 2850
m di Qilian, yang menunjukkan hubungan non-linear antara elevasi dan curah
hujan. Selain itu, gradien presipitasi menunjukkan peningkatan signifikan secara
temporal, terutama setelah periode perubahan abrupt pada sekitar tahun 1970–
1980-an, yang mengindikasikan tren “humidifikasi” di wilayah pegunungan
tersebut. Secara musiman, hubungan antara elevasi dan curah hujan paling kuat
terjadi pada bulan Juni–Agustus, sedangkan pada musim dingin hubungan
tersebut dapat melemah atau bahkan berbalik. Penelitian ini juga menjelaskan
bahwa kelembapan relatif memiliki korelasi positif terhadap PG, sementara
kecepatan angin cenderung berkorelasi negatif, menunjukkan interaksi kompleks
antara faktor atmosfer dan topografi dalam pembentukan presipitasi. Secara
metodologis, pendekatan regresi linear digunakan untuk mengestimasi PG, serta
uji Mann-Kendall untuk mendeteksi tren dan perubahan abrupt. Temuan ini
menegaskan bahwa distribusi curah hujan di wilayah pegunungan tidak hanya
bergantung pada elevasi, tetapi juga pada dinamika atmosfer regional dan
interaksi orografis, sehingga memiliki implikasi penting dalam pemodelan
hidrologi dan analisis iklim di daerah dengan topografi kompleks.

15

8. Penelitian “Skilful precipitation nowcasting using deep generative models of
radar” oleh Ravuri et al. (2021) [6].
Penelitian ini mengembangkan pendekatan nowcasting curah hujan
berbasis deep learning menggunakan model generatif probabilistik dari data
radar untuk prediksi jangka sangat pendek (0–90 menit). Permasalahan utama
dalam nowcasting konvensional adalah ketergantungan pada metode adveksi
berbasis radar yang sulit menangkap fenomena non-linear seperti konveksi, serta
keterbatasan model deep learning sebelumnya yang menghasilkan prediksi kabur
(blurry) pada lead time panjang. Untuk mengatasi hal tersebut, penelitian ini
mengusulkan Deep Generative Model of Radar (DGMR) berbasis conditional
Generative Adversarial Network (GAN) yang mampu menghasilkan prediksi
probabilistik dengan mempertahankan konsistensi spasial dan temporal. Model
menggunakan empat citra radar sebelumnya (20 menit terakhir) sebagai input
dan menghasilkan hingga 18 frame prediksi (90 menit ke depan). Berdasarkan
evaluasi kuantitatif seperti Critical Success Index (CSI), Continuous Ranked
Probability Score (CRPS), serta analisis spektral, model DGMR menunjukkan
performa lebih baik dibanding metode baseline seperti PySTEPS dan UNet,
terutama dalam mempertahankan struktur curah hujan skala kecil dan intensitas
tinggi. Selain itu, berdasarkan studi evaluasi oleh lebih dari 50 meteorolog
profesional, model ini dipilih sebagai metode terbaik pada sekitar 89–93% kasus,
menunjukkan peningkatan signifikan dalam nilai operasional dan pengambilan
keputusan. Penelitian ini menunjukan bahwa model menggabungkan generator,
spatial discriminator, dan temporal discriminator untuk menjaga kualitas
prediksi. Meskipun demikian, tantangan masih ada dalam memprediksi curah
hujan intensitas tinggi pada lead time panjang. Penelitian ini menegaskan bahwa
pendekatan generatif berbasis deep learning memiliki potensi besar dalam
meningkatkan akurasi dan utilitas nowcasting dibanding metode tradisional.
9. Penelitian “FuXi-ENS: A machine learning model for efficient and accurate
ensemble weather prediction” oleh Zhong et al. (2025) [18].
Penelitian ini mengembangkan model prediksi cuaca berbasis machine
learning yang dirancang khusus untuk ensemble forecasting guna mengatasi
keterbatasan metode Numerical Weather Prediction (NWP) konvensional dalam
16

menangani ketidakpastian dan kebutuhan komputasi tinggi. Permasalahan utama
pada sistem ensemble tradisional adalah keterbatasan jumlah anggota ensemble
akibat biaya komputasi, sehingga distribusi probabilitas kondisi cuaca tidak
dapat direpresentasikan secara optimal. Untuk mengatasi hal tersebut, penelitian
ini mengusulkan FuXi-ENS, sebuah model berbasis variational autoencoder
(VAE) yang mampu menghasilkan prediksi ensemble global hingga 15 hari ke
depan dengan resolusi spasial 0.25°. Model ini menggunakan fungsi loss
kombinasi antara Continuous Ranked Probability Score (CRPS) dan KullbackLeibler (KL) divergence untuk menghasilkan perturbasi yang bergantung pada
kondisi

atmosfer

(flow-dependent

perturbations),

sehingga

mampu

merepresentasikan ketidakpastian secara lebih realistis dibanding pendekatan
deterministik. Berdasarkan hasil evaluasi terhadap berbagai metrik seperti
RMSE, Anomaly Correlation Coefficient (ACC), CRPS, dan Brier Score, FuXiENS menunjukkan performa yang lebih baik dibandingkan sistem ensemble
ECMWF, khususnya pada lead time pendek hingga menengah. Dalam penelitian
ini terlihat bahwa FuXi-ENS memiliki RMSE yang lebih rendah dan ACC yang
lebih tinggi dibanding ECMWF-ENS, terutama pada variabel suhu permukaan
(T2M). Selain itu, model ini juga menunjukkan keunggulan dalam prediksi
kejadian ekstrem seperti siklon tropis dan gelombang panas, dengan peningkatan
akurasi hingga 10–25% pada beberapa kasus. Dari sisi efisiensi, model ini sangat
signifikan karena mampu menghasilkan prediksi 15 hari hanya dalam sekitar 10
detik per anggota ensemble menggunakan GPU, jauh lebih cepat dibanding
metode NWP tradisional. Meskipun demikian, model ini masih memiliki
keterbatasan dalam estimasi spread ensemble yang cenderung underdispersed,
terutama pada lead time awal. Penelitian ini menunjukkan bahwa pendekatan
machine learning berbasis probabilistik memiliki potensi besar untuk
menggantikan atau melengkapi sistem prediksi cuaca konvensional dalam skala
global.
10. Penelitian “Deep Learning Model for Precipitation Nowcasting Based on
Residual and Attention Mechanisms” oleh Zhang et al. (2025) [23].
Penelitian ini mengusulkan model deep learning bernama RA-UNet untuk
meningkatkan akurasi prediksi curah hujan jangka pendek (nowcasting) berbasis
17

data radar. Permasalahan utama yang diangkat adalah keterbatasan metode
tradisional seperti optical flow dalam menangkap dinamika pembentukan dan
peluruhan hujan, serta kelemahan model deep learning sebelumnya seperti UNet yang cenderung menghasilkan prediksi yang kabur (blur) dan kurang akurat
dalam merepresentasikan intensitas hujan ekstrem. Untuk mengatasi hal
tersebut, RA-UNet dikembangkan dengan mengintegrasikan arsitektur U-Net
dengan residual network (ResNet) dan convolutional block attention module
(CBAM), sehingga mampu menangkap fitur spasial dan temporal secara lebih
efektif. Selain itu, model ini menggunakan depthwise separable convolution
untuk meningkatkan efisiensi komputasi tanpa mengorbankan performa. Model
dilatih menggunakan data radar reflectivity dari jaringan CINRAD di China
dengan resolusi temporal 6 menit dan total sekitar 37.000 citra radar.
Berdasarkan hasil evaluasi, RA-UNet menunjukkan peningkatan performa yang
signifikan dibandingkan model pembanding, dengan penurunan Mean Absolute
Error (MAE) sekitar 7% dan penurunan False Alarm Ratio (FAR) hingga 20%.
Penelitian ini menunjukan bahwa RA-UNet secara konsisten memiliki nilai
MAE lebih rendah dan SSIM lebih tinggi dibanding U-Net dan optical flow pada
berbagai lead time. Selain itu, model ini juga mampu mempertahankan struktur
spasial curah hujan dan lebih akurat dalam memprediksi area hujan lebat (>40
dBZ), meskipun masih cenderung meremehkan intensitas ekstrem (>50 dBZ).
Secara keseluruhan, penelitian ini menunjukkan bahwa kombinasi residual
learning dan attention mechanism mampu meningkatkan kemampuan model
dalam menangkap dinamika kompleks sistem presipitasi, terutama untuk
prediksi jangka pendek hingga 3 jam ke depan.
Tabel 2.1
No
Judul
1
Precipitation
Nowcasting
with
Generative
Diffusion
Models [29]

Tabel Penelitian Terdahulu
Metode
Hasil
Generative
Pendekatan
Ensemble
ensemble GED
Diffusion (GED)
secara konsisten
berbasis Diffusion mengungguli
Models dengan
baseline pada
post-processing U- metrik MSE dan
Net menggunakan recall untuk
dataset ERA5.
prediksi hingga 3
jam.
18

Keterangan
Pendekatan
generatif
probabilistik
untuk
nowcasting
presipitasi
kurang sensitif
terhadap
ekstrem.

No
Judul
2
Skilful
Nowcasting of
Extreme
Precipitation
with
NowcastNet [7]

Metode
Integrasi evolution
network berbasis
fisika dan
generative network
berbasis latent
stochastic
sampling dalam
kerangka generatif
bersyarat.

Hasil
Peningkatan
signifikan pada
intensitas hujan
> 16 mm/jam;
dipilih sebagai
model terbaik
oleh 67–76%
meteorolog
profesional
dibanding
baseline.

Keterangan
Model hybrid
untuk
nowcasting
presipitasi
ekstrem

3

FuXi-ENS: A
machine
learning model
for efficient
and accurate
ensemble
weather
prediction [18]

Ensemble weather
prediction berbasis
Variational
Autoencoder
(VAE) dengan
fungsi loss CRPS
dan KL
divergence.

Mengungguli
sistem ensemble
ECMWF pada
lead time
pendekmenengah
dengan
peningkatan
akurasi 10–25%
pada kejadian
ekstrem seperti
siklon.

Pendekatan
ensemble
probabilistik
berbasis
machine
learning

4

Nowcasting
Probabilistik
Presipitasi
Untuk Mitigasi
Risiko Pendaki
di Gunung
Gede–
Pangrango
Menggunakan
RetrievalAugmented
Diffusion
Model Dengan
SpatioTemporal
Graph
Conditioning

RetrievalAugmented
Diffusion Model
terintegrasi dengan
Spatio-Temporal
Graph
Conditioning
(Graph Attention
Network)

-

Penelitian Saat
Ini

19

Berdasarkan Tabel 2.2, ketiga penelitian tersebut menunjukkan kemajuan
penting dalam peramalan cuaca berbasis machine learning probabilistik. FuXi-ENS
[18] menekankan efisiensi dan akurasi ensemble prediction, Asperti et al. (2023)
[29] mengembangkan generative diffusion models untuk nowcasting presipitasi,
sedangkan Zhang et al. (2023) [7] fokus pada nowcasting presipitasi ekstrem
dengan pendekatan hybrid generatif. Meskipun ketiganya memberikan kontribusi
signifikan, penelitian tersebut masih terbatas pada data radar resolusi tinggi atau
wilayah non-pegunungan, serta belum mengintegrasikan retrieval-based historical
analogs dan spatio-temporal graph conditioning secara simultan dalam konteks
mikrocuaca pegunungan tropis. Penelitian ini mengevaluasi kombinasi tersebut
untuk konteks data reanalysis titik-lokasi di Gunung Gede–Pangrango.

2.2

Landasan Teori
Landasan teori dalam penelitian ini disusun untuk memberikan fondasi ilmiah

bagi pengembangan model nowcasting probabilistik presipitasi menggunakan
Retrieval-Augmented

Diffusion

Model

dengan

Spatio-Temporal

Graph

Conditioning. Bab ini membahas konsep-konsep utama yang menjadi dasar
penelitian, mulai dari prinsip nowcasting pada skala waktu jam-an, dinamika
presipitasi orografis di pegunungan tropis, hingga mekanisme model generatif
probabilistik berbasis diffusion. Selanjutnya, dijelaskan pula pendekatan retrievalbased historical analogs dan Spatio-Temporal Graph Conditioning dengan
representasi lima node yang dirancang untuk menangkap ketergantungan spasial
multidirectional di sekitar titik prediksi utama. Pembahasan ditutup dengan analisis
risiko hipotermia pada aktivitas pendakian di Gunung Gede–Pangrango serta
relevansinya dengan sistem peringatan dini berbasis nowcasting probabilistik.
Dengan landasan teori yang terfokus ini, penelitian diharapkan mampu
menghasilkan model yang tidak hanya unggul secara teknis, tetapi juga memberikan
kontribusi nyata dalam mitigasi risiko keselamatan pendaki di kawasan pegunungan
tropis.

20

2.2.1

Konsep Nowcasting dan Skala Waktu Prediksi Atmosfer
Nowcasting merupakan salah satu bidang paling dinamis dalam ilmu

prakiraan cuaca modern karena beroperasi pada batas prediktabilitas atmosfer yang
paling sensitif terhadap kondisi awal [6]. Pada skala waktu sangat pendek, atmosfer
tidak lagi sepenuhnya didominasi oleh dinamika sinoptik berskala besar, melainkan
juga belum sepenuhnya stabil dalam rezim statistik jangka menengah. Posisi antara
dua rezim ini menjadikan nowcasting sebagai domain yang memerlukan
pendekatan khusus dibandingkan prakiraan harian atau mingguan. Dalam konteks
penelitian ini yang berfokus pada mitigasi risiko pendakian di wilayah pegunungan
tropis, pemahaman konseptual mengenai nowcasting skala waktu per jam menjadi
landasan teoritis yang krusial [7].
Tabel 2.2
Skala
Prakiraan

Perbandingan Skala Prakiraan Atmosfer disusun [7] [19].
Contoh
Horizon
Karakteristik Utama
Keluaran

0–6 jam

Bergantung kuat pada
kondisi awal dan observasi
terkini; menargetkan
fenomena lokal yang
berubah cepat.

Prediksi
presipitasi satu
jam ke depan.

Short-range
forecasting

6–72 jam

Memadukan observasi dan
prediksi numerik untuk
evolusi cuaca jangka
pendek.

Prakiraan hujan
harian dan
peringatan
cuaca.

Mediumrange
forecasting

Lebih dari
72 jam
hingga
sekitar 15
hari

Menekankan dinamika
atmosfer skala besar dan
prediksi ensemble.

Prakiraan
probabilistik
regional atau
global.

Nowcasting

2.2.1.2 Definisi dan Ruang Lingkup Nowcasting
Dalam meteorologi operasional, nowcasting memiliki pengertian yang relatif
spesifik dan dibedakan dari prakiraan cuaca pada skala waktu lain. Organisasi
Meteorologi Dunia (WMO) mendefinisikan nowcasting sebagai prakiraan cuaca
dengan detail lokal pada skala mesoscale dan skala yang lebih kecil, mencakup
kondisi saat ini hingga sekitar 6 jam ke depan. Definisi ini menekankan tiga aspek
utama: fokus pada horizon waktu sangat pendek (0–6 jam), resolusi spasial tinggi

21

dengan detail lokal yang relevan untuk kejadian berisiko tinggi, serta
ketergantungan kuat pada informasi pengamatan terkini beresolusi tinggi [6].
Dalam penelitian ini, nowcasting difokuskan pada prediksi satu jam ke depan
dengan menggunakan jendela observasi 6 jam terakhir sebagai input [7].
Pendekatan ini sesuai dengan definisi WMO sekaligus selaras dengan karakteristik
dinamika cuaca di kawasan pegunungan, di mana ancaman utama seperti hujan
orografis dan peningkatan kecepatan angin dapat berkembang dengan sangat cepat
[11]. Horizon satu jam memberikan keseimbangan optimal antara kecepatan
respons dan akurasi prediksi untuk mendukung mitigasi risiko pendakian [1].
Horizon nowcasting yang pendek ini secara konseptual berada pada rezim
initial condition-dominated, di mana akurasi prediksi sangat bergantung pada
kualitas data observasi terkini dan kemampuan model dalam menangkap pola
evolusi cuaca jangka pendek [6], [7]. Berbeda dengan short-range forecasting (6–
72 jam) yang lebih bergantung pada model Numerical Weather Prediction (NWP),
nowcasting lebih mengandalkan data-driven learning dan ekstrapolasi dari
observasi resolusi tinggi [30]. Pemahaman ini menjadi dasar pemilihan pendekatan
probabilistik generatif dalam penelitian ini.
2.2.1.3 Karakteristik Dinamika pada Skala Waktu Per Jam di Wilayah
Pegunungan
Skala waktu 0–6 jam erat kaitannya dengan dinamika atmosfer pada
mesoscale dan convective scale [7]. Pada skala ini, proses konvektif seperti
convective initiation sangat sensitif terhadap fluktuasi kecil dalam kelembapan,
suhu, dan dinamika angin lokal [13]. Di wilayah orografis tropis seperti Gunung
Gede–Pangrango, dinamika ini semakin kompleks karena adanya efek orografis
lifting, siklus diurnal konveksi yang kuat, serta interaksi angin lereng yang
menyebabkan variabilitas spasial dan temporal yang tinggi [11], [14].
Lifetime sistem konvektif umumnya berada pada orde puluhan menit hingga
beberapa jam [5]. Sel konvektif tunggal dapat hidup sekitar 30–60 menit, sedangkan
sistem konveksi mesoskal dapat bertahan 3–6 jam atau lebih. Oleh karena itu,
prakiraan pada horizon satu jam harus mampu merepresentasikan pembentukan sel
baru, pergerakan sel yang ada, serta perubahan intensitas presipitasi secara akurat

22

dan cepat [4], [20]. Rapid error growth akibat sifat nonlinier dan kaotik atmosfer
semakin membatasi prediktabilitas deterministik pada skala ini [16].
Di kawasan Gunung Gede–Pangrango, efek topografi memperkuat dinamika
mesoskal [11]. Gradien elevasi yang curam menyebabkan perbedaan kondisi
meteorologis yang signifikan dalam jarak horizontal yang relatif pendek [14].
Variabilitas spasial yang tinggi ini menjadikan prakiraan per jam sebagai skala
paling relevan untuk mitigasi risiko pendakian, karena ancaman cuaca dapat
berkembang dan berubah secara cepat sepanjang lintasan pendakian dari hilir
menuju puncak [1].
2.2.1.4 Keterbatasan Pendekatan Deterministik dan Relevansi Pendekatan
Probabilistik
Model Numerical Weather Prediction (NWP) dan pendekatan deterministik
berbasis machine learning sering mengalami keterbatasan signifikan pada lead time
pendek untuk fenomena konvektif [16]. Siklus pembaruan model yang relatif
lambat, resolusi grid yang terbatas, serta kecenderungan menghasilkan prediksi
yang terlalu halus (over-smoothing) menyebabkan under-prediction pada kejadian
presipitasi ekstrem [17], [30]. Model regresi yang dioptimalkan dengan fungsi
kerugian berbasis mean squared error cenderung bias ke nilai rata-rata distribusi,
sehingga kurang sensitif terhadap distribusi heavy-tailed presipitasi yang menjadi
ciri khas hujan orografis di pegunungan [6], [29].
Selain itu, pendekatan deterministik sulit merepresentasikan ketidakpastian
intrinsik atmosfer pada skala per jam [7], [16]. Ketidakpastian ini berasal dari initial
condition uncertainty, model structural error, observation error, dan chaotic error
growth yang tumbuh secara nonlinier [19], [31]. Akibatnya, model deterministik
sering gagal memberikan informasi yang cukup bagi pengambilan keputusan
berbasis risiko, terutama dalam konteks keselamatan pendaki di lingkungan
pegunungan yang dinamis [1].
Pendekatan probabilistik, seperti generative diffusion models, menawarkan
solusi yang lebih sesuai karena mampu memodelkan distribusi kemungkinan
kondisi atmosfer masa depan daripada hanya menghasilkan satu trajektori
deterministik [29]. Pendekatan ini selaras dengan sifat stokastik atmosfer pada skala
per jam dan mendukung pengambilan keputusan berbasis risiko, khususnya dalam
23

mitigasi hipotermia pendaki [1]. Oleh karena itu, pemilihan Retrieval-Augmented
Diffusion Model dengan Spatio-Temporal Graph Conditioning dalam penelitian ini
memiliki justifikasi teoritis yang kuat [7].
Tabel 2.3
Aspek

Perbandingan Keluaran Deterministik dan Probabilistik pada
Nowcasting [4], [19], [27].
Pendekatan
Deterministik

Pendekatan Probabilistik

Keluaran

Satu prediksi titik untuk
setiap waktu dan lokasi.

Ensemble atau distribusi
beberapa prediksi yang
mungkin.

Ketidakpastian

Tidak dinyatakan secara
eksplisit.

Direpresentasikan melalui
sebaran antar anggota ensemble.

Kegunaan

Menilai nilai prakiraan
tunggal.

Mendukung keputusan berbasis
peluang dan tingkat risiko.

Metrik utama

MAE, RMSE, korelasi.

CRPS, Brier Score, reliability
diagram, serta metrik
deterministik.

Keterbatasan

Cenderung meratakan
nilai pada kondisi tidak
pasti.

Memerlukan evaluasi kalibrasi
dan biaya sampling lebih tinggi.

2.2.2

Dinamika Presipitasi Orografis di Pegunungan Tropis
Presipitasi orografis merupakan fenomena utama yang membentuk pola curah

hujan di wilayah pegunungan, terutama di daerah tropis seperti Gunung Gede–
Pangrango [11], [32]. Interaksi antara aliran angin lembab dengan topografi
menghasilkan angkat paksa (orographic lifting) yang memicu pembentukan awan
dan hujan intens [33]. Di kawasan tropis, proses ini sering diperkuat oleh
kelembapan tinggi dan ketidakstabilan atmosfer, sehingga menghasilkan distribusi
presipitasi yang sangat heterogen dalam skala spasial dan temporal yang kecil [11],
[29]. Pemahaman dinamika ini menjadi krusial dalam penelitian nowcasting
probabilistik karena presipitasi orografis cenderung berkembang cepat dan sulit
diprediksi dengan model deterministik.
Di Gunung Gede–Pangrango, variabilitas elevasi yang tajam dari dataran
rendah hingga puncak menciptakan gradien meteorologis yang signifikan [32]. Efek
orografis tidak hanya meningkatkan intensitas hujan di lereng angin (windward
24

slope), tetapi juga menghasilkan bayangan hujan (rain shadow) di sisi lereng
lindung (leeward) [11], [33]. Fenomena ini menyebabkan perbedaan kondisi cuaca
yang ekstrem dalam jarak horizontal relatif pendek, sehingga sangat relevan dengan
risiko hipotermia pendaki yang melintasi berbagai elevasi dalam waktu singkat
[28].
Dinamika presipitasi orografis di pegunungan tropis juga dipengaruhi oleh
siklus diurnal dan interaksi dengan monsun, yang mempercepat convective
initiation pada skala per jam [34]. Proses ini menghasilkan presipitasi dengan
distribusi heavy-tailed, di mana kejadian ekstrem lebih sering terjadi dibandingkan
distribusi normal [6], [7]. Oleh karena itu, model nowcasting yang digunakan dalam
penelitian ini harus mampu menangkap ketergantungan spasial-temporal antar
elevasi (hilir, lereng, dan puncak) melalui pendekatan Spatio-Temporal Graph
Conditioning.
2.2.2.1 Mekanisme Orographic Lifting dan Convective Initiation
Mekanisme utama presipitasi orografis adalah angkat paksa udara lembab
oleh topografi, yang menyebabkan pendinginan adiabatik dan kondensasi [33]. Di
wilayah tropis, angkat orografis ini sering memicu convective initiation ketika
udara mencapai tingkat kebebasan konveksi (Level of Free Convection) [11].
Kecepatan angin lintang lereng (cross-slope wind) memainkan peran penting;
semakin kuat angin, semakin dalam penetrasi gangguan gravitasi orografis yang
mendinginkan dan melembabkan troposfer bawah, sehingga meningkatkan curah
hujan hingga 20–30% per m/s peningkatan kecepatan angin [29], [33].
Berdasarkan sintesis mekanisme orographic lifting oleh Nicolas dan Boos
[32], hubungan konseptual antara intensitas presipitasi orografis dan faktor
atmosfer serta topografi dapat dinyatakan sebagai berikut:
𝑃𝑜𝑟𝑜𝑔𝑟𝑎𝑓𝑖𝑠 = 𝑓(𝑈𝑐𝑟𝑜𝑠𝑠 , 𝑞, 𝐻, 𝑆, 𝐶)
dengan:
• 𝑃𝑜𝑟𝑜𝑔𝑟𝑎𝑓𝑖𝑠 = intensitas presipitasi orografis;
• 𝑈𝑐𝑟𝑜𝑠𝑠 = komponen kecepatan angin yang tegak lurus terhadap lereng;
• 𝑞 = kelembapan udara;
• 𝐻 = karakteristik elevasi/topografi;
• 𝑆 = orientasi dan kemiringan lereng;
25

(2.1)

• 𝐶 = kondisi konvektif atmosfer.
Rumus ini merupakan representasi konseptual, bukan persamaan empiris
untuk menghitung curah hujan di Gunung Gede-Pangrango. Sumber: disusun dari
Nicolas dan Boos [32].
Di Gunung Gede–Pangrango, gradien elevasi yang curam memperkuat proses
lifting ini, terutama pada lereng selatan dan barat yang sering menjadi windward
terhadap aliran monsun [32]. Interaksi antara angin lembab dengan topografi
menghasilkan konvergensi lokal yang memicu sel konvektif baru dalam waktu
singkat. Proses ini sangat dinamis pada skala per jam, sehingga memerlukan
pendekatan probabilistik untuk merepresentasikan ketidakpastian convective
initiation [6], [7].
Selain mekanis, faktor termal seperti pemanasan radiasi lereng pada siang hari
juga berkontribusi terhadap convective initiation di pegunungan tropis [34].
Kombinasi forcing mekanis dan termal ini menghasilkan variabilitas spasial yang
tinggi, di mana hujan lebat dapat terkonsentrasi di lereng tertentu sementara area
berdekatan tetap relatif kering [11]. Pemahaman mekanisme ganda ini mendukung
penggunaan graph neural network untuk memodelkan ketergantungan antar titik
elevasi yang berbeda dalam arsitektur model penelitian ini.
2.2.2.2 Variabilitas Spasial dan Temporal Presipitasi Orografis
Presipitasi orografis di pegunungan tropis menunjukkan variabilitas spasial
yang sangat tinggi akibat perbedaan elevasi, orientasi lereng, dan interaksi angin
[14], [32]. Di Gunung Gede–Pangrango, curah hujan cenderung meningkat seiring
kenaikan elevasi pada lereng angin hingga mencapai maksimum di zona tengah
lereng, kemudian menurun di puncak atau sisi lindung [11], [33]. Variabilitas ini
menciptakan microclimate yang berbeda antar elevasi, yang berdampak langsung
pada kondisi termal dan kelembapan yang dialami pendaki.
Secara temporal, presipitasi orografis di wilayah tropis sering mengikuti
siklus diurnal yang kuat, dengan puncak hujan pada sore hingga malam hari akibat
pemanasan siang dan pelepasan ketidakstabilan [34]. Namun, pada kondisi angin
lintas lereng yang kuat, siklus diurnal dapat tergantikan oleh forcing mekanis yang
lebih dominan, sehingga hujan dapat terjadi kapan saja dalam skala per jam [33].

26

Karakteristik ini menjadikan nowcasting satu jam ke depan sangat relevan untuk
mendeteksi perubahan mendadak yang berbahaya bagi keselamatan pendakian.
Distribusi temporal presipitasi orografis juga cenderung heavy-tailed, dengan
kejadian ekstrem yang jarang namun berdampak besar [6], [7]. Kejadian hujan lebat
yang dipicu orografi sering bersifat quasi-stationary, di mana sistem hujan bertahan
lama di lereng tertentu akibat blocking dan lifting berulang [17], [32]. Variabilitas
ini menekankan pentingnya pendekatan retrieval-augmented untuk menangkap
analog historis kejadian ekstrem dalam model diffusion yang dikembangkan.
2.2.2.3 Relevansi dengan Nowcasting Probabilistik dan Mitigasi Risiko
Dinamika presipitasi orografis yang kompleks dan cepat berubah membatasi
kemampuan model deterministik dalam memprediksi kejadian ekstrem di
pegunungan tropis [7], [16]. Model regresi cenderung under-predict presipitasi
berintensitas tinggi karena smoothing efek, sementara pendekatan probabilistik
mampu menghasilkan distribusi kemungkinan yang lebih realistis [29]. Hal ini
sangat penting untuk nowcasting di Gunung Gede–Pangrango, di mana hujan
orografis dapat mempercepat kehilangan panas tubuh pendaki.
Spatio-temporal variability yang tinggi antar elevasi memerlukan representasi
graph-based untuk menangkap ketergantungan antar node [7]. Pendekatan SpatioTemporal Graph Conditioning memungkinkan model mempelajari bagaimana
presipitasi di satu elevasi memengaruhi elevasi lain melalui message passing,
sehingga meningkatkan sensitivitas terhadap kejadian ekstrem [6], [7].
Dengan demikian, pemahaman mendalam tentang dinamika presipitasi
orografis memberikan landasan teoritis yang kuat bagi Retrieval-Augmented
Diffusion Model dalam menghasilkan prediksi probabilistik yang dapat mendukung
sistem peringatan dini mitigasi risiko hipotermia pendaki di kawasan pegunungan
tropis seperti Gunung Gede–Pangrango [1], [33].
2.2.3

Model Generatif Probabilistik dengan Diffusion Models
Model generatif probabilistik semakin menjadi pilihan utama dalam

nowcasting presipitasi karena kemampuannya menangani ketidakpastian intrinsik
atmosfer pada skala per jam [6], [29]. Berbeda dengan model deterministik yang
cenderung menghasilkan prediksi rata-rata dan over-smoothing, model generatif
mampu menghasilkan distribusi kemungkinan skenario cuaca masa depan yang
27

lebih realistis [19], [31]. Dalam konteks penelitian ini, pendekatan diffusion models
dipilih karena keunggulannya dalam memodelkan distribusi heavy-tailed presipitasi
orografis di pegunungan tropis seperti Gunung Gede–Pangrango [7], [29].
Diffusion models bekerja berdasarkan proses forward yang secara bertahap
menambahkan noise Gaussian ke data hingga menjadi noise murni, diikuti proses
reverse (denoising) yang mempelajari cara mengembalikan noise menjadi data asli
secara probabilistik [19], [29]. Proses ini memungkinkan model menghasilkan
sampel yang beragam, sehingga sangat sesuai untuk menghasilkan ensemble
prediksi nowcasting yang mencerminkan ketidakpastian alamiah atmosfer [6], [7].
Kemampuan ini menjadi landasan penting bagi pengembangan RetrievalAugmented Diffusion Model dalam penelitian ini.
Mengacu pada formulasi forward diffusion yang digunakan oleh Asperti et al.
[27], proses penambahan noise Gaussian secara bertahap pada data dapat
dinyatakan sebagai berikut:
𝑞(𝑥𝑡 |𝑥𝑡−1 ) = 𝑁(𝑥𝑡 ; √(1 − 𝛽𝑡 ) 𝑥𝑡−1 , 𝛽𝑡 𝐼)

(2.2)

Selanjutnya, sesuai objektif pelatihan diffusion model yang dijelaskan oleh
Asperti et al. [27], fungsi kerugian untuk meminimalkan perbedaan antara noise
aktual dan noise prediksi dapat dinyatakan sebagai berikut:
2

𝐿𝑑𝑖𝑓𝑓 = 𝐸𝑥0 ,𝜖,𝑡 [||𝜖 − 𝜖𝜃(𝑥𝑡,𝑡,𝑐) || ]

(2.3)

2

dengan:
• 𝑥0 = data target asli;
• 𝑥𝑡 = data pada diffusion timestep 𝑡;
• 𝛽𝑡 = varians noise pada timestep 𝑡;
• 𝐼 = matriks identitas;
• 𝜖 = noise Gaussian yang ditambahkan;
• 𝜖 𝜃 = noise yang diprediksi model dengan parameter theta; dan
• 𝑐 = kondisi tambahan, berupa representasi historis, spasial, dan retrieval.
Penerapan diffusion models pada nowcasting presipitasi telah menunjukkan
peningkatan performa signifikan dibandingkan pendekatan generatif sebelumnya
seperti Generative Adversarial Networks (GAN), terutama dalam hal kualitas
sampel dan stabilitas pelatihan [27], [30]. Model ini mampu menghasilkan prediksi
28

yang lebih tajam dan lebih sensitif terhadap kejadian ekstrem, yang sangat
dibutuhkan untuk mitigasi risiko hipotermia pendaki di kawasan dengan dinamika
orografis yang kompleks [5], [29].
2.2.3.1 Prinsip Kerja dan Keunggulan Diffusion Models untuk Nowcasting
Diffusion models terdiri dari dua proses utama yaitu forward diffusion
process dan reverse denoising process [27]. Pada proses forward, data input seperti
citra radar atau field presipitasi secara bertahap ditambahkan noise hingga menjadi
distribusi Gaussian murni. Proses reverse kemudian mempelajari langkah demi
langkah untuk menghilangkan noise tersebut dan merekonstruksi data asli yang
kondisional terhadap observasi terkini [4], [17].
Keunggulan utama diffusion models dibandingkan model generatif lain
adalah kemampuannya menghindari mode collapse dan menghasilkan sampel
dengan kualitas tinggi serta variasi yang realistis [27], [30]. Dalam nowcasting
presipitasi, kemampuan ini sangat penting karena presipitasi pada skala per jam
bersifat sangat stokastik dan memiliki distribusi heavy-tailed [5], [29]. Model ini
mampu

menghasilkan

multiple

plausible

realizations

yang

lebih

baik

merepresentasikan ketidakpastian prediksi.
Selain itu, diffusion models dapat dengan mudah dikondisikan terhadap
variabel tambahan seperti kondisi atmosfer sebelumnya, angin, kelembapan, dan
informasi spasial elevasi [4], [27]. Kondisional ini memungkinkan integrasi dengan
Spatio-Temporal Graph Conditioning, sehingga model dapat mempelajari
ketergantungan antar titik observasi di Gunung Gede–Pangrango [5], [17].
2.2.3.2 Keterbatasan Model Deterministik dan Relevansi Pendekatan
Generatif
Pendekatan deterministik berbasis regresi atau Numerical Weather Prediction
(NWP) sering mengalami keterbatasan pada lead time pendek karena
kecenderungan menghasilkan prediksi yang terlalu halus dan kurang sensitif
terhadap kejadian ekstrem [14], [15]. Model ini bias terhadap nilai rata-rata
distribusi, sehingga under-predict intensitas presipitasi tinggi yang sering memicu
risiko hipotermia pada pendaki [29], [30].
Machine learning weather prediction models saat ini juga masih menghadapi
tantangan dalam merepresentasikan ketidakpastian dan rapid error growth pada
29

skala per jam [14], [17]. Pendekatan probabilistik generatif menjadi solusi karena
mampu memodelkan distribusi penuh kemungkinan daripada hanya satu nilai
prediksi tunggal [27], [4].
Dalam konteks pegunungan tropis, di mana presipitasi orografis bersifat
sangat lokal dan cepat berubah, pendekatan generatif seperti diffusion models
menawarkan fleksibilitas yang lebih tinggi untuk menangkap pola kompleks yang
sulit dimodelkan secara fisika murni [5], [9]. Hal ini menjadikan diffusion models
sebagai pilihan yang sesuai untuk mendukung sistem peringatan dini berbasis
risiko.
2.2.3.3 Relevansi dengan Retrieval-Augmented Diffusion Model
Retrieval-Augmented Diffusion Model menggabungkan kekuatan diffusion
models dengan mekanisme retrieval historis untuk meningkatkan kualitas prediksi
pada kejadian ekstrem [27], [30]. Dengan mengambil analog historis serupa dari
database kejadian masa lalu, model dapat memperkaya proses denoising sehingga
lebih akurat dalam memprediksi presipitasi intens di kawasan orografis [4], [5].
Integrasi retrieval mechanism membantu mengatasi keterbatasan data
pelatihan pada kejadian langka (rare events) yang sangat berbahaya bagi pendaki
[17], [29]. Kombinasi ini memungkinkan model menghasilkan ensemble prediksi
yang lebih terkalibrasi dan sensitif terhadap kondisi ekstrem di Gunung Gede–
Pangrango.
Dengan demikian, Retrieval-Augmented Diffusion Model dengan Spatio-Temporal
Graph Conditioning digunakan untuk mengevaluasi kemampuan diffusion model
dalam menghasilkan prediksi probabilistik yang beragam pada lingkungan
pegunungan tropis yang dinamis [27], [5], [28].

2.2.4

Retrieval-Based Historical Analogs
Retrieval-based historical analogs merupakan pendekatan penting dalam

meningkatkan kualitas prediksi pada kejadian ekstrem yang jarang terjadi dalam
data pelatihan [27], [30]. Pendekatan ini bekerja dengan mencari kondisi historis
yang mirip (analogs) dari database observasi masa lalu, kemudian menggunakan
informasi tersebut untuk mengkondisikan proses generasi model [4], [17]. Dalam
penelitian ini, mekanisme retrieval diintegrasikan ke dalam arsitektur Diffusion
30

Model untuk meningkatkan akurasi dan keandalan prediksi probabilistik presipitasi
pada skala per jam di kawasan Gunung Gede–Pangrango [5], [27].
Teknik retrieval memanfaatkan kemiripan fitur antara kondisi atmosfer saat
ini dengan kejadian historis menggunakan jarak Euclidean L2 pada ruang fitur yang
telah

direduksi.

Pencarian

nearest

neighbor

dilakukan

dengan

FAISS

`IndexFlatL2`. Dengan mengambil beberapa analogs terbaik, model dapat
memperoleh informasi tambahan mengenai evolusi presipitasi, angin, dan
kelembapan yang pernah terjadi pada kondisi serupa di masa lalu [17], [4].
Pendekatan ini membantu menyediakan konteks historis, namun tidak
menghilangkan keterbatasan jumlah kejadian presipitasi ekstrem dalam data.
Dalam pemeringkatan analog historis menggunakan Euclidean L2, jarak
antara kondisi saat ini dan kandidat analog dapat dihitung sebagai berikut:
𝑑(𝑥𝑞 , 𝑥𝑖 ) = √∑𝑚
𝑗=1 (𝑥𝑞,𝑗 − 𝑥𝑖,𝑗 )

2

(2.4)

dengan:
𝑥𝑞 = vektor fitur kondisi cuaca saat ini;
𝑥𝑖 = vektor fitur kejadian historis ke-i;
𝑚 = jumlah dimensi fitur; dan
𝑑(𝑥𝑞 , 𝑥𝑖 ) = jarak Euclidean L2.
Sebanyak 𝑘 kejadian historis dengan nilai jarak terkecil dipilih sebagai analog
[17].
Integrasi retrieval mechanism dengan diffusion model menghasilkan
arsitektur Retrieval-Augmented Diffusion Model untuk mengevaluasi manfaat
konteks historis terhadap prediksi ensemble [27], [5]. Pendekatan ini
menggabungkan kemampuan generatif diffusion dengan pengetahuan historis
spesifik lokasi pegunungan tropis [16], [30].
2.2.4.1 Prinsip Kerja Retrieval-Based Historical Analogs
Retrieval-based historical analogs bekerja dengan membangun database
indeks dari seluruh data historis observasi atmosfer menggunakan struktur data
efisien seperti FAISS (Facebook AI Similarity Search) [27], [30]. Pada saat
inferensi, kondisi atmosfer saat ini (current state) di-query ke database untuk
31

mencari k nearest neighbors berdasarkan kesamaan fitur multivariat, termasuk
presipitasi lag, kelembapan, suhu, dan kecepatan angin [17], [4].
Setelah analogs historis berhasil diambil, informasi dari analogs tersebut
kemudian digunakan sebagai conditioning tambahan pada proses denoising
diffusion model [5], [27]. Kondisional ini membantu model membimbing proses
reverse diffusion menuju trajektori yang lebih realistis, terutama ketika menghadapi
kejadian di luar distribusi umum data pelatihan [29], [30].
Keunggulan utama pendekatan ini adalah kemampuannya meningkatkan
performa prediksi pada rare events tanpa harus meningkatkan ukuran model secara
signifikan [16], [17]. Dalam konteks nowcasting satu jam ke depan, retrieval
analogs memberikan konteks historis yang berharga untuk memprediksi evolusi
presipitasi orografis di sekitar Gunung Gede–Pangrango.
2.2.4.2 Keunggulan Retrieval-Augmented dibandingkan Pure Generative
Models
Pure generative models seperti diffusion models standar sering mengalami
kesulitan dalam memprediksi kejadian ekstrem karena keterbatasan jumlah sampel
langka dalam dataset pelatihan [14], [27]. Akibatnya, model cenderung
menghasilkan prediksi yang kurang tajam atau bias terhadap pola rata-rata [15],
[29].
Retrieval-Augmented

approach

mengatasi

masalah

tersebut

dengan

menyuntikkan pengetahuan historis langsung ke dalam proses generasi [27], [30].
Dengan menyediakan analogs yang relevan, model dapat mempelajari pola transisi
dari kondisi serupa di masa lalu, sehingga meningkatkan kemampuan generalisasi
pada kasus ekstrem yang berbahaya bagi pendaki [5], [17].
Selain itu, pendekatan retrieval juga meningkatkan interpretabilitas model
karena dapat menunjukkan analogs historis mana yang paling berpengaruh terhadap
prediksi saat ini [16], [4]. Hal ini memberikan nilai tambah bagi pengembangan
sistem decision support untuk mitigasi risiko hipotermia di kawasan pegunungan.
2.2.5

Spatio-Temporal Graph Conditioning pada Representasi Data Elevasi
Representasi data spatio-temporal menjadi sangat penting dalam nowcasting

presipitasi di wilayah pegunungan karena adanya ketergantungan kuat antar lokasi
yang berbeda posisi relatif terhadap titik observasi utama [9], [12]. Dalam penelitian
32

ini, lima titik observasi direpresentasikan sebagai node dalam graph, yaitu MAIN,
UP, DOWN, LEFT, dan RIGHT, dengan node MAIN sebagai target prediksi utama
[5], [31]. Pendekatan Spatio-Temporal Graph Conditioning memungkinkan model
mempelajari hubungan spasial dan temporal antar kelima node ini secara simultan,
sehingga lebih mampu menangkap dinamika lokal presipitasi orografis di sekitar
lokasi pendakian Gunung Gede–Pangrango [17], [30].
Graph Neural Network (GNN), khususnya Graph Attention Network (GAT),
merepresentasikan kelima titik observasi sebagai node dalam graph, dengan node
MAIN sebagai pusat perhatian utama [14], [5]. Node UP, DOWN, LEFT, dan
RIGHT merepresentasikan posisi relatif di sekitar node utama, sehingga
memungkinkan model menangkap pengaruh angin, propagasi awan, dan presipitasi
dari berbagai arah terhadap kondisi di titik MAIN [9], [12]. Struktur graph ini lebih
fleksibel dan komprehensif dibandingkan representasi linier sederhana untuk
memodelkan lingkungan mikro di pegunungan tropis.
Integrasi Spatio-Temporal Graph Conditioning ke dalam arsitektur RetrievalAugmented Diffusion Model meningkatkan kemampuan model dalam menangani
variabilitas spasial tinggi yang menjadi ciri khas presipitasi orografis [31], [9].
Pendekatan ini sangat relevan untuk mitigasi risiko hipotermia karena
memungkinkan prediksi yang lebih akurat pada titik MAIN, yang dapat mewakili
posisi pendaki atau lokasi krusial di sepanjang jalur pendakian [28], [15].
2.2.5.1 Representasi Graph untuk Data Elevasi dengan Lima Node
Dalam model ini, lima titik observasi direpresentasikan sebagai node dalam
graph spatio-temporal, yaitu node MAIN sebagai target utama, serta node UP,
DOWN, LEFT, dan RIGHT yang merepresentasikan posisi relatif di sekitarnya
[12], [31]. Setiap node dilengkapi dengan fitur meteorologis seperti presipitasi lag,
suhu udara, kelembapan relatif, kecepatan angin, dan variabel atmosfer lainnya
pada beberapa waktu observasi sebelumnya [5], [17]. Node MAIN menjadi pusat
perhatian karena menjadi lokasi utama yang diprediksi.
Graph Attention Network (GAT) digunakan untuk memproses graph ini
karena kemampuannya memberikan bobot perhatian yang adaptif terhadap node
tetangga [14], [30]. Melalui mekanisme attention, model dapat mempelajari
seberapa besar pengaruh node UP, DOWN, LEFT, dan RIGHT terhadap evolusi
33

kondisi cuaca di node MAIN pada horizon satu jam ke depan [5], [17]. Struktur
graph menggunakan topologi star: node MAIN terhubung dua arah dengan UP,
DOWN, LEFT, dan RIGHT, sedangkan node tetangga tidak saling terhubung.
Tabel 2.4
Node

Definisi Node pada Representasi Graph Lima Node [9], [12], [31]
Posisi Relatif
Fungsi dalam Model
Koneksi Graph
terhadap MAIN
Titik prediksi
utama.

Menyediakan target
curah hujan, kecepatan
angin, dan kelembapan
relatif pada $t+1$.

Terhubung dua arah
dengan UP, DOWN,
LEFT, RIGHT.

UP

Utara dari MAIN.

Menyediakan konteks
kondisi atmosfer dari
arah utara.

Terhubung dua arah
dengan MAIN

DOWN

Selatan dari
MAIN.

LEFT

Barat dari MAIN.

RIGHT

Timur dari
MAIN.

MAIN

Menyediakan konteks
kondisi atmosfer dari
arah selatan.
Menyediakan konteks
kondisi atmosfer dari
arah barat.
Menyediakan konteks
kondisi atmosfer dari
arah timur.

Terhubung dua arah
dengan MAIN
Terhubung dua arah
dengan MAIN
Terhubung dua arah
dengan MAIN

2.2.5.2 Keunggulan Spatio-Temporal Graph dibandingkan Pendekatan
Konvensional
Pendekatan sequence-based atau grid-based konvensional sering kali kurang
efektif dalam menangkap ketergantungan spasial non-Euclidean yang kompleks di
wilayah pegunungan [14], [15]. Model berbasis CNN atau LSTM umumnya
mengasumsikan hubungan spasial tetap atau hanya lokal, sehingga sulit
memodelkan interaksi multidirectional dari berbagai arah terhadap satu titik utama
[29], [30].
Spatio-Temporal Graph Conditioning mampu mengatasi keterbatasan
tersebut dengan secara eksplisit memodelkan hubungan antar node melalui message
passing dan attention mechanism [5], [17]. Pendekatan ini memungkinkan model
mempelajari pola propagasi presipitasi orografis dan pengaruh angin dari berbagai
arah secara lebih akurat pada skala per jam [9], [31].
34

Integrasi graph conditioning dengan diffusion model memungkinkan proses
denoising dilakukan secara kondisional terhadap struktur graph lima node, sehingga
menghasilkan prediksi ensemble yang lebih konsisten secara spasial dan lebih
sensitif terhadap kondisi lokal di sekitar node MAIN [27], [5]. Keunggulan ini
sangat penting untuk meningkatkan akurasi prediksi pada titik krusial yang
digunakan untuk mitigasi risiko.
2.2.6

Risiko Hipotermia dan Mitigasi Pendakian
Risiko hipotermia merupakan ancaman serius bagi keselamatan pendaki di

pegunungan, termasuk di kawasan tropis seperti Gunung Gede–Pangrango [28].
Hipotermia terjadi ketika suhu inti tubuh manusia turun di bawah 35°C akibat
kehilangan panas yang melebihi produksi panas tubuh [28], [34]. Di pegunungan
tropis, hipotermia sering disebabkan oleh kombinasi hujan deras, angin kencang,
dan kelembapan tinggi yang mempercepat kehilangan panas melalui konduksi,
konveksi, dan evaporasi, meskipun suhu udara tidak ekstrem seperti di pegunungan
tinggi bersalju [28], [22].
Kondisi cuaca di Gunung Gede–Pangrango yang dinamis, terutama
presipitasi orografis disertai angin, dapat menyebabkan hipotermia “wet-cold”
(hipotermia basah-dingin) dengan cepat [28]. Pakaian yang basah kehilangan
kemampuan isolasi hingga 90%, sementara angin memperbesar efek wind chill,
sehingga mempercepat penurunan suhu tubuh pendaki meskipun berada di elevasi
sedang hingga tinggi [28], [34]. Kejadian ini semakin berbahaya karena pendaki
sering berada dalam kondisi kelelahan setelah berjalan jauh, sehingga kemampuan
tubuh untuk menghasilkan panas menurun secara signifikan [22].
Nowcasting probabilistik presipitasi, kecepatan angin, dan kelembapan relatif
menjadi sangat relevan sebagai sistem peringatan dini. Prediksi satu jam ke depan
dengan pendekatan probabilistik memungkinkan pendeteksian dini kombinasi
faktor berisiko tinggi sebelum kondisi hipotermia berkembang [4], [5]. Penelitian
ini menghubungkan output model Retrieval-Augmented Diffusion dengan SpatioTemporal Graph Conditioning secara langsung dengan mitigasi risiko keselamatan
pendaki.

35

2.2.6.1 Mekanisme Hipotermia di Lingkungan Pegunungan Tropis
Hipotermia di pegunungan tropis berbeda dengan kasus klasik di daerah
dingin karena suhu udara relatif hangat, namun kelembapan tinggi dan hujan deras
menjadi faktor dominan [28]. Mekanisme utama meliputi konduksi panas langsung
ke pakaian dan kulit yang basah, konveksi yang dipercepat oleh angin, serta
evaporasi yang terus berlangsung pada pakaian lembab [28], [34]. Kombinasi ketiga
mekanisme ini menyebabkan kehilangan panas yang cepat bahkan pada suhu udara
di atas 10–15°C.
Di Gunung Gede–Pangrango, efek orografis sering menghasilkan hujan lebat
secara tiba-tiba disertai peningkatan kecepatan angin seiring kenaikan elevasi [9],
[31]. Kondisi ini sangat berbahaya bagi pendaki karena perubahan cuaca dapat
terjadi dalam waktu singkat sepanjang jalur pendakian, dari zona hilir hingga
menuju puncak [28]. Selain itu, kelelahan fisik dan penurunan kadar gula darah
akibat aktivitas berkepanjangan semakin memperbesar kerentanan terhadap
hipotermia [22], [34].
Gejala awal hipotermia seperti menggigil, kelelahan, dan penurunan
koordinasi motorik sering diabaikan pendaki karena dianggap sebagai kelelahan
biasa, padahal kondisi ini dapat berkembang cepat menjadi hipotermia sedang
hingga berat yang mengancam nyawa [28]. Oleh karena itu, pemantauan dini
melalui prediksi cuaca probabilistik menjadi krusial.
2.2.6.2 Faktor Meteorologis Pemicu Risiko Hipotermia
Faktor meteorologis utama yang memicu hipotermia di pegunungan tropis
adalah kombinasi curah hujan tinggi, kecepatan angin, dan kelembapan relatif yang
tinggi [28], [5]. Hujan orografis yang intens membasahi pakaian secara cepat,
sementara angin memperbesar efek convective heat loss [9], [31]. Kelembapan
udara yang tinggi menghambat penguapan keringat dan memperburuk sensasi
dingin pada tubuh.
Tabel 2.5

Faktor Meteorologis terkait Paparan Dingin Pendaki [28], [22]

36

Variabel
Model

Peran Informasi
Nowcasting

Presipitasi

Membasahi pakaian
dan meningkatkan
kehilangan panas
Curah hujan.
melalui konduksi serta
evaporasi.

Memberi peringatan
peluang hujan pada
satu jam berikutnya.

Kecepatan
angin

Meningkatkan
kehilangan panas
melalui konveksi dan
efek wind chill.

Mengidentifikasi
kondisi angin yang
berpotensi
memperburuk
paparan.

Kelembapan
relatif

Memengaruhi
Kelembapan
penguapan dan
relatif.
kondisi pakaian basah.

Faktor

Dampak Potensial

Kecepatan
angin.

Melengkapi
interpretasi kondisi
basah dan
kenyamanan termal.

Pada node utama (MAIN) yang menjadi fokus prediksi model ini, variabilitas
dari node UP, DOWN, LEFT, dan RIGHT dapat memengaruhi kondisi lokal secara
signifikan [5], [17]. Misalnya, angin dari arah tertentu dapat membawa hujan lebih
deras ke titik MAIN, sehingga meningkatkan risiko hipotermia dalam waktu singkat
[28]. Pendekatan lima node pada Spatio-Temporal Graph Conditioning
memungkinkan model menangkap dinamika multidirectional ini dengan lebih baik.
Kombinasi variabel tersebut sulit diprediksi secara akurat dengan model
deterministik karena sifat stokastik dan heavy-tailed dari presipitasi orografis [4],
[27]. Oleh karena itu, pendekatan probabilistik yang menghasilkan distribusi
kemungkinan menjadi lebih sesuai untuk mendukung pengambilan keputusan
keselamatan pendaki.
2.2.6.3 Relevansi Nowcasting Probabilistik sebagai Mitigasi Risiko
Nowcasting probabilistik presipitasi dengan horizon satu jam ke depan dapat
menyediakan informasi bagi tindakan preventif, seperti mencari tempat berteduh,
mengganti pakaian basah, atau memutuskan untuk turun sebelum kondisi
memburuk [28], [26]. Output model berupa distribusi probabilitas pada node MAIN
dapat menjadi masukan bagi pengembangan indeks risiko hipotermia multi-faktor
pada penelitian lanjutan; indeks tersebut bukan keluaran yang diimplementasikan
dalam penelitian ini.
37

Integrasi Retrieval-Augmented Diffusion Model dengan Spatio-Temporal
Graph Conditioning memungkinkan prediksi yang lebih sensitif terhadap kejadian
ekstrem yang berpotensi memicu hipotermia [27], [5]. Pendekatan ini tidak hanya
meningkatkan akurasi teknis, tetapi juga memberikan nilai praktis langsung sebagai
decision support system untuk pengelola taman nasional dan komunitas pendaki
Gunung Gede–Pangrango.
Dengan demikian, penelitian ini menghubungkan kemajuan teknologi
nowcasting probabilistik dengan kebutuhan keselamatan manusia di lingkungan
pegunungan tropis. Model yang dikembangkan diharapkan dapat berkontribusi
pada pengurangan kejadian hipotermia dan peningkatan keselamatan aktivitas
pendakian melalui sistem peringatan dini berbasis data yang akurat dan tepat waktu
[28], [3], [26].
2.2.7

Visualisasi Web Berbasis Vue.js dan FastAPI
Visualisasi berbasis web berperan sebagai lapisan penyajian yang

menghubungkan keluaran model nowcasting dengan pengguna akhir. Lapisan ini
memungkinkan informasi berupa prediksi titik, distribusi ensemble, probabilitas
kejadian hujan, dan indikator risiko disajikan secara terstruktur serta mudah
diinterpretasikan tanpa mengubah proses pelatihan maupun inferensi model. Dalam
pengembangan

dashboard,

penggunaan

framework

front-end

mendukung

konsistensi antarmuka, pemisahan komponen tampilan, dan pembaruan data secara
responsif [35].
Vue.js digunakan sebagai landasan front-end untuk merancang dashboard
yang memuat kartu indikator kondisi node MAIN, grafik deret waktu, visualisasi
probabilitas, dan tabel ringkasan prediksi. Pendekatan berbasis komponen
memungkinkan setiap elemen antarmuka dikembangkan secara modular, sehingga
perubahan pada satu komponen tidak memengaruhi keseluruhan halaman. Vue
memiliki kinerja rendering yang kompetitif, struktur pengembangan sederhana,
serta efisiensi penggunaan sumber daya yang memadai untuk aplikasi web
responsif. Karakteristik tersebut relevan bagi penyajian hasil nowcasting yang perlu
diperbarui secara periodik dan dipahami oleh pengguna nonteknis[35].

38

FastAPI digunakan sebagai lapisan backend untuk menyediakan antarmuka
pemrograman aplikasi yang menyalurkan artefak hasil inferensi kepada dashboard.
Framework ini mendukung validasi data berbasis Python type hints, dokumentasi
API otomatis, dan pemrosesan asinkron, sehingga sesuai untuk penyediaan layanan
data yang terstruktur [36]. Integrasi FastAPI sebagai backend Python dan Vue
sebagai antarmuka memungkinkan pemisahan yang jelas antara modul algoritme,
layanan data, dan visualisasi hasil pada sistem model atmosfer. Dalam arah
pengembangan penelitian ini, FastAPI dapat menyediakan endpoint untuk hasil
prediksi probabilistik, sedangkan Vue.js menyajikannya sebagai grafik dan
indikator risiko. Implementasi dashboard tidak termasuk ruang lingkup evaluasi
model; subbab ini menjadi landasan konseptual bagi pengembangan sistem
penyampaian informasi pada tahap lanjutan [36].
2.3

Analisis Celah Penelitian
Penelitian terkini dalam nowcasting presipitasi telah menunjukkan kemajuan

melalui penerapan model generatif probabilistik, khususnya diffusion models [4],
[27]. Model-model tersebut menyediakan prediksi ensemble, tetapi performanya
tetap bergantung pada data, metrik, dan karakteristik kejadian ekstrem [5], [17].
Sebagian besar studi masih berfokus pada data radar resolusi tinggi di wilayah
dataran atau lautan, dengan sedikit perhatian pada karakteristik presipitasi orografis
di pegunungan tropis yang memiliki variabilitas spasial dan temporal yang sangat
tinggi [9], [31].
Meskipun beberapa penelitian telah mengintegrasikan aspek fisika orografis
ke dalam model nowcasting, pendekatan yang digunakan umumnya masih berbasis
grid-based atau sequence modeling yang kurang efektif dalam menangkap
ketergantungan multidirectional antar lokasi pada gradien elevasi yang curam [12],
[14]. Selain itu, model diffusion yang ada sering kali mengalami kesulitan dalam
memprediksi kejadian ekstrem karena keterbatasan data langka (rare events),
sehingga cenderung under-predict intensitas presipitasi tinggi yang menjadi pemicu
utama risiko hipotermia [27], [29]. Celah ini semakin terlihat pada konteks
pegunungan tropis seperti Gunung Gede–Pangrango, di mana interaksi angin lintas
lereng dan siklus diurnal memainkan peran penting.

39

Penelitian sebelumnya juga jarang menggabungkan mekanisme retrieval
historical analogs dengan diffusion models secara spesifik untuk nowcasting satu
jam ke depan di lingkungan orografis [30], [16]. Kebanyakan studi hanya
menggunakan pure generative approach atau retrieval sederhana tanpa integrasi
yang mendalam dengan representasi graph untuk menangani variabilitas spasial dari
berbagai arah. Selain itu, aplikasi model nowcasting probabilistik untuk mitigasi
risiko keselamatan manusia, khususnya hipotermia pada aktivitas pendakian di
pegunungan tropis, masih sangat terbatas [28], [26]. Mayoritas penelitian lebih
berorientasi pada aspek teknis akurasi model daripada penerapan langsung sebagai
sistem peringatan dini berbasis risiko.
2.3.1.1 Celah pada Representasi Spasial dan Temporal
Pendekatan konvensional seperti CNN, LSTM, atau transformer murni belum
optimal dalam memodelkan ketergantungan spasial non-Euclidean yang kompleks
di wilayah pegunungan [14], [15]. Representasi lima node (MAIN, UP, DOWN,
LEFT, RIGHT) dengan node MAIN sebagai target utama belum banyak
dieksplorasi dalam literatur nowcasting presipitasi orografis [5], [9]. Celah ini
menyebabkan model sulit menangkap pengaruh multidirectional dari berbagai arah
terhadap titik prediksi utama.
2.3.1.2 Celah pada Penanganan Kejadian Ekstrem dan Ketidakpastian
Banyak model diffusion saat ini masih kurang optimal dalam menangani
distribusi heavy-tailed presipitasi ekstrem di lingkungan tropis tanpa bantuan
informasi historis yang relevan [27], [29]. Penggunaan retrieval-based historical
analogs yang dikombinasikan dengan proses denoising diffusion masih jarang
diterapkan, terutama pada horizon nowcasting satu jam dengan data reanalysis atau
stasiun permukaan [30], [17].
2.3.1.3 Celah pada Aplikasi Mitigasi Risiko Pendakian
Terdapat kesenjangan antara kemajuan teknologi nowcasting probabilistik
dengan aplikasi mitigasi keselamatan pendakian di pegunungan tropis [28], [3].
Pengembangan indeks risiko hipotermia berbasis multi-faktor dari keluaran
probabilistik merupakan peluang penelitian lanjutan, bukan komponen yang
diimplementasikan dalam penelitian ini [26], [22].

40

Berdasarkan analisis celah tersebut, penelitian ini mengusulkan RetrievalAugmented Diffusion Model dengan Spatio-Temporal Graph Conditioning
menggunakan representasi lima node dengan MAIN sebagai target utama.
Penelitian mengevaluasi retrieval historical analogs, diffusion probabilistik, dan
representasi graph untuk dinamika orografis Gunung Gede–Pangrango; manfaatnya
terhadap mitigasi keselamatan ditafsirkan dari hasil evaluasi, bukan diasumsikan
sebelumnya.

41

BAB III
METODE PENELITIAN
3.1

Tahapan Penelitian
Tahapan penelitian disusun secara berurutan untuk mengembangkan dan

mengevaluasi Retrieval-Augmented Diffusion Model dengan Spatio-Temporal
Graph Conditioning dalam nowcasting probabilistik presipitasi satu jam ke depan.
Tahapan tersebut mencakup studi literatur, pengambilan data, preprocessing,
pembagian dataset temporal, pembentukan fitur dan representasi graf, perancangan
arsitektur model, pelatihan, inferensi, evaluasi, serta skenario eksperimen. Alur
keseluruhan tahapan penelitian ditunjukkan pada Gambar 3.1.

Gambar 3.1 Alur Tahapan Penelitian
Tahap awal penelitian adalah studi literatur untuk mengidentifikasi,
menganalisis, dan mensintesis kajian terdahulu yang relevan dengan pengembangan
model nowcasting probabilistik presipitasi berbasis deep learning. Fokus kajian
mencakup: (1) teknik nowcasting presipitasi pada skala jam-an, (2) penerapan
model generatif probabilistik khususnya diffusion models, serta (3) pendekatan
spatio-temporal graph dan retrieval-augmented mechanism dalam pemodelan
cuaca.
Tahap selanjutnya adalah Pengambilan Data, yaitu pengumpulan data
meteorologi per jam dari Open-Meteo Archive API berbasis ERA5 reanalysis pada
lima titik observasi spasial (MAIN, UP, DOWN, LEFT, RIGHT). Tahap ini
42

menjadi fondasi utama karena kualitas dan kelengkapan data sangat menentukan
keberhasilan tahapan berikutnya.
Selanjutnya dilakukan Preprocessing Data yang meliputi validasi struktur,
pembersihan, pembentukan fitur lag, transformasi logaritmik pada presipitasi, serta
normalisasi data menggunakan Standard Scaling untuk memastikan data siap
digunakan dalam pelatihan model.
Kemudian dilakukan Pembagian Dataset secara temporal (time-based split)
untuk memisahkan data menjadi training, validation, dan testing set. Pembagian ini
penting untuk menjaga aspek temporal order dan mencegah data leakage pada
evaluasi model.
Tahap Pembentukan Fitur dan Graf melibatkan rekayasa fitur serta
pembuatan struktur graph spatio-temporal dengan lima node di mana node MAIN
menjadi target prediksi utama. Graf ini merepresentasikan ketergantungan spasial
multidirectional antar node.
Selanjutnya adalah Perancangan Arsitektur Model yang mendefinisikan
keseluruhan

komponen

Retrieval-Augmented

Diffusion

Model,

termasuk

mekanisme retrieval historical analogs dan Spatio-Temporal Graph Conditioning
menggunakan Graph Attention Network (GAT).
Proses Proses Pelatihan Model dilakukan dengan menggunakan weighted
denoising loss untuk meningkatkan sensitivitas model terhadap kejadian presipitasi
ekstrem serta teknik conditional diffusion berdasarkan observasi historis dan
struktur graph.
Setelah model dilatih, dilakukan Prosedur Inferensi yang menghasilkan
prediksi probabilistik satu jam ke depan berupa distribusi sampel ensemble pada
node MAIN.
Tahap

Evaluasi

Model

kemudian

dilakukan

menggunakan

metrik

deterministik (MAE, RMSE), metrik probabilistik (CRPS), serta metrik berbasis
threshold (POD, FAR, CSI) untuk menilai performa model secara komprehensif.
Tahap terakhir, Pemetaan Skenario Eksperimen Model dilakukan untuk
menguji berbagai variasi konfigurasi model, hyperparameter, dan baseline
perbandingan guna menganalisis kontribusi masing-masing komponen terhadap
peningkatan performa.
43

Dengan alur tahapan yang terstruktur seperti yang ditunjukkan pada Gambar
3.1, penelitian ini memastikan bahwa setiap komponen model dikembangkan secara
bertahap dan dapat dievaluasi secara ilmiah.
3.2

Pengambilan Data
Pengambilan data merupakan tahap awal yang sangat krusial dalam penelitian

ini karena menjadi fondasi bagi seluruh proses pengembangan model. Penelitian ini
menggunakan data sekunder berupa deret waktu multivariat dengan referensi
spasial tetap yang diperoleh melalui layanan Open-Meteo Archive API berbasis
reanalisis ERA5. Pemilihan ERA5 didasarkan pada konsistensi historis jangka
panjang, kelengkapan variabel meteorologis, serta resolusi spasial dan temporal
yang seragam [31], [34]. ERA5 adalah reanalysis historis, bukan sumber observasi
real-time; karena itu penggunaannya dibatasi untuk pelatihan dan evaluasi
metodologis, bukan operasi nowcasting langsung. Proses pengambilan data melalui
Open-Meteo Archive API dapat dilihat pada Gambar 3.2 berikut.

Gambar 3.2

Diagram Proses Pengambilan Data ERA5 melalui Open-Meteo
API

Gambar 3.2 mengilustrasikan alur pengambilan data yang dimulai dari
penentuan Koordinat Node, permintaan data melalui Open-Meteo API, perolehan
ERA5 Dataset, dilanjutkan dengan Validasi Data Grid, Penggabungan Data Semua
Node, hingga menghasilkan Dataset Penelitian yang siap digunakan. Diagram ini
menekankan mekanisme validasi grid center untuk mencegah duplikasi spasial serta
konsistensi penggabungan data dari kelima node.
Data dikumpulkan dengan resolusi temporal per jam (hourly) mencakup
periode 1 Januari 2005 hingga 31 Desember 2025. Pengambilan data dilakukan
44

pada lima titik grid tetap yang mewakili wilayah studi Gunung Gede–Pangrango.
Satu titik ditetapkan sebagai node utama (MAIN) yang berfungsi sebagai target
prediksi, sedangkan empat titik lainnya berperan sebagai node konteks spasial (UP,
DOWN, LEFT, dan RIGHT) untuk memperkaya informasi multidirectional di
sekitar node utama.
Posisi spasial kelima node observasi yang digunakan dalam penelitian ini
ditampilkan pada Gambar 3.3.

Gambar 3.3

Posisi Lima Node Pengamatan pada Wilayah Studi Gunung Gede–
Pangrango
Gambar 3.3 memperlihatkan distribusi spasial kelima node di sekitar kawasan

Gunung Gede–Pangrango. Node MAIN diposisikan sebagai titik representatif
utama, sementara node UP, DOWN, LEFT, dan RIGHT dirancang untuk
menangkap variasi kondisi atmosfer dari berbagai arah dan elevasi di sekitar node
utama. Koordinat masing-masing node ditentukan secara strategis dan ditunjukkan
pada Tabel 3.1.
Tabel 3.1

Koordinat Node

45

Node

Latitude

Longitude

MAIN

-6.75

107.00

UP

-6.50

107.00

DOWN

-7.00

107.00

LEFT

-6.75

106.75

RIGHT

-6.75

107.25

Untuk menjamin akurasi representasi spasial, dilakukan validasi terhadap grid
center yang dikembalikan oleh API. Proses validasi ini bertujuan mencegah grid
collision, yaitu kondisi ketika dua koordinat berbeda jatuh pada grid ERA5 yang
sama. Apabila ditemukan duplikasi, proses pengambilan data dihentikan secara
otomatis dan koordinat ditinjau ulang.
Data dari kelima node kemudian digabungkan menjadi satu dataset terpadu
dengan urutan node yang konsisten, yaitu [MAIN, UP, DOWN, LEFT, RIGHT]
pada setiap timestamp. Konsistensi urutan ini sangat penting untuk menjaga
kesesuaian representasi graph spatio-temporal pada tahap selanjutnya.
Karakteristik keseluruhan dataset yang berhasil diperoleh ditunjukkan pada
Tabel 3.2.
Tabel 3.2
Karakteristik Dataset Penelitian
Karakteristik
Nilai
Sumber Data

Open-Meteo (ERA5)

Resolusi Temporal

1 Jam

Periode Data

2005–2025

Jumlah Node

5

Jumlah Timestamp

184.080

Total Observasi

920.400

Jumlah Variabel

21

Dengan prosedur pengambilan data yang terstruktur, tervalidasi, dan
didukung visualisasi spasial, dataset yang dihasilkan memiliki integritas spasial dan
temporal yang tinggi, sehingga siap digunakan pada tahapan preprocessing dan
pemodelan selanjutnya.

46

3.3

Preprocessing Data
Tahap preprocessing data merupakan proses krusial yang bertujuan untuk

meningkatkan kualitas, konsistensi, dan kesesuaian data sebelum digunakan dalam
pelatihan model Retrieval-Augmented Diffusion dengan Spatio-Temporal Graph
Conditioning. Proses ini meliputi validasi struktur, harmonisasi, pembentukan fitur
temporal, transformasi distribusi, dan normalisasi. Dengan preprocessing yang
teliti, potensi kesalahan akibat ketidaksesuaian data dapat diminimalisir, sehingga
stabilitas dan performa model dapat ditingkatkan [31], [34].

Gambar 3.4

Flowchart Preprocessing Data

Gambar 3.4 menunjukan alur preprocessing data secara keseluruhan, mulai
dari dataset mentah hingga dataset siap pemodelan. Diagram ini menunjukkan
urutan tahapan yang saling terkait untuk memastikan integritas data sepanjang
pipeline.
Tahap pertama adalah validasi struktur data. Pemeriksaan dilakukan terhadap
kelengkapan lima node pada setiap timestamp, urutan node yang harus konsisten
([MAIN, UP, DOWN, LEFT, RIGHT]), kesesuaian nama kolom, serta keberadaan
nilai tidak valid. Validasi ini bersifat fail-fast, yaitu proses dihentikan apabila
ditemukan ketidaksesuaian untuk mencegah propagasi kesalahan ke tahap
berikutnya.
Selanjutnya dilakukan pembentukan fitur temporal. Untuk menangkap
ketergantungan waktu pada horizon nowcasting satu jam, dibuat fitur lag curah
hujan satu jam sebelumnya (precipitation_lag1). Fitur ini dihitung menggunakan
persamaan berikut:
47

𝑃𝑟𝑒𝑐𝑖𝑝𝑖𝑡𝑎𝑡𝑖𝑜𝑛𝐿𝑎𝑔1𝑡 = 𝑃𝑟𝑒𝑐𝑖𝑝𝑖𝑡𝑎𝑡𝑖𝑜𝑛𝑡−1

(3.1)

Dengan:
• 𝑃𝑟𝑒𝑐𝑖𝑝𝑖𝑡𝑎𝑡𝑖𝑜𝑛𝐿𝑎𝑔1𝑡 = nilai curah hujan satu jam sebelumnya pada waktu 𝑡
• 𝑃𝑟𝑒𝑐𝑖𝑝𝑖𝑡𝑎𝑡𝑖𝑜𝑛𝑡−1 = nilai curah hujan pada waktu sebelumnya (𝑡 − 1)
Pada timestamp pertama setiap node, nilai lag diinisialisasi dengan nol.
Penambahan fitur lag ini penting karena curah hujan memiliki autokorelasi
temporal yang kuat pada skala jam-an [5], [4].
Tahap berikutnya adalah transformasi variabel curah hujan. Variabel ini
memiliki distribusi yang sangat skewed dan didominasi oleh nilai nol dengan sedikit
kejadian ekstrem. Untuk mengurangi skewness dan meningkatkan stabilitas
pelatihan, dilakukan transformasi logaritmik menggunakan fungsi:
𝑦 ′ = log(1 + 𝑦)

(3.2)

dengan:
• 𝑦 = nilai curah hujan asli
• 𝑦 ′ = nilai curah hujan setelah transformasi
Transformasi ini membantu model lebih baik dalam mempelajari pola
presipitasi orografis yang bersifat heavy-tailed. Sebagai contoh untuk nilai curah
hujan sebesar 2.40 mm maka 𝑦 = 2.40 𝑚𝑚, diperoleh nilai curah hujan setelah
transformasi sebagai berikut 𝑦′ = 𝑙𝑜𝑔(1 + 2.40) = 1.2238
Setelah transformasi, seluruh fitur dan variabel target dinormalisasi
menggunakan Standard Scaling. Proses normalisasi dilakukan dengan persamaan:
𝑧=

𝑦′ − 𝜇
𝜎+𝜀

(3.3)

dengan:
• 𝑧 = nilai hasil normalisasi
• 𝑦 ′ = nilai curah hujan setelah transformasi
• 𝜇 = rata-rata (mean) dari data pelatihan
• 𝜎 = simpangan baku (standard deviation) dari data pelatihan
• 𝜀 = konstanta kecil untuk menghindari pembagian dengan nol
Parameter 𝜇 dan 𝜎 dihitung hanya dari data pelatihan untuk mencegah data
leakage. Parameter yang sama kemudian diterapkan pada data validasi dan data uji.
Dengan 𝑦′ = 1.2238, 𝜇 = 0.60, 𝜎 = 0.90, 𝜖 = 0.000001, diperoleh nilai hasil
normalisasi sebagai berikut 𝑧 =

1.2238 − 0.60
0.90 + 0.000001

48

= 0.6931.

Melalui serangkaian tahapan preprocessing yang terstruktur ini, data yang
dihasilkan tidak hanya memiliki kualitas tinggi, tetapi juga telah disesuaikan secara
statistik agar sesuai dengan karakteristik arsitektur model probabilistik yang sensitif
terhadap distribusi dan ketergantungan spasial-temporal. Proses ini memastikan
bahwa input yang diberikan kepada model mencerminkan dinamika atmosfer nyata
di kawasan Gunung Gede–Pangrango secara optimal.
3.4

Pembagian Dataset Temporal
Setelah melalui tahap preprocessing, dataset selanjutnya dibagi menjadi tiga

himpunan yaitu data pelatihan (training set), data validasi (validation set), dan data
pengujian (test set). Pembagian ini bertujuan untuk mendukung proses pelatihan
model, pemilihan hyperparameter, serta evaluasi performa yang objektif dan tidak
bias terhadap data yang belum pernah dilihat oleh model. Karena data meteorologis
bersifat deret waktu dengan ketergantungan temporal yang kuat, penelitian ini
menggunakan pembagian temporal (time-based split) daripada pembagian acak [4],
[17].
Pendekatan temporal split dipilih untuk menjaga urutan kronologis data,
sehingga model hanya mempelajari pola dari periode masa lalu dan dievaluasi pada
periode masa depan. Skema pembagian dataset yang digunakan dalam penelitian
ini ditunjukkan pada Gambar 3.5.

Gambar 3.5

Ilustrasi Pembagian Dataset Temporal

Gambar 3.5 menampilkan diagram garis waktu yang menggambarkan
pembagian dataset secara kronologis menjadi tiga periode utama. Visualisasi ini
memperjelas bahwa model dilatih hanya menggunakan data historis dan diuji pada
data masa depan yang sepenuhnya tidak terlibat dalam proses pelatihan.
Salah satu tujuan utama pembagian temporal adalah mencegah data leakage.
Data leakage terjadi ketika informasi dari masa depan secara tidak sengaja
digunakan selama pelatihan, sehingga menghasilkan estimasi performa yang terlalu

49

optimistis [14], [17]. Dalam penelitian ini, beberapa langkah pencegahan
diterapkan, antara lain:
• Pembagian dilakukan secara kronologis tanpa pengacakan (shuffle).
• Statistik normalisasi dihitung hanya dari data pelatihan.
• Basis data retrieval hanya dibangun dari data pelatihan.
• Data validasi dan pengujian tidak digunakan dalam proses pelatihan maupun
pemilihan hyperparameter.
Dengan skema pembagian temporal ini, model dilatih sesuai dengan kondisi
operasional nowcasting yang sesungguhnya, di mana prediksi satu jam ke depan
hanya dapat dibuat berdasarkan informasi observasi hingga waktu saat ini.
Pendekatan ini meningkatkan reliabilitas evaluasi dan mendekatkan hasil penelitian
dengan aplikasi nyata di lapangan untuk mitigasi risiko hipotermia pendaki di
Gunung Gede–Pangrango.
3.5

Pembentukan Fitur dan Representasi Graf
Setelah dataset melalui tahap preprocessing dan pembagian temporal, langkah

selanjutnya adalah membentuk representasi data yang sesuai dengan arsitektur
model

Retrieval-Augmented

Diffusion

dengan

Spatio-Temporal

Graph

Conditioning. Pada tahap ini, data tidak lagi diperlakukan sebagai tabel
konvensional, melainkan direpresentasikan sebagai urutan graf (graph sequence)
yang mengintegrasikan informasi spasial dan temporal secara simultan. Proses ini
mencakup pembentukan fitur input, pembuatan jendela waktu, serta konstruksi
struktur graf yang merepresentasikan hubungan antar node observasi [5], [30].
3.5.1

Pembentukan Fitur Input
Fitur input dipilih berdasarkan ketersediaan pada ERA5 dan relevansinya

terhadap dinamika presipitasi orografis serta mitigasi risiko hipotermia. Fitur
dibedakan menjadi dua kelompok: fitur dinamis (berubah terhadap waktu) dan fitur
statis (nilai tetap). Elevasi berfungsi sebagai fitur statis yang memberikan informasi
topografi pada setiap node.
Tabel 3.3
No
1

Daftar Fitur Input Model

Variabel
Temperature 2m

Keterangan
Suhu udara
50

No

Variabel

Keterangan

2

Relative Humidity 2m

Kelembapan relatif

3

Dew Point 2m

Titik embun

4

Surface Pressure

Tekanan permukaan

5

Wind Speed 10m

Kecepatan angin

6

Wind Direction 10m

Arah angin

7

Cloud Cover

Tutupan awan

8

Precipitation Lag-1

Curah hujan 1 jam sebelumnya

9

Elevation

Ketinggian node

Secara matematis, vektor fitur untuk node ke-n n n pada waktu t t t dapat
dituliskan sebagai:
3.5.2

Pembentukan Jendela Waktu (Temporal Window)
Untuk menangkap ketergantungan temporal, data disusun menggunakan

pendekatan sliding window. Setiap sampel input terdiri dari beberapa observasi
historis sebagai konteks prediksi satu jam ke depan. Jendela observasi dapat
dinyatakan sebagai:
𝑋𝑡 = {𝑥𝑡−𝐿+1 , 𝑥𝑡−𝐿+2 , … , 𝑥𝑡 }

(3.4)

Dengan:
• 𝑋𝑡 = jendela observasi pada waktu 𝑡
• 𝐿 = panjang jendela historis
• 𝑥𝑡 = vektor fitur pada waktu 𝑡
Melalui pendekatan ini, model dapat mempelajari perubahan kondisi atmosfer
yang terjadi dalam beberapa jam sebelumnya sebelum menghasilkan prediksi pada
waktu berikutnya.
3.5.3

Representasi Graf Spasial
Hubungan spasial antar titik pengamatan direpresentasikan dalam bentuk

graf. Penelitian ini menggunakan lima node dengan node MAIN sebagai target
prediksi utama, sedangkan node UP, DOWN, LEFT, dan RIGHT berfungsi sebagai
51

konteks spasial multidirectional. Struktur graf menggunakan topologi star dengan
delapan edge terarah: MAIN↔UP, MAIN↔DOWN, MAIN↔LEFT, dan
MAIN↔RIGHT. Node tetangga tidak memiliki edge langsung satu sama lain.
3.5.4

Representasi Spasio-Temporal
Setelah struktur graf terbentuk, informasi spasial dan temporal digabungkan

menjadi rangkaian graf (graph sequence). Pada setiap langkah waktu digunakan
struktur graf yang sama, sedangkan nilai fitur pada masing-masing node diperbarui
sesuai kondisi meteorologis aktual. Secara konseptual, rangkaian graf dapat
dituliskan sebagai:
𝑔 = {𝐺𝑡−𝐿+1 , 𝐺𝑡−𝐿+2 , … , 𝐺𝑡 }

(3.5)

dengan:
• 𝐺𝑡 = graf pada waktu 𝑡
• 𝑔 = urutan graf yang menjadi input model
Melalui tahap pembentukan fitur dan representasi graf ini, data berhasil
diubah menjadi format yang sesuai dengan kebutuhan Graph Neural Network dan
conditional diffusion model. Representasi spasio-temporal yang dihasilkan
memungkinkan model mempelajari baik ketergantungan antar lokasi maupun
evolusi kondisi atmosfer dari waktu ke waktu, sehingga mendukung prediksi
probabilistik yang lebih akurat untuk mitigasi risiko hipotermia di Gunung Gede–
Pangrango [5], [30], [17].
3.6

Perancangan Arsitektur Model
Arsitektur model yang dikembangkan dalam penelitian ini merupakan

integrasi antara tiga komponen utama, yaitu Spatio-Temporal Graph Neural
Network (STGNN), Retrieval-Augmented Mechanism, dan Conditional Diffusion
Model. Desain ini dirancang khusus untuk menangani karakteristik presipitasi
orografis di pegunungan tropis serta menghasilkan prediksi probabilistik satu jam
ke depan yang akurat dan sensitif terhadap kejadian ekstrem [27], [5], [30].
3.6.1

Spatio-Temporal Graph Neural Network (STGNN)
Komponen pertama adalah Spatio-Temporal Graph Neural Network yang

bertugas mengekstraksi fitur spasial dan temporal dari kelima node observasi. Pada
setiap timestep, Graph Attention Network (GAT) digunakan untuk memproses

52

struktur graf dan mempelajari bobot hubungan antar node secara adaptif. Proses
encoding spasial dapat dinyatakan dengan:
ℎ𝜏 = 𝐺𝐴𝑇(𝐺𝜏 )

(3.6)

dengan:
• 𝐺𝜏 = graf pada waktu 𝜏
• ℎ𝜏 = representasi spasial hasil encoding
Representasi dari seluruh timestep dalam jendela historis kemudian
diagregasi menggunakan mekanisme temporal attention untuk menghasilkan
representasi spasio-temporal akhir:
ℎ𝑡𝐺 = 𝑇𝑒𝑚𝑝𝑜𝑟𝑎𝑙𝐴𝑡𝑡𝑒𝑛𝑡𝑖𝑜𝑛(ℎ𝑡−𝐿+1 , … , ℎ𝑡 )

(3.7)

dengan:
• ℎ𝑡𝐺 = representasi spasio-temporal akhir pada waktu 𝑡
• 𝐿 = panjang jendela observasi historis
Komponen STGNN memungkinkan model memahami bagaimana kondisi di
node UP, DOWN, LEFT, dan RIGHT memengaruhi node MAIN [5], [30].
3.6.2

Retrieval-Augmented
Untuk meningkatkan kemampuan model dalam menangani kejadian ekstrem,

mekanisme retrieval historical analogs digunakan. Komponen ini mencari kondisi
historis yang paling mirip dengan kondisi saat ini dari basis data pelatihan. Proses
retrieval dapat diformulasikan sebagai:
𝑅𝑡 = 𝑘𝑁𝑁(𝑐𝑡 , 𝐷𝑡𝑟𝑎𝑖𝑛 , 𝑘)

(3.8)

dengan:
• 𝑐𝑡 = representasi kondisi atmosfer saat ini
• 𝐷𝑡𝑟𝑎𝑖𝑛 = basis data historis dari data pelatihan
• 𝑘 = jumlah analog historis yang diambil
• 𝑅𝑡 = himpunan analog historis yang ditemukan
Mekanisme ini sangat penting untuk mengatasi data scarcity pada presipitasi
ekstrem di wilayah pegunungan tropis [27], [17].
3.6.3

Conditional Diffusion Model
Komponen inti model adalah Conditional Diffusion Model yang bertugas

menghasilkan distribusi prediksi probabilistik. Model ini mempelajari proses

53

forward diffusion (penambahan noise) dan reverse diffusion (penghilangan noise)
secara kondisional. Persamaan forward diffusion adalah:
𝑥𝑡 = √𝛼̅𝑡 𝑥0 + √1 − 𝛼̅𝑡 𝜖

(3.9)

dengan:
• 𝑥0 = data asli
• 𝑥𝑡 = data setelah penambahan noise pada timestep 𝑡
• 𝜖 = noise Gaussian
Model kemudian dilatih untuk memprediksi noise tersebut dengan kondisi
tambahan dari STGNN dan retrieval:
𝜖̂𝜃 = 𝑓𝜃 (𝑥𝑡 , 𝑡, ℎ𝑡𝐺 , 𝑅𝑡 )

(3.10)

dengan:
• 𝜖̂𝜃 = prediksi noise oleh model
• ℎ𝑡𝐺 = representasi dari STGNN
• 𝑅𝑡 = hasil retrieval
3.6.4

Integrasi Antar Komponen Model
Ketiga komponen bekerja secara terintegrasi. Representasi dari STGNN dan

hasil retrieval menjadi conditioning input bagi diffusion model, sehingga prediksi
yang dihasilkan tidak hanya probabilistik tetapi juga kontekstual terhadap dinamika
spasial dan historis di kawasan Gunung Gede–Pangrango. Pendekatan ini
memungkinkan model mempelajari distribusi kondisional berikut:
𝑀𝐴𝐼𝑁 |𝑋 )
𝑝(𝑦𝑡+1
𝑡

(3.11)

dengan:
𝑀𝐴𝐼𝑁
• 𝑦𝑡+1
= prediksi pada node MAIN

• 𝑋𝑡 = jendela observasi historis
Dengan arsitektur yang terintegrasi ini, model menghasilkan sampel prediksi
probabilistik untuk dievaluasi dari sisi akurasi titik, kualitas distribusi, dan deteksi
kejadian hujan.
3.7

Prosedur Pelatihan Model
Setelah arsitektur model dirancang, tahap pelatihan dilakukan untuk

mengoptimalkan parameter model agar mampu mempelajari pola kompleks dari
data spasio-temporal serta menghasilkan prediksi probabilistik yang akurat. Proses
54

pelatihan menggunakan data pelatihan yang telah melalui preprocessing dan
representasi graf, dengan tujuan utama meminimalkan kesalahan prediksi sekaligus
meningkatkan sensitivitas terhadap kejadian ekstrem yang berpotensi memicu
hipotermia pendaki [5], [27].
Proses pelatihan diawali dengan pembentukan sampel berupa graph sequence
dari jendela historis. Setiap sampel input terdiri dari observasi selama beberapa
timestep sebelumnya beserta struktur graf lima node, dengan node MAIN sebagai
target prediksi. Model kemudian melakukan forward pass yang meliputi ekstraksi
representasi oleh STGNN, pencarian analog historis melalui retrieval mechanism,
dan prediksi noise pada Conditional Diffusion Model.
Optimasi parameter dilakukan menggunakan algoritma AdamW dengan
mekanisme backpropagation dan weight decay 1e-4. Parameter normalisasi dan
basis retrieval hanya dihitung dari data pelatihan untuk mencegah data leakage.
Pada akhir setiap epoch, model dievaluasi menggunakan data validasi. Model
dengan performa validasi terbaik (berdasarkan loss dan metrik lain) disimpan
sebagai model akhir yang akan digunakan pada tahap inferensi.
Weighted denoising loss memberikan bobot 5 kali untuk target ternormalisasi
dengan nilai absolut lebih dari 1,0 dan 10 kali untuk nilai absolut lebih dari 3,0.
Pembobotan ini meningkatkan penekanan pada target besar, tetapi tidak mengatasi
kelangkaan informasi kejadian ekstrem dalam data. Dengan auxiliary task wet/dry
dan pemilihan model berdasarkan validasi, proses pelatihan ditujukan untuk
meningkatkan sensitivitas terhadap kejadian berisiko tinggi [27], [5], [17].
3.8

Prosedur Inferensi
Setelah model terbaik diperoleh dari tahap pelatihan, proses inferensi

dilakukan untuk menghasilkan prediksi probabilistik satu jam ke depan. Pada tahap
ini, model menerima data observasi terbaru sebagai input dan menghasilkan
distribusi prediksi untuk variabel target (curah hujan, kecepatan angin, dan
kelembapan relatif) pada node MAIN. Prosedur inferensi mengikuti pendekatan
one-step rolling forecasting yang sesuai dengan kondisi operasional nowcasting di
lapangan [4], [5], [27].
Proses inferensi dimulai dengan pembentukan jendela observasi historis
(historical context window). Data dari lima node disusun dalam urutan tetap
55

[MAIN, UP, DOWN, LEFT, RIGHT] selama beberapa timestep sebelumnya.
Jendela observasi ini dapat dinyatakan sebagai:
𝑋𝑡 = {𝐺𝑡−𝐿+1 , 𝐺𝑡−𝐿+2 , … , 𝐺𝑡 }

(3.12)

Dengan:
• 𝑋𝑡 = jendela observasi pada waktu 𝑡
• 𝐺𝑡 = graf pada waktu 𝑡
• 𝐿 = panjang jendela historis
Jendela ini kemudian diproses oleh Spatio-Temporal Graph Neural Network
(STGNN) untuk menghasilkan representasi konteks spasial dan temporal.
Selanjutnya dilakukan retrieval historical analogs untuk mencari kondisi historis
yang paling mirip dengan kondisi saat ini. Representasi dari STGNN dan hasil
retrieval kemudian dijadikan kondisi tambahan pada Conditional Diffusion Model.
Proses reverse diffusion dilakukan untuk menghasilkan ensemble prediksi melalui
sampling berulang dengan inisialisasi noise yang berbeda.
Hasil prediksi kemudian dikembalikan ke skala asli menggunakan
transformasi invers logaritmik untuk variabel curah hujan. Selain itu, output dari
wet/dry auxiliary head digunakan sebagai rain gate untuk menyaring prediksi hujan
palsu.
Prosedur inferensi ini dirancang agar sesuai dengan kondisi operasional nyata,
di mana prediksi selalu dibuat berdasarkan data observasi terkini tanpa akses ke
informasi masa depan. Dengan pendekatan ini, model mampu menghasilkan
prediksi probabilistik yang andal dan dapat digunakan langsung untuk mendukung
sistem peringatan dini mitigasi risiko hipotermia pendaki di Gunung Gede–
Pangrango [27], [5], [17].
3.9

Metode Evaluasi Model
Evaluasi model dilakukan untuk mengukur kemampuan sistem dalam

memprediksi kondisi meteorologis pada node utama (MAIN) secara akurat dan
andal. Mengingat model menghasilkan prediksi dalam bentuk distribusi
probabilistik, evaluasi tidak hanya mencakup akurasi titik (point forecast), tetapi
juga kualitas distribusi probabilitas dan kemampuan deteksi kejadian ekstrem.
Seluruh evaluasi dilakukan secara eksklusif menggunakan data pengujian (test set)
56

yang tidak pernah digunakan selama pelatihan maupun pemilihan model [4], [5],
[27].
Evaluasi model dibagi menjadi tiga kelompok utama: evaluasi deterministik,
evaluasi probabilistik, dan evaluasi berbasis kejadian. Selain itu, dilakukan pula
analisis reliabilitas distribusi prediksi.
a. Evaluasi Deterministik
Evaluasi deterministik mengukur kedekatan antara prediksi titik (median dari
ensemble) dengan nilai observasi aktual. Metrik yang digunakan meliputi:
1. Mean Absolute Error (MAE)
2. Root Mean Square Error (RMSE)
3. Pearson Correlation Coefficient (𝒓)
b. Evaluasi Probabilistik
Evaluasi probabilistik menilai kualitas distribusi prediksi yang dihasilkan
oleh diffusion model.
1. Continuous Ranked Probability Score (CRPS)
2. Brier Score
c. Evaluasi Berbasis Kejadian Hujan
Evaluasi ini menilai kemampuan model mendeteksi kejadian hujan pada
berbagai tingkat intensitas (ambang 2 mm, 5 mm, dan 10 mm). Metrik yang
digunakan adalah:
1. Probability of Detection (POD): Proporsi kejadian hujan yang berhasil dideteksi.
2. False Alarm Ratio (FAR): Proporsi alarm palsu.
3. Critical Success Index (CSI): Ukuran keseluruhan keberhasilan deteksi.
d. Evaluasi Reliabilitas Prediksi
Reliabilitas dinilai menggunakan Reliability Diagram yang membandingkan
probabilitas prediksi dengan frekuensi kejadian aktual. Diagram ini menunjukkan
sejauh mana probabilitas yang dihasilkan model sesuai dengan realisasi observasi
di lapangan.
Tabel 3.4
Kelompok Evaluasi
Deterministik

Ringkasan Metrik Evaluasi

Metrik Utama
MAE, RMSE,
Pearson Correlation
57

Tujuan Utama
Akurasi prediksi titik

Kelompok Evaluasi

Metrik Utama

Tujuan Utama

Probabilistik

CRPS, Brier Score

Kualitas distribusi probabilitas

Berbasis Kejadian

POD, FAR, CSI

Kemampuan deteksi kejadian
hujan

Reliabilitas

Reliability Diagram

Kalibrasi probabilitas

Evaluasi dilakukan dengan pendekatan one-step rolling forecast pada data
pengujian. Model menggunakan enam jam observasi terakhir untuk memprediksi
satu jam berikutnya (𝑡 + 1). Untuk hasil utama, titik evaluasi diambil setiap 11 jam,
parameter ini hanya mengatur jarak antar sampel evaluasi untuk mengurangi bias
akibat autokorelasi, bukan horizon prediksi. Evaluasi penuh setiap jam tetap
didukung melalui `eval_step=1`. Seluruh skenario menggunakan titik evaluasi yang
sama. Dalam penggunaan operasional, apabila sumber data near real-time tersedia,
model dapat dijalankan setiap jam dengan rolling window enam jam terakhir.
Dengan metode evaluasi yang komprehensif ini, penelitian dapat memberikan
gambaran yang lengkap mengenai performa model, baik dari sisi akurasi, kalibrasi
probabilitas, maupun kemampuan deteksi kejadian ekstrem yang relevan dengan
mitigasi risiko hipotermia pendaki di Gunung Gede–Pangrango [4], [5], [27], [29].
3.10

Skenario Eksperimen
Skenario eksperimen dirancang untuk mengevaluasi secara sistematis

kontribusi masing-masing komponen dalam arsitektur model yang diusulkan.
Evaluasi tidak hanya dilakukan terhadap model akhir, tetapi juga terhadap beberapa
konfigurasi bertahap. Pendekatan ini memungkinkan identifikasi pengaruh setiap
komponen (STGNN, Retrieval-Augmented Mechanism, dan Conditional Diffusion
Model) terhadap peningkatan performa prediksi [5], [27], [30].
Seluruh skenario dijalankan menggunakan data pengujian yang sama dan
mengikuti prosedur inferensi one-step rolling forecasting yang telah dijelaskan
sebelumnya. Tujuan utama eksperimen adalah (1) membandingkan model yang
diusulkan dengan baseline, (2) mengukur kontribusi masing-masing komponen, dan
(3) menilai efektivitas integrasi keseluruhan komponen dalam mendukung

58

nowcasting probabilistik untuk mitigasi risiko hipotermia. Enam skenario
eksperimen yang dijalankan ditunjukan pada Tabel 3.5
Tabel 3.5

Matriks Skenario Eksperimen dan Pengujian

Skenario
Eksperimen

Konfigurasi
Model

Output/Parameter
Pengujian

Persistence
Baseline

Nilai 𝒕
digunakan
sebagai prediksi
𝒕+𝟏

Target: presipitasi, angin,
kelembapan;
horizon: t+1;
input: observasi t

-

Input node MAIN; target 3
variabel; learning rate; epoch

-

Jumlah diffusion step; jumlah
sampel ensemble; learning
rate; epoch

-

Parameter Diffusion Only + k
analog; metrik Euclidean L2;
basis retrieval train-only

-

MLP
Baseline

Diffusion
Model Only

Diffusion +
Retrieval

Diffusion +
STGNN

Full Model

Fitur historis
node MAIN
tanpa graph dan
retrieval
Conditional
diffusion tanpa
STGNN dan
retrieval
Conditional
diffusion
dengan retrieval
analog historis
Conditional
diffusion
dengan SpatioTemporal GNN
Conditional
diffusion,
retrieval, dan
STGNN

Parameter Diffusion Only + 5
node; topologi star; jumlah
head GAT; dimensi
embedding
Parameter Diffusion +
Retrieval + STGNN; k analog;
jumlah sampel ensemble; raingate threshold

Hasil
Pengujian

-

-

Perbandingan antar skenario dilakukan menggunakan seluruh metrik evaluasi
yang telah dijelaskan pada subbab sebelumnya (MAE, RMSE, Pearson Correlation,
CRPS, Brier Score, POD, FAR, CSI, serta Reliability Diagram). Dengan
menggunakan metrik yang sama, peningkatan performa dapat dikaitkan langsung
dengan penambahan komponen tertentu.
Hasil eksperimen disajikan dalam bentuk tabel perbandingan metrik, grafik
visualisasi prediksi versus observasi, reliability diagram, serta analisis kontribusi
komponen (ablation study). Pendekatan ini memungkinkan penelitian tidak hanya
menunjukkan bahwa model yang diusulkan lebih baik, tetapi juga menjelaskan
mengapa dan komponen mana yang memberikan kontribusi paling signifikan.

59

Melalui skenario eksperimen yang terstruktur ini, penelitian membandingkan
kontribusi Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph
Conditioning terhadap baseline dalam konteks nowcasting presipitasi di Gunung
Gede–Pangrango [27], [5], [30].
3.11

Rancangan Visualisasi Hasil Prediksi Berbasis Web
Visualisasi hasil prediksi dirancang sebagai sarana penyampaian informasi

nowcasting yang dapat mendukung interpretasi keluaran model oleh pengguna.
Rancangan ini memisahkan proses pemodelan dari proses penyajian hasil: pipeline
penelitian menghasilkan artefak prediksi probabilistik, sedangkan aplikasi web
menampilkan ringkasan informasi tersebut dalam bentuk yang mudah dipahami.
Pengembangan visualisasi berbasis web ditetapkan sebagai arah lanjutan penelitian
dan tidak menjadi bagian dari evaluasi kinerja model pada subbab sebelumnya.
Arsitektur yang direncanakan menggunakan FastAPI sebagai backend dan
Vue.js sebagai frontend. FastAPI berperan menyediakan endpoint terstruktur untuk
mengakses hasil inferensi, meliputi waktu prediksi, nilai median ensemble, rentang
ketidakpastian, probabilitas kejadian hujan, serta status peringatan. Vue.js berperan
merender data tersebut menjadi komponen antarmuka yang interaktif, seperti kartu
ringkasan kondisi cuaca, grafik deret waktu, indikator tingkat risiko, dan tabel detail
prediksi. Pemisahan backend dan frontend ini menjaga modularitas aplikasi,
sehingga pembaruan pada model atau antarmuka dapat dilakukan tanpa mengubah
komponen lain secara langsung [36], [37].

Gambar 3.6 Mockup Dashboard Visualisasi Hasil Nowcasting
60

Gambar 3.6 menunjukkan rancangan awal dashboard visualisasi hasil
nowcasting. Halaman dirancang untuk menampilkan informasi lokasi dan waktu
pembaruan, kondisi cuaca terkini, prediksi presipitasi, kecepatan angin, serta
kelembapan relatif pada node MAIN. Grafik prediksi digunakan untuk
membandingkan perubahan antarwaktu, sedangkan indikator risiko menyajikan
interpretasi ringkas terhadap kondisi yang berpotensi membahayakan pendaki.
Dengan rancangan tersebut, keluaran probabilistik model tidak hanya tersedia
sebagai nilai numerik, tetapi dapat dikomunikasikan sebagai informasi pendukung
pengambilan keputusan sebelum dan selama aktivitas pendakian.
Pada tahap implementasi lanjutan, dashboard perlu memvalidasi respons API,
menampilkan status data terakhir, dan membedakan secara jelas antara observasi
dengan prediksi. Apabila sumber data near real-time tersedia, FastAPI dapat
memperbarui data prediksi secara berkala dan Vue.js menampilkan perubahan
tersebut pada antarmuka. Namun, karena penelitian ini menggunakan ERA5
sebagai data reanalysis historis, mockup yang ditampilkan berfungsi sebagai
rancangan konseptual visualisasi dan bukan sistem peringatan dini operasional.

61

DAFTAR PUSTAKA
[1]

E. Procter, H. Brugger, and M. Burtscher, “Accidental hypothermia in
recreational activities in the mountains : A narrative review,” no. August,
pp. 2464–2472, 2018, doi: 10.1111/sms.13294.

[2]

Aminullah, A. Ayumar, A. Y. Kasma, and E. Samaliwu, “Overview Of
Knowledge About First Aid For Hypothermia In Mountain Climbers In
Makassar,” vol. XVI, pp. 180–190, 2025.

[3]

D. Luciano and D. L. Giovanna, “Rainfall nowcasting model for early
warning systems applied to a case over Central Italy,” Nat. Hazards, vol.
112, no. 1, pp. 501–520, 2022, doi: 10.1007/s11069-021-05191-w.

[4]

P. Chavula, F. Kayusi, G. Lungu, and A. Uwimbabazi, “The Current
Landscape of Early Warning Systems and Traditional Approaches to
Disaster Detection,” 2025, doi: 10.62486/latia202577.

[5]

M. Reichstein et al., “Early warning of complex climate risk with
integrated artificial intelligence,” Nat. Commun., no. May 2024, 2025, doi:
10.1038/s41467-025-57640-w.

[6]

S. Ravuri et al., “Skilful precipitation nowcasting using deep generative
models of radar,” Nature, vol. 597, no. February, 2021, doi:
10.1038/s41586-021-03854-z.

[7]

Y. Zhang, M. Long, K. Chen, L. Xing, and R. Jin, “Skilful nowcasting of
extreme precipitation with NowcastNet,” vol. 619, no. September 2022,
2023, doi: 10.1038/s41586-023-06184-4.

[8]

A. Mazza, A. Antonini, S. Melani, and A. Ortolani, “A Radar-Based Fast
Code for Rainfall Nowcasting over the Tuscany Region,” pp. 1–24, 2025.

[9]

A. P. Lakspriyanti, M. Ekayani, and A. Sunkar, “CARRYING CAPACITY
ASSESSMENT OF CIBEUREUM WATERFALL TOURISM IN
GUNUNG GEDE PANGRANGO NATIONAL PARK,” vol. 25, no. 3, pp.
203–211, 2020, doi: 10.29244/medkon.25.3.203-211.

[10]

D. Sun, J. Wu, H. Huang, R. Wang, F. Liang, and H. Xinhua, “Prediction of
Short-Time Rainfall Based on Deep Learning,” vol. 2021, 2021, doi:
10.1155/2021/6664413.

[11]

Q. Nicolas and W. R. Boos, “Understanding the Spatiotemporal Variability
62

of Tropical Orographic Rainfall Using Convective Plume Buoyancy,” pp.
1737–1757, 2024, doi: 10.1175/JCLI-D-23-0340.1.
[12]

C. L. Tsai, G. Lee, K. Kim, H. Park, H. Park, and W. Bang, “Association
Between Upstream Conditions and the Intensity of Orographic Precipitation
in the Main Mountain Ranges of South Korea,” 2025, doi:
10.1029/2024EA003989.

[13]

L. Silva, R. Célleri, and M. Córdova, “Diurnal to seasonal meteorological
cycles along an equatorial Andean elevational gradient,” vol. 48, pp. 33–48,
2025.

[14]

X. Zhang, M. Xu, S. Kang, H. Wu, and H. Han, “The spatiotemporal
variability in precipitation gradients based on meteorological station
observations in mountainous areas of Northwest China,” 2023.

[15]

Y. Jiang et al., “Characterizing basin-scale precipitation gradients in the
Third Pole region using a high-resolution atmospheric simulation-based
dataset,” pp. 4587–4601, 2022.

[16]

M. Bonavita, “On Some Limitations of Current Machine Learning Weather
Prediction Models,” 2024, doi: 10.1029/2023GL107377.

[17]

Z. Ben Bouallègue et al., “The Rise of Data-Driven Weather Forecasting: A
First Statistical Assessment of Machine Learning–Based Weather Forecasts
in an Operational-Like Context,” pp. 864–883, 2024, doi: 10.1175/BAMSD-23-0162.1.

[18]

X. Zhong et al., “FuXi-ENS : A machine learning model for efficient and
accurate ensemble weather prediction,” pp. 1–14, 2025, doi:
10.1126/sciadv.adu2854.

[19]

I. Price et al., “Probabilistic weather forecasting with machine learning,”
Nature, vol. 637, no. April 2024, pp. 8–11, 2025, doi: 10.1038/s41586-02408252-9.

[20]

O. A. Wani et al., “Predicting rainfall using machine learning , deep
learning , and time series models across an altitudinal gradient in the NorthWestern Himalayas,” 2024.

[21]

L. He et al., “Deep Learning-Based Feature Importance for Rainfall
Nowcast Driven by GNSS,” IEEE J. Sel. Top. Appl. Earth Obs. Remote
63

Sens., vol. 18, pp. 26688–26698, 2025, doi:
10.1109/JSTARS.2025.3621857.
[22]

H. Kim, S. Yoon, Y. H. Gu, S. H. Kim, and Y. Choi, “A Hybrid Approach
to Physical and Deep Learning Models for Radar-Based Precipitation
Nowcasting,” IEEE Trans. Geosci. Remote Sens., vol. 63, pp. 1–16, 2025,
doi: 10.1109/TGRS.2025.3560454.

[23]

Z. Zhang, Q. Song, M. Duan, H. Liu, and J. Huo, “Deep Learning Model
for Precipitation Nowcasting Based on Residual and Attention
Mechanisms,” pp. 1–17, 2025.

[24]

J. H. Meltzer and J. D. Forrester, “Human-Factor Risk Mitigation in
Outdoor Climbing Areas : Survey of Existing Policies in Regulated
Climbing Areas,” vol. 32, no. 4, 2021, doi: 10.1016/j.wem.2021.08.001.

[25]

E. Rahmah, E. K. S. H. Muntasib, and S. B. Rushayati, “Tourism Hazard
Mitigation in Mount Rinjani National Park ,” vol. 2, no. 2, pp. 5–19, 2021,
doi: 10.22158/sshsr.v2n2p5.

[26]

K. Hanly and G. Mcdowell, “The Evolution of ‘ Riskscapes ’: 100 years of
climate change and mountaineering activity in the Lake Louise area of the
Canadian Rockies,” pp. 1–25, 2023.

[27]

P. M. Zarinabegam, I. Kale, P. Deshmukh, and A. Khilare, “IOT-Based
Advanced Monitoring System for Outdoor Safety,” no. May, 2025, doi:
10.22214/ijraset.2025.70379.

[28]

G. Ayzel and M. Heistermann, “Brief communication : Training of AIbased nowcasting models for rainfall early warning should take into
account user requirements,” pp. 41–47, 2025.

[29]

A. Asperti, F. Merizzi, A. Paparella, and G. Pedrazzi, “Precipitation
nowcasting with generative diffusion models,” Appl. Intell., 2024, doi:
https://doi.org/10.1007/s10489-024-06048-y.

[30]

S. P. Ashok and S. Pekkat, “A systematic quantitative review on the
performance of some of the recent short-term rainfall forecasting
techniques,” J. Water Clim. Chang., vol. 13, no. 8, 2022, doi:
10.2166/wcc.2022.302.

[31]

S. Haji-aghajany, W. Rohm, P. Lipinski, and M. Kryza, “Beyond the
64

horizon : A comprehensive analysis of artificial intelligence-based weather
forecasting models,” Eng. Appl. Artif. Intell., vol. 162, no. PA, p. 112335,
2025, doi: 10.1016/j.engappai.2025.112335.
[32]

M. Dumont, M. Saadi, L. Oudin, P. Lachassagne, B. Nugraha, and A.
Fadillah, “Assessing rainfall global products reliability for water resource
management in a tropical volcanic mountainous catchment,” J. Hydrol.
Reg. Stud., vol. 40, no. February, 2022, doi: 10.1016/j.ejrh.2022.101037.

[33]

Q. Nicolas and W. R. Boos, “Sensitivity of tropical orographic precipitation
to wind speed with implications for future projections,” pp. 231–244, 2025.

[34]

E. Yulihastin, T. W. Hadi, N. S. Ningsih, and M. R. Syahputra, “Early
morning peaks in the diurnal cycle of precipitation over the northern coast
of West Java and possible influencing factors,” pp. 231–242, 2020.

[35]

F. A. F. Sofi’ie and A. Qoiriah, “Analisis Perbandingan Framework FrontEnd Javascript React dan Vue Pada Pengembangan Website Analisis
Kebutuhan Pembuatan Sistem Pengumpulan Data Analisa Data,” vol. 05,
pp. 157–164, 2023.

[36]

J. Chen, “Application Technology of Atmospheric Dispersion Models
Based on FastAPI + Vue,” vol. 6, no. 11, pp. 120–126, 2023, doi:
10.25236/AJETS.2023.061118.

65

