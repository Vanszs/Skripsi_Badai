

✅ RECOMMENDED: 2-3 Years of Data = 17,520-26,280 samples

Data Composition:
├─ Total time: 24 months - 36 months
├─ Frequency: 1 hourly reading per hour = 24 readings/day
├─ Formula: 24 readings/day × 365 days/year × 2-3 years
│
├─ Train: 80% = 14,016-21,024 samples
├─ Test: 20% = 3,504-5,256 samples
│
└─ Class distribution (ideal):
    ├─ Normal (Class 0): 70% = 12,264-18,396 samples
    ├─ Anomaly (Class 1): 20% = 3,504-5,256 samples
    └─ Tropical Storm (Class 2): 10% = 1,752-2,628 samples

Literature Support: [743][747][748]
- Min for stable models: 500-1,000 samples
- Recommended: 3,000-5,000 samples
- Your dataset: 17,520+ = 3.5-5x more → VERY ROBUST ✅

WHY 2-3 YEARS?
• Covers 2-3 tropical cyclone seasons (more events in dataset)
• More statistically stable F1 scores
• Better generalization to unseen data
• Publication-quality dataset
• Still feasible timeline (download historical data quickly)

Collection Time: 1-2 weeks (download + preprocessing)

═══════════════════════════════════════════════════════════════════════════════
⏱️ CHECKPOINT FREQUENCY RECOMMENDATION:
═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = Saved model state during training (weights, biases)
WHY? Resume interrupted training, select best model, track progress

📌 MODEL A: AUTOENCODER

Frequency: Every 1 EPOCH
├─ Total epochs: 100
├─ Total checkpoints: 100 files
├─ Time per epoch: 1-2 minutes (GPU)
├─ Total training: ~100-200 minutes = 1.5-3 hours
│
└─ Strategy:
    ├─ Monitor: validation MSE (reconstruction error)
    ├─ Save strategy: Keep best 10 checkpoints
    ├─ Storage per checkpoint: 50 MB
    └─ Total storage: ~500 MB

📌 MODEL B: XGBOOST

Frequency: FINAL MODEL ONLY (no frequent checkpoints)
├─ Total boost rounds: 500
├─ Time: ~1-2 hours
│
└─ Strategy:
    ├─ Monitor: Early stopping (stops if no improvement for 20 rounds)
    ├─ Save: Final best model
    ├─ Storage: ~5-10 MB per model
    └─ Total: ~1 file

Python Code:

📌 MODEL C: PSO-NN

Frequency: Every 10 PSO ITERATIONS
├─ Total iterations: 100
├─ Checkpoints: 10 files (every 10 iterations)
├─ Time per 10 iterations: 5-10 minutes (GPU)
├─ Total training: ~300-600 minutes = 5-10 hours
│
└─ Strategy:
    ├─ Monitor: Validation loss
    ├─ Save best 5 checkpoints
    ├─ Storage per checkpoint: 10 MB
    └─ Total storage: ~50-100 MB
═══════════════════════════════════════════════════════════════════════════════
DAERAH TERBAIK UNTUK DATASET BADAI TROPIS
═══════════════════════════════════════════════════════════════════════════════

✅ REKOMENDASI #1 (TERBAIK): SOUTH CHINA SEA (15°N, 112°E)

STATISTIK:
├─ Badai Tropis per Tahun: 8-10 [751]
├─ Tingkat Landfall: Sangat Tinggi (75%+ dari total WNP) [751]
├─ Data Historis: 50+ tahun (1970-2024) ✅
├─ Kualitas Data: ⭐⭐⭐⭐⭐ Excellent
├─ API Coverage: ⭐⭐⭐⭐ Good
└─ Novelty Penelitian: ⭐⭐⭐⭐⭐ High

MENGAPA TERBAIK?
✅ Frekuensi Optimal (8-10/tahun)
   • Bukan terlalu banyak (overwhelming)
   • Bukan terlalu sedikit (insufficient events)
   
✅ Data Berkualitas Tinggi [750][751]
   • JTWC (Joint Typhoon Warning Center) - track tercakup
   • Tokyo Meteorological Agency - good records
   • China Meteorological Administration - excellent coverage
   • IBTrACS - consensus dataset tersedia
   
✅ Geographic Scope Manageable
   • Area: 3°N-25°N, 100°E-125°E (moderate size)
   • Easier to label regional context
   • Consistent tropical monsoon regime
   
✅ Banyak Station Monitoring
   • Hong Kong: 22.3°N, 114.2°E (EXCELLENT)
   • Da Nang, Vietnam: 16°N, 107.6°E (GOOD)
   • Haikou, China: 19.8°N, 110.3°E (GOOD)
   • Hanoi, Vietnam: 21°N, 105.8°E (GOOD)

UNTUK 2-3 TAHUN DATASET:
• 8-10 cyclones/tahun × 3 tahun = 24-30 cyclone events
• Per event: 5-7 hari dengan pressure anomaly
• Total cyclone samples: ~150-200 (Class 2)
• Class 0 (Normal): ~12,000 samples
• Class 1 (Anomaly): ~3,500 samples
→ SUFFICIENT untuk training robust model ✅

═══════════════════════════════════════════════════════════════════════════════
📊 PERBANDINGAN DENGAN DAERAH LAIN:
═══════════════════════════════════════════════════════════════════════════════

[chart:768]

Dari chart di atas terlihat:
• South China Sea: SEIMBANG (9 TCs, data quality 5, scope manageable)
• Western Pacific: LEBIH BANYAK (26 TCs, tapi scope terlalu besar)
• Philippines: LEBIH SEDIKIT (7 TCs, data quality 4)
• Vietnam: TOO NARROW (6 TCs, data 3, geographic scope terlalu kecil)
• Indonesia: TOO LIMITED (4 TCs, data 3)

═══════════════════════════════════════════════════════════════════════════════
🥈 REKOMENDASI #2 (ALTERNATIF): WESTERN NORTH PACIFIC (15°N, 150°E)

STATISTIK:
├─ Badai Tropis per Tahun: 26-27 [765]
├─ Data Historis: 50+ tahun
├─ Kualitas Data: ⭐⭐⭐⭐⭐ Excellent
└─ Cakupan: Paling comprehensive di dunia

MENGAPA BAGUS?
✅ Paling aktif secara global [765]
✅ Data berkualitas tertinggi (JTWC fokus)
✅ Banyak variasi cyclone (weak, strong, super typhoon)
✅ 26-27 events/tahun = 78-81 events untuk 3 tahun

KEKURANGAN:
❌ Geographic scope TERLALU BESAR
❌ Lebih sulit untuk labeling regional
❌ Lebih banyak data yang harus diproses
❌ Cyclone properties lebih variable

REKOMENDASI: Gunakan jika ingin comprehensive analysis
(bukan untuk focused regional thesis)

═══════════════════════════════════════════════════════════════════════════════
🥉 REKOMENDASI #3 (OPSIONAL): PHILIPPINES (13°N, 125°E)

STATISTIK:
├─ Badai Tropis per Tahun: 6-8
├─ Data Historis: 30+ tahun
├─ Agency: PAGASA (excellent local agency)
├─ Kualitas Data: ⭐⭐⭐⭐
└─ Geographic Scope: SANGAT KECIL (country-level)

MENGAPA BAGUS?
✅ Banyak meteorological stations (multiple islands)
✅ Local expertise dari PAGASA
✅ Easier to get local ground truth data

KEKURANGAN:
❌ Terlalu sedikit cyclone events (6-8/tahun)
❌ Untuk 3 tahun = hanya 18-24 events
❌ Geographic scope terlalu narrow untuk generalization

REKOMENDASI: Jika ingin penelitian COUNTRY-SPECIFIC
(bukan untuk generalization ke region lain)

═══════════════════════════════════════════════════════════════════════════════
❌ TIDAK DIREKOMENDASIKAN:
═══════════════════════════════════════════════════════════════════════════════

❌ Vietnam (5-6/tahun) - TOO FEW events
❌ Indonesia (3-5/tahun) - TOO FEW + variable behavior
❌ North Atlantic (6-7/tahun) - Different climate regime, less novelty
❌ Indian Ocean (4-5/tahun) - Limited data, variable patterns

═══════════════════════════════════════════════════════════════════════════════
🎯 FINAL DECISION: SOUTH CHINA SEA ✅✅✅
═══════════════════════════════════════════════════════════════════════════════

KOORDINAT UTAMA: 15°N, 112°E (Central South China Sea)

Batas-batas region:
├─ Northwest: 25°N, 100°E (Vietnam coast)
├─ Northeast: 25°N, 125°E (Taiwan)
├─ Southeast: 3°N, 125°E (Philippines)
└─ Southwest: 3°N, 100°E (Malaysia)

ALASAN DIPILIH:
1. ✅ Frekuensi optimal (8-10 TCs/tahun)
2. ✅ Data berkualitas tinggi (50+ tahun tersedia)
3. ✅ Geographic scope manageable
4. ✅ Publication quality research
5. ✅ Feasible untuk timeline thesis 3-4 bulan
6. ✅ Novelty tinggi untuk Southeast Asia focus
7. ✅ OpenWeatherMap API coverage bagus

═══════════════════════════════════════════════════════════════════════════════
📥 DATA SOURCES UNTUK SOUTH CHINA SEA:
═══════════════════════════════════════════════════════════════════════════════

1️⃣ ERA5 HISTORICAL PRESSURE DATA (Recommended)
   Source: https://cds.climate.copernicus.eu/
   Region: 3°N-25°N, 100°E-125°E
   Time: 2022-2024 (3 years)
   Variables: pressure, temperature, wind, humidity
   Format: NetCDF/GRIB
   Cost: FREE ✅
   Download time: ~2-3 days

2️⃣ IBTrACS CYCLONE LABELS
   Source: https://www.ncei.noaa.gov/products/international-best-track-archive-climate-stewardship-ibtracs/
   Time: 2022-2024
   Format: CSV
   Coverage: All TC tracks in SCS
   Cost: FREE ✅
   Download time: 1 hour

3️⃣ OPENWEATHERMAP API (Real-time ongoing)
   Link: https://openweathermap.org/api/one-call-3
   Primary station: Hong Kong (22.3°N, 114.2°E)
   Backup stations:
   ├─ Da Nang, Vietnam (16°N, 107.6°E)
   ├─ Haikou, China (19.8°N, 110.3°E)
   └─ Hanoi, Vietnam (21°N, 105.8°E)
   Cost: FREE tier (1000 calls/day) ✅

═══════════════════════════════════════════════════════════════════════════════
✅ EXPECTED DATASET COMPOSITION (3 YEARS SCS):
═══════════════════════════════════════════════════════════════════════════════

Total samples: 26,280 (3 years × 365 days × 24 hours)

Train/Test split (temporal):
├─ Train: 2022-2023 (17,520 samples)
└─ Test: 2024 (8,760 samples)

Class distribution:
├─ Class 0 (Normal): 70% = 18,396 samples
├─ Class 1 (Anomaly): 20% = 5,256 samples
└─ Class 2 (Tropical Storm): 10% = 2,628 samples

Cyclone events:
├─ 2022: ~8 cyclone events
├─ 2023: ~9 cyclone events
└─ 2024: ~8 cyclone events (partial)
TOTAL: 24-25 cyclone events untuk labeling

═══════════════════════════════════════════════════════════════════════════════
🚀 NEXT STEPS:
═══════════════════════════════════════════════════════════════════════════════

1. Start downloading ERA5 data:
   https://cds.climate.copernicus.eu/
   Region: 3-25N, 100-125E
   Time: 2022-2024

2. Download IBTrACS for SCS cyclone events:
   https://www.ncei.noaa.gov/products/international-best-track-archive-climate-stewardship-ibtracs/

3. Set up OpenWeatherMap API:
   https://openweathermap.org/api/one-call-3

4. Preprocessing (Week 1):
   ✓ Load ERA5 + API data
   ✓ Normalize pressure readings
   ✓ Create 72-hour sliding windows
   ✓ Apply 3-class labels from IBTrACS
   ✓ Train/test split (temporal)

5. Model training (Week 2):
   ✓ Autoencoder (1-3 hours)
   ✓ XGBoost (2-3 hours)
   ✓ PSO-NN (6-11 hours overnight)

6. Results & Writing (Week 3-4):
   ✓ Evaluation metrics
   ✓ Paper writing

═══════════════════════════════════════════════════════════════════════════════
