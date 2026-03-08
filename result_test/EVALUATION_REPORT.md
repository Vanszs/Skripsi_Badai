# Hasil Evaluasi 4 Skenario — Honest Comparison

**Tanggal:** 2026-03-08 11:59
**Test Period:** 2022-2025
**Eval Step:** 24h (daily)
**Ensemble Size:** 30 (untuk diffusion)
**Nodes:** Puncak, Lereng_Cibodas, Hilir_Cianjur

## Skenario yang Dibandingkan

| # | Skenario | Deskripsi |
|---|----------|-----------|
| 1 | Persistence | Prediksi = nilai jam sebelumnya (naive baseline) |
| 2 | MLP Baseline | MLP 3-layer, evaluasi per-node (apple-to-apple) |
| 3 | Diffusion (Pure) | Model diffusion TANPA hybrid blending |
| 4 | Diffusion+Hybrid | Diffusion + persistence blending (w=0.9/0.9/0.7) |

## Metrik Agregasi (Semua Node)

| Skenario | Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |
|----------|----------|------|-----|------|------|-------|-----|-----|-----|
| Persistence | precipitation | 2.0775 | 0.8236 | 0.4585 | N/A | 0.0219 | 0.0588 | 0.8889 | 0.0400 |
| Persistence | wind_speed | 1.9889 | 1.4695 | 0.8356 | N/A | 0.0875 | 0.6439 | 0.2009 | 0.5542 |
| Persistence | humidity | 5.0580 | 3.5539 | 0.9444 | N/A | 0.0909 | 0.4316 | 0.3508 | 0.3500 |
| MLP Baseline | precipitation | 11.1134 | 4.5086 | 0.1733 | N/A | 0.1452 | 0.3529 | 0.9611 | 0.0363 |
| MLP Baseline | wind_speed | 3.1828 | 2.4988 | 0.7316 | N/A | 0.2404 | 0.8903 | 0.5959 | 0.3849 |
| MLP Baseline | humidity | 8.6947 | 6.5711 | 0.8278 | N/A | 0.0887 | 0.3780 | 0.2985 | 0.3256 |
| Diffusion (Pure) | precipitation | 2.6128 | 1.5682 | -0.0103 | 1.4699 | 0.0155 | 0.0000 | N/A | 0.0000 |
| Diffusion (Pure) | wind_speed | 4.4664 | 3.2103 | 0.0668 | 3.1127 | 0.1689 | 0.0000 | N/A | 0.0000 |
| Diffusion (Pure) | humidity | 16.8542 | 12.1100 | 0.2540 | 11.7777 | 0.1654 | 0.3190 | 0.7657 | 0.1562 |
| Diffusion+Hybrid | precipitation | 2.0088 | 0.8488 | 0.4543 | 0.8301 | 0.0203 | 0.0588 | 0.8636 | 0.0429 |
| Diffusion+Hybrid | wind_speed | 2.0087 | 1.5151 | 0.8328 | 1.4885 | 0.0990 | 0.5054 | 0.1662 | 0.4592 |
| Diffusion+Hybrid | humidity | 6.7037 | 5.1597 | 0.9235 | 4.9981 | 0.0808 | 0.3512 | 0.2471 | 0.3149 |

## Metrik Per Node: Puncak

| Skenario | Variable | RMSE | MAE | Corr |
|----------|----------|------|-----|------|
| Persistence | precipitation | 1.8486 | 0.7955 | 0.4896 |
| Persistence | wind_speed | 1.5619 | 1.1951 | 0.8753 |
| Persistence | humidity | 5.5820 | 3.9548 | 0.9052 |
| MLP Baseline | precipitation | 2.4709 | 1.0881 | 0.0328 |
| MLP Baseline | wind_speed | 3.1593 | 2.7779 | 0.8454 |
| MLP Baseline | humidity | 7.2545 | 5.4659 | 0.8884 |
| Diffusion (Pure) | precipitation | 2.3573 | 1.1889 | 0.1685 |
| Diffusion (Pure) | wind_speed | 4.5118 | 3.1871 | -0.5111 |
| Diffusion (Pure) | humidity | 16.4092 | 14.6878 | 0.2528 |
| Diffusion+Hybrid | precipitation | 1.8301 | 0.8005 | 0.4872 |
| Diffusion+Hybrid | wind_speed | 1.5900 | 1.2457 | 0.8703 |
| Diffusion+Hybrid | humidity | 6.3348 | 5.3594 | 0.9062 |

## Metrik Per Node: Lereng_Cibodas

| Skenario | Variable | RMSE | MAE | Corr |
|----------|----------|------|-----|------|
| Persistence | precipitation | 1.8486 | 0.7955 | 0.4896 |
| Persistence | wind_speed | 1.5619 | 1.1951 | 0.8753 |
| Persistence | humidity | 5.2868 | 3.7540 | 0.9065 |
| MLP Baseline | precipitation | 1.8073 | 0.9417 | 0.4260 |
| MLP Baseline | wind_speed | 1.5521 | 1.2132 | 0.8827 |
| MLP Baseline | humidity | 4.9539 | 3.8111 | 0.9142 |
| Diffusion (Pure) | precipitation | 1.9537 | 1.0350 | 0.2447 |
| Diffusion (Pure) | wind_speed | 3.4043 | 2.3712 | 0.3184 |
| Diffusion (Pure) | humidity | 8.3598 | 5.7449 | 0.7389 |
| Diffusion+Hybrid | precipitation | 1.7987 | 0.7801 | 0.4966 |
| Diffusion+Hybrid | wind_speed | 1.5784 | 1.2271 | 0.8701 |
| Diffusion+Hybrid | humidity | 5.4991 | 4.0653 | 0.8963 |

## Metrik Per Node: Hilir_Cianjur

| Skenario | Variable | RMSE | MAE | Corr |
|----------|----------|------|-----|------|
| Persistence | precipitation | 2.4725 | 0.8797 | 0.4362 |
| Persistence | wind_speed | 2.6435 | 2.0181 | 0.7421 |
| Persistence | humidity | 4.2000 | 2.9528 | 0.9637 |
| MLP Baseline | precipitation | 19.0041 | 11.4959 | 0.3510 |
| MLP Baseline | wind_speed | 4.2428 | 3.5053 | 0.6332 |
| MLP Baseline | humidity | 12.2321 | 10.4364 | 0.9315 |
| Diffusion (Pure) | precipitation | 3.3327 | 2.4807 | -0.2122 |
| Diffusion (Pure) | wind_speed | 5.2820 | 4.0725 | 0.0350 |
| Diffusion (Pure) | humidity | 22.6504 | 15.8972 | -0.7226 |
| Diffusion+Hybrid | precipitation | 2.3498 | 0.9659 | 0.4236 |
| Diffusion+Hybrid | wind_speed | 2.6619 | 2.0726 | 0.7401 |
| Diffusion+Hybrid | humidity | 8.0281 | 6.0543 | 0.9456 |

## Analisis Pemenang per Metrik

| Variable | Best RMSE | Best MAE | Best Corr |
|----------|-----------|----------|-----------|
| precipitation | Diffusion+Hybrid (2.0088) | Persistence (0.8236) | Persistence (0.4585) |
| wind_speed | Persistence (1.9889) | Persistence (1.4695) | Persistence (0.8356) |
| humidity | Persistence (5.0580) | Persistence (3.5539) | Persistence (0.9444) |

## Interpretasi

- **Persistence** adalah baseline paling sederhana: prediksi = nilai jam sebelumnya.
- Jika model lain tidak signifikan lebih baik dari persistence, berarti model belum memberikan added value.
- **Diffusion+Hybrid** dengan w=0.9 berarti 90% prediksi berasal dari persistence — sehingga hasilnya akan mirip persistence.
- **Diffusion (Pure)** menunjukkan kemampuan sebenarnya dari model diffusion tanpa 'bantuan' persistence.
- Perbandingan yang fair: **MLP Baseline vs Diffusion (Pure)** — keduanya murni model learning.

## Plot yang Dihasilkan

1. `plots/rmse_comparison.png` — Bar chart RMSE per variabel
2. `plots/correlation_comparison.png` — Bar chart Correlation per variabel
3. `plots/mae_comparison.png` — Bar chart MAE per variabel
4. `plots/timeseries_*.png` — Time series perbandingan 4 skenario
5. `plots/scatter_*.png` — Scatter plot actual vs predicted
6. `plots/rmse_heatmap.png` — Heatmap RMSE semua skenario
