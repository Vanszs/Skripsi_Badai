# Evaluation Report

## Retrieval-Augmented Diffusion Model + Spatio-Temporal Graph Conditioning
### Gunung Gede-Pangrango Nowcasting System

---

## 1. Konfigurasi Evaluasi

| Parameter | Value |
|-----------|-------|
| Device | cuda |
| Test Period | 2022-2025 |
| Nodes | Puncak, Lereng_Cibodas, Hilir_Cianjur |
| Ensemble Size | 30 |
| Eval Step | 24h (daily) |
| DDIM Steps | 20 |
| Heavy Rain Threshold | 10.0 mm/jam |
| Evaluation Time | 566.6s |

## 2. Data Statistics

| Node | Total Hours | Eval Samples | Valid | NaN Skipped |
|------|-------------|--------------|-------|-------------|
| Puncak | 26,321 | 1,097 | 1,097 | 0 |
| Lereng_Cibodas | 26,321 | 1,097 | 1,097 | 0 |
| Hilir_Cianjur | 26,321 | 1,097 | 1,097 | 0 |
| **Total** | **78,963** | **3,291** | **3,291** | **0** |

## 3. Metrik Per Node

### Puncak

| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |
|----------|------|-----|------|------|-------|-----|-----|-----|
| precipitation | 1.8285 | 0.8024 | 0.4888 | 0.7864 | 0.0163 | 0.0000 | 1.0000 | 0.0000 |
| wind_speed | 1.5882 | 1.2444 | 0.8705 | 1.2181 | 0.0593 | 0.5122 | 0.0870 | 0.4884 |
| humidity | 6.3387 | 5.3698 | 0.9059 | 5.2345 | 0.1179 | 0.2881 | 0.1500 | 0.2742 |

### Lereng_Cibodas

| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |
|----------|------|-----|------|------|-------|-----|-----|-----|
| precipitation | 1.7944 | 0.7780 | 0.4992 | 0.7609 | 0.0155 | 0.0000 | 1.0000 | 0.0000 |
| wind_speed | 1.5798 | 1.2279 | 0.8699 | 1.2013 | 0.0561 | 0.5528 | 0.0933 | 0.5231 |
| humidity | 5.5209 | 4.0812 | 0.8953 | 3.9133 | 0.1202 | 0.4180 | 0.3070 | 0.3527 |

### Hilir_Cianjur

| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |
|----------|------|-----|------|------|-------|-----|-----|-----|
| precipitation | 2.3504 | 0.9672 | 0.4235 | 0.9427 | 0.0294 | 0.1429 | 0.8333 | 0.0833 |
| wind_speed | 2.6620 | 2.0709 | 0.7402 | 2.0440 | 0.1811 | 0.4839 | 0.2228 | 0.4249 |
| humidity | 8.0307 | 6.0560 | 0.9450 | 5.8586 | 0.0055 | 0.1429 | 0.5000 | 0.1250 |

## 4. Metrik Agregasi (Semua Node)

| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |
|----------|------|-----|------|------|-------|-----|-----|-----|
| precipitation | 2.0073 | 0.8492 | 0.4553 | 0.8300 | 0.0204 | 0.0588 | 0.8750 | 0.0417 |
| wind_speed | 2.0087 | 1.5144 | 0.8328 | 1.4878 | 0.0988 | 0.5054 | 0.1662 | 0.4592 |
| humidity | 6.7120 | 5.1690 | 0.9231 | 5.0021 | 0.0812 | 0.3512 | 0.2557 | 0.3134 |

## 5. Interpretasi

### Metrik Deterministik
- **RMSE** (Root Mean Square Error): Semakin rendah semakin baik. Sensitif terhadap outlier.
- **MAE** (Mean Absolute Error): Semakin rendah semakin baik. Robust terhadap outlier.
- **Corr** (Pearson Correlation): Mendekati 1.0 = model menangkap pola temporal dengan baik.

### Metrik Probabilistik
- **CRPS**: Semakin rendah semakin baik. Mengukur kualitas distribusi prediksi.
- **Brier Score**: Semakin rendah semakin baik (0 = sempurna). Mengukur kalibrasi probabilitas event.
- **POD** (Probability of Detection): Mendekati 1.0 = model mendeteksi event dengan baik.
- **FAR** (False Alarm Ratio): Mendekati 0.0 = sedikit false alarm.
- **CSI** (Critical Success Index): Mendekati 1.0 = keseimbangan POD dan FAR.

## 6. Output Files

```
results/
  diffusion_results/diffusion_metrics.json
  probabilistic_metrics.json
  tables/metrics_summary.csv
  tables/metrics_summary.json
  plots/rain_prediction_vs_actual.png
  plots/wind_prediction_vs_actual.png
  plots/humidity_prediction_vs_actual.png
  plots/scatter_actual_vs_predicted.png
  plots/ensemble_spread.png
  plots/reliability_diagram.png
  evaluation_report.md
```