# Evaluation Report - Main Node Only

- **main_node_name**: `MAIN`
- **main_node_identifier**: `(-6.75, 107.0)`
- **node_order**: `['MAIN', 'UP', 'DOWN', 'LEFT', 'RIGHT']`
- **graph_topology**: `star`
- **target_node_policy**: `main_node_only`
- **context_policy**: `main_node_context`
- **model_mode**: `era5`
- **data_path**: `data/raw/pangrango_era5_5node_2005_2025.parquet`
- **eval_step**: `11`
- **max_eval_samples**: `0`
- **samples_per_scenario**: `3189`
- **num_ensemble**: `30`
- **seq_len**: `6`
- **crps_estimator**: `fair_unbiased`
- **mlp_crps_type**: `deterministic_single_pass`
- **rain_specialization_enabled**: `True`
- **rain_occurrence_threshold_mm**: `0.1`
- **rain_probability_threshold**: `0.5`

## PRECIPITATION

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 0.6847 | 0.2187 | 0.6856 | 0.2187 |
| mlp_baseline | 0.7316 | 0.2823 | 0.5509 | 0.2823 |
| diff_only | 0.9254 | 0.3779 | 0.4172 | 0.2958 |
| diff_retrieval | 0.8723 | 0.3813 | 0.4980 | 0.3139 |
| diff_gnn | 0.7713 | 0.3530 | 0.5136 | 0.2815 |
| full_model | 0.8466 | 0.3721 | 0.5079 | 0.3011 |

## WIND_SPEED

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 1.1153 | 0.8368 | 0.8489 | 0.8368 |
| mlp_baseline | 1.1292 | 0.8858 | 0.8321 | 0.8858 |
| diff_only | 1.0824 | 0.8234 | 0.8488 | 0.5768 |
| diff_retrieval | 1.1111 | 0.8477 | 0.8524 | 0.5940 |
| diff_gnn | 1.0780 | 0.8156 | 0.8497 | 0.5766 |
| full_model | 1.1124 | 0.8542 | 0.8530 | 0.5965 |

## HUMIDITY

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 4.4942 | 2.9061 | 0.9380 | 2.9061 |
| mlp_baseline | 3.8033 | 2.8186 | 0.9583 | 2.8186 |
| diff_only | 4.3347 | 3.1230 | 0.9419 | 2.2319 |
| diff_retrieval | 4.3979 | 3.1150 | 0.9404 | 2.2324 |
| diff_gnn | 4.3443 | 3.1438 | 0.9423 | 2.2473 |
| full_model | 4.3914 | 3.1180 | 0.9408 | 2.2446 |

## Precipitation Threshold Metrics

### Threshold 2.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.4959 | 0.5041 | 0.3297 | 0.0389 |
| mlp_baseline | 0.0894 | 0.5417 | 0.0809 | 0.0392 |
| diff_only | 0.2764 | 0.6909 | 0.1709 | 0.0391 |
| diff_retrieval | 0.4797 | 0.7065 | 0.2226 | 0.0458 |
| diff_gnn | 0.2683 | 0.6733 | 0.1728 | 0.0383 |
| full_model | 0.3984 | 0.7263 | 0.1937 | 0.0451 |

### Threshold 5.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.3810 | 0.6522 | 0.2222 | 0.0088 |
| mlp_baseline | 0.0000 | 1.0000 | 0.0000 | 0.0069 |
| diff_only | 0.0476 | 0.8333 | 0.0385 | 0.0089 |
| diff_retrieval | 0.0476 | 0.8889 | 0.0345 | 0.0110 |
| diff_gnn | 0.0476 | 0.5000 | 0.0455 | 0.0086 |
| full_model | 0.0476 | 0.8750 | 0.0357 | 0.0104 |

### Threshold 10.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.5000 | 0.0000 | 0.5000 | 0.0003 |
| mlp_baseline | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_only | 0.0000 | 1.0000 | 0.0000 | 0.0021 |
| diff_retrieval | 0.0000 | 1.0000 | 0.0000 | 0.0026 |
| diff_gnn | 0.0000 | 1.0000 | 0.0000 | 0.0020 |
| full_model | 0.0000 | 1.0000 | 0.0000 | 0.0025 |

