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
- **seed**: `1`
- **crps_estimator**: `fair_unbiased`
- **mlp_crps_type**: `deterministic_single_pass`
- **rain_specialization_enabled**: `True`
- **rain_occurrence_threshold_mm**: `0.1`
- **rain_probability_threshold**: `0.6000000238418579`

## PRECIPITATION

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 0.6847 | 0.2187 | 0.6856 | 0.2187 |
| mlp_baseline | 0.7417 | 0.2686 | 0.5461 | 0.2686 |
| diff_only | 0.8718 | 0.4132 | 0.5345 | 0.2893 |
| diff_retrieval | 0.8957 | 0.4219 | 0.5303 | 0.2960 |
| diff_gnn | 0.7802 | 0.3415 | 0.6320 | 0.2463 |
| full_model | 0.7920 | 0.3530 | 0.6390 | 0.2518 |

## WIND_SPEED

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 1.1153 | 0.8368 | 0.8489 | 0.8368 |
| mlp_baseline | 1.1557 | 0.9114 | 0.8315 | 0.9114 |
| diff_only | 1.1713 | 0.9023 | 0.8479 | 0.6222 |
| diff_retrieval | 1.1735 | 0.9063 | 0.8492 | 0.6265 |
| diff_gnn | 0.9753 | 0.7395 | 0.8904 | 0.5174 |
| full_model | 0.9862 | 0.7477 | 0.8889 | 0.5223 |

## HUMIDITY

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 4.4942 | 2.9061 | 0.9380 | 2.9061 |
| mlp_baseline | 4.5479 | 3.3326 | 0.9556 | 3.3326 |
| diff_only | 4.3285 | 2.8149 | 0.9428 | 2.0345 |
| diff_retrieval | 4.3786 | 2.8500 | 0.9423 | 2.0584 |
| diff_gnn | 2.9672 | 2.0310 | 0.9730 | 1.4442 |
| full_model | 2.9899 | 2.0537 | 0.9728 | 1.4551 |

## Precipitation Threshold Metrics

### Threshold 2.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.4959 | 0.5041 | 0.3297 | 0.0389 |
| mlp_baseline | 0.0569 | 0.5000 | 0.0538 | 0.0386 |
| diff_only | 0.6341 | 0.7665 | 0.2058 | 0.0592 |
| diff_retrieval | 0.6585 | 0.7775 | 0.1995 | 0.0614 |
| diff_gnn | 0.7317 | 0.6715 | 0.2932 | 0.0477 |
| full_model | 0.7398 | 0.7065 | 0.2661 | 0.0494 |

### Threshold 5.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.3810 | 0.6522 | 0.2222 | 0.0088 |
| mlp_baseline | 0.0000 | 1.0000 | 0.0000 | 0.0069 |
| diff_only | 0.0476 | 0.6667 | 0.0435 | 0.0074 |
| diff_retrieval | 0.0476 | 0.5000 | 0.0455 | 0.0075 |
| diff_gnn | 0.0476 | 0.8571 | 0.0370 | 0.0076 |
| full_model | 0.0476 | 0.8000 | 0.0400 | 0.0071 |

### Threshold 10.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.5000 | 0.0000 | 0.5000 | 0.0003 |
| mlp_baseline | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_only | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_retrieval | 0.0000 | nan | 0.0000 | 0.0006 |
| diff_gnn | 0.0000 | nan | 0.0000 | 0.0006 |
| full_model | 0.0000 | nan | 0.0000 | 0.0007 |

