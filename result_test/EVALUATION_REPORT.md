# Evaluation Report - Main Node Only

- **main_node_name**: `MAIN`
- **main_node_identifier**: `(-6.75, 107.0)`
- **node_order**: `['MAIN', 'UP', 'DOWN', 'LEFT', 'RIGHT']`
- **graph_topology**: `star`
- **target_node_policy**: `main_node_only`
- **context_policy**: `main_node_context`
- **model_mode**: `era5`
- **data_path**: `data/raw/pangrango_era5_5node_2005_2025.parquet`
- **eval_step**: `24`
- **num_ensemble**: `30`
- **seq_len**: `6`
- **rain_specialization_enabled**: `True`
- **rain_occurrence_threshold_mm**: `0.1`
- **rain_probability_threshold**: `0.550000011920929`

## PRECIPITATION

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 1.3777 | 0.7344 | 0.5990 | nan |
| mlp_baseline | 1.3038 | 0.7662 | 0.4223 | 0.7348 |
| diff_only | 1.5116 | 0.7847 | 0.3112 | 0.7740 |
| diff_retrieval | 1.5176 | 0.7876 | 0.2936 | 0.7796 |
| diff_gnn | 1.7388 | 1.3226 | 0.4115 | 1.5169 |
| full_model | 1.6903 | 1.2893 | 0.4292 | 1.5896 |

## WIND_SPEED

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 1.4424 | 1.0924 | 0.8378 | nan |
| mlp_baseline | 1.2971 | 1.0211 | 0.8482 | 0.9618 |
| diff_only | 1.2434 | 0.9341 | 0.8571 | 0.9424 |
| diff_retrieval | 1.2522 | 0.9374 | 0.8567 | 0.9470 |
| diff_gnn | 1.3091 | 0.9904 | 0.8471 | 1.0910 |
| full_model | 1.3051 | 0.9853 | 0.8494 | 1.0947 |

## HUMIDITY

| Scenario | RMSE | MAE | Corr | CRPS |
|---|---:|---:|---:|---:|
| persistence | 5.2005 | 4.1665 | 0.9357 | nan |
| mlp_baseline | 4.2661 | 3.4671 | 0.9362 | 3.4067 |
| diff_only | 3.9944 | 3.4042 | 0.9599 | 3.3000 |
| diff_retrieval | 4.0265 | 3.4208 | 0.9598 | 3.3277 |
| diff_gnn | 3.1156 | 2.4979 | 0.9595 | 2.6745 |
| full_model | 3.1387 | 2.5234 | 0.9591 | 2.7192 |

## Precipitation Threshold Metrics

### Threshold 2.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.4877 | 0.3767 | 0.3767 | 0.1573 |
| mlp_baseline | 0.0673 | 0.5455 | 0.0622 | 0.1462 |
| diff_only | 0.0448 | 0.5000 | 0.0429 | 0.1255 |
| diff_retrieval | 0.0448 | 0.6000 | 0.0420 | 0.1278 |
| diff_gnn | 0.8610 | 0.7500 | 0.2403 | 0.2275 |
| full_model | 0.8206 | 0.7493 | 0.2377 | 0.2223 |

### Threshold 5.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | 0.1356 | 0.7714 | 0.0930 | 0.0534 |
| mlp_baseline | 0.0000 | nan | 0.0000 | 0.0239 |
| diff_only | 0.0000 | nan | 0.0000 | 0.0242 |
| diff_retrieval | 0.0000 | nan | 0.0000 | 0.0240 |
| diff_gnn | 0.0286 | 0.8750 | 0.0238 | 0.0469 |
| full_model | 0.0286 | 0.9000 | 0.0227 | 0.0462 |

### Threshold 10.0 mm

| Scenario | POD | FAR | CSI | Brier |
|---|---:|---:|---:|---:|
| persistence | nan | nan | nan | 0.0000 |
| mlp_baseline | nan | nan | nan | 0.0000 |
| diff_only | nan | nan | nan | 0.0000 |
| diff_retrieval | nan | nan | nan | 0.0001 |
| diff_gnn | nan | nan | nan | 0.0012 |
| full_model | nan | nan | nan | 0.0011 |

