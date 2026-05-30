# Canonical Active System (5-Node Star, Main-Node Target)

Effective date: `2026-04-20`

## Data Source and Grid Policy

- API: Open-Meteo archive
- Mandatory model mode: `models=era5`
- Spatial identity policy: `grid_center_strict`
- Canonical node identity is based on returned grid center lat/lon, not free-text labels

## Canonical Node Set and Order

Mandatory node order: `[MAIN, UP, DOWN, LEFT, RIGHT]`

- `MAIN`: `(-6.75, 107.00)`
- `UP`: `(-6.50, 107.00)`
- `DOWN`: `(-7.00, 107.00)`
- `LEFT`: `(-6.75, 106.75)`
- `RIGHT`: `(-6.75, 107.25)`

Context note:

- `DOWN` is retained by design even when returned grid elevation is maritime (`elevation <= 0`), and this context is explicitly documented in grid validation report.

Main node canonical identifier:

- Coordinate: `(-6.75, 107.00)`
- Name: `MAIN`
- Alias is optional and non-canonical

## Graph Topology

- Topology: `star`
- Directed edges: `MAIN<->UP`, `MAIN<->DOWN`, `MAIN<->LEFT`, `MAIN<->RIGHT`
- Total directed edges: `8`

## Target and Context Policy

- `target_node_policy = main_node_only`
- `context_policy = main_node_context`
- Node averaging for target/retrieval context is forbidden in active pipeline

## Active Dataset Artifacts

- Canonical full dataset:
  - `data/raw/pangrango_era5_5node_2005_2025.parquet`
- Canonical smoke dataset:
  - `data/raw/pangrango_era5_5node_2024_2024.parquet`
- Grid validation report:
  - `data/raw/pangrango_era5_5node_grid_validation.json`

Legacy 3-node datasets are archived and not active.
