"""
Script to Generate Complete Pipeline Jupyter Notebook

Run this script to create the notebook:
    python scripts/generate_notebook.py
"""

import json
import os


def create_notebook():
    """Generate a complete Jupyter notebook for the pipeline."""
    
    cells = []
    
    # ==========================================================================
    # HEADER
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Probabilistic Nowcasting Hujan Ekstrem Sitaro\n",
            "## Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning\n",
            "\n",
            "---\n",
            "\n",
            "### Pipeline Overview\n",
            "\n",
            "| No | Pipeline | Input | Output |\n",
            "|----|----------|-------|--------|\n",
            "| 1 | Data Ingestion | Koordinat + Waktu | DataFrame |\n",
            "| 2 | Temporal Split | DataFrame | Train/Val/Test |\n",
            "| 3 | Preprocessing | Raw Data | Normalized Tensors |\n",
            "| 4 | Sliding Window | Flat Tensor | Graph Sequences |\n",
            "| 5 | Retrieval Database | Features | FAISS Index |\n",
            "| 6 | Spatio-Temporal GNN | Graph Seq | Embedding |\n",
            "| 7 | Diffusion Training | All Conditioning | Trained Model |\n",
            "| 8 | Probabilistic Inference | Current Weather | 50 Rain Samples |"
        ]
    })
    
    # ==========================================================================
    # IMPORTS
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## 0. Setup & Imports"]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import sys\n",
            "import os\n",
            "sys.path.insert(0, os.path.abspath('..'))\n",
            "\n",
            "import torch\n",
            "import torch.nn as nn\n",
            "import pandas as pd\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from datetime import datetime\n",
            "from tqdm.notebook import tqdm\n",
            "\n",
            "# Custom modules\n",
            "from src.data.ingest import fetch_era5_data, SITARO_NODES\n",
            "from src.data.temporal_loader import TemporalGraphDataset, collate_temporal_graphs\n",
            "from src.models.gnn import SpatioTemporalGNN, SpatialGNN, TemporalAttention\n",
            "from src.models.diffusion import ConditionalDiffusionModel, RainForecaster\n",
            "from src.retrieval.base import RetrievalDatabase\n",
            "\n",
            "# Settings\n",
            "plt.style.use('seaborn-v0_8-whitegrid')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
            "plt.rcParams['font.size'] = 11\n",
            "\n",
            "print(f'PyTorch version: {torch.__version__}')\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "print(f'Device: {\"cuda\" if torch.cuda.is_available() else \"cpu\"}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 1: DATA INGESTION
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 1: Data Ingestion\n",
            "\n",
            "Mengambil data cuaca historis dari Open-Meteo Archive API (ERA5 Reanalysis).\n",
            "\n",
            "**Variabel yang diambil:**\n",
            "- Target: `precipitation` (mm/jam)\n",
            "- Dynamic: temperature, humidity, pressure, wind, cloudcover\n",
            "- Static: elevation, land_sea_mask\n",
            "- Derived: precipitation_lag1, precipitation_lag3"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Load data (or fetch if not exists)\n",
            "DATA_PATH = '../data/raw/sitaro_era5_2005_2025.parquet'\n",
            "\n",
            "try:\n",
            "    df = pd.read_parquet(DATA_PATH)\n",
            "    print(f'✓ Data loaded from {DATA_PATH}')\n",
            "except FileNotFoundError:\n",
            "    print('Data not found. Fetching from Open-Meteo...')\n",
            "    df = fetch_era5_data()\n",
            "\n",
            "print(f'\\nDataset Shape: {df.shape}')\n",
            "print(f'Date Range: {df[\"date\"].min()} to {df[\"date\"].max()}')\n",
            "print(f'\\nColumns:')\n",
            "for i, col in enumerate(df.columns, 1):\n",
            "    print(f'  {i:2d}. {col}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Node Locations\n",
            "fig, ax = plt.subplots(figsize=(8, 8))\n",
            "\n",
            "nodes = SITARO_NODES.copy()\n",
            "ax.scatter(nodes['lon'], nodes['lat'], s=200, c='red', marker='^', edgecolors='black', linewidths=2)\n",
            "\n",
            "for i, name in enumerate(nodes['name']):\n",
            "    ax.annotate(name, (nodes['lon'].iloc[i], nodes['lat'].iloc[i]), \n",
            "                fontsize=12, ha='center', va='bottom', fontweight='bold',\n",
            "                xytext=(0, 10), textcoords='offset points')\n",
            "\n",
            "ax.set_xlabel('Longitude (°E)', fontsize=12)\n",
            "ax.set_ylabel('Latitude (°N)', fontsize=12)\n",
            "ax.set_title('Kepulauan Sitaro - Node Locations', fontsize=14, fontweight='bold')\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Precipitation Distribution\n",
            "fig, axes = plt.subplots(1, 4, figsize=(18, 4))\n",
            "\n",
            "# Raw distribution\n",
            "axes[0].hist(df['precipitation'], bins=100, color='steelblue', edgecolor='white')\n",
            "axes[0].set_xlabel('Precipitation (mm/h)')\n",
            "axes[0].set_ylabel('Frequency')\n",
            "axes[0].set_title('Raw Distribution (Heavy-tailed)')\n",
            "axes[0].set_yscale('log')\n",
            "\n",
            "# Log-transformed\n",
            "precip_log = np.log1p(df['precipitation'])\n",
            "axes[1].hist(precip_log, bins=100, color='forestgreen', edgecolor='white')\n",
            "axes[1].set_xlabel('log(1 + Precipitation)')\n",
            "axes[1].set_ylabel('Frequency')\n",
            "axes[1].set_title('Log-Transformed Distribution')\n",
            "\n",
            "# High intensity events (>5mm/h) - Primary threshold\n",
            "high_threshold = 5  # mm/h\n",
            "high_mask = df['precipitation'] > high_threshold\n",
            "axes[2].bar(['Normal', 'High (>5mm)'], \n",
            "            [len(df) - high_mask.sum(), high_mask.sum()],\n",
            "            color=['steelblue', 'orange'])\n",
            "axes[2].set_ylabel('Count')\n",
            "axes[2].set_title(f'High Intensity Rain (>5mm): {high_mask.sum()} ({high_mask.mean()*100:.2f}%)')\n",
            "\n",
            "# Very high intensity events (>10mm/h) - Secondary threshold\n",
            "very_high_threshold = 10  # mm/h\n",
            "very_high_mask = df['precipitation'] > very_high_threshold\n",
            "axes[3].bar(['Normal', 'Very High (>10mm)'], \n",
            "            [len(df) - very_high_mask.sum(), very_high_mask.sum()],\n",
            "            color=['steelblue', 'crimson'])\n",
            "axes[3].set_ylabel('Count')\n",
            "axes[3].set_title(f'Very High Intensity (>10mm): {very_high_mask.sum()} ({very_high_mask.mean()*100:.4f}%)')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f'\\nPrecipitation Statistics:')\n",
            "print(df['precipitation'].describe())\n",
            "print(f'\\n--- High Intensity Thresholds ---')\n",
            "print(f'> 5 mm/h (High Intensity): {high_mask.sum():,} events ({high_mask.mean()*100:.4f}%)')\n",
            "print(f'> 10 mm/h (Very High Intensity): {very_high_mask.sum():,} events ({very_high_mask.mean()*100:.4f}%)')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 2: TEMPORAL SPLIT
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 2: Temporal Split (CRITICAL!)\n",
            "\n",
            "⚠️ **MENCEGAH DATA LEAKAGE** - Random shuffle TIDAK digunakan!\n",
            "\n",
            "| Split | Periode | Proporsi |\n",
            "|-------|---------|----------|\n",
            "| Training | 2005-2018 | 67% |\n",
            "| Validation | 2019-2021 | 14% |\n",
            "| Test | 2022-2025 | 19% |"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def temporal_split(df, train_end='2018-12-31', val_end='2021-12-31'):\n",
            "    \"\"\"\n",
            "    Split DataFrame berdasarkan waktu, BUKAN random.\n",
            "    \"\"\"\n",
            "    df = df.copy()\n",
            "    df['date'] = pd.to_datetime(df['date'])\n",
            "    \n",
            "    # Remove timezone if present (to avoid tz-naive vs tz-aware comparison)\n",
            "    if df['date'].dt.tz is not None:\n",
            "        df['date'] = df['date'].dt.tz_localize(None)\n",
            "    \n",
            "    train_end_dt = pd.to_datetime(train_end)\n",
            "    val_end_dt = pd.to_datetime(val_end)\n",
            "    \n",
            "    train_mask = df['date'] <= train_end_dt\n",
            "    val_mask = (df['date'] > train_end_dt) & (df['date'] <= val_end_dt)\n",
            "    test_mask = df['date'] > val_end_dt\n",
            "    \n",
            "    return df[train_mask].copy(), df[val_mask].copy(), df[test_mask].copy()\n",
            "\n",
            "# Apply temporal split\n",
            "TRAIN_END = '2018-12-31'\n",
            "VAL_END = '2021-12-31'\n",
            "\n",
            "train_df, val_df, test_df = temporal_split(df, TRAIN_END, VAL_END)\n",
            "\n",
            "print('TEMPORAL SPLIT RESULTS:')\n",
            "print(f'  Training:   {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%) | {train_df[\"date\"].min().date()} to {train_df[\"date\"].max().date()}')\n",
            "print(f'  Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%) | {val_df[\"date\"].min().date()} to {val_df[\"date\"].max().date()}')\n",
            "print(f'  Test:       {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%) | {test_df[\"date\"].min().date()} to {test_df[\"date\"].max().date()}')\n",
            "\n",
            "# Verify no overlap\n",
            "print(f'\\n✓ Train max < Val min: {train_df[\"date\"].max() < val_df[\"date\"].min()}')\n",
            "print(f'✓ Val max < Test min: {val_df[\"date\"].max() < test_df[\"date\"].min()}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Temporal Split Timeline\n",
            "fig, ax = plt.subplots(figsize=(14, 3))\n",
            "\n",
            "# Create timeline\n",
            "years = range(2005, 2026)\n",
            "colors = ['#2ecc71'] * 14 + ['#f39c12'] * 3 + ['#e74c3c'] * 4  # green, orange, red\n",
            "\n",
            "ax.barh(0, 14, left=0, height=0.5, color='#2ecc71', label='Training (2005-2018)')\n",
            "ax.barh(0, 3, left=14, height=0.5, color='#f39c12', label='Validation (2019-2021)')\n",
            "ax.barh(0, 4, left=17, height=0.5, color='#e74c3c', label='Test (2022-2025)')\n",
            "\n",
            "# Add year labels\n",
            "for i, year in enumerate([2005, 2010, 2015, 2018, 2019, 2021, 2022, 2025]):\n",
            "    if year <= 2018:\n",
            "        x = year - 2005\n",
            "    elif year <= 2021:\n",
            "        x = 14 + (year - 2019)\n",
            "    else:\n",
            "        x = 17 + (year - 2022)\n",
            "    ax.axvline(x, color='black', linestyle='--', alpha=0.3)\n",
            "    ax.text(x, 0.35, str(year), ha='center', fontsize=10)\n",
            "\n",
            "ax.set_xlim(-0.5, 21.5)\n",
            "ax.set_ylim(-0.5, 0.8)\n",
            "ax.set_yticks([])\n",
            "ax.set_xlabel('Year Index', fontsize=12)\n",
            "ax.set_title('Temporal Data Split (NO RANDOM SHUFFLE)', fontsize=14, fontweight='bold')\n",
            "ax.legend(loc='upper right')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 3: PREPROCESSING
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 3: Preprocessing\n",
            "\n",
            "1. **Log Transform Target**: `y_log = log(1 + y)`\n",
            "2. **Z-Score Normalization**: `x_norm = (x - μ) / σ`\n",
            "\n",
            "⚠️ **Stats dihitung HANYA dari Training Set!**"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Define feature columns\n",
            "FEATURE_COLS = [\n",
            "    'temperature_2m',\n",
            "    'relative_humidity_2m', \n",
            "    'dewpoint_2m',\n",
            "    'surface_pressure',\n",
            "    'wind_speed_10m',\n",
            "    'wind_direction_10m',\n",
            "    'cloudcover',\n",
            "    'precipitation_lag1',\n",
            "    'precipitation_lag3',\n",
            "    'elevation',\n",
            "]\n",
            "\n",
            "# Use only available columns\n",
            "feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]\n",
            "print(f'Using {len(feature_cols)} features:')\n",
            "for col in feature_cols:\n",
            "    print(f'  - {col}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "def compute_stats_from_train(train_df, feature_cols):\n",
            "    \"\"\"\n",
            "    Compute normalization stats ONLY from training data.\n",
            "    \"\"\"\n",
            "    # Target stats (with log transform)\n",
            "    target_log = np.log1p(train_df['precipitation'].values)\n",
            "    t_mean = torch.tensor(target_log.mean(), dtype=torch.float32)\n",
            "    t_std = torch.tensor(target_log.std(), dtype=torch.float32)\n",
            "    \n",
            "    # Feature stats\n",
            "    features = train_df[feature_cols].values\n",
            "    c_mean = torch.tensor(features.mean(axis=0), dtype=torch.float32)\n",
            "    c_std = torch.tensor(features.std(axis=0), dtype=torch.float32)\n",
            "    \n",
            "    return {'t_mean': t_mean, 't_std': t_std, 'c_mean': c_mean, 'c_std': c_std}\n",
            "\n",
            "# Compute stats from TRAINING ONLY\n",
            "stats = compute_stats_from_train(train_df, feature_cols)\n",
            "\n",
            "print('Normalization Stats (from Training Data ONLY):')\n",
            "print(f'  Target Log Mean: {stats[\"t_mean\"]:.4f}')\n",
            "print(f'  Target Log Std:  {stats[\"t_std\"]:.4f}')\n",
            "print(f'\\nFeature Means:')\n",
            "for i, col in enumerate(feature_cols):\n",
            "    print(f'  {col}: μ={stats[\"c_mean\"][i]:.2f}, σ={stats[\"c_std\"][i]:.2f}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Before vs After Normalization\n",
            "fig, axes = plt.subplots(2, 5, figsize=(16, 6))\n",
            "\n",
            "sample_features = feature_cols[:5]  # First 5 features\n",
            "\n",
            "for i, col in enumerate(sample_features):\n",
            "    # Before\n",
            "    axes[0, i].hist(train_df[col].dropna(), bins=50, color='steelblue', alpha=0.7)\n",
            "    axes[0, i].set_title(f'{col}\\n(Before)', fontsize=9)\n",
            "    axes[0, i].tick_params(labelsize=8)\n",
            "    \n",
            "    # After\n",
            "    normalized = (train_df[col].values - stats['c_mean'][i].numpy()) / (stats['c_std'][i].numpy() + 1e-5)\n",
            "    axes[1, i].hist(normalized, bins=50, color='forestgreen', alpha=0.7)\n",
            "    axes[1, i].set_title(f'{col}\\n(After: μ≈0, σ≈1)', fontsize=9)\n",
            "    axes[1, i].tick_params(labelsize=8)\n",
            "\n",
            "plt.suptitle('Normalization: Before vs After', fontsize=14, fontweight='bold')\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 4: SLIDING WINDOW
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 4: Sliding Window & Graph Construction\n",
            "\n",
            "- **Sequence Length**: 6 timesteps\n",
            "- **Nodes**: 3 (Siau, Tagulandang, Biaro)\n",
            "- **Edges**: Fully connected (6 edges)\n",
            "\n",
            "```\n",
            "Input:  [Graph_t-5, Graph_t-4, Graph_t-3, Graph_t-2, Graph_t-1, Graph_t]\n",
            "Target: precipitation(t+1)\n",
            "```"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Create datasets\n",
            "SEQ_LEN = 6\n",
            "BATCH_SIZE = 32\n",
            "\n",
            "train_dataset = TemporalGraphDataset(\n",
            "    df=train_df,\n",
            "    feature_cols=feature_cols,\n",
            "    seq_len=SEQ_LEN,\n",
            "    stats=stats\n",
            ")\n",
            "\n",
            "val_dataset = TemporalGraphDataset(\n",
            "    df=val_df,\n",
            "    feature_cols=feature_cols,\n",
            "    seq_len=SEQ_LEN,\n",
            "    stats=stats  # SAME stats from training!\n",
            ")\n",
            "\n",
            "print(f'Training Dataset: {len(train_dataset)} samples')\n",
            "print(f'Validation Dataset: {len(val_dataset)} samples')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Inspect one sample\n",
            "graphs, target, context = train_dataset[0]\n",
            "\n",
            "print('Sample Structure:')\n",
            "print(f'  Graphs: {len(graphs)} timesteps')\n",
            "print(f'  Graph[0].x shape: {graphs[0].x.shape} (nodes × features)')\n",
            "print(f'  Graph[0].edge_index shape: {graphs[0].edge_index.shape}')\n",
            "print(f'  Target shape: {target.shape}')\n",
            "print(f'  Context shape: {context.shape}')\n",
            "\n",
            "# Visualize edge connections\n",
            "print(f'\\nEdge Index (COO format):')\n",
            "print(f'  Source nodes: {graphs[0].edge_index[0].tolist()}')\n",
            "print(f'  Target nodes: {graphs[0].edge_index[1].tolist()}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Graph Structure\n",
            "import networkx as nx\n",
            "\n",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n",
            "\n",
            "# Create graph\n",
            "G = nx.DiGraph()\n",
            "node_names = ['Siau', 'Tagulandang', 'Biaro']\n",
            "G.add_nodes_from(node_names)\n",
            "\n",
            "edge_index = graphs[0].edge_index.numpy()\n",
            "for src, tgt in zip(edge_index[0], edge_index[1]):\n",
            "    G.add_edge(node_names[src], node_names[tgt])\n",
            "\n",
            "# Draw graph\n",
            "pos = {'Siau': (0, 1), 'Tagulandang': (0.5, 0.5), 'Biaro': (0, 0)}\n",
            "nx.draw(G, pos, ax=axes[0], with_labels=True, node_color='lightblue', \n",
            "        node_size=2000, font_size=10, font_weight='bold',\n",
            "        arrows=True, arrowsize=20, edge_color='gray')\n",
            "axes[0].set_title('Graph Structure (Fully Connected)', fontsize=12, fontweight='bold')\n",
            "\n",
            "# Show sliding window\n",
            "timesteps = ['t-5', 't-4', 't-3', 't-2', 't-1', 't']\n",
            "axes[1].bar(timesteps, [1]*6, color='steelblue', alpha=0.7)\n",
            "axes[1].axvline(5.5, color='red', linestyle='--', linewidth=2)\n",
            "axes[1].text(5.7, 0.5, 'Predict\\nt+1', fontsize=10, color='red', fontweight='bold')\n",
            "axes[1].set_xlabel('Timestep')\n",
            "axes[1].set_ylabel('Graph Input')\n",
            "axes[1].set_title('Sliding Window (seq_len=6)', fontsize=12, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 5: RETRIEVAL DATABASE
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 5: Retrieval Database (FAISS)\n",
            "\n",
            "⚠️ **Index HANYA dari Training Data!**\n",
            "\n",
            "- Algorithm: Flat L2 (brute-force)\n",
            "- K neighbors: 3\n",
            "- Distance: Euclidean"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Build retrieval database from TRAINING ONLY\n",
            "train_features = train_df[feature_cols].values\n",
            "train_features_norm = (train_features - stats['c_mean'].numpy()) / (stats['c_std'].numpy() + 1e-5)\n",
            "\n",
            "retrieval_db = RetrievalDatabase(embedding_dim=len(feature_cols))\n",
            "retrieval_db.add_items(train_features_norm, train_features_norm)\n",
            "\n",
            "print(f'FAISS Index built with {len(train_features_norm):,} training samples')\n",
            "print(f'Embedding dimension: {len(feature_cols)}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Test retrieval\n",
            "K_NEIGHBORS = 3\n",
            "\n",
            "# Query with a sample\n",
            "query = train_features_norm[1000:1001]  # Single query\n",
            "retrieved = retrieval_db.query(query, k=K_NEIGHBORS)\n",
            "\n",
            "print(f'Query shape: {query.shape}')\n",
            "print(f'Retrieved shape: {retrieved.shape} (batch × k × features)')\n",
            "print(f'\\nK={K_NEIGHBORS} Nearest Neighbors found!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 6: SPATIO-TEMPORAL GNN
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 6: Spatio-Temporal GNN\n",
            "\n",
            "**Architecture:**\n",
            "1. **Spatial**: 2-layer GAT (Graph Attention Network)\n",
            "2. **Temporal**: Multi-head Self-Attention\n",
            "3. **Output**: Graph embedding [B, 64]"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Initialize models\n",
            "HIDDEN_DIM = 128\n",
            "GRAPH_DIM = 64\n",
            "CONTEXT_DIM = len(feature_cols)\n",
            "RETRIEVAL_DIM = CONTEXT_DIM * K_NEIGHBORS\n",
            "\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "\n",
            "# SpatioTemporal GNN\n",
            "st_gnn = SpatioTemporalGNN(\n",
            "    node_features=CONTEXT_DIM,\n",
            "    hidden_dim=HIDDEN_DIM // 2,\n",
            "    output_dim=GRAPH_DIM,\n",
            "    num_gat_heads=4,\n",
            "    num_attn_heads=4,\n",
            "    seq_len=SEQ_LEN\n",
            ").to(device)\n",
            "\n",
            "print(f'SpatioTemporalGNN Parameters: {sum(p.numel() for p in st_gnn.parameters()):,}')\n",
            "print(f'\\nArchitecture:')\n",
            "print(st_gnn)"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Test forward pass\n",
            "from torch.utils.data import DataLoader\n",
            "\n",
            "train_loader = DataLoader(\n",
            "    train_dataset, \n",
            "    batch_size=4, \n",
            "    shuffle=True,\n",
            "    collate_fn=collate_temporal_graphs\n",
            ")\n",
            "\n",
            "batched_graphs, targets, contexts = next(iter(train_loader))\n",
            "batched_graphs = [g.to(device) for g in batched_graphs]\n",
            "\n",
            "with torch.no_grad():\n",
            "    graph_emb = st_gnn(batched_graphs)\n",
            "\n",
            "print(f'Input: {len(batched_graphs)} graph batches')\n",
            "print(f'Output Graph Embedding: {graph_emb.shape}')\n",
            "print(f'\\n✓ SpatioTemporalGNN forward pass successful!')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 7: DIFFUSION TRAINING
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 7: Conditional Diffusion Model Training\n",
            "\n",
            "**Conditioning:**\n",
            "1. Time embedding (sinusoidal)\n",
            "2. Context embedding (current weather)\n",
            "3. Retrieval embedding (historical analogs)\n",
            "4. Graph embedding (spatio-temporal) ← **Thesis novelty!**"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Initialize Diffusion Model\n",
            "diff_model = ConditionalDiffusionModel(\n",
            "    input_dim=1,\n",
            "    context_dim=CONTEXT_DIM,\n",
            "    retrieval_dim=RETRIEVAL_DIM,\n",
            "    graph_dim=GRAPH_DIM,\n",
            "    hidden_dim=HIDDEN_DIM\n",
            ")\n",
            "\n",
            "forecaster = RainForecaster(diff_model, device=device)\n",
            "\n",
            "print(f'DiffusionModel Parameters: {sum(p.numel() for p in diff_model.parameters()):,}')\n",
            "print(f'\\nTotal Trainable Parameters: {sum(p.numel() for p in st_gnn.parameters()) + sum(p.numel() for p in diff_model.parameters()):,}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Training loop (mini version for demo)\n",
            "EPOCHS = 3  # Reduced for notebook demo\n",
            "\n",
            "train_loader = DataLoader(\n",
            "    train_dataset, \n",
            "    batch_size=BATCH_SIZE, \n",
            "    shuffle=True,\n",
            "    collate_fn=collate_temporal_graphs\n",
            ")\n",
            "\n",
            "# Combined optimizer\n",
            "all_params = list(st_gnn.parameters()) + list(diff_model.parameters())\n",
            "optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)\n",
            "\n",
            "print(f'Starting training for {EPOCHS} epochs...')\n",
            "print(f'Batches per epoch: {len(train_loader)}')\n",
            "\n",
            "losses = []\n",
            "\n",
            "for epoch in range(EPOCHS):\n",
            "    st_gnn.train()\n",
            "    forecaster.model.train()\n",
            "    epoch_loss = 0\n",
            "    \n",
            "    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')\n",
            "    for batched_graphs, targets, contexts in pbar:\n",
            "        batched_graphs = [g.to(device) for g in batched_graphs]\n",
            "        targets = targets.to(device)\n",
            "        contexts = contexts.to(device)\n",
            "        \n",
            "        # Get graph embedding\n",
            "        graph_emb = st_gnn(batched_graphs)\n",
            "        \n",
            "        # Get retrieval\n",
            "        with torch.no_grad():\n",
            "            retrieved = retrieval_db.query(contexts.cpu().numpy(), k=K_NEIGHBORS)\n",
            "            retrieved = retrieved.to(device)\n",
            "        \n",
            "        # Diffusion step\n",
            "        optimizer.zero_grad()\n",
            "        noise = torch.randn_like(targets).to(device)\n",
            "        timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()\n",
            "        noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)\n",
            "        \n",
            "        noise_pred = forecaster.model(\n",
            "            noisy_target, timesteps, contexts, retrieved, graph_emb\n",
            "        )\n",
            "        \n",
            "        loss = forecaster.criterion(noise_pred, noise)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        \n",
            "        epoch_loss += loss.item()\n",
            "        pbar.set_postfix(loss=f'{loss.item():.4f}')\n",
            "    \n",
            "    avg_loss = epoch_loss / len(train_loader)\n",
            "    losses.append(avg_loss)\n",
            "    print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Training Loss\n",
            "plt.figure(figsize=(10, 4))\n",
            "plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=8)\n",
            "plt.xlabel('Epoch')\n",
            "plt.ylabel('Average Loss (MSE)')\n",
            "plt.title('Training Loss Curve', fontsize=14, fontweight='bold')\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # PIPELINE 8: PROBABILISTIC INFERENCE
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Pipeline 8: Probabilistic Inference\n",
            "\n",
            "Generate 50 probabilistic rain samples for uncertainty quantification."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Inference function\n",
            "@torch.no_grad()\n",
            "def run_inference(graphs_sequence, context, st_gnn, forecaster, retrieval_db, stats, num_samples=50):\n",
            "    st_gnn.eval()\n",
            "    forecaster.model.eval()\n",
            "    \n",
            "    # Graph embedding\n",
            "    graph_emb = st_gnn(graphs_sequence)\n",
            "    \n",
            "    # Retrieval\n",
            "    retrieved = retrieval_db.query(context.cpu().numpy(), k=K_NEIGHBORS)\n",
            "    retrieved = retrieved.to(device)\n",
            "    \n",
            "    # Sample\n",
            "    samples = forecaster.sample(\n",
            "        condition=context,\n",
            "        retrieved=retrieved,\n",
            "        graph_emb=graph_emb,\n",
            "        num_samples=num_samples\n",
            "    )\n",
            "    \n",
            "    # Denormalize\n",
            "    samples_log = samples * stats['t_std'] + stats['t_mean']\n",
            "    samples_mm = torch.expm1(samples_log)\n",
            "    samples_mm = torch.clamp(samples_mm, min=0)\n",
            "    \n",
            "    return samples_mm.cpu().numpy().flatten()\n",
            "\n",
            "# Get a sample for inference\n",
            "graphs, target, context = train_dataset[100]\n",
            "batched_graphs = [g.unsqueeze(0).to(device) if hasattr(g, 'unsqueeze') else g.to(device) for g in graphs]\n",
            "\n",
            "# Need to batch graphs properly\n",
            "from torch_geometric.data import Batch\n",
            "batched_graphs = [Batch.from_data_list([g]).to(device) for g in graphs]\n",
            "context = context.unsqueeze(0).to(device)\n",
            "\n",
            "# Run inference\n",
            "predictions = run_inference(\n",
            "    batched_graphs, context, st_gnn, forecaster, retrieval_db, stats, num_samples=50\n",
            ")\n",
            "\n",
            "print(f'Generated {len(predictions)} probabilistic samples')\n",
            "print(f'\\nPrediction Statistics:')\n",
            "print(f'  Mean: {predictions.mean():.2f} mm')\n",
            "print(f'  Std:  {predictions.std():.2f} mm')\n",
            "print(f'  Min:  {predictions.min():.2f} mm')\n",
            "print(f'  Max:  {predictions.max():.2f} mm')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    cells.append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# Visualize: Probabilistic Predictions\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "\n",
            "# Histogram of samples\n",
            "axes[0].hist(predictions, bins=20, color='steelblue', edgecolor='white', alpha=0.7)\n",
            "axes[0].axvline(predictions.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {predictions.mean():.2f}')\n",
            "axes[0].set_xlabel('Predicted Precipitation (mm)')\n",
            "axes[0].set_ylabel('Frequency')\n",
            "axes[0].set_title('50 Probabilistic Samples', fontsize=12, fontweight='bold')\n",
            "axes[0].legend()\n",
            "\n",
            "# Box plot\n",
            "bp = axes[1].boxplot(predictions, vert=True, patch_artist=True)\n",
            "bp['boxes'][0].set_facecolor('lightblue')\n",
            "axes[1].set_ylabel('Predicted Precipitation (mm)')\n",
            "axes[1].set_title('Distribution (Box Plot)', fontsize=12, fontweight='bold')\n",
            "\n",
            "# Percentiles\n",
            "percentiles = [10, 25, 50, 75, 90]\n",
            "perc_values = [np.percentile(predictions, p) for p in percentiles]\n",
            "axes[2].bar([f'P{p}' for p in percentiles], perc_values, color='forestgreen', edgecolor='white')\n",
            "axes[2].set_ylabel('Precipitation (mm)')\n",
            "axes[2].set_title('Percentiles (Uncertainty)', fontsize=12, fontweight='bold')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Extreme event probability\n",
            "thresholds = [50, 100, 150]\n",
            "print('\\nExtreme Event Probabilities:')\n",
            "for t in thresholds:\n",
            "    prob = (predictions > t).mean() * 100\n",
            "    print(f'  P(Rain > {t}mm): {prob:.1f}%')"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Summary\n",
            "\n",
            "### Pipeline Completed:\n",
            "\n",
            "1. ✅ **Data Ingestion**: 20 tahun data ERA5\n",
            "2. ✅ **Temporal Split**: Train 2005-2018 / Val 2019-2021 / Test 2022-2025\n",
            "3. ✅ **Preprocessing**: Log transform + Z-score normalization\n",
            "4. ✅ **Sliding Window**: 6-timestep graph sequences\n",
            "5. ✅ **Retrieval Database**: FAISS dengan k=3 neighbors\n",
            "6. ✅ **Spatio-Temporal GNN**: GAT + TemporalAttention\n",
            "7. ✅ **Diffusion Training**: DDPM with all conditioning\n",
            "8. ✅ **Probabilistic Inference**: 50 ensemble samples\n",
            "\n",
            "### Key Features:\n",
            "\n",
            "- 🔒 **No Data Leakage**: Strict temporal split\n",
            "- 📊 **Probabilistic**: Uncertainty quantification\n",
            "- 🌐 **Graph-based**: Spatio-temporal dependencies\n",
            "- 🔍 **Retrieval-Augmented**: Historical analog search"
        ]
    })
    
    # ==========================================================================
    # Create notebook structure
    # ==========================================================================
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }
    
    return notebook


if __name__ == "__main__":
    notebook = create_notebook()
    
    output_path = "notebooks/complete_pipeline.ipynb"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Notebook created: {output_path}")
