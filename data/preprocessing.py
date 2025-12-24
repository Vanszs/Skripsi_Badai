import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def label_anomalies(df, window_hours=None):
    """
    Generates 3-Class Labels based on WMO Saffir-Simpson Wind Scale.
    
    Standard (WMO/BMKG):
    - Tropical Depression: ~11-17 m/s (22-33 knots)
    - Tropical Storm: >= 18 m/s (34+ knots)
    
    Classes:
    - 0 (Normal): Wind Speed < 11 m/s
    - 1 (Anomaly/Depression): 11.0 <= Wind Speed < 18.0 m/s
    - 2 (Tropical Storm): Wind Speed >= 18.0 m/s
    
    Args:
        df: DataFrame with 'wind_speed' (m/s)
        window_hours: Not used for wind threshold, kept for compatibility.
        
    Returns:
        Series with labels {0, 1, 2}
    """
    # 1. Initialize all as Normal (0)
    labels = np.zeros(len(df), dtype=int)
    
    # Adjusted for ERA5 Hourly Mean (Smoothed Data)
    # Real-world TCs often show lower grid-cell averages than point gusts.
    
    # 2. Assign Class 2 (Storm / Near Gale)
    # Lowered to 15.0 m/s (~30 knots) to capture ERA5 storm signals
    labels[df['wind_speed'] >= 15.0] = 2
    
    # 3. Assign Class 1 (Anomaly / Depression)
    # Lowered to 10.0 m/s (~20 knots)
    mask_anomaly = (df['wind_speed'] >= 10.0) & (df['wind_speed'] < 15.0)
    labels[mask_anomaly] = 1
    
    return labels

def add_derived_features(df):
    """
    Adds Domain-Specific Features for Hybrid+ Model.
    """
    df = df.copy()
    
    # 1. Pressure Dynamics
    df['pressure_gradient'] = df['pressure'].diff().fillna(0) # 1h gradient
    df['pressure_ma24'] = df['pressure'].rolling(window=24).mean().bfill()
    df['pressure_std24'] = df['pressure'].rolling(window=24).std().fillna(0)
    
    # 2. Wind Kinetics
    # Kinetic Energy = 0.5 * density * velocity^2
    # Standard Air Density at Sea Level = 1.225 kg/m^3
    AIR_DENSITY = 1.225
    df['wind_kinetic'] = 0.5 * AIR_DENSITY * (df['wind_speed'] ** 2)
    
    # Gust Factor (Gustiness)
    # Avoid division by zero
    df['gust_factor'] = df['wind_gust'] / (df['wind_speed'] + 1e-6)
    
    # 3. Temporal Embeddings (Cyclical)
    if 'dt' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['dt']):
             df['dt'] = pd.to_datetime(df['dt'])
        
        df['hour_sin'] = np.sin(2 * np.pi * df['dt'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['dt'].dt.hour / 24)
    
    return df

def validate_dataset(X, y):
    """
    Validates dataset quality before training.
    """
    # Check completeness
    if np.isnan(X).sum() > 0:
        raise ValueError("❌ Missing values found in dataset")
        
    # Check temporal continuity (heuristic: size check)
    if len(X) < 100:
        raise ValueError("❌ Dataset too small for training")
    
    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"✅ Data Validation Passed. Class Distribution: {dist}")
    
    # Check for severe imbalance
    total = len(y)
    for label, count in dist.items():
        if (count / total) < 0.05:
            print(f"⚠️ WARNING: Class {label} is severely underrepresented (<5%)")

def normalize_features(train_df, test_df, feature_cols):
    """
    Fits scaler on Train, Transforms Test to prevent leakage.
    Returns normalized dataframes and the scaler.
    """
    scaler = StandardScaler()
    
    # Fit on training data only
    train_matrix = scaler.fit_transform(train_df[feature_cols])
    test_matrix = scaler.transform(test_df[feature_cols])
    
    # Convert back to DataFrame to keep indices/structure if needed, 
    # but here we usually return numpy arrays for modeling
    return train_matrix, test_matrix, scaler

def create_sliding_window(data, labels, window_size):
    """
    Creates (samples, window_size, features) for X 
    and (samples,) for y.
    
    Args:
        data: Feature matrix (N, F)
        labels: Label vector (N,)
        window_size: Int (e.g. 72)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        # Label is the state AT the end of the window (prediction target)
        y.append(labels[i + window_size]) 
        
    return np.array(X), np.array(y)
