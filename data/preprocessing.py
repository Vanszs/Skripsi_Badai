import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def label_anomalies(df, window_hours=1):
    """
    Generates 3-Class Labels based on Pressure Drop Rate (Meteorological Standard).
    
    Logic:
    - Delta P (hourly drop) = P(t-1) - P(t)
    - Class 0 (Normal): Drop < 1.0 hPa/hr
    - Class 1 (Anomaly): 1.0 <= Drop < 3.0 hPa/hr (Rapid Descent)
    - Class 2 (Storm): Drop >= 3.0 hPa/hr (Explosive Intensification)
    
    Returns:
        Series with labels {0, 1, 2}
    """
    # Calculate hourly pressure change (negative diff means drop)
    # We want positive value for drop magnitude
    # diff = current - prev. If current < prev (drop), diff is negative.
    # drop = prev - current = -diff
    pressure_change = -df['pressure'].diff(periods=window_hours)
    
    # Initialize with 0 (Normal)
    labels = np.zeros(len(df), dtype=int)
    
    # Apply Thresholds
    # Class 2: Storm
    labels[pressure_change >= 3.0] = 2
    
    # Class 1: Anomaly (where not already Storm)
    mask_anomaly = (pressure_change >= 1.0) & (pressure_change < 3.0)
    labels[mask_anomaly] = 1
    
    # Fill NaN at start (due to diff) with 0
    labels = np.nan_to_num(labels, 0)
    
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
    # Kinetic Energy ~ 0.5 * mass * velocity^2 -> Proportional to speed squared
    df['wind_kinetic'] = df['wind_speed'] ** 2
    
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
