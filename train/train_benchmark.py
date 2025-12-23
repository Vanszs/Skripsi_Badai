import numpy as np
import xgboost as xgb
from models.autoencoder import Autoencoder
import time

def train_hybrid_benchmark(X_train, y_train, input_size, latent_size=5, ae_epochs=50, ae_lr=0.01):
    """
    Trains a Hybrid Autoencoder + XGBoost model (Multi-Class).
    
    1. Autoencoder learns compression (unsupervised).
    2. Features = Latent Vector + Reconstruction Error.
    3. XGBoost predicts class (0, 1, 2) from these features.
    """
    print(f"Starting Hubrid Benchmark Training...")
    start_time = time.time()
    
    # --- Phase 1: Train Autoencoder ---
    print(f"1. Training Autoencoder (Latent={latent_size})...")
    ae = Autoencoder(input_size, latent_size)
    ae_history = ae.train(X_train, epochs=ae_epochs, lr=ae_lr)
    
    # --- Phase 2: Feature Extraction ---
    # XGBoost gets: Compressed Latent + Anomaly Score (MSE)
    # Note: Preprocessing already added domain features to X_train, 
    # so the AE compresses those too! 
    # Ideally, domain features should bypass AE, but for simplicity we compress everything 
    # AND append the error signal.
    print("2. Extracting Hybrid Features...")
    X_train_augmented = ae.get_features(X_train)
    
    # --- Phase 3: Train XGBoost (Multi-Class) ---
    print("3. Training XGBoost Classifier (Multi-Class)...")
    
    # XGBoost Config for 3 Classes
    params = {
        'objective': 'multi:softmax',
        'num_class': 3,  # Normal, Anomaly, Storm
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'seed': 42
    }
    
    dtrain = xgb.DMatrix(X_train_augmented, label=y_train)
    bst = xgb.train(params, dtrain, num_boost_round=100)
    
    end_time = time.time()
    print(f"Benchmark Training Complete. Time: {end_time - start_time:.2f}s")
    
    return {
        'autoencoder': ae,
        'xgboost': bst
    }, ae_history

def predict_hybrid(model_pack, X):
    """
    Predict using the hybrid pipeline.
    """
    ae = model_pack['autoencoder']
    bst = model_pack['xgboost']
    
    # 1. Transform features with AE
    X_augmented = ae.get_features(X)
    
    # 2. Predict with XGBoost
    dtest = xgb.DMatrix(X_augmented)
    preds = bst.predict(dtest) # Returns class indices directly for multi:softmax
    
    return preds.astype(int)
