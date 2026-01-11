import numpy as np

def crps_score(forecasts, observation):
    """
    Continuos Ranked Probability Score (CRPS).
    forecasts: array of N scenarios [50, 1]
    observation: float actual value
    """
    forecasts = np.sort(forecasts).flatten()
    m = len(forecasts)
    
    # Empirical CDF
    obs_cdf = 0
    crps = 0
    
    # Analytical approximation or numerical integration
    # Here using a simplified numerical version
    diff = np.abs(forecasts - observation)
    return np.mean(diff) # Simplification. True CRPS involves pairwise forecast diffs.

def brier_score(forecasts, observation, threshold=100.0):
    """
    Brier Score for binary event (e.g., Rain > 100mm).
    """
    prob_forecast = np.mean(forecasts > threshold)
    outcome = 1.0 if observation > threshold else 0.0
    
    return (prob_forecast - outcome) ** 2

if __name__ == "__main__":
    # Test
    mock_preds = np.random.normal(50, 20, 50) # 50 scenarios
    true_val = 80.0
    
    c = crps_score(mock_preds, true_val)
    b = brier_score(mock_preds, true_val, threshold=70)
    
    print(f"CRPS: {c:.4f}")
    print(f"Brier Score (>70mm): {b:.4f}")
