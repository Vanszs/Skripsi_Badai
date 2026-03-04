"""
Metrik Evaluasi Probabilistik untuk Nowcasting Hujan Lebat

Metrik yang diimplementasikan:
1. CRPS (Continuous Ranked Probability Score)
2. Brier Score (untuk event hujan lebat)
3. POD (Probability of Detection)
4. FAR (False Alarm Ratio)
5. CSI (Critical Success Index)

Metrik deterministik (tetap dipertahankan):
- RMSE
- MAE
- Correlation

Referensi:
- Wilks, D.S. (2019). Statistical Methods in the Atmospheric Sciences.
- Hersbach, H. (2000). Decomposition of the CRPS for ensemble prediction systems.
"""

import numpy as np
from typing import Dict, Optional


# ==============================================================================
# METRIK DETERMINISTIK
# ==============================================================================

def compute_rmse(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Root Mean Square Error (NaN-safe)."""
    mask = ~(np.isnan(predictions) | np.isnan(observations))
    if mask.sum() == 0:
        return float('nan')
    return float(np.sqrt(np.mean((predictions[mask] - observations[mask]) ** 2)))


def compute_mae(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Mean Absolute Error (NaN-safe)."""
    mask = ~(np.isnan(predictions) | np.isnan(observations))
    if mask.sum() == 0:
        return float('nan')
    return float(np.mean(np.abs(predictions[mask] - observations[mask])))


def compute_correlation(predictions: np.ndarray, observations: np.ndarray) -> float:
    """Pearson Correlation Coefficient (NaN-safe)."""
    mask = ~(np.isnan(predictions) | np.isnan(observations))
    if mask.sum() < 3:
        return 0.0
    p, o = predictions[mask], observations[mask]
    if np.std(p) < 1e-10 or np.std(o) < 1e-10:
        return 0.0
    return float(np.corrcoef(p, o)[0, 1])


# ==============================================================================
# METRIK PROBABILISTIK
# ==============================================================================

def compute_crps(ensemble_samples: np.ndarray, observations: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score (CRPS).
    
    CRPS mengukur kualitas prediksi probabilistik. 
    Semakin rendah CRPS, semakin baik prediksi.
    
    Menggunakan formula:
    CRPS = E|X - y| - 0.5 * E|X - X'|
    di mana X dan X' adalah sampel independen dari distribusi prediksi,
    dan y adalah observasi.
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble] - sampel ensemble per timestep
        observations: [N_timesteps] - nilai observasi
    
    Returns:
        float: rata-rata CRPS
    """
    n_timesteps = len(observations)
    crps_values = []
    
    for t in range(n_timesteps):
        samples = ensemble_samples[t]  # [N_ensemble]
        obs = observations[t]
        
        # Skip NaN samples
        valid = ~np.isnan(samples)
        if np.isnan(obs) or valid.sum() < 2:
            continue
        samples = samples[valid]
        
        # Term 1: E|X - y|
        term1 = np.mean(np.abs(samples - obs))
        
        # Term 2: 0.5 * E|X - X'|
        n = len(samples)
        if n > 1:
            # Efficient computation using sorted samples
            sorted_samples = np.sort(samples)
            # E|X - X'| = (2/n^2) * sum_i (2i - n - 1) * x_(i)
            indices = np.arange(1, n + 1)
            term2 = np.sum((2 * indices - n - 1) * sorted_samples) / (n * n)
        else:
            term2 = 0.0
        
        crps = term1 - 0.5 * abs(term2)
        crps_values.append(crps)
    
    return float(np.mean(crps_values))


def compute_brier_score(ensemble_samples: np.ndarray, observations: np.ndarray,
                         threshold: float = 10.0) -> float:
    """
    Brier Score untuk event hujan lebat.
    
    Brier Score = (1/N) * sum((p_i - o_i)^2)
    
    di mana:
    - p_i = probabilitas prediksi (fraksi ensemble yang melebihi threshold)
    - o_i = 1 jika observasi melebihi threshold, 0 jika tidak
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble]
        observations: [N_timesteps]
        threshold: batas curah hujan lebat (mm/jam), default 10 mm/jam
    
    Returns:
        float: Brier Score (0 = sempurna, 1 = terburuk)
    """
    n_timesteps = len(observations)
    brier_values = []
    
    for t in range(n_timesteps):
        samples = ensemble_samples[t]
        obs = observations[t]
        
        # Probabilitas prediksi: fraksi ensemble > threshold
        prob_pred = np.mean(samples > threshold)
        
        # Observasi biner
        obs_binary = 1.0 if obs > threshold else 0.0
        
        brier = (prob_pred - obs_binary) ** 2
        brier_values.append(brier)
    
    return float(np.mean(brier_values))


def compute_pod(ensemble_samples: np.ndarray, observations: np.ndarray,
                threshold: float = 10.0, prob_threshold: float = 0.5) -> float:
    """
    Probability of Detection (POD) / Hit Rate.
    
    POD = Hits / (Hits + Misses)
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble]
        observations: [N_timesteps]
        threshold: batas curah hujan lebat (mm/jam)
        prob_threshold: probabilitas minimum untuk dianggap prediksi positif
    
    Returns:
        float: POD (0-1, 1 = sempurna)
    """
    hits = 0
    misses = 0
    
    for t in range(len(observations)):
        obs_event = observations[t] > threshold
        pred_prob = np.mean(ensemble_samples[t] > threshold)
        pred_event = pred_prob >= prob_threshold
        
        if obs_event:
            if pred_event:
                hits += 1
            else:
                misses += 1
    
    if hits + misses == 0:
        return float('nan')  # Tidak ada event hujan lebat
    
    return float(hits / (hits + misses))


def compute_far(ensemble_samples: np.ndarray, observations: np.ndarray,
                threshold: float = 10.0, prob_threshold: float = 0.5) -> float:
    """
    False Alarm Ratio (FAR).
    
    FAR = False Alarms / (Hits + False Alarms)
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble]
        observations: [N_timesteps]
        threshold: batas curah hujan lebat (mm/jam)
        prob_threshold: probabilitas minimum untuk dianggap prediksi positif
    
    Returns:
        float: FAR (0-1, 0 = sempurna)
    """
    hits = 0
    false_alarms = 0
    
    for t in range(len(observations)):
        obs_event = observations[t] > threshold
        pred_prob = np.mean(ensemble_samples[t] > threshold)
        pred_event = pred_prob >= prob_threshold
        
        if pred_event:
            if obs_event:
                hits += 1
            else:
                false_alarms += 1
    
    if hits + false_alarms == 0:
        return float('nan')  # Tidak ada prediksi positif
    
    return float(false_alarms / (hits + false_alarms))


def compute_csi(ensemble_samples: np.ndarray, observations: np.ndarray,
                threshold: float = 10.0, prob_threshold: float = 0.5) -> float:
    """
    Critical Success Index (CSI) / Threat Score.
    
    CSI = Hits / (Hits + Misses + False Alarms)
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble]
        observations: [N_timesteps]
        threshold: batas curah hujan lebat (mm/jam)
        prob_threshold: probabilitas minimum untuk dianggap prediksi positif
    
    Returns:
        float: CSI (0-1, 1 = sempurna)
    """
    hits = 0
    misses = 0
    false_alarms = 0
    
    for t in range(len(observations)):
        obs_event = observations[t] > threshold
        pred_prob = np.mean(ensemble_samples[t] > threshold)
        pred_event = pred_prob >= prob_threshold
        
        if obs_event and pred_event:
            hits += 1
        elif obs_event and not pred_event:
            misses += 1
        elif not obs_event and pred_event:
            false_alarms += 1
    
    denominator = hits + misses + false_alarms
    if denominator == 0:
        return float('nan')
    
    return float(hits / denominator)


# ==============================================================================
# FUNGSI AGREGASI
# ==============================================================================

def compute_all_metrics(ensemble_samples: np.ndarray, 
                        observations: np.ndarray,
                        deterministic_predictions: Optional[np.ndarray] = None,
                        heavy_rain_threshold: float = 10.0,
                        prob_threshold: float = 0.5) -> Dict[str, float]:
    """
    Hitung semua metrik evaluasi sekaligus.
    
    Args:
        ensemble_samples: [N_timesteps, N_ensemble] - sampel ensemble prediksi
        observations: [N_timesteps] - observasi aktual
        deterministic_predictions: [N_timesteps] - prediksi deterministik (median ensemble jika None)
        heavy_rain_threshold: batas curah hujan lebat (mm/jam)
        prob_threshold: probabilitas minimum untuk deteksi event
    
    Returns:
        Dict dengan semua metrik
    """
    if deterministic_predictions is None:
        deterministic_predictions = np.median(ensemble_samples, axis=1)
    
    metrics = {}
    
    # Metrik deterministik
    metrics['rmse'] = compute_rmse(deterministic_predictions, observations)
    metrics['mae'] = compute_mae(deterministic_predictions, observations)
    metrics['correlation'] = compute_correlation(deterministic_predictions, observations)
    
    # Metrik probabilistik
    metrics['crps'] = compute_crps(ensemble_samples, observations)
    metrics['brier_score'] = compute_brier_score(ensemble_samples, observations, heavy_rain_threshold)
    metrics['pod'] = compute_pod(ensemble_samples, observations, heavy_rain_threshold, prob_threshold)
    metrics['far'] = compute_far(ensemble_samples, observations, heavy_rain_threshold, prob_threshold)
    metrics['csi'] = compute_csi(ensemble_samples, observations, heavy_rain_threshold, prob_threshold)
    
    return metrics


def compute_reliability_data(ensemble_samples: np.ndarray, observations: np.ndarray,
                              threshold: float = 10.0, n_bins: int = 10) -> Dict:
    """
    Hitung data untuk reliability diagram.
    
    Returns:
        Dict dengan 'forecast_probs', 'observed_freqs', 'bin_counts'
    """
    # Hitung probabilitas prediksi per timestep
    pred_probs = np.array([np.mean(ensemble_samples[t] > threshold) 
                           for t in range(len(observations))])
    obs_binary = (observations > threshold).astype(float)
    
    # Binning
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    forecast_probs = []
    observed_freqs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        count = np.sum(mask)
        bin_counts.append(int(count))
        
        if count > 0:
            forecast_probs.append(float(np.mean(pred_probs[mask])))
            observed_freqs.append(float(np.mean(obs_binary[mask])))
        else:
            forecast_probs.append(float(bin_centers[i]))
            observed_freqs.append(float('nan'))
    
    return {
        'forecast_probs': forecast_probs,
        'observed_freqs': observed_freqs,
        'bin_counts': bin_counts,
        'bin_edges': bin_edges.tolist()
    }
