"""
Joint Stability-Intensity Weighting functions.

This module provides the canonical implementation of the weighting mechanism
used across all formant analysis scripts. The weighting focuses analysis on
acoustically relevant, stable, and intense portions of audio.

Formula:
    weights = w_intensity * w_stability * gate_mask
    
Where:
    - w_intensity = clip(intensity - noise_floor, 0, 30)^2
    - w_stability = 1 / (instability + smoothing)
    - instability = sum(|dF/dt| / F) for F1, F2, F3
    - gate_mask = intensity >= soft_gate_threshold
"""

import numpy as np
from .config import (
    DEFAULT_NOISE_FLOOR,
    DEFAULT_SOFT_GATE_THRESHOLD,
    DEFAULT_INTENSITY_CLIP_DB,
    DEFAULT_INTENSITY_EXPONENT,
    DEFAULT_STABILITY_SMOOTHING
)


def compute_joint_weights(
    f1_arr: np.ndarray,
    f2_arr: np.ndarray,
    f3_arr: np.ndarray,
    time_arr: np.ndarray,
    intensity_arr: np.ndarray,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    soft_gate_threshold: float = DEFAULT_SOFT_GATE_THRESHOLD,
    intensity_clip_db: float = DEFAULT_INTENSITY_CLIP_DB,
    intensity_exponent: float = DEFAULT_INTENSITY_EXPONENT,
    stability_smoothing: float = DEFAULT_STABILITY_SMOOTHING
) -> dict:
    """
    Compute Joint Stability-Intensity Weights for formant analysis.
    
    This is the canonical implementation used across all analysis scripts.
    It focuses analysis on stable, loud vocalic segments while ignoring
    transient or silent parts.
    
    Args:
        f1_arr: Array of F1 frequencies (Hz)
        f2_arr: Array of F2 frequencies (Hz)  
        f3_arr: Array of F3 frequencies (Hz)
        time_arr: Array of time values (seconds)
        intensity_arr: Array of intensity values (dB)
        noise_floor: Intensity below which frames are down-weighted (dB)
        soft_gate_threshold: Intensity below which frames get zero weight (dB)
        intensity_clip_db: Maximum intensity above floor to consider (dB)
        intensity_exponent: Exponent for intensity weighting (default 2.0)
        stability_smoothing: Smoothing constant for stability weight
    
    Returns:
        Dictionary containing:
            - frame_weights: Raw (unnormalized) weights
            - frame_weights_norm: Normalized weights (sum to 1)
            - instability: Normalized formant instability per frame
            - n_eff: Effective number of frames
            - confidence: n_eff / total frames (0-1)
            - weight_entropy: Normalized entropy (0=concentrated, 1=uniform)
    """
    # 1. Compute gradients using np.gradient for proper edge handling
    df1 = np.abs(np.gradient(f1_arr, time_arr))
    df2 = np.abs(np.gradient(f2_arr, time_arr))
    df3 = np.abs(np.gradient(f3_arr, time_arr))
    
    # 2. Normalized Instability: |dF/dt| / F (dimensionless)
    instability = (df1 / f1_arr) + (df2 / f2_arr) + (df3 / f3_arr)
    
    # 3. Compute weight components
    gate_mask = intensity_arr >= soft_gate_threshold
    
    # Clip intensity above floor to prevent burst dominance
    intensity_above_floor = np.clip(intensity_arr - noise_floor, 0, intensity_clip_db)
    w_intensity = intensity_above_floor ** intensity_exponent
    
    # Stability weight: inverse of instability
    w_stability = 1.0 / (instability + stability_smoothing)
    
    # 4. Combined weight
    weights = w_intensity * w_stability * gate_mask
    
    # 5. Fallback if all weights are zero
    if np.sum(weights) == 0:
        weights = np.ones_like(intensity_arr)
    
    # 6. Normalize
    weights_norm = weights / np.sum(weights)
    
    # 7. Diagnostics
    sum_w = np.sum(weights)
    sum_w_sq = np.sum(weights**2)
    n_eff = (sum_w**2) / sum_w_sq if sum_w_sq > 0 else 0
    confidence = np.clip(n_eff / len(f1_arr), 0, 1)
    weight_entropy = compute_weight_entropy(weights_norm)
    
    return {
        'frame_weights': weights,
        'frame_weights_norm': weights_norm,
        'instability': instability,
        'n_eff': n_eff,
        'confidence': confidence,
        'weight_entropy': weight_entropy
    }


def compute_weight_entropy(weights_norm: np.ndarray) -> float:
    """
    Compute normalized entropy of weights.
    
    This diagnostic helps detect pathological weighting where only
    a few frames dominate the analysis.
    
    Args:
        weights_norm: Normalized weights (must sum to 1)
    
    Returns:
        Normalized entropy value:
            - 0.0 = Fully concentrated (one frame dominates)
            - 1.0 = Uniform (all frames equally weighted)
            - Values in between indicate partial concentration
    """
    p = weights_norm
    entropy = -np.sum(p * np.log(p + 1e-12))
    return entropy / np.log(len(p)) if len(p) > 1 else 0.0


def compute_weighted_stats(
    values: np.ndarray,
    weights_norm: np.ndarray
) -> dict:
    """
    Compute weighted mean and variance for an array of values.
    
    Args:
        values: Array of values
        weights_norm: Normalized weights (must sum to 1)
    
    Returns:
        Dictionary with 'mean', 'var', 'std'
    """
    mean = np.average(values, weights=weights_norm)
    var = np.average((values - mean)**2, weights=weights_norm)
    return {
        'mean': mean,
        'var': var,
        'std': np.sqrt(var)
    }
