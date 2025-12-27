"""
Core formant extraction functions using Praat.

This module provides the canonical formant extraction implementation
used across all analysis scripts. It uses Praat (via parselmouth) for
formant estimation and applies Joint Stability-Intensity Weighting.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

from .weighting import compute_joint_weights, compute_weighted_stats
from .config import (
    DEFAULT_TIME_STEP,
    DEFAULT_MAX_FORMANTS,
    DEFAULT_MAX_FORMANT_FREQ,
    DEFAULT_WINDOW_LENGTH,
    DEFAULT_STABILITY_SMOOTHING
)


def extract_formants_with_weights(
    audio_path: str,
    time_step: float = DEFAULT_TIME_STEP,
    max_formants: int = DEFAULT_MAX_FORMANTS,
    max_formant_freq: float = DEFAULT_MAX_FORMANT_FREQ,
    window_length: float = DEFAULT_WINDOW_LENGTH,
    stability_smoothing: float = DEFAULT_STABILITY_SMOOTHING,
    return_raw_arrays: bool = True
) -> dict:
    """
    Extract formant frequencies with Joint Stability-Intensity Weighting.
    
    This is the canonical implementation used across all analysis scripts.
    It extracts F1, F2, F3 from an audio file and computes weighted statistics
    that focus on stable, high-intensity vocalic segments.
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length in seconds
        stability_smoothing: Smoothing constant for stability weights
        return_raw_arrays: If True, include raw value arrays in output
    
    Returns:
        Dictionary containing:
            - f1_mean, f2_mean, f3_mean: Weighted means (Hz)
            - f1_std, f2_std, f3_std: Weighted standard deviations
            - f1_median_unweighted, etc.: Unweighted medians (diagnostic)
            - n_frames: Number of valid frames
            - duration: Audio duration (seconds)
            - frame_weights, frame_weights_norm: Weights
            - n_eff, confidence, weight_entropy: Diagnostics
            - f1_values, f2_values, f3_values, time_values, intensity_values: 
              Raw arrays (if return_raw_arrays=True)
        
        Returns None if extraction fails or too few frames.
    """
    try:
        # Load audio and create Praat objects
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        formant = call(sound, "To Formant (burg)",
                       time_step, max_formants, max_formant_freq, window_length, 50.0)
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        n_frames = call(formant, "Get number of frames")
        
        # Collect all valid frames
        f1_values, f2_values, f3_values = [], [], []
        time_values, intensity_values = [], []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            # Get intensity at this time
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens):
                    intens = 0.0
            except:
                intens = 0.0
            
            # Only include valid (non-NaN, positive) values
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        # Require minimum number of frames
        if len(f1_values) < 3:
            return None
        
        # Convert to numpy arrays
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        time_arr = np.array(time_values)
        intensity_arr = np.array(intensity_values)
        
        # Compute weights using the canonical weighting function
        weight_data = compute_joint_weights(
            f1_arr, f2_arr, f3_arr, time_arr, intensity_arr,
            stability_smoothing=stability_smoothing
        )
        
        weights_norm = weight_data['frame_weights_norm']
        
        # Compute weighted statistics for each formant
        f1_stats = compute_weighted_stats(f1_arr, weights_norm)
        f2_stats = compute_weighted_stats(f2_arr, weights_norm)
        f3_stats = compute_weighted_stats(f3_arr, weights_norm)
        
        # Build result dictionary
        result = {
            # Weighted means
            'f1_mean': f1_stats['mean'],
            'f2_mean': f2_stats['mean'],
            'f3_mean': f3_stats['mean'],
            
            # Weighted standard deviations
            'f1_std': f1_stats['std'],
            'f2_std': f2_stats['std'],
            'f3_std': f3_stats['std'],
            
            # Unweighted medians (for diagnostic purposes)
            'f1_median_unweighted': np.median(f1_arr),
            'f2_median_unweighted': np.median(f2_arr),
            'f3_median_unweighted': np.median(f3_arr),
            
            # Metadata
            'n_frames': len(f1_values),
            'duration': duration,
            
            # Weights and diagnostics
            'frame_weights': weight_data['frame_weights'],
            'frame_weights_norm': weight_data['frame_weights_norm'],
            'n_eff': weight_data['n_eff'],
            'confidence': weight_data['confidence'],
            'weight_entropy': weight_data['weight_entropy'],
        }
        
        # Optionally include raw arrays for per-frame analysis
        if return_raw_arrays:
            result.update({
                'f1_values': f1_arr,
                'f2_values': f2_arr,
                'f3_values': f3_arr,
                'time_values': time_arr,
                'intensity_values': intensity_arr,
            })
        
        return result
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def extract_raw_formant_trajectory(
    audio_path: str,
    time_step: float = 0.005,  # Finer resolution for trajectory analysis
    max_formants: int = DEFAULT_MAX_FORMANTS,
    max_formant_freq: float = DEFAULT_MAX_FORMANT_FREQ,
    window_length: float = DEFAULT_WINDOW_LENGTH,
    intensity_threshold: float = 50.0
) -> dict:
    """
    Extract raw formant trajectory data for temporal analysis.
    
    Unlike extract_formants_with_weights, this function is optimized for
    trajectory analysis (dF/dt, curvature, etc.) with finer time resolution
    and returns data suitable for computing derivatives.
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step (default 0.005s for finer resolution)
        max_formants: Maximum number of formants
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length (seconds)
        intensity_threshold: Minimum intensity threshold (dB)
    
    Returns:
        Dictionary containing:
            - f1_values, f2_values, f3_values: Formant arrays (Hz)
            - time_values: Time array (seconds)
            - intensity_values: Intensity array (dB)
            - n_frames: Number of valid frames
            - duration: Audio duration
            - time_step: Time step used
        
        Returns None if extraction fails.
    """
    try:
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        formant = call(sound, "To Formant (burg)",
                       time_step, max_formants, max_formant_freq, window_length, 50.0)
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        n_frames = call(formant, "Get number of frames")
        
        f1_values, f2_values, f3_values = [], [], []
        time_values, intensity_values = [], []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens):
                    intens = 0.0
            except:
                intens = 0.0
            
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        if len(f1_values) < 5:  # Need more frames for trajectory analysis
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        time_arr = np.array(time_values)
        intensity_arr = np.array(intensity_values)
        
        # Apply intensity threshold filter
        valid_mask = intensity_arr >= intensity_threshold
        if np.sum(valid_mask) < 5:
            # Fall back to all frames if too few pass threshold
            valid_mask = np.ones(len(f1_arr), dtype=bool)
        
        return {
            'f1_values': f1_arr[valid_mask],
            'f2_values': f2_arr[valid_mask],
            'f3_values': f3_arr[valid_mask],
            'time_values': time_arr[valid_mask],
            'intensity_values': intensity_arr[valid_mask],
            'n_frames': int(np.sum(valid_mask)),
            'duration': duration,
            'time_step': time_step
        }
        
    except Exception as e:
        print(f"Error extracting trajectory from {audio_path}: {e}")
        return None
