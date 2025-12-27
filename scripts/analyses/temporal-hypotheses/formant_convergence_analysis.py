#!/usr/bin/env python3
"""
Temporal/Dynamic Hypothesis Analysis: Formant Convergence/Divergence

This script analyzes formant trajectories to measure convergence/divergence patterns.

Key Metrics:
- Distance metric: |F2-F1| over time
- Convergence rate: d(|F2-F1|)/dt

Hypothesis: 
- /a/ (अ) = convergent (F2-F1 distance decreases over time)
- /i/ (इ) = divergent (F2-F1 distance increases over time)

Modes:
- Two-file comparison: Compare convergence patterns between two files
- Batch folder: Compare all files against a reference
- Golden files: Analyze all golden files across phonemes
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
from pathlib import Path
from scipy import stats

# Add parent directory to path to import common package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from common package
from common import (
    configure_matplotlib,
    extract_raw_formant_trajectory,
    compute_joint_weights,
    apply_dark_theme,
    create_styled_figure,
    style_legend,
    COLORS,
    tqdm,
    HAS_SEABORN,
    HAS_TQDM
)

# Import seaborn for enhanced plotting (optional)
if HAS_SEABORN:
    import seaborn as sns

# Configure matplotlib for Devanagari support
configure_matplotlib()


def extract_formant_trajectory(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                                max_formant_freq: float = 5500.0, window_length: float = 0.025,
                                intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant frequencies (F1, F2) trajectory from an audio file.
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length in seconds
        intensity_threshold: Minimum intensity (dB) for a frame to be considered
    
    Returns:
        Dictionary containing formant trajectories and time values
    """
    try:
        # Load the audio file with Praat
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        # Create a Formant object
        formant = call(sound, "To Formant (burg)",
                       time_step,       # Time step
                       max_formants,    # Max number of formants
                       max_formant_freq,  # Maximum formant frequency
                       window_length,   # Window length
                       50.0)            # Pre-emphasis from (Hz)
        
        # Create Intensity object for low-energy filtering
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        
        # Get the number of frames
        n_frames = call(formant, "Get number of frames")
        
        # Collect formant values for all frames
        f1_values = []
        f2_values = []
        time_values = []
        intensity_values = []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            
            # Get intensity at this time
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens):
                    intens = 0.0
            except:
                intens = 60.0  # Default if intensity extraction fails
            
            # Only include valid (non-undefined) values above intensity threshold
            if not np.isnan(f1) and not np.isnan(f2):
                if f1 > 0 and f2 > 0 and intens >= intensity_threshold:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        if len(f1_values) < 5:  # Need at least 5 frames for trajectory analysis
            return None
        
        return {
            'f1_values': np.array(f1_values),
            'f2_values': np.array(f2_values),
            'time_values': np.array(time_values),
            'intensity_values': np.array(intensity_values),
            'n_frames': len(f1_values),
            'duration': duration
        }
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def compute_convergence_metrics(trajectory_data: dict) -> dict:
    """
    Compute convergence/divergence metrics from formant trajectory.
    
    Metrics:
    - |F2-F1| distance at each time point
    - d(|F2-F1|)/dt: Rate of change of distance (convergence rate)
    - Classification: CONVERGENT (negative rate), DIVERGENT (positive rate), STABLE
    
    Refined Method (Method 3):
    - Uses Joint Stability-Intensity Weighting for robust regression
    - Weight = (max(0, intensity - 50)^2) / (instability + epsilon)
    - Instability is frequency-normalized: sum(|dF/dt| / F)
    
    Args:
        trajectory_data: Dictionary from extract_formant_trajectory()
    
    Returns:
        Dictionary containing convergence metrics
    """
    if trajectory_data is None:
        return None
    
    f1 = trajectory_data['f1_values']
    f2 = trajectory_data['f2_values']
    time = trajectory_data['time_values']
    intensity = trajectory_data['intensity_values']
    
    # Compute |F2-F1| distance at each time point
    f2_f1_distance = np.abs(f2 - f1)
    
    # --- Weighting Logic (Method 3: Joint Stability-Intensity) ---
    
    # 1. Calculate Instability (Frequency-Normalized)
    # Using np.gradient for proper edge handling
    d_f1 = np.gradient(f1, time)
    d_f2 = np.gradient(f2, time)
    
    # Normalized instability: |dF/F|
    instability = (np.abs(d_f1) / f1) + (np.abs(d_f2) / f2)
    
    # 2. Calculate Weights - Joint Stability-Intensity
    noise_floor = 50.0
    stability_smoothing = 0.1
    soft_gate_threshold = 30.0
    intensity_clip_db = 30.0  # Clip to prevent burst dominance
    intensity_exponent = 2.0
    
    gate_mask = intensity >= soft_gate_threshold
    
    # Clip intensity above floor to prevent single loud frames from dominating
    intensity_above_floor = np.clip(intensity - noise_floor, 0, intensity_clip_db)
    w_intensity = intensity_above_floor ** intensity_exponent
    w_stability = 1.0 / (instability + stability_smoothing)
    
    # Joint Weight
    weights = w_intensity * w_stability * gate_mask
    
    # Handle edge case where all weights are zero
    if np.sum(weights) == 0:
        # Fallback to unweighted if valid data exists, else return None
        if len(time) > 0:
            weights = np.ones_like(time)
        else:
            return None
    
    # Normalize weights
    weights_norm = weights / np.sum(weights)
    
    # Weight entropy (0=concentrated, 1=uniform)
    p = weights_norm
    weight_entropy = -np.sum(p * np.log(p + 1e-12))
    weight_entropy_norm = weight_entropy / np.log(len(p)) if len(p) > 1 else 0

    # --- Metrics Logic ---

    # 1. Weighted Regression (Primary Metric)
    try:
        # np.polyfit with weights (w=weights) minimizes weighted least squares
        coeffs, cov = np.polyfit(time, f2_f1_distance, deg=1, w=weights, cov=True)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        convergence_rate = slope
        
        # Calculate Weighted R-squared
        predicted = slope * time + intercept
        residuals = f2_f1_distance - predicted
        weighted_ss_res = np.sum(weights * residuals**2)
        weighted_mean = np.average(f2_f1_distance, weights=weights)
        weighted_ss_tot = np.sum(weights * (f2_f1_distance - weighted_mean)**2)
        
        r_squared = 1 - (weighted_ss_res / weighted_ss_tot) if weighted_ss_tot > 0 else 0
        
    except Exception as e:
        print(f"Weighted regression failed: {e}")
        convergence_rate = 0.0
        r_squared = 0.0
        predicted = f2_f1_distance # Fallback

    # 2. Unweighted Regression (Legacy/Debug Metric)
    try:
        slope_legacy, _, r_value_legacy, p_value_legacy, _ = stats.linregress(time, f2_f1_distance)
        convergence_rate_legacy = slope_legacy
    except:
        convergence_rate_legacy = 0.0
        p_value_legacy = 1.0

    # 3. Diagnostics
    sum_w = np.sum(weights)
    sum_w_sq = np.sum(weights**2)
    n_eff = (sum_w**2) / sum_w_sq if sum_w_sq > 0 else 0
    
    # Confidence Score: Min(Fraction of effective frames, Weight Entropy-ish metric)
    n_total = len(time)
    confidence = np.clip(n_eff / n_total, 0, 1)

    # Classification based on WEIGHTED convergence rate
    # Threshold: 50 Hz/s is considered significant
    RATE_THRESHOLD = 50.0  # Hz/s
    
    if convergence_rate < -RATE_THRESHOLD:
        classification = "CONVERGENT"
    elif convergence_rate > RATE_THRESHOLD:
        classification = "DIVERGENT"
    else:
        classification = "STABLE"
    
    # Compute additional statistics (Weighted)
    distance_mean = np.average(f2_f1_distance, weights=weights)
    distance_std = np.sqrt(np.average((f2_f1_distance - distance_mean)**2, weights=weights))
    distance_start = predicted[0] 
    distance_end = predicted[-1]
    distance_change = distance_end - distance_start
    
    # Normalized convergence (percentage change per second)
    if distance_mean > 0:
        normalized_rate = (convergence_rate / distance_mean) * 100  # %/second
    else:
        normalized_rate = 0.0
    
    # Frame rates
    d_distance = np.diff(f2_f1_distance)
    dt = np.diff(time)
    dt = np.where(dt == 0, 1e-6, dt)  # Avoid division by zero
    frame_rates = d_distance / dt
    
    return {
        # Distance metrics (Weighted)
        'f2_f1_distance': f2_f1_distance,
        'f2_f1_distance_mean': distance_mean,
        'f2_f1_distance_std': distance_std,
        'f2_f1_distance_start': distance_start,
        'f2_f1_distance_end': distance_end,
        'f2_f1_distance_change': distance_change,
        
        # Convergence rate metrics (Weighted)
        'convergence_rate': convergence_rate,  # Hz/second
        'convergence_rate_normalized': normalized_rate,  # %/second
        'convergence_r_squared': r_squared,
        
        # Legacy/Debug Metrics (Unweighted)
        'convergence_rate_unweighted': convergence_rate_legacy,
        'convergence_p_value_unweighted': p_value_legacy,
        'convergence_p_value': p_value_legacy, # For backward compatibility
        
        # Diagnostics
        'n_eff': n_eff,
        'confidence': confidence,
        'weight_entropy': weight_entropy_norm,
        'frame_weights': weights,
        'frame_weights_norm': weights_norm,
        
        # Frame-by-frame rates
        'frame_rates': frame_rates,
        'frame_rate_mean': np.mean(frame_rates),
        'frame_rate_std': np.std(frame_rates),
        
        # Classification
        'classification': classification,
        
        # Time info
        'time_values': time,
        'analysis_duration': time[-1] - time[0]
    }


def analyze_audio_file(audio_path: str) -> dict:
    """
    Complete convergence/divergence analysis for a single audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary with all trajectory data and convergence metrics
    """
    # Using 8000 Hz max formant frequency
    trajectory_data = extract_formant_trajectory(audio_path, max_formant_freq=8000.0)
    
    if trajectory_data is None:
        return None
    
    metrics = compute_convergence_metrics(trajectory_data)
    
    if metrics is None:
        return None
    
    # Combine all data
    result = {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        'n_frames': trajectory_data['n_frames'],
        'duration': trajectory_data['duration'],
        
        # Key convergence metrics
        'f2_f1_distance_mean': metrics['f2_f1_distance_mean'],
        'f2_f1_distance_std': metrics['f2_f1_distance_std'],
        'f2_f1_distance_start': metrics['f2_f1_distance_start'],
        'f2_f1_distance_end': metrics['f2_f1_distance_end'],
        'f2_f1_distance_change': metrics['f2_f1_distance_change'],
        
        'convergence_rate': metrics['convergence_rate'],
        'convergence_rate_normalized': metrics['convergence_rate_normalized'],
        'convergence_r_squared': metrics['convergence_r_squared'],
        'convergence_p_value': metrics['convergence_p_value'],
        
        'n_eff': metrics.get('n_eff', 0),
        'confidence': metrics.get('confidence', 0),
        
        'classification': metrics['classification'],
        'analysis_duration': metrics['analysis_duration'],
        
        # Raw data for plotting
        '_trajectory': trajectory_data,
        '_metrics': metrics
    }
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """
    Compare convergence patterns between two audio files.
    
    Args:
        file1_path: Path to first audio file
        file2_path: Path to second audio file
        output_dir: Directory to save results
    
    Returns:
        DataFrame with comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing: {os.path.basename(file1_path)}")
    result1 = analyze_audio_file(file1_path)
    
    print(f"Analyzing: {os.path.basename(file2_path)}")
    result2 = analyze_audio_file(file2_path)
    
    if result1 is None or result2 is None:
        print("Error: Could not analyze one or both files")
        return None
    
    # Create comparison DataFrame
    metrics = [
        'f2_f1_distance_mean', 'f2_f1_distance_std',
        'f2_f1_distance_start', 'f2_f1_distance_end', 'f2_f1_distance_change',
        'convergence_rate', 'convergence_rate_normalized',
        'convergence_r_squared', 'convergence_p_value',
        'classification', 'analysis_duration'
    ]
    
    comparison_data = []
    for metric in metrics:
        val1 = result1.get(metric, np.nan)
        val2 = result2.get(metric, np.nan)
        
        if isinstance(val1, str):
            comparison_data.append({
                'metric': metric,
                f'{os.path.basename(file1_path)}': val1,
                f'{os.path.basename(file2_path)}': val2,
                'difference': 'N/A',
            })
        else:
            diff = val1 - val2 if not np.isnan(val1) and not np.isnan(val2) else np.nan
            comparison_data.append({
                'metric': metric,
                f'{os.path.basename(file1_path)}': val1,
                f'{os.path.basename(file2_path)}': val2,
                'difference': diff,
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'convergence_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print N_eff for verification
    if 'n_eff' in result1:
        print(f"File 1 Effective Frames (N_eff): {result1['n_eff']:.2f}")
    if 'n_eff' in result2:
        print(f"File 2 Effective Frames (N_eff): {result2['n_eff']:.2f}")
    
    # Create visualization
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing convergence patterns between two files."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    
    # Colors
    color1 = '#FF6B6B'  # Coral red
    color2 = '#4ECDC4'  # Teal
    
    # Get trajectory data
    traj1 = result1['_trajectory']
    traj2 = result2['_trajectory']
    metrics1 = result1['_metrics']
    metrics2 = result2['_metrics']
    
    # 1. F2-F1 Distance Over Time
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    # Normalize time to percentage for comparison
    t1_norm = (traj1['time_values'] - traj1['time_values'][0]) / (traj1['time_values'][-1] - traj1['time_values'][0]) * 100
    t2_norm = (traj2['time_values'] - traj2['time_values'][0]) / (traj2['time_values'][-1] - traj2['time_values'][0]) * 100
    
    ax.plot(t1_norm, metrics1['f2_f1_distance'], color=color1, linewidth=2, label=file1_name, alpha=0.8)
    ax.plot(t2_norm, metrics2['f2_f1_distance'], color=color2, linewidth=2, label=file2_name, alpha=0.8)
    
    ax.set_xlabel('Time (% of vowel)', color='#EAEAEA')
    ax.set_ylabel('|F2 - F1| Distance (Hz)', color='#EAEAEA')
    ax.set_title('F2-F1 Distance Trajectory', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Convergence Rate Comparison
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    rates = [result1['convergence_rate'], result2['convergence_rate']]
    names = [file1_name, file2_name]
    colors = [color1, color2]
    
    bars = ax.barh(names, rates, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
    
    # Add classification labels
    for i, (rate, name) in enumerate(zip(rates, names)):
        classification = result1['classification'] if i == 0 else result2['classification']
        ax.text(rate, i, f'  {classification}', va='center', color='#EAEAEA', fontsize=10)
    
    ax.axvline(x=0, color='#FFD93D', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Convergence Rate (Hz/s)', color='#EAEAEA')
    ax.set_ylabel('File', color='#EAEAEA')
    ax.set_title('Convergence Rate (d|F2-F1|/dt)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add annotation
    ax.annotate('← CONVERGENT', xy=(-50, -0.5), fontsize=9, color='#4ECDC4', ha='right')
    ax.annotate('DIVERGENT →', xy=(50, -0.5), fontsize=9, color='#FF6B6B', ha='left')
    
    # 3. F1-F2 Trajectory in Vowel Space
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    # Plot F1 vs F2 trajectory with arrows showing direction
    ax.scatter(traj1['f2_values'], traj1['f1_values'], c=t1_norm, cmap='Reds', s=30, alpha=0.7, label=file1_name)
    ax.scatter(traj2['f2_values'], traj2['f1_values'], c=t2_norm, cmap='Blues', s=30, alpha=0.7, label=file2_name)
    
    # Mark start and end points
    ax.scatter([traj1['f2_values'][0]], [traj1['f1_values'][0]], c='white', s=100, marker='o', edgecolors=color1, linewidths=2, zorder=5)
    ax.scatter([traj1['f2_values'][-1]], [traj1['f1_values'][-1]], c=color1, s=100, marker='X', edgecolors='white', linewidths=2, zorder=5)
    ax.scatter([traj2['f2_values'][0]], [traj2['f1_values'][0]], c='white', s=100, marker='o', edgecolors=color2, linewidths=2, zorder=5)
    ax.scatter([traj2['f2_values'][-1]], [traj2['f1_values'][-1]], c=color2, s=100, marker='X', edgecolors='white', linewidths=2, zorder=5)
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)', color='#EAEAEA')
    ax.set_ylabel('F1 (Hz)', color='#EAEAEA')
    ax.set_title('F1-F2 Trajectory (○=start, ✕=end)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA', loc='upper left')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    FORMANT CONVERGENCE/DIVERGENCE ANALYSIS
    ========================================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    F2-F1 DISTANCE (Hz):
    ─────────────────────────────────────
    Mean:  {result1['f2_f1_distance_mean']:.1f} vs {result2['f2_f1_distance_mean']:.1f}
    Start: {result1['f2_f1_distance_start']:.1f} vs {result2['f2_f1_distance_start']:.1f}
    End:   {result1['f2_f1_distance_end']:.1f} vs {result2['f2_f1_distance_end']:.1f}
    Change:{result1['f2_f1_distance_change']:+.1f} vs {result2['f2_f1_distance_change']:+.1f}
    
    CONVERGENCE RATE (d|F2-F1|/dt):
    ─────────────────────────────────────
    Rate (Hz/s): {result1['convergence_rate']:+.1f} vs {result2['convergence_rate']:+.1f}
    Rate (%/s):  {result1['convergence_rate_normalized']:+.2f} vs {result2['convergence_rate_normalized']:+.2f}
    R²:          {result1['convergence_r_squared']:.3f} vs {result2['convergence_r_squared']:.3f}
    
    CLASSIFICATION:
    ─────────────────────────────────────
    {file1_name}: {result1['classification']}
    {file2_name}: {result2['classification']}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'convergence_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """
    Compare all audio files in a folder against a pinned reference file.
    
    Args:
        folder_path: Path to folder containing audio files
        reference_file: Path to the reference (pinned) file to compare against
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all comparison results
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all wav files in the folder
    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
    
    if not wav_files:
        print(f"Error: No .wav files found in {folder_path}")
        return None
    
    # Analyze the reference file first
    print(f"\n{'='*60}")
    print(f"REFERENCE FILE: {os.path.basename(reference_file)}")
    print(f"{'='*60}")
    
    ref_result = analyze_audio_file(reference_file)
    if ref_result is None:
        print(f"Error: Could not analyze reference file: {reference_file}")
        return None
    
    # Compare each file against the reference
    all_comparisons = []
    successful_results = []
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        # Skip if it's the reference file itself
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        
        if result is None:
            continue
        
        successful_results.append(result)
        
        # Compute comparison metrics
        comparison = {
            'filename': os.path.basename(wav_file),
            'f2_f1_distance_mean': result['f2_f1_distance_mean'],
            'f2_f1_distance_std': result['f2_f1_distance_std'],
            'convergence_rate': result['convergence_rate'],
            'convergence_rate_normalized': result['convergence_rate_normalized'],
            'classification': result['classification'],
            'convergence_r_squared': result['convergence_r_squared'],
            
            # Differences from reference
            'rate_diff_from_ref': result['convergence_rate'] - ref_result['convergence_rate'],
            'distance_mean_diff_from_ref': result['f2_f1_distance_mean'] - ref_result['f2_f1_distance_mean'],
        }
        
        all_comparisons.append(comparison)
    
    if not all_comparisons:
        print("Error: No files could be analyzed")
        return None
    
    # Create results DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'batch_convergence_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create batch visualization
    create_batch_plots(ref_result, successful_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    
    # Colors
    ref_color = '#FFD93D'  # Gold for reference
    convergent_color = '#4ECDC4'  # Teal for convergent
    divergent_color = '#FF6B6B'  # Coral for divergent
    stable_color = '#888888'  # Gray for stable
    
    def get_color(classification):
        if classification == 'CONVERGENT':
            return convergent_color
        elif classification == 'DIVERGENT':
            return divergent_color
        return stable_color
    
    # 1. Convergence Rate Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    rates = [r['convergence_rate'] for r in all_results]
    colors = [get_color(r['classification']) for r in all_results]
    
    ax.axhline(y=ref_result['convergence_rate'], color=ref_color, linestyle='--', linewidth=2, 
               label=f'Reference: {ref_name}')
    ax.scatter(range(len(rates)), rates, c=colors, s=50, alpha=0.8)
    ax.axhline(y=0, color='#666', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Convergence Rate (Hz/s)', color='#EAEAEA')
    ax.set_title('Convergence Rate vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Classification Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    class_counts = comparison_df['classification'].value_counts()
    class_colors = [get_color(c) for c in class_counts.index]
    
    wedges, texts, autotexts = ax.pie(class_counts.values, labels=class_counts.index, 
                                       colors=class_colors, autopct='%1.1f%%',
                                       textprops={'color': '#EAEAEA'})
    ax.set_title('Classification Distribution', color='#EAEAEA', fontweight='bold')
    
    # 3. Distance vs Rate Scatter
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    distances = [r['f2_f1_distance_mean'] for r in all_results]
    rates = [r['convergence_rate'] for r in all_results]
    colors = [get_color(r['classification']) for r in all_results]
    
    ax.scatter(distances, rates, c=colors, s=60, alpha=0.8, edgecolors='white', linewidths=0.5)
    ax.scatter([ref_result['f2_f1_distance_mean']], [ref_result['convergence_rate']], 
               c=ref_color, s=150, marker='*', edgecolors='white', linewidths=2, 
               label=f'Reference', zorder=5)
    
    ax.axhline(y=0, color='#666', linestyle='-', linewidth=1, alpha=0.5)
    ax.axhline(y=50, color='#444', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=-50, color='#444', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Mean |F2-F1| Distance (Hz)', color='#EAEAEA')
    ax.set_ylabel('Convergence Rate (Hz/s)', color='#EAEAEA')
    ax.set_title('Distance vs Convergence Rate', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    n_convergent = sum(1 for r in all_results if r['classification'] == 'CONVERGENT')
    n_divergent = sum(1 for r in all_results if r['classification'] == 'DIVERGENT')
    n_stable = sum(1 for r in all_results if r['classification'] == 'STABLE')
    
    avg_rate = np.mean([r['convergence_rate'] for r in all_results])
    std_rate = np.std([r['convergence_rate'] for r in all_results])
    
    summary_text = f"""
    BATCH CONVERGENCE ANALYSIS SUMMARY
    ===================================
    
    Reference: {ref_name}
    Reference Rate: {ref_result['convergence_rate']:+.1f} Hz/s
    Reference Class: {ref_result['classification']}
    
    FILES ANALYZED: {n_files}
    ─────────────────────────────────
    Convergent:  {n_convergent} ({n_convergent/n_files*100:.1f}%)
    Divergent:   {n_divergent} ({n_divergent/n_files*100:.1f}%)
    Stable:      {n_stable} ({n_stable/n_files*100:.1f}%)
    
    CONVERGENCE RATE STATISTICS:
    ─────────────────────────────────
    Mean Rate:   {avg_rate:+.1f} Hz/s
    Std Dev:     {std_rate:.1f} Hz/s
    
    INTERPRETATION:
    ─────────────────────────────────
    Negative rate = Convergent (F2→F1)
    Positive rate = Divergent (F2←F1)
    Threshold: ±50 Hz/s
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_convergence_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Batch visualization saved to: {plot_path}")


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Find and compare all golden files across different phonemes.
    
    Args:
        cleaned_data_dir: Path to the cleaned data directory (e.g., data/02_cleaned)
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all golden file convergence data
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all golden files
    golden_pattern = os.path.join(cleaned_data_dir, '*', '*_golden_*.wav')
    golden_files = glob.glob(golden_pattern)
    
    if not golden_files:
        print(f"Error: No golden files found in {cleaned_data_dir}")
        return None
    
    print(f"\nFound {len(golden_files)} golden files")
    print("=" * 60)
    
    # Analyze each golden file
    all_results = []
    
    for golden_file in tqdm(sorted(golden_files), desc="Analyzing golden files"):
        # Extract phoneme from parent directory
        phoneme = os.path.basename(os.path.dirname(golden_file))
        
        result = analyze_audio_file(golden_file)
        
        if result is None:
            continue
        
        result['phoneme'] = phoneme
        all_results.append(result)
    
    if not all_results:
        print("Error: No golden files could be analyzed")
        return None
    
    # Create DataFrame (exclude internal trajectory/metrics data)
    df_data = []
    for r in all_results:
        df_data.append({k: v for k, v in r.items() if not k.startswith('_')})
    
    df = pd.DataFrame(df_data)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'golden_convergence_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_golden_comparison_plots(df, all_results, output_dir)
    
    return df


def create_golden_comparison_plots(df: pd.DataFrame, all_results: list, output_dir: str):
    """Create visualization comparing all golden files across phonemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    # Colors
    convergent_color = '#4ECDC4'
    divergent_color = '#FF6B6B'
    stable_color = '#888888'
    
    def get_color(classification):
        if classification == 'CONVERGENT':
            return convergent_color
        elif classification == 'DIVERGENT':
            return divergent_color
        return stable_color
    
    # 1. Convergence Rate by Phoneme
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    # Group by phoneme and get mean convergence rate
    phoneme_rates = df.groupby('phoneme')['convergence_rate'].mean().sort_values()
    
    colors = [get_color('CONVERGENT' if r < -50 else 'DIVERGENT' if r > 50 else 'STABLE') 
              for r in phoneme_rates.values]
    
    bars = ax.barh(range(len(phoneme_rates)), phoneme_rates.values, color=colors, 
                   alpha=0.8, edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(phoneme_rates)))
    ax.set_yticklabels(phoneme_rates.index, fontsize=10)
    ax.axvline(x=0, color='#FFD93D', linestyle='-', linewidth=2, alpha=0.8)
    ax.axvline(x=-50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Convergence Rate (Hz/s)', color='#EAEAEA', fontsize=11)
    ax.set_title('Convergence Rate by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Distance vs Rate Scatter with Phoneme Labels
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    for _, row in df.iterrows():
        color = get_color(row['classification'])
        ax.scatter(row['f2_f1_distance_mean'], row['convergence_rate'], 
                   c=color, s=100, alpha=0.7, edgecolors='white', linewidths=0.5)
        ax.annotate(row['phoneme'], (row['f2_f1_distance_mean'], row['convergence_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='#EAEAEA')
    
    ax.axhline(y=0, color='#FFD93D', linestyle='-', linewidth=2, alpha=0.8)
    ax.axhline(y=-50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Mean |F2-F1| Distance (Hz)', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('Convergence Rate (Hz/s)', color='#EAEAEA', fontsize=11)
    ax.set_title('Distance vs Convergence Rate', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')
    
    # 3. Classification by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    # Create stacked bar for each phoneme
    phoneme_class = df.groupby(['phoneme', 'classification']).size().unstack(fill_value=0)
    
    if not phoneme_class.empty:
        phoneme_class = phoneme_class.reindex(columns=['CONVERGENT', 'STABLE', 'DIVERGENT'], fill_value=0)
        
        phonemes = phoneme_class.index.tolist()
        x = np.arange(len(phonemes))
        width = 0.6
        
        bottom = np.zeros(len(phonemes))
        for cls, color in [('CONVERGENT', convergent_color), ('STABLE', stable_color), ('DIVERGENT', divergent_color)]:
            if cls in phoneme_class.columns:
                values = phoneme_class[cls].values
                ax.bar(x, values, width, label=cls, bottom=bottom, color=color, alpha=0.8)
                bottom += values
        
        ax.set_xticks(x)
        ax.set_xticklabels(phonemes, fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('Phoneme', color='#EAEAEA', fontsize=11)
        ax.set_ylabel('Count', color='#EAEAEA', fontsize=11)
        ax.set_title('Classification by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
        ax.tick_params(colors='#EAEAEA')
        ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_total = len(df)
    n_convergent = (df['classification'] == 'CONVERGENT').sum()
    n_divergent = (df['classification'] == 'DIVERGENT').sum()
    n_stable = (df['classification'] == 'STABLE').sum()
    
    # Find most convergent and divergent phonemes
    phoneme_avg = df.groupby('phoneme')['convergence_rate'].mean()
    most_convergent = phoneme_avg.idxmin() if len(phoneme_avg) > 0 else 'N/A'
    most_divergent = phoneme_avg.idxmax() if len(phoneme_avg) > 0 else 'N/A'
    
    summary_text = f"""
    GOLDEN FILES CONVERGENCE ANALYSIS
    ==================================
    
    Total Phonemes: {df['phoneme'].nunique()}
    Total Files: {n_total}
    
    CLASSIFICATION SUMMARY:
    ─────────────────────────────────
    Convergent: {n_convergent} ({n_convergent/n_total*100:.1f}%)
    Divergent:  {n_divergent} ({n_divergent/n_total*100:.1f}%)
    Stable:     {n_stable} ({n_stable/n_total*100:.1f}%)
    
    RATE STATISTICS:
    ─────────────────────────────────
    Mean:   {df['convergence_rate'].mean():+.1f} Hz/s
    Std:    {df['convergence_rate'].std():.1f} Hz/s
    Min:    {df['convergence_rate'].min():+.1f} Hz/s
    Max:    {df['convergence_rate'].max():+.1f} Hz/s
    
    HYPOTHESIS CHECK:
    ─────────────────────────────────
    Most Convergent: {most_convergent}
      Rate: {phoneme_avg.get(most_convergent, 0):+.1f} Hz/s
    Most Divergent: {most_divergent}
      Rate: {phoneme_avg.get(most_divergent, 0):+.1f} Hz/s
    
    Hypothesis: /a/ = convergent, /i/ = divergent
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_convergence_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    # Generate seaborn plots if available
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn-based visualizations for golden files."""
    
    BG_COLOR = '#111111'
    TEXT_COLOR = '#eaeaea'
    ACCENT_COLOR = '#e17100'
    BORDER_COLOR = '#333333'
    
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['figure.facecolor'] = BG_COLOR
    plt.rcParams['axes.facecolor'] = BG_COLOR
    plt.rcParams['axes.edgecolor'] = BORDER_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR
    
    # 1. Convergence Rate by Phoneme with Raincloud/Strip Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Sort phonemes by mean convergence rate
    order = df.groupby('phoneme')['convergence_rate'].mean().sort_values().index.tolist()
    
    palette = {'CONVERGENT': '#4ECDC4', 'STABLE': '#888888', 'DIVERGENT': '#FF6B6B'}
    
    sns.stripplot(data=df, x='phoneme', y='convergence_rate', hue='classification',
                  palette=palette, order=order, ax=ax, size=8, alpha=0.8,
                  edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0, color=ACCENT_COLOR, linestyle='--', linewidth=2, alpha=0.9)
    ax.axhline(y=-50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    ax.axhline(y=50, color='#666', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_title('Convergence Rate by Phoneme', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('Convergence Rate (Hz/s)', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    plt.xticks(rotation=45, ha='right')
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_convergence_by_phoneme.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 2. Distance vs Rate with Regression
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sns.scatterplot(data=df, x='f2_f1_distance_mean', y='convergence_rate', 
                    hue='phoneme', style='classification', s=150, palette='bright',
                    ax=ax, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0, color=ACCENT_COLOR, linestyle='--', linewidth=2, alpha=0.9)
    
    ax.set_title('F2-F1 Distance vs Convergence Rate', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Mean |F2-F1| Distance (Hz)', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('Convergence Rate (Hz/s)', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_distance_vs_rate.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-02_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Formant Convergence/Divergence Analysis: Measure |F2-F1| distance dynamics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files (e.g., /a/ vs /i/)
  python formant_convergence_analysis.py --file1 data/02_cleaned/अ/अ_golden_043.wav --file2 data/02_cleaned/इ/इ_golden_036.wav

  # Batch mode: Compare all files in a folder against a reference file
  python formant_convergence_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python formant_convergence_analysis.py --golden-compare data/02_cleaned

Hypothesis:
  - /a/ (अ) should show CONVERGENT pattern (negative rate: F2→F1)
  - /i/ (इ) should show DIVERGENT pattern (positive rate: F2←F1)
        """
    )
    
    # Single file comparison mode
    parser.add_argument('--file1', type=str, 
                        help='Path to first audio file (for single comparison mode)')
    parser.add_argument('--file2', type=str, 
                        help='Path to second audio file (for single comparison mode)')
    
    # Batch folder comparison mode
    parser.add_argument('--folder', type=str,
                        help='Path to folder containing audio files (for batch mode)')
    parser.add_argument('--reference', type=str,
                        help='Path to reference/pinned file to compare against (for batch mode)')
    
    # Golden files comparison mode
    parser.add_argument('--golden-compare', type=str, dest='golden_compare',
                        help='Path to cleaned data folder to compare all golden files (e.g., data/02_cleaned)')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: results/formant_convergence_analysis/{mode})')
    
    args = parser.parse_args()
    
    # Determine mode
    batch_mode = args.folder is not None and args.reference is not None
    single_mode = args.file1 is not None and args.file2 is not None
    golden_mode = args.golden_compare is not None
    
    if not batch_mode and not single_mode and not golden_mode:
        print("Error: Please specify one of:")
        print("  - Single mode: --file1 and --file2")
        print("  - Batch mode: --folder and --reference")
        print("  - Golden mode: --golden-compare <cleaned_data_dir>")
        parser.print_help()
        return
    
    # Generate default output directory based on mode
    if args.output_dir is None:
        base_dir = "results/formant_convergence_analysis"
        if golden_mode:
            output_dir = f"{base_dir}/golden"
        elif batch_mode:
            folder_name = os.path.basename(os.path.normpath(args.folder))
            output_dir = f"{base_dir}/batch/{folder_name}"
        else:
            name1 = os.path.splitext(os.path.basename(args.file1))[0]
            name2 = os.path.splitext(os.path.basename(args.file2))[0]
            output_dir = f"{base_dir}/compare/{name1}_vs_{name2}"
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("FORMANT CONVERGENCE/DIVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: /a/ = convergent, /i/ = divergent")
    print(f"Metric: d(|F2-F1|)/dt - Rate of formant distance change")
    print("=" * 60)
    
    if batch_mode:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            return
        
        print(f"\nMode: BATCH COMPARISON")
        print(f"Folder: {args.folder}")
        print(f"Reference: {args.reference}")
        
        results_df = batch_compare_folder(args.folder, args.reference, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("BATCH ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Files compared: {len(results_df)}")
            
            # Count classifications
            class_counts = results_df['classification'].value_counts()
            print(f"\nClassification breakdown:")
            for cls, count in class_counts.items():
                print(f"  {cls}: {count}")
    
    elif golden_mode:
        if not os.path.isdir(args.golden_compare):
            print(f"Error: Directory not found: {args.golden_compare}")
            return
        
        print(f"\nMode: GOLDEN FILES COMPARISON")
        print(f"Directory: {args.golden_compare}")
        
        results_df = compare_all_golden_files(args.golden_compare, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("GOLDEN FILES ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Phonemes analyzed: {results_df['phoneme'].nunique()}")
            
            # Show hypothesis check
            print("\n--- HYPOTHESIS CHECK ---")
            phoneme_avg = results_df.groupby('phoneme')['convergence_rate'].mean()
            
            # Check for /a/ variants
            a_variants = ['अ', 'आ']
            for p in a_variants:
                if p in phoneme_avg.index:
                    rate = phoneme_avg[p]
                    status = "✓" if rate < 0 else "✗"
                    print(f"{p}: {rate:+.1f} Hz/s - {'CONVERGENT' if rate < -50 else 'STABLE' if rate < 50 else 'DIVERGENT'} {status}")
            
            # Check for /i/ variants
            i_variants = ['इ', 'ई']
            for p in i_variants:
                if p in phoneme_avg.index:
                    rate = phoneme_avg[p]
                    status = "✓" if rate > 0 else "✗"
                    print(f"{p}: {rate:+.1f} Hz/s - {'CONVERGENT' if rate < -50 else 'STABLE' if rate < 50 else 'DIVERGENT'} {status}")
    
    else:
        if not os.path.exists(args.file1):
            print(f"Error: File not found: {args.file1}")
            return
        if not os.path.exists(args.file2):
            print(f"Error: File not found: {args.file2}")
            return
        
        results_df = compare_two_files(args.file1, args.file2, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
