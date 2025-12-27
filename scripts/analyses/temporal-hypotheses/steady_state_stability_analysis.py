#!/usr/bin/env python3
"""
Temporal Hypothesis Analysis: Steady-State Stability

This script analyzes formant stability (variance) across the entire utterance (0-100%)
to identify true vowel regions without introducing arbitrary bias.

Metrics computed:
- Variance (σ²) of F1, F2, F3 across multiple windows
- Standard deviation (σ) for each formant
- Coefficient of Variation (CV = σ/μ) for each formant

Hypothesis: True vowel sounds should exhibit low variance (low CV) in stable regions.
The script identifies which region has the lowest variance automatically.
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

# Add parent directory to path to import common package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from common package
from common import (
    configure_matplotlib,
    compute_joint_weights,
    apply_dark_theme,
    create_styled_figure,
    style_legend,
    COLORS,
    tqdm,
    HAS_SEABORN,
    HAS_TQDM
)

# Configure matplotlib for Devanagari support
configure_matplotlib()

# Define analysis windows (start%, end%)
ANALYSIS_WINDOWS = [
    (0, 100, 'full'),        # Full utterance baseline
    (0, 20, '0-20%'),
    (20, 40, '20-40%'),
    (40, 60, '40-60%'),
    (60, 80, '60-80%'),
    (80, 100, '80-100%'),
    # Overlapping windows
    (10, 30, '10-30%'),
    (30, 50, '30-50%'),
    (50, 70, '50-70%'),
    (70, 90, '70-90%'),
    # Special regions
    (30, 70, 'middle-40%'),
    (20, 80, 'middle-60%'),
    (10, 90, 'middle-80%'),
]


def extract_formants_full(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                          max_formant_freq: float = 5500.0, window_length: float = 0.025,
                          intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant frequencies (F1, F2, F3) and compute frame weights.
    
    Refined Method (Method 3):
    - Computes Joint Stability-Intensity Weights for the entire file
    - These weights are returned for use in windowed analysis
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length in seconds
        intensity_threshold: Minimum intensity (dB) (Legacy parameter, replaced by soft gate)
    
    Returns:
        Dictionary containing full formant time-series data and weights
    """
    try:
        # Load the audio file with Praat
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        # Create a Formant object
        formant = call(sound, "To Formant (burg)",
                       time_step,
                       max_formants,
                       max_formant_freq,
                       window_length,
                       50.0)
        
        # Create Intensity object
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        
        # Get the number of frames
        n_frames = call(formant, "Get number of frames")
        
        f1_values = []
        f2_values = []
        f3_values = []
        time_values = []
        intensity_values = []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            # Get intensity
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens): intens = 0.0
            except:
                intens = 0.0
            
            # Only include valid values
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        if len(f1_values) < 3:
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        time_arr = np.array(time_values)
        intensity_arr = np.array(intensity_values)
        
        # --- Compute Joint Stability-Intensity Weights ---
        
        # 1. Gradient (dF/dt) using np.gradient for proper edge handling
        df1 = np.abs(np.gradient(f1_arr, time_arr))
        df2 = np.abs(np.gradient(f2_arr, time_arr))
        df3 = np.abs(np.gradient(f3_arr, time_arr))
        
        # Normalized Instability: |dF/dt| / F (dimensionless)
        instability = (df1 / f1_arr) + (df2 / f2_arr) + (df3 / f3_arr)
        
        # 2. Weights - Joint Stability-Intensity
        noise_floor = 50.0
        stability_smoothing = 0.1
        soft_gate_threshold = 30.0
        intensity_clip_db = 30.0  # Clip to prevent burst dominance
        intensity_exponent = 2.0
        
        intensity_above_floor = np.clip(intensity_arr - noise_floor, 0, intensity_clip_db)
        w_intensity = intensity_above_floor ** intensity_exponent
        w_stability = 1.0 / (instability + stability_smoothing)
        gate_mask = intensity_arr >= soft_gate_threshold
        
        weights = w_intensity * w_stability * gate_mask
        
        # Fallback
        if np.sum(weights) == 0:
            weights = np.ones_like(time_arr)
        
        # Normalize weights
        weights_norm = weights / np.sum(weights)
        
        # Weight entropy (0=concentrated, 1=uniform)
        p = weights_norm
        weight_entropy = -np.sum(p * np.log(p + 1e-12))
        weight_entropy_norm = weight_entropy / np.log(len(p)) if len(p) > 1 else 0
        
        return {
            'f1_values': f1_arr,
            'f2_values': f2_arr,
            'f3_values': f3_arr,
            'time_values': time_arr,
            'intensity_values': intensity_arr,
            'frame_weights': weights,
            'frame_weights_norm': weights_norm,
            'weight_entropy': weight_entropy_norm,
            'n_frames': len(f1_arr),
            'duration': duration,
            'intensity_threshold': intensity_threshold
        }
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def compute_window_stability(formant_data: dict, start_pct: float, end_pct: float) -> dict:
    """
    Compute weighted stability metrics for a specific time window.
    
    Refined Method (Method 3):
    - Uses pre-computed joint weights for all stats (mean, std, var)
    - Returns N_eff and Confidence
    
    Args:
        formant_data: Dictionary containing full formant time-series and weights
        start_pct: Start percentage of utterance (0-100)
        end_pct: End percentage of utterance (0-100)
    
    Returns:
        Dictionary containing weighted stability metrics
    """
    if formant_data is None:
        return None
    
    duration = formant_data['duration']
    time_arr = formant_data['time_values']
    f1_arr = formant_data['f1_values']
    f2_arr = formant_data['f2_values']
    f3_arr = formant_data['f3_values']
    weights_full = formant_data['frame_weights']
    
    # Calculate time bounds
    start_time = duration * (start_pct / 100.0)
    end_time = duration * (end_pct / 100.0)
    
    # Select frames within the window
    # Note: We rely on weights for filtering, but time selection is still hard cutoff
    mask = (time_arr >= start_time) & (time_arr <= end_time)
    
    if np.sum(mask) < 2:
        return None
        
    f1_window = f1_arr[mask]
    f2_window = f2_arr[mask]
    f3_window = f3_arr[mask]
    weights_window = weights_full[mask]
    
    # Check if we have any valid weights in this window
    if np.sum(weights_window) == 0:
        # If all weights zero (e.g. silence), fall back to uniform for this window?
        # Or just return None/Undefined?
        # Better to return None as this window has no valid speech info
        return None
        
    # Normalize weights for this window
    weights_norm = weights_window / np.sum(weights_window)
    
    # Diagnostics
    sum_w = np.sum(weights_window)
    sum_w_sq = np.sum(weights_window**2)
    n_eff = (sum_w**2) / sum_w_sq if sum_w_sq > 0 else 0
    confidence = np.clip(n_eff / len(weights_window), 0, 1) if len(weights_window) > 0 else 0
    
    # Compute Weighted Mean
    f1_mean = np.average(f1_window, weights=weights_norm)
    f2_mean = np.average(f2_window, weights=weights_norm)
    f3_mean = np.average(f3_window, weights=weights_norm)
    
    # Compute Weighted Variance
    # Var = sum(w * (x - mean)^2) / sum(w)
    # Since weights_norm sum to 1, term is just sum(w * (x-mean)^2)
    f1_var = np.average((f1_window - f1_mean)**2, weights=weights_norm)
    f2_var = np.average((f2_window - f2_mean)**2, weights=weights_norm)
    f3_var = np.average((f3_window - f3_mean)**2, weights=weights_norm)
    
    # Weighted Standard Deviation
    f1_std = np.sqrt(f1_var)
    f2_std = np.sqrt(f2_var)
    f3_std = np.sqrt(f3_var)
    
    # Coefficient of Variation (CV = σ/μ)
    f1_cv = (f1_std / f1_mean) * 100 if f1_mean > 0 else np.nan
    f2_cv = (f2_std / f2_mean) * 100 if f2_mean > 0 else np.nan
    f3_cv = (f3_std / f3_mean) * 100 if f3_mean > 0 else np.nan
    
    # Combined stability score (lower = more stable)
    combined_cv = (f1_cv + f2_cv + f3_cv) / 3
    
    return {
        'start_pct': start_pct,
        'end_pct': end_pct,
        'n_frames': len(f1_window),
        'n_eff': n_eff,
        'confidence': confidence,
        
        # Mean values (Weighted)
        'f1_mean': f1_mean,
        'f2_mean': f2_mean,
        'f3_mean': f3_mean,
        
        # Standard deviation (Weighted)
        'f1_std': f1_std,
        'f2_std': f2_std,
        'f3_std': f3_std,
        
        # Variance (Weighted)
        'f1_var': f1_var,
        'f2_var': f2_var,
        'f3_var': f3_var,
        
        # Coefficient of Variation (%)
        'f1_cv': f1_cv,
        'f2_cv': f2_cv,
        'f3_cv': f3_cv,
        
        # Combined stability score
        'combined_cv': combined_cv,
    }


def analyze_audio_file(audio_path: str) -> dict:
    """
    Complete stability analysis for a single audio file across all windows.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary with full stability analysis results
    """
    # Extract formants with higher max frequency for robustness
    max_freq = 8000.0
    formant_data = extract_formants_full(audio_path, max_formant_freq=max_freq)
    
    if formant_data is None:
        return None
    
    # Analyze all windows
    windows_results = []
    for start_pct, end_pct, window_name in ANALYSIS_WINDOWS:
        window_result = compute_window_stability(formant_data, start_pct, end_pct)
        if window_result is not None:
            window_result['window_name'] = window_name
            windows_results.append(window_result)
    
    if not windows_results:
        return None
    
    # Find the most stable window (lowest combined CV, excluding 'full')
    non_full_windows = [w for w in windows_results if w['window_name'] != 'full']
    if non_full_windows:
        most_stable = min(non_full_windows, key=lambda x: x['combined_cv'])
    else:
        most_stable = windows_results[0]
    
    # Get full utterance results for reference
    full_result = next((w for w in windows_results if w['window_name'] == 'full'), None)
    
    return {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        'duration': formant_data['duration'],
        'total_frames': formant_data['n_frames'],
        'windows': windows_results,
        'most_stable_window': most_stable['window_name'],
        'most_stable_cv': most_stable['combined_cv'],
        'full_utterance': full_result,
        # Raw data for visualization
        'f1_values': formant_data['f1_values'],
        'f2_values': formant_data['f2_values'],
        'f3_values': formant_data['f3_values'],
        'time_values': formant_data['time_values'],
    }


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """
    Compare stability metrics between two audio files.
    
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
    
    # Create comparison DataFrame for all windows
    comparison_data = []
    
    for w1 in result1['windows']:
        w2 = next((w for w in result2['windows'] if w['window_name'] == w1['window_name']), None)
        if w2 is None:
            continue
        
        comparison_data.append({
            'window': w1['window_name'],
            f'{os.path.basename(file1_path)}_f1_cv': w1['f1_cv'],
            f'{os.path.basename(file2_path)}_f1_cv': w2['f1_cv'],
            f'{os.path.basename(file1_path)}_f2_cv': w1['f2_cv'],
            f'{os.path.basename(file2_path)}_f2_cv': w2['f2_cv'],
            f'{os.path.basename(file1_path)}_f3_cv': w1['f3_cv'],
            f'{os.path.basename(file2_path)}_f3_cv': w2['f3_cv'],
            f'{os.path.basename(file1_path)}_combined_cv': w1['combined_cv'],
            f'{os.path.basename(file2_path)}_combined_cv': w2['combined_cv'],
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'stability_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Print N_eff for verification (using full utterance)
    if result1['full_utterance'] and 'n_eff' in result1['full_utterance']:
        print(f"File 1 Full Utterance Effective Frames (N_eff): {result1['full_utterance']['n_eff']:.2f}")
    if result2['full_utterance'] and 'n_eff' in result2['full_utterance']:
        print(f"File 2 Full Utterance Effective Frames (N_eff): {result2['full_utterance']['n_eff']:.2f}")
    
    # Create visualization
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing stability between two files."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    
    # Colors
    color1 = '#FF6B6B'  # Coral red
    color2 = '#4ECDC4'  # Teal
    
    # 1. Formant Trajectories (File 1)
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    time_pct1 = (result1['time_values'] / result1['duration']) * 100
    ax.plot(time_pct1, result1['f1_values'], color='#FF6B6B', alpha=0.8, label='F1', linewidth=1.5)
    ax.plot(time_pct1, result1['f2_values'], color='#4ECDC4', alpha=0.8, label='F2', linewidth=1.5)
    ax.plot(time_pct1, result1['f3_values'], color='#FFD93D', alpha=0.8, label='F3', linewidth=1.5)
    
    # Highlight most stable region
    stable_window = next((w for w in ANALYSIS_WINDOWS if w[2] == result1['most_stable_window']), None)
    if stable_window:
        ax.axvspan(stable_window[0], stable_window[1], alpha=0.2, color='#4ECDC4', label=f'Most Stable: {result1["most_stable_window"]}')
    
    ax.set_xlabel('Utterance %', color='#EAEAEA')
    ax.set_ylabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_title(f'Formant Trajectory: {file1_name}', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA', fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Formant Trajectories (File 2)
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    time_pct2 = (result2['time_values'] / result2['duration']) * 100
    ax.plot(time_pct2, result2['f1_values'], color='#FF6B6B', alpha=0.8, label='F1', linewidth=1.5)
    ax.plot(time_pct2, result2['f2_values'], color='#4ECDC4', alpha=0.8, label='F2', linewidth=1.5)
    ax.plot(time_pct2, result2['f3_values'], color='#FFD93D', alpha=0.8, label='F3', linewidth=1.5)
    
    stable_window = next((w for w in ANALYSIS_WINDOWS if w[2] == result2['most_stable_window']), None)
    if stable_window:
        ax.axvspan(stable_window[0], stable_window[1], alpha=0.2, color='#4ECDC4', label=f'Most Stable: {result2["most_stable_window"]}')
    
    ax.set_xlabel('Utterance %', color='#EAEAEA')
    ax.set_ylabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_title(f'Formant Trajectory: {file2_name}', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA', fontsize=8)
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. CV Comparison by Window
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    windows = [w['window_name'] for w in result1['windows']]
    cv1 = [w['combined_cv'] for w in result1['windows']]
    cv2 = [next((w2['combined_cv'] for w2 in result2['windows'] if w2['window_name'] == w['window_name']), np.nan) for w in result1['windows']]
    
    x = np.arange(len(windows))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cv1, width, label=file1_name, color=color1, alpha=0.8)
    bars2 = ax.bar(x + width/2, cv2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Window', color='#EAEAEA')
    ax.set_ylabel('Combined CV (%)', color='#EAEAEA')
    ax.set_title('Stability Comparison (Lower = More Stable)', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(windows, rotation=45, ha='right', fontsize=8)
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444', axis='y')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    full1 = result1['full_utterance']
    full2 = result2['full_utterance']
    
    summary_text = f"""
    STEADY-STATE STABILITY ANALYSIS
    ================================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    MOST STABLE REGIONS:
    ─────────────────────────────────
    File 1: {result1['most_stable_window']} (CV={result1['most_stable_cv']:.2f}%)
    File 2: {result2['most_stable_window']} (CV={result2['most_stable_cv']:.2f}%)
    
    FULL UTTERANCE STABILITY:
    ─────────────────────────────────
    File 1: F1 CV={full1['f1_cv']:.2f}%, F2 CV={full1['f2_cv']:.2f}%, F3 CV={full1['f3_cv']:.2f}%
    File 2: F1 CV={full2['f1_cv']:.2f}%, F2 CV={full2['f2_cv']:.2f}%, F3 CV={full2['f3_cv']:.2f}%
    
    INTERPRETATION:
    ─────────────────────────────────
    Lower CV = More stable formant
    True vowels should show low CV in stable regions
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'stability_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """
    Compare all audio files in a folder against a reference file.
    
    Args:
        folder_path: Path to folder containing audio files
        reference_file: Path to the reference file
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all comparison results
    """
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all wav files
    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
    
    if not wav_files:
        print(f"Error: No .wav files found in {folder_path}")
        return None
    
    # Analyze reference file
    print(f"\n{'='*60}")
    print(f"REFERENCE FILE: {os.path.basename(reference_file)}")
    print(f"{'='*60}")
    
    ref_result = analyze_audio_file(reference_file)
    if ref_result is None:
        print(f"Error: Could not analyze reference file: {reference_file}")
        return None
    
    # Analyze all files
    all_results = []
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        if result is None:
            continue
        
        all_results.append(result)
    
    if not all_results:
        print("Error: No files could be analyzed")
        return None
    
    # Create comparison DataFrame
    comparison_data = []
    
    for result in all_results:
        row = {
            'filename': result['filename'],
            'most_stable_window': result['most_stable_window'],
            'most_stable_cv': result['most_stable_cv'],
        }
        
        # Add full utterance metrics
        if result['full_utterance']:
            full = result['full_utterance']
            row['f1_cv_full'] = full['f1_cv']
            row['f2_cv_full'] = full['f2_cv']
            row['f3_cv_full'] = full['f3_cv']
            row['combined_cv_full'] = full['combined_cv']
        
        # Compare to reference
        if ref_result['full_utterance']:
            ref_full = ref_result['full_utterance']
            row['ref_combined_cv'] = ref_full['combined_cv']
            row['cv_diff_from_ref'] = row.get('combined_cv_full', np.nan) - ref_full['combined_cv']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save results
    csv_path = os.path.join(output_dir, 'batch_stability_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_batch_plots(ref_result, all_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    
    # Colors
    ref_color = '#FFD93D'  # Gold for reference
    other_color = '#4ECDC4'  # Teal for others
    
    # 1. Combined CV Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_cv = ref_result['full_utterance']['combined_cv'] if ref_result['full_utterance'] else 0
    other_cvs = comparison_df['combined_cv_full'].dropna().values
    
    ax.axhline(y=ref_cv, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_cvs)), sorted(other_cvs), c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index (sorted by CV)', color='#EAEAEA')
    ax.set_ylabel('Combined CV (%)', color='#EAEAEA')
    ax.set_title('Stability Distribution vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Most Stable Window Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    window_counts = comparison_df['most_stable_window'].value_counts()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(window_counts)))
    
    bars = ax.bar(range(len(window_counts)), window_counts.values, color=colors, alpha=0.8)
    ax.set_xticks(range(len(window_counts)))
    ax.set_xticklabels(window_counts.index, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('Window', color='#EAEAEA')
    ax.set_ylabel('Count', color='#EAEAEA')
    ax.set_title('Distribution of Most Stable Windows', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. CV by Formant (Boxplot)
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    box_data = [
        comparison_df['f1_cv_full'].dropna().values,
        comparison_df['f2_cv_full'].dropna().values,
        comparison_df['f3_cv_full'].dropna().values,
    ]
    
    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=['F1', 'F2', 'F3'])
    colors_box = ['#FF6B6B', '#4ECDC4', '#FFD93D']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set_color('#EAEAEA')
    
    ax.set_xlabel('Formant', color='#EAEAEA')
    ax.set_ylabel('CV (%)', color='#EAEAEA')
    ax.set_title('CV Distribution by Formant', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    avg_cv = comparison_df['combined_cv_full'].mean()
    std_cv = comparison_df['combined_cv_full'].std()
    
    summary_text = f"""
    BATCH STABILITY ANALYSIS SUMMARY
    =================================
    
    Reference: {ref_name}
    Files analyzed: {n_files}
    
    REFERENCE STABILITY:
    ─────────────────────────────────
    Combined CV: {ref_cv:.2f}%
    Most Stable Window: {ref_result['most_stable_window']}
    
    OTHER FILES STATISTICS:
    ─────────────────────────────────
    Average CV: {avg_cv:.2f}% ± {std_cv:.2f}%
    Min CV: {comparison_df['combined_cv_full'].min():.2f}%
    Max CV: {comparison_df['combined_cv_full'].max():.2f}%
    
    INTERPRETATION:
    ─────────────────────────────────
    Lower CV = More stable formant
    Consistent most_stable_window across files
    suggests predictable vowel structure
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_stability_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Find and compare stability metrics for all golden files across phonemes.
    
    Args:
        cleaned_data_dir: Path to the cleaned data directory
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all golden file stability data
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
        phoneme = os.path.basename(os.path.dirname(golden_file))
        
        result = analyze_audio_file(golden_file)
        if result is None:
            continue
        
        # Flatten the result for DataFrame
        row = {
            'phoneme': phoneme,
            'filename': result['filename'],
            'file_path': result['file_path'],
            'duration': result['duration'],
            'most_stable_window': result['most_stable_window'],
            'most_stable_cv': result['most_stable_cv'],
        }
        
        # Add full utterance metrics
        if result['full_utterance']:
            full = result['full_utterance']
            row['f1_mean'] = full['f1_mean']
            row['f2_mean'] = full['f2_mean']
            row['f3_mean'] = full['f3_mean']
            row['f1_cv'] = full['f1_cv']
            row['f2_cv'] = full['f2_cv']
            row['f3_cv'] = full['f3_cv']
            row['combined_cv'] = full['combined_cv']
        
        # Add window-specific data
        for window in result['windows']:
            prefix = window['window_name'].replace('-', '_').replace('%', '')
            row[f'{prefix}_cv'] = window['combined_cv']
        
        all_results.append(row)
    
    if not all_results:
        print("Error: No golden files could be analyzed")
        return None
    
    df = pd.DataFrame(all_results)
    
    # Save results
    csv_path = os.path.join(output_dir, 'golden_stability_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_golden_plots(df, output_dir)
    
    return df


def create_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization for golden files stability analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('#111111')
    
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # 1. Combined CV by Phoneme
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    sorted_df = df.sort_values('combined_cv')
    bars = ax.barh(range(len(sorted_df)), sorted_df['combined_cv'].values, 
                   color=plt.cm.rainbow(np.linspace(0, 1, len(sorted_df))), alpha=0.8)
    ax.set_yticks(range(len(sorted_df)))
    ax.set_yticklabels(sorted_df['phoneme'].values, fontsize=10)
    
    ax.set_xlabel('Combined CV (%)', color='#EAEAEA', fontsize=11)
    ax.set_title('Stability by Phoneme (Lower = More Stable)', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444', axis='x')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. CV by Formant (Scatter)
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['f1_cv'], row['f2_cv'], c=[colors[i]], s=150, alpha=0.8,
                   edgecolors='white', linewidths=1)
        ax.annotate(row['phoneme'], (row['f1_cv'], row['f2_cv']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color='#EAEAEA')
    
    ax.set_xlabel('F1 CV (%)', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('F2 CV (%)', color='#EAEAEA', fontsize=11)
    ax.set_title('F1 vs F2 Stability', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. Most Stable Window Distribution
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    window_counts = df['most_stable_window'].value_counts()
    window_colors = plt.cm.rainbow(np.linspace(0, 1, len(window_counts)))
    
    bars = ax.bar(range(len(window_counts)), window_counts.values, color=window_colors, alpha=0.8)
    ax.set_xticks(range(len(window_counts)))
    ax.set_xticklabels(window_counts.index, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel('Window', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('Count', color='#EAEAEA', fontsize=11)
    ax.set_title('Most Stable Window Distribution Across Phonemes', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    # Top 5 most stable
    top5_stable = df.nsmallest(5, 'combined_cv')[['phoneme', 'combined_cv', 'most_stable_window']]
    top5_lines = [f"{row['phoneme']}: CV={row['combined_cv']:.2f}% ({row['most_stable_window']})" 
                  for _, row in top5_stable.iterrows()]
    
    # Top 5 least stable
    top5_unstable = df.nlargest(5, 'combined_cv')[['phoneme', 'combined_cv', 'most_stable_window']]
    unstable_lines = [f"{row['phoneme']}: CV={row['combined_cv']:.2f}% ({row['most_stable_window']})" 
                      for _, row in top5_unstable.iterrows()]
    
    summary_text = f"""
    GOLDEN FILES STABILITY SUMMARY
    ==============================
    
    Total phonemes: {len(df)}
    
    MOST STABLE PHONEMES:
    ─────────────────────────────────
{chr(10).join(f'    {line}' for line in top5_lines)}
    
    LEAST STABLE PHONEMES:
    ─────────────────────────────────
{chr(10).join(f'    {line}' for line in unstable_lines)}
    
    STATISTICS:
    ─────────────────────────────────
    Avg Combined CV: {df['combined_cv'].mean():.2f}%
    Std Combined CV: {df['combined_cv'].std():.2f}%
    Most common stable window: {df['most_stable_window'].mode().values[0] if len(df['most_stable_window'].mode()) > 0 else 'N/A'}
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_stability_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    # Create seaborn heatmap if available
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn-based visualizations for golden files."""
    
    BG_COLOR = '#111111'
    TEXT_COLOR = '#eaeaea'
    BORDER_COLOR = '#333333'
    
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['figure.facecolor'] = BG_COLOR
    plt.rcParams['axes.facecolor'] = BG_COLOR
    
    # 1. Stability Heatmap by Window
    window_cols = [col for col in df.columns if col.endswith('_cv') and col not in ['f1_cv', 'f2_cv', 'f3_cv', 'combined_cv', 'most_stable_cv']]
    
    if window_cols:
        fig, ax = plt.subplots(figsize=(16, max(8, len(df) * 0.4)))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        
        heatmap_data = df.set_index('phoneme')[window_cols]
        # Clean column names for display
        heatmap_data.columns = [col.replace('_cv', '').replace('_', '-') for col in heatmap_data.columns]
        
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn_r', 
                    ax=ax, cbar_kws={'label': 'CV (%)'}, annot_kws={'size': 8})
        
        ax.set_title('Stability Heatmap: CV by Phoneme and Window', fontsize=14, color=TEXT_COLOR, fontweight='bold')
        ax.set_xlabel('Window', fontsize=11, color=TEXT_COLOR)
        ax.set_ylabel('Phoneme', fontsize=11, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/01_stability_heatmap.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close()
    
    # 2. F1/F2/F3 CV Comparison
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    melt_df = df.melt(id_vars=['phoneme'], value_vars=['f1_cv', 'f2_cv', 'f3_cv'],
                      var_name='Formant', value_name='CV')
    melt_df['Formant'] = melt_df['Formant'].map({'f1_cv': 'F1', 'f2_cv': 'F2', 'f3_cv': 'F3'})
    
    sns.barplot(data=melt_df, x='phoneme', y='CV', hue='Formant', 
                palette=['#FF6B6B', '#4ECDC4', '#FFD93D'], ax=ax)
    
    ax.set_title('CV by Formant Across Phonemes', fontsize=14, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=11, color=TEXT_COLOR)
    ax.set_ylabel('CV (%)', fontsize=11, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    plt.xticks(rotation=45, ha='right')
    ax.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_formant_cv_comparison.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-02_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Hypothesis Analysis: Steady-State Stability',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python steady_state_stability_analysis.py --file1 male.wav --file2 female.wav

  # Batch mode: Compare all files in a folder against a reference file
  python steady_state_stability_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python steady_state_stability_analysis.py --golden-compare data/02_cleaned
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
                        help='Path to cleaned data folder to compare all golden files')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
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
    
    # Generate default output directory
    if args.output_dir is None:
        base_dir = "results/steady_state_stability"
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
    print("STEADY-STATE STABILITY ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: True vowels show low variance (low CV)")
    print(f"in stable regions of the utterance")
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
            print(f"Phonemes analyzed: {len(results_df)}")
            print(f"\nMost stable phoneme: {results_df.loc[results_df['combined_cv'].idxmin(), 'phoneme']}")
            print(f"Least stable phoneme: {results_df.loc[results_df['combined_cv'].idxmax(), 'phoneme']}")
    
    else:
        if not os.path.exists(args.file1):
            print(f"Error: File not found: {args.file1}")
            return
        if not os.path.exists(args.file2):
            print(f"Error: File not found: {args.file2}")
            return
        
        print(f"\nMode: SINGLE COMPARISON")
        
        results_df = compare_two_files(args.file1, args.file2, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("COMPARISON COMPLETE")
            print("=" * 60)
            print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
