#!/usr/bin/env python3
"""
Formant-Based Invariant Analysis: Frequency Ratios (Scale-Invariant)

This script analyzes formant frequencies (F1, F2, F3) from audio files and computes
scale-invariant ratios that are hypothesized to collapse male/female voice differences.

Ratios computed:
- F1/F2, F2/F3, F1/F3 (linear ratios)
- log(F2/F1), log(F3/F2) (log ratios)

Hypothesis: These ratios should produce similar values across different speakers
(male/female) for the same phoneme, revealing vowel-intrinsic properties.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import common package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from common package
from common import (
    configure_matplotlib,
    extract_formants_with_weights,
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

# Import visualizer for integration
try:
    from formant_visualizer import generate_all_figures
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def extract_formants(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                      max_formant_freq: float = 5500.0, window_length: float = 0.025,
                      stability_smoothing: float = 0.1, intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant frequencies (F1, F2, F3) from an audio file using Praat algorithms.
    
    This is a thin wrapper around the canonical implementation in common.formant_extraction.
    
    Refined Method (Method 3):
    - Uses Joint Stability-Intensity Weighting (Intensity^2 / Instability)
    - Computes weighted means and standard deviations
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz). Use ~5000 for male, ~5500 for female
        window_length: Analysis window length in seconds
        stability_smoothing: Smoothing constant for stability weight calculation
        intensity_threshold: Minimum intensity (dB) (Legacy, not used)
    
    Returns:
        Dictionary containing formant statistics with stability-weighted means
    """
    return extract_formants_with_weights(
        audio_path=audio_path,
        time_step=time_step,
        max_formants=max_formants,
        max_formant_freq=max_formant_freq,
        window_length=window_length,
        stability_smoothing=stability_smoothing,
        return_raw_arrays=True
    )


def compute_formant_ratios(formant_data: dict) -> dict:
    """
    Compute scale-invariant formant ratios from formant data.
    
    Args:
        formant_data: Dictionary containing formant statistics from extract_formants()
    
    Returns:
        Dictionary containing:
        - Linear ratios: F1/F2, F2/F3, F1/F3
        - Log ratios: log(F2/F1), log(F3/F2)
        - Both mean-based and median-based calculations
    """
    if formant_data is None:
        return None
    
    # Use mean values for ratio computation
    f1_mean = formant_data['f1_mean']
    f2_mean = formant_data['f2_mean']
    f3_mean = formant_data['f3_mean']
    
    # Use median values (more robust to outliers) - Note: these are unweighted
    f1_med = formant_data['f1_median_unweighted']
    f2_med = formant_data['f2_median_unweighted']
    f3_med = formant_data['f3_median_unweighted']
    
    # Linear ratios (using means)
    f1_f2_ratio_mean = f1_mean / f2_mean
    f2_f3_ratio_mean = f2_mean / f3_mean
    f1_f3_ratio_mean = f1_mean / f3_mean
    
    # Linear ratios (using medians)
    f1_f2_ratio_med = f1_med / f2_med
    f2_f3_ratio_med = f2_med / f3_med
    f1_f3_ratio_med = f1_med / f3_med
    
    # Log ratios (using means)
    log_f2_f1_mean = np.log(f2_mean / f1_mean)
    log_f3_f2_mean = np.log(f3_mean / f2_mean)
    
    # Log ratios (using medians)
    log_f2_f1_med = np.log(f2_med / f1_med)
    log_f3_f2_med = np.log(f3_med / f2_med)
    
    # Also compute per-frame ratios for analyzing distribution
    f1_vals = formant_data['f1_values']
    f2_vals = formant_data['f2_values']
    f3_vals = formant_data['f3_values']
    
    frame_f1_f2_ratios = f1_vals / f2_vals
    frame_f2_f3_ratios = f2_vals / f3_vals
    frame_log_f2_f1 = np.log(f2_vals / f1_vals)
    frame_log_f3_f2 = np.log(f3_vals / f2_vals)
    
    return {
        # Mean-based ratios
        'f1_f2_ratio_mean': f1_f2_ratio_mean,
        'f2_f3_ratio_mean': f2_f3_ratio_mean,
        'f1_f3_ratio_mean': f1_f3_ratio_mean,
        'log_f2_f1_mean': log_f2_f1_mean,
        'log_f3_f2_mean': log_f3_f2_mean,
        
        # Median-based ratios
        'f1_f2_ratio_median': f1_f2_ratio_med,
        'f2_f3_ratio_median': f2_f3_ratio_med,
        'f1_f3_ratio_median': f1_f3_ratio_med,
        'log_f2_f1_median': log_f2_f1_med,
        'log_f3_f2_median': log_f3_f2_med,
        
        # F3-F2 difference (useful for retroflexion detection)
        'f3_f2_diff_mean': f3_mean - f2_mean,
        'f3_f2_diff_median': f3_med - f2_med,
        
        # Per-frame ratio statistics
        'frame_f1_f2_ratios': frame_f1_f2_ratios,
        'frame_f2_f3_ratios': frame_f2_f3_ratios,
        'frame_log_f2_f1': frame_log_f2_f1,
        'frame_log_f3_f2': frame_log_f3_f2,
        'frame_f1_f2_std': np.std(frame_f1_f2_ratios),
        'frame_f2_f3_std': np.std(frame_f2_f3_ratios),
    }


def analyze_audio_file(audio_path: str) -> dict:
    """
    Complete formant ratio analysis for a single audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary with all formant data and ratios
    """
    # max_formant_freq is required by Praat's Burg algorithm
    # Using 8000 Hz (well above typical F3) to avoid any artificial limits
    max_freq = 8000.0
    
    formant_data = extract_formants(audio_path, max_formant_freq=max_freq)
    
    if formant_data is None:
        return None
    
    ratios = compute_formant_ratios(formant_data)
    
    if ratios is None:
        return None
    
    # Combine all data
    result = {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        **{k: v for k, v in formant_data.items() if not k.endswith('_values') and k != 'time_values'},
        **{k: v for k, v in ratios.items() if not k.startswith('frame_')},
    }
    
    # Add frame-level statistics
    result['frame_f1_f2_std'] = ratios['frame_f1_f2_std']
    result['frame_f2_f3_std'] = ratios['frame_f2_f3_std']
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """
    Compare formant ratios between two audio files.
    
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
        'f1_mean', 'f2_mean', 'f3_mean',
        'f1_f2_ratio_mean', 'f2_f3_ratio_mean', 'f1_f3_ratio_mean',
        'log_f2_f1_mean', 'log_f3_f2_mean',
        'f1_f2_ratio_median', 'f2_f3_ratio_median', 'f1_f3_ratio_median',
        'log_f2_f1_median', 'log_f3_f2_median',
    ]
    
    comparison_data = []
    for metric in metrics:
        val1 = result1.get(metric, np.nan)
        val2 = result2.get(metric, np.nan)
        diff = abs(val1 - val2)
        pct_diff = (diff / ((val1 + val2) / 2)) * 100 if (val1 + val2) != 0 else np.nan
        
        comparison_data.append({
            'metric': metric,
            f'{os.path.basename(file1_path)}': val1,
            f'{os.path.basename(file2_path)}': val2,
            'absolute_difference': diff,
            'percent_difference': pct_diff,
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'formant_ratio_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing formant ratios between two files."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    
    # Colors
    color1 = '#FF6B6B'  # Coral red
    color2 = '#4ECDC4'  # Teal
    
    # 1. Raw Formants Comparison (Bar chart)
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(3)
    width = 0.35
    
    formants1 = [result1['f1_mean'], result1['f2_mean'], result1['f3_mean']]
    formants2 = [result2['f1_mean'], result2['f2_mean'], result2['f3_mean']]
    
    bars1 = ax.bar(x - width/2, formants1, width, label=file1_name, color=color1, alpha=0.8)
    bars2 = ax.bar(x + width/2, formants2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Formant', color='#EAEAEA')
    ax.set_ylabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_title('Raw Formant Frequencies', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['F1', 'F2', 'F3'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Linear Ratios Comparison
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ratios1 = [result1['f1_f2_ratio_mean'], result1['f2_f3_ratio_mean'], result1['f1_f3_ratio_mean']]
    ratios2 = [result2['f1_f2_ratio_mean'], result2['f2_f3_ratio_mean'], result2['f1_f3_ratio_mean']]
    
    bars1 = ax.bar(x - width/2, ratios1, width, label=file1_name, color=color1, alpha=0.8)
    bars2 = ax.bar(x + width/2, ratios2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Ratio Type', color='#EAEAEA')
    ax.set_ylabel('Ratio Value', color='#EAEAEA')
    ax.set_title('Linear Formant Ratios (Scale-Invariant)', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['F1/F2', 'F2/F3', 'F1/F3'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Log Ratios Comparison
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    x2 = np.arange(2)
    
    log_ratios1 = [result1['log_f2_f1_mean'], result1['log_f3_f2_mean']]
    log_ratios2 = [result2['log_f2_f1_mean'], result2['log_f3_f2_mean']]
    
    bars1 = ax.bar(x2 - width/2, log_ratios1, width, label=file1_name, color=color1, alpha=0.8)
    bars2 = ax.bar(x2 + width/2, log_ratios2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Log Ratio Type', color='#EAEAEA')
    ax.set_ylabel('Log Ratio Value', color='#EAEAEA')
    ax.set_title('Log Formant Ratios', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(['log(F2/F1)', 'log(F3/F2)'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. F1-F2 Vowel Space Plot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    ax.scatter([result1['f2_mean']], [result1['f1_mean']], s=200, c=color1, 
               label=file1_name, marker='o', edgecolors='white', linewidths=2)
    ax.scatter([result2['f2_mean']], [result2['f1_mean']], s=200, c=color2, 
               label=file2_name, marker='s', edgecolors='white', linewidths=2)
    
    # Invert axes for traditional vowel space representation
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    ax.set_xlabel('F2 (Hz)', color='#EAEAEA')
    ax.set_ylabel('F1 (Hz)', color='#EAEAEA')
    ax.set_title('F1-F2 Vowel Space', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')
    
    # 5. Ratio Difference Visualization
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    
    ratio_names = ['F1/F2', 'F2/F3', 'F1/F3', 'log(F2/F1)', 'log(F3/F2)']
    differences = [
        abs(result1['f1_f2_ratio_mean'] - result2['f1_f2_ratio_mean']),
        abs(result1['f2_f3_ratio_mean'] - result2['f2_f3_ratio_mean']),
        abs(result1['f1_f3_ratio_mean'] - result2['f1_f3_ratio_mean']),
        abs(result1['log_f2_f1_mean'] - result2['log_f2_f1_mean']),
        abs(result1['log_f3_f2_mean'] - result2['log_f3_f2_mean']),
    ]
    
    colors = ['#FF6B6B' if d > 0.1 else '#4ECDC4' for d in differences]
    bars = ax.barh(ratio_names, differences, color=colors, alpha=0.8)
    
    ax.set_xlabel('Absolute Difference', color='#EAEAEA')
    ax.set_ylabel('Ratio Type', color='#EAEAEA')
    ax.set_title('Ratio Differences (Lower = More Invariant)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.axvline(x=0.1, color='#FFD93D', linestyle='--', alpha=0.7, label='Threshold (0.1)')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 6. Summary Statistics Table
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    # Create summary text
    summary_text = f"""
    FORMANT RATIO INVARIANCE ANALYSIS
    ==================================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    RAW FORMANTS (Hz):
    ─────────────────────────────────
    F1: {result1['f1_mean']:.1f} vs {result2['f1_mean']:.1f}  (Δ={abs(result1['f1_mean']-result2['f1_mean']):.1f})
    F2: {result1['f2_mean']:.1f} vs {result2['f2_mean']:.1f}  (Δ={abs(result1['f2_mean']-result2['f2_mean']):.1f})
    F3: {result1['f3_mean']:.1f} vs {result2['f3_mean']:.1f}  (Δ={abs(result1['f3_mean']-result2['f3_mean']):.1f})
    
    SCALE-INVARIANT RATIOS:
    ─────────────────────────────────
    F1/F2: {result1['f1_f2_ratio_mean']:.4f} vs {result2['f1_f2_ratio_mean']:.4f}  (Δ={abs(result1['f1_f2_ratio_mean']-result2['f1_f2_ratio_mean']):.4f})
    F2/F3: {result1['f2_f3_ratio_mean']:.4f} vs {result2['f2_f3_ratio_mean']:.4f}  (Δ={abs(result1['f2_f3_ratio_mean']-result2['f2_f3_ratio_mean']):.4f})
    F1/F3: {result1['f1_f3_ratio_mean']:.4f} vs {result2['f1_f3_ratio_mean']:.4f}  (Δ={abs(result1['f1_f3_ratio_mean']-result2['f1_f3_ratio_mean']):.4f})
    
    LOG RATIOS:
    ─────────────────────────────────
    log(F2/F1): {result1['log_f2_f1_mean']:.4f} vs {result2['log_f2_f1_mean']:.4f}  (Δ={abs(result1['log_f2_f1_mean']-result2['log_f2_f1_mean']):.4f})
    log(F3/F2): {result1['log_f3_f2_mean']:.4f} vs {result2['log_f3_f2_mean']:.4f}  (Δ={abs(result1['log_f3_f2_mean']-result2['log_f3_f2_mean']):.4f})
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'formant_ratio_comparison.png')
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
    
    for wav_file in wav_files:
        # Skip if it's the reference file itself
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        print(f"\nAnalyzing: {os.path.basename(wav_file)}")
        result = analyze_audio_file(wav_file)
        
        if result is None:
            print(f"  ⚠ Skipped (could not extract formants)")
            continue
        
        successful_results.append(result)
        
        # Compute comparison metrics
        metrics = [
            'f1_mean', 'f2_mean', 'f3_mean',
            'f1_f2_ratio_mean', 'f2_f3_ratio_mean', 'f1_f3_ratio_mean',
            'log_f2_f1_mean', 'log_f3_f2_mean',
            'f1_f2_ratio_median', 'f2_f3_ratio_median', 'f1_f3_ratio_median',
            'log_f2_f1_median', 'log_f3_f2_median',
        ]
        
        comparison = {'filename': os.path.basename(wav_file)}
        
        for metric in metrics:
            val_ref = ref_result.get(metric, np.nan)
            val_file = result.get(metric, np.nan)
            diff = abs(val_ref - val_file)
            pct_diff = (diff / ((val_ref + val_file) / 2)) * 100 if (val_ref + val_file) != 0 else np.nan
            
            comparison[f'{metric}'] = val_file
            comparison[f'{metric}_diff'] = diff
            comparison[f'{metric}_pct_diff'] = pct_diff
        
        all_comparisons.append(comparison)
    
    if not all_comparisons:
        print("Error: No files could be analyzed")
        return None
    
    # Create results DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'batch_comparison_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Create summary statistics
    summary_metrics = ['f1_f2_ratio_mean', 'f2_f3_ratio_mean', 'f1_f3_ratio_mean',
                       'log_f2_f1_mean', 'log_f3_f2_mean']
    
    summary_data = []
    for metric in summary_metrics:
        pct_col = f'{metric}_pct_diff'
        if pct_col in df.columns:
            summary_data.append({
                'metric': metric,
                'mean_pct_diff': df[pct_col].mean(),
                'std_pct_diff': df[pct_col].std(),
                'min_pct_diff': df[pct_col].min(),
                'max_pct_diff': df[pct_col].max(),
                'median_pct_diff': df[pct_col].median(),
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'batch_comparison_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
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
    other_color = '#4ECDC4'  # Teal for others
    
    # 1. F1/F2 Ratio Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_ratio = ref_result['f1_f2_ratio_mean']
    other_ratios = [r['f1_f2_ratio_mean'] for r in all_results]
    
    ax.axhline(y=ref_ratio, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_ratios)), other_ratios, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('F1/F2 Ratio', color='#EAEAEA')
    ax.set_title('F1/F2 Ratio vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. F2/F3 Ratio Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ref_ratio = ref_result['f2_f3_ratio_mean']
    other_ratios = [r['f2_f3_ratio_mean'] for r in all_results]
    
    ax.axhline(y=ref_ratio, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_ratios)), other_ratios, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('F2/F3 Ratio', color='#EAEAEA')
    ax.set_title('F2/F3 Ratio vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Percent Difference Boxplot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    pct_diff_cols = ['f1_f2_ratio_mean_pct_diff', 'f2_f3_ratio_mean_pct_diff', 
                     'f1_f3_ratio_mean_pct_diff', 'log_f2_f1_mean_pct_diff', 'log_f3_f2_mean_pct_diff']
    pct_diff_labels = ['F1/F2', 'F2/F3', 'F1/F3', 'log(F2/F1)', 'log(F3/F2)']
    
    box_data = [comparison_df[col].dropna().values for col in pct_diff_cols if col in comparison_df.columns]
    
    bp = ax.boxplot(box_data, patch_artist=True, labels=pct_diff_labels[:len(box_data)])
    for patch in bp['boxes']:
        patch.set_facecolor(other_color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set_color('#EAEAEA')
    
    ax.set_xlabel('Ratio Type', color='#EAEAEA')
    ax.set_ylabel('% Difference from Reference', color='#EAEAEA')
    ax.set_title('Distribution of % Differences', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Summary Statistics
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    
    # Calculate stats
    stats_lines = []
    for col, label in zip(pct_diff_cols, pct_diff_labels):
        if col in comparison_df.columns:
            mean_diff = comparison_df[col].mean()
            std_diff = comparison_df[col].std()
            stats_lines.append(f"{label}: {mean_diff:.2f}% ± {std_diff:.2f}%")
    
    summary_text = f"""
    BATCH COMPARISON SUMMARY
    ========================
    
    Reference: {ref_name}
    Files analyzed: {n_files}
    
    AVERAGE % DIFFERENCE FROM REFERENCE:
    ────────────────────────────────────
    {chr(10).join(f'    {line}' for line in stats_lines)}
    
    INTERPRETATION:
    ────────────────────────────────────
    Lower % difference = More scale-invariant
    across different speakers.
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_comparison.png')
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
        DataFrame with all golden file formant data
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
        filename = os.path.basename(golden_file)
        
        result = analyze_audio_file(golden_file)
        
        if result is None:
            continue
        
        result['phoneme'] = phoneme
        all_results.append(result)
    
    if not all_results:
        print("Error: No golden files could be analyzed")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'golden_files_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_golden_comparison_plots(df, output_dir)
    
    return df


def create_golden_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization comparing all golden files across phonemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    # Use a colormap for different phonemes
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # 1. F1-F2 Vowel Space (Traditional)
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['f2_mean'], row['f1_mean'], 
                   c=[colors[i]], s=150, alpha=0.8,
                   edgecolors='white', linewidths=1)
        ax.annotate(row['phoneme'], (row['f2_mean'], row['f1_mean']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, color='#EAEAEA')
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('F1 (Hz)', color='#EAEAEA', fontsize=11)
    ax.set_title('F1-F2 Vowel Space (Golden Files)', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')
    
    # 2. F1/F2 Ratio by Phoneme
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    phonemes = df['phoneme'].tolist()
    ratios = df['f1_f2_ratio_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('F1/F2 Ratio', color='#EAEAEA', fontsize=11)
    ax.set_title('F1/F2 Ratio by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. F2/F3 Ratio by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    ratios = df['f2_f3_ratio_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('F2/F3 Ratio', color='#EAEAEA', fontsize=11)
    ax.set_title('F2/F3 Ratio by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    # Create summary
    summary_lines = []
    for _, row in df.iterrows():
        summary_lines.append(
            f"{row['phoneme']}: F1={row['f1_mean']:.0f}, F2={row['f2_mean']:.0f}, "
            f"F1/F2={row['f1_f2_ratio_mean']:.3f}"
        )
    
    summary_text = f"""
    GOLDEN FILES COMPARISON
    =======================
    
    Total phonemes: {len(df)}
    
    FORMANT SUMMARY:
    ────────────────────────────────────
{chr(10).join(f'    {line}' for line in summary_lines[:20])}
    {'... and more' if len(summary_lines) > 20 else ''}
    
    RATIO STATISTICS:
    ────────────────────────────────────
    F1/F2 range: {df['f1_f2_ratio_mean'].min():.3f} - {df['f1_f2_ratio_mean'].max():.3f}
    F2/F3 range: {df['f2_f3_ratio_mean'].min():.3f} - {df['f2_f3_ratio_mean'].max():.3f}
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_files_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    # Generate additional seaborn-based plots if available
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn-based visualizations for golden files with high contrast."""
    # COLOR SCHEME aligned with sound-topology dark mode
    BG_COLOR = '#111111'
    TEXT_COLOR = '#eaeaea'
    ACCENT_COLOR = '#e17100'  # Orange - use sparingly for important elements
    BORDER_COLOR = '#333333'  # Subtle border
    
    plt.style.use('dark_background')
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['figure.facecolor'] = BG_COLOR
    plt.rcParams['axes.facecolor'] = BG_COLOR
    plt.rcParams['axes.edgecolor'] = BORDER_COLOR
    plt.rcParams['axes.labelcolor'] = TEXT_COLOR
    plt.rcParams['text.color'] = TEXT_COLOR
    plt.rcParams['xtick.color'] = TEXT_COLOR
    plt.rcParams['ytick.color'] = TEXT_COLOR
    plt.rcParams['grid.color'] = '#2a2a2a'
    plt.rcParams['grid.alpha'] = 0.5
    
    # 1. Vowel Space Map with seaborn (F1 vs F2)
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.scatterplot(data=df, x='f2_mean', y='f1_mean', hue='phoneme', 
                    style='phoneme', s=250, palette='bright', legend='brief', ax=ax,
                    edgecolor='white', linewidth=0.5)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_title('Golden Vowel Space (F1 vs F2)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('F2 Frequency (Tongue Frontness)', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('F1 Frequency (Jaw Opening)', fontsize=13, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_vowel_space_map.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 2. Ratio vs F3 Separation Analysis
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.scatterplot(data=df, x='f1_f2_ratio_mean', y='f3_mean', hue='phoneme', 
                    style='phoneme', s=250, palette='bright', legend='brief', ax=ax,
                    edgecolor='white', linewidth=0.5)
    ax.set_title('Separation Analysis: F1/F2 Ratio vs F3', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('F1/F2 Ratio (Invariant Shape)', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('F3 Frequency (Retroflexion Indicator)', fontsize=13, color=TEXT_COLOR)
    ax.axvline(x=0.42, color=ACCENT_COLOR, linestyle='--', alpha=0.9, linewidth=2, label='Ratio Threshold')
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_ratio_vs_f3_separation.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 3. F1/F2 Ratio by Phoneme - USE BARPLOT with VIBRANT COLORS
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sorted_df = df.sort_values('f1_f2_ratio_mean')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    bars = ax.bar(range(n), sorted_df['f1_f2_ratio_mean'].values,
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('F1/F2 Ratio by Phoneme (sorted by mean)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('F1/F2 Ratio', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_ratio_stability.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 4. F3-F2 Difference by Phoneme - BARPLOT with VIBRANT COLORS
    if 'f3_f2_diff_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(18, 8))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        
        sorted_df = df.sort_values('f3_f2_diff_mean')
        n = len(sorted_df)
        rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
        
        bars = ax.bar(range(n), sorted_df['f3_f2_diff_mean'].values,
                      color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
        
        ax.set_xticks(range(n))
        ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
        ax.set_title('F3-F2 Difference by Phoneme (Retroflexion Indicator)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
        ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
        ax.set_ylabel('F3 - F2 (Hz)', fontsize=14, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_COLOR, labelsize=11)
        ax.axhline(y=df['f3_f2_diff_mean'].mean(), color=ACCENT_COLOR, linestyle='--', 
                   linewidth=2, alpha=0.9, label='Mean')
        ax.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
        ax.grid(True, axis='y', alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/04_f3_f2_diff_retroflexion.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close()
    
    # 5. Separation Check Plot for target vowels (A vs Ri analysis)
    target_phonemes = ['अ', 'ॠ', 'इ', 'उ', 'ऋ', 'आ', 'ए', 'ओ']
    subset = df[df['phoneme'].isin(target_phonemes)]
    
    if not subset.empty and 'f3_f2_diff_mean' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor(BG_COLOR)
        ax.set_facecolor(BG_COLOR)
        sns.scatterplot(data=subset, x='f1_f2_ratio_mean', y='f3_f2_diff_mean', 
                        hue='phoneme', style='phoneme', s=300, palette='bright', ax=ax,
                        edgecolor='white', linewidth=0.5)
        ax.set_title("Separation Check: 'A' (अ) vs 'Ri' (ॠ/ऋ)", fontsize=18, color=TEXT_COLOR, fontweight='bold')
        ax.set_xlabel("F1 / F2 Ratio (Invariant Shape)", fontsize=13, color=TEXT_COLOR)
        ax.set_ylabel("F3 - F2 Frequency Gap (Hz)", fontsize=13, color=TEXT_COLOR)
        ax.axvline(x=0.41, color=ACCENT_COLOR, linestyle='--', linewidth=2, alpha=0.9, label='Ratio Threshold')
        ax.tick_params(colors=TEXT_COLOR, labelsize=11)
        ax.grid(True, alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1)
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                           facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/05_separation_check_a_vs_ri.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close()
        
        # Print statistical verdict
        print("\n--- STATISTICAL VERDICT: 'A' vs 'Ri' ---")
        verdict_subset = df[df['phoneme'].isin(['अ', 'ॠ', 'ऋ'])]
        if not verdict_subset.empty:
            summary = verdict_subset.groupby('phoneme')[['f1_f2_ratio_mean', 'f3_f2_diff_mean']].mean()
            print(summary.to_string())
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-05_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Formant-Based Invariant Analysis: Compare frequency ratios between audio files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python formant_ratio_analysis.py --file1 male.wav --file2 female.wav

  # Custom output directory
  python formant_ratio_analysis.py --file1 audio1.wav --file2 audio2.wav --output_dir ./results

  # Batch mode: Compare all files in a folder against a reference file
  python formant_ratio_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Batch mode with custom output
  python formant_ratio_analysis.py --folder data/02_cleaned/इ --reference data/02_cleaned/इ/इ_golden_036.wav --output_dir results/इ_analysis

  # Golden comparison mode: Compare all golden files across phonemes
  python formant_ratio_analysis.py --golden-compare data/02_cleaned
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
                        help='Output directory for results (default: results/formant_ratio_analysis/{mode})')
    parser.add_argument('--no-visual', action='store_true', dest='no_visual',
                        help='Skip generating visualization figures')
    
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
        base_dir = "results/formant_ratio_analysis"
        if golden_mode:
            output_dir = f"{base_dir}/golden"
        elif batch_mode:
            # Use folder name for batch mode
            folder_name = os.path.basename(os.path.normpath(args.folder))
            output_dir = f"{base_dir}/batch/{folder_name}"
        else:
            # Single mode: file1_vs_file2
            name1 = os.path.splitext(os.path.basename(args.file1))[0]
            name2 = os.path.splitext(os.path.basename(args.file2))[0]
            output_dir = f"{base_dir}/compare/{name1}_vs_{name2}"
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("FORMANT-BASED INVARIANT ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: Frequency ratios (F1/F2, F2/F3, log ratios)")
    print(f"should be scale-invariant across different speakers")
    print("=" * 60)
    
    if batch_mode:
        # Batch folder analysis
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
            
            # Show best and worst matches
            if 'f1_f2_ratio_mean_pct_diff' in results_df.columns:
                best_idx = results_df['f1_f2_ratio_mean_pct_diff'].idxmin()
                worst_idx = results_df['f1_f2_ratio_mean_pct_diff'].idxmax()
                
                print(f"\nBest match (F1/F2 ratio): {results_df.loc[best_idx, 'filename']}")
                print(f"  Difference: {results_df.loc[best_idx, 'f1_f2_ratio_mean_pct_diff']:.2f}%")
                
                print(f"\nWorst match (F1/F2 ratio): {results_df.loc[worst_idx, 'filename']}")
                print(f"  Difference: {results_df.loc[worst_idx, 'f1_f2_ratio_mean_pct_diff']:.2f}%")
            
            # Generate visualizations for ALL files in batch (parallel processing)
            # Only generate figures relevant to ratio analysis: 1 (Temporal), 2 (Formant Structure), 4 (Ratios)
            if HAS_VISUALIZER and not args.no_visual:
                print(f"\nGenerating visualization figures for {len(results_df) + 1} files (parallel, 3 figs each)...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                
                # Build file list: (audio_path, subfolder)
                file_list = []
                
                # Add reference file
                ref_filename = os.path.splitext(os.path.basename(args.reference))[0]
                file_list.append((args.reference, ref_filename))
                
                # Add compared files
                for _, row in results_df.iterrows():
                    filename = os.path.splitext(row['filename'])[0]
                    file_path = os.path.join(args.folder, row['filename'])
                    if os.path.exists(file_path):
                        file_list.append((file_path, filename))
                
                successful = generate_batch_figures(file_list, visual_base, workers=4, figures=[1, 2, 4])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    elif golden_mode:
        # Golden files comparison
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
            
            # Show ratio statistics
            print(f"\nF1/F2 Ratio Range: {results_df['f1_f2_ratio_mean'].min():.3f} - {results_df['f1_f2_ratio_mean'].max():.3f}")
            print(f"F2/F3 Ratio Range: {results_df['f2_f3_ratio_mean'].min():.3f} - {results_df['f2_f3_ratio_mean'].max():.3f}")
            
            # Generate visualizations for ALL golden files (parallel processing)
            # Only generate figures relevant to ratio analysis: 1 (Temporal), 2 (Formant Structure), 4 (Ratios)
            if HAS_VISUALIZER and not args.no_visual and len(results_df) > 0:
                print(f"\nGenerating visualization figures for {len(results_df)} files (parallel, 3 figs each)...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                
                # Build file list: (audio_path, subfolder)
                file_list = []
                for _, row in results_df.iterrows():
                    phoneme = row['phoneme']
                    filename = os.path.splitext(row['filename'])[0]
                    subfolder = os.path.join(phoneme, filename)
                    file_list.append((row['file_path'], subfolder))
                
                successful = generate_batch_figures(file_list, visual_base, workers=4, figures=[1, 2, 4])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    else:
        # Single file comparison
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
            print("\n" + "=" * 60)
            print("HYPOTHESIS EVALUATION")
            print("=" * 60)
            
            # Evaluate which ratios show the best invariance
            ratio_metrics = results_df[results_df['metric'].str.contains('ratio|log')]
            best_invariant = ratio_metrics.loc[ratio_metrics['percent_difference'].idxmin()]
            
            print(f"\nMost invariant ratio: {best_invariant['metric']}")
            print(f"  Percent difference: {best_invariant['percent_difference']:.2f}%")
            
            # Check if hypothesis holds (ratios should differ less than raw formants)
            raw_formant_df = results_df[results_df['metric'].isin(['f1_mean', 'f2_mean', 'f3_mean'])]
            avg_raw_pct_diff = raw_formant_df['percent_difference'].mean()
            avg_ratio_pct_diff = ratio_metrics['percent_difference'].mean()
            
            print(f"\nAverage % difference in raw formants: {avg_raw_pct_diff:.2f}%")
            print(f"Average % difference in ratios: {avg_ratio_pct_diff:.2f}%")
            
            if avg_ratio_pct_diff < avg_raw_pct_diff:
                print("\n✓ HYPOTHESIS SUPPORTED: Ratios show better invariance than raw formants")
            else:
                print("\n✗ HYPOTHESIS NOT SUPPORTED: Ratios do not show better invariance")
            
            # Generate visualizations for both files
            if HAS_VISUALIZER and not args.no_visual:
                print("\nGenerating visualization figures...")
                visual_dir = os.path.join(output_dir, 'visual')
                generate_all_figures(args.file1, os.path.join(visual_dir, 'file1'))
                generate_all_figures(args.file2, os.path.join(visual_dir, 'file2'))


if __name__ == "__main__":
    main()

