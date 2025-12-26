#!/usr/bin/env python3
"""
Formant-Based Invariant Analysis: Formant Dispersion

This script analyzes formant frequencies (F1, F2, F3) from audio files and computes
formant dispersion metrics that are hypothesized to be characteristic signatures for vowels.

Metrics computed:
- σ_formant = std_dev(F1, F2, F3) - Standard deviation of the three formants
- Dispersion ratio: (F3-F1) / F2 - Normalized spread of formants
- Range: F3 - F1 - Absolute formant range

Hypothesis: Vowels have characteristic dispersion signatures that can help
distinguish between different vowel qualities.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import parselmouth
from parselmouth.praat import call
from pathlib import Path

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

# Import visualizer for integration
try:
    from formant_visualizer import generate_all_figures
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False

# Configure matplotlib to use Noto Sans Devanagari for proper script rendering
DEVANAGARI_FONT_PATH = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT_PATH):
    fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


def extract_formants(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                      max_formant_freq: float = 5500.0, window_length: float = 0.025,
                      stability_smoothing: float = 50.0, intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant frequencies (F1, F2, F3) from an audio file using Praat algorithms.
    Uses dynamic stability weighting instead of static trimming.
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
                intens = 60.0
            
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
        intensity_arr = np.array(intensity_values)
        
        # Calculate stability weights
        n = len(f1_arr)
        instability = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                delta_f1 = abs(f1_arr[1] - f1_arr[0])
                delta_f2 = abs(f2_arr[1] - f2_arr[0])
            elif i == n - 1:
                delta_f1 = abs(f1_arr[n-1] - f1_arr[n-2])
                delta_f2 = abs(f2_arr[n-1] - f2_arr[n-2])
            else:
                delta_f1 = abs(f1_arr[i+1] - f1_arr[i-1])
                delta_f2 = abs(f2_arr[i+1] - f2_arr[i-1])
            instability[i] = delta_f1 + delta_f2
        
        weights = 1.0 / (instability + stability_smoothing)
        weights[intensity_arr < intensity_threshold] = 0.0
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n) / n
        
        f1_weighted_mean = np.average(f1_arr, weights=weights)
        f2_weighted_mean = np.average(f2_arr, weights=weights)
        f3_weighted_mean = np.average(f3_arr, weights=weights)
        
        f1_weighted_var = np.average((f1_arr - f1_weighted_mean)**2, weights=weights)
        f2_weighted_var = np.average((f2_arr - f2_weighted_mean)**2, weights=weights)
        f3_weighted_var = np.average((f3_arr - f3_weighted_mean)**2, weights=weights)
        
        return {
            'f1_mean': f1_weighted_mean,
            'f2_mean': f2_weighted_mean,
            'f3_mean': f3_weighted_mean,
            'f1_median': np.median(f1_arr),
            'f2_median': np.median(f2_arr),
            'f3_median': np.median(f3_arr),
            'f1_std': np.sqrt(f1_weighted_var),
            'f2_std': np.sqrt(f2_weighted_var),
            'f3_std': np.sqrt(f3_weighted_var),
            'f1_values': f1_arr,
            'f2_values': f2_arr,
            'f3_values': f3_arr,
            'time_values': np.array(time_values),
            'stability_weights': weights,
            'n_frames': len(f1_values),
            'duration': duration
        }
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def compute_formant_dispersion(formant_data: dict) -> dict:
    """
    Compute formant dispersion metrics from formant data.
    
    Metrics:
    - σ_formant: Standard deviation of F1, F2, F3 (mean-based)
    - Dispersion ratio: (F3-F1) / F2
    - Formant range: F3 - F1
    """
    if formant_data is None:
        return None
    
    f1_mean = formant_data['f1_mean']
    f2_mean = formant_data['f2_mean']
    f3_mean = formant_data['f3_mean']
    
    f1_med = formant_data['f1_median']
    f2_med = formant_data['f2_median']
    f3_med = formant_data['f3_median']
    
    # σ_formant = std_dev(F1, F2, F3) using means
    sigma_formant_mean = np.std([f1_mean, f2_mean, f3_mean])
    sigma_formant_median = np.std([f1_med, f2_med, f3_med])
    
    # Dispersion ratio: (F3-F1) / F2
    dispersion_ratio_mean = (f3_mean - f1_mean) / f2_mean
    dispersion_ratio_median = (f3_med - f1_med) / f2_med
    
    # Formant range: F3 - F1
    formant_range_mean = f3_mean - f1_mean
    formant_range_median = f3_med - f1_med
    
    # Per-frame dispersion metrics
    f1_vals = formant_data['f1_values']
    f2_vals = formant_data['f2_values']
    f3_vals = formant_data['f3_values']
    
    frame_sigma = np.array([np.std([f1, f2, f3]) for f1, f2, f3 in zip(f1_vals, f2_vals, f3_vals)])
    frame_dispersion_ratio = (f3_vals - f1_vals) / f2_vals
    frame_range = f3_vals - f1_vals
    
    return {
        # Mean-based metrics
        'sigma_formant_mean': sigma_formant_mean,
        'dispersion_ratio_mean': dispersion_ratio_mean,
        'formant_range_mean': formant_range_mean,
        
        # Median-based metrics
        'sigma_formant_median': sigma_formant_median,
        'dispersion_ratio_median': dispersion_ratio_median,
        'formant_range_median': formant_range_median,
        
        # Per-frame statistics
        'frame_sigma': frame_sigma,
        'frame_dispersion_ratio': frame_dispersion_ratio,
        'frame_range': frame_range,
        'frame_sigma_std': np.std(frame_sigma),
        'frame_dispersion_ratio_std': np.std(frame_dispersion_ratio),
        'frame_range_std': np.std(frame_range),
    }


def analyze_audio_file(audio_path: str) -> dict:
    """Complete formant dispersion analysis for a single audio file."""
    max_freq = 8000.0
    formant_data = extract_formants(audio_path, max_formant_freq=max_freq)
    
    if formant_data is None:
        return None
    
    dispersion = compute_formant_dispersion(formant_data)
    
    if dispersion is None:
        return None
    
    result = {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        **{k: v for k, v in formant_data.items() if not k.endswith('_values') and k != 'time_values' and k != 'stability_weights'},
        **{k: v for k, v in dispersion.items() if not k.startswith('frame_')},
    }
    
    result['frame_sigma_std'] = dispersion['frame_sigma_std']
    result['frame_dispersion_ratio_std'] = dispersion['frame_dispersion_ratio_std']
    result['frame_range_std'] = dispersion['frame_range_std']
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """Compare formant dispersion between two audio files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing: {os.path.basename(file1_path)}")
    result1 = analyze_audio_file(file1_path)
    
    print(f"Analyzing: {os.path.basename(file2_path)}")
    result2 = analyze_audio_file(file2_path)
    
    if result1 is None or result2 is None:
        print("Error: Could not analyze one or both files")
        return None
    
    metrics = [
        'f1_mean', 'f2_mean', 'f3_mean',
        'sigma_formant_mean', 'dispersion_ratio_mean', 'formant_range_mean',
        'sigma_formant_median', 'dispersion_ratio_median', 'formant_range_median',
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
    
    csv_path = os.path.join(output_dir, 'formant_dispersion_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing formant dispersion between two files."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    
    color1 = '#FF6B6B'
    color2 = '#4ECDC4'
    
    # 1. Raw Formants
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(3)
    width = 0.35
    
    formants1 = [result1['f1_mean'], result1['f2_mean'], result1['f3_mean']]
    formants2 = [result2['f1_mean'], result2['f2_mean'], result2['f3_mean']]
    
    ax.bar(x - width/2, formants1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, formants2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Formant', color='#EAEAEA')
    ax.set_ylabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_title('Raw Formant Frequencies', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['F1', 'F2', 'F3'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    _style_axis(ax)
    
    # 2. σ_formant Comparison
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(2)
    
    sigma1 = [result1['sigma_formant_mean'], result1['sigma_formant_median']]
    sigma2 = [result2['sigma_formant_mean'], result2['sigma_formant_median']]
    
    ax.bar(x - width/2, sigma1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, sigma2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Calculation Method', color='#EAEAEA')
    ax.set_ylabel('σ_formant (Hz)', color='#EAEAEA')
    ax.set_title('Formant Dispersion (σ)', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean-based', 'Median-based'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    _style_axis(ax)
    
    # 3. Dispersion Ratio Comparison
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    
    disp1 = [result1['dispersion_ratio_mean'], result1['dispersion_ratio_median']]
    disp2 = [result2['dispersion_ratio_mean'], result2['dispersion_ratio_median']]
    
    ax.bar(x - width/2, disp1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, disp2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Calculation Method', color='#EAEAEA')
    ax.set_ylabel('(F3-F1)/F2', color='#EAEAEA')
    ax.set_title('Dispersion Ratio', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean-based', 'Median-based'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    _style_axis(ax)
    
    # 4. Formant Range Comparison
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    range1 = [result1['formant_range_mean'], result1['formant_range_median']]
    range2 = [result2['formant_range_mean'], result2['formant_range_median']]
    
    ax.bar(x - width/2, range1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, range2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Calculation Method', color='#EAEAEA')
    ax.set_ylabel('F3 - F1 (Hz)', color='#EAEAEA')
    ax.set_title('Formant Range', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Mean-based', 'Median-based'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    _style_axis(ax)
    
    # 5. Difference Visualization
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    
    metric_names = ['σ_formant', 'Dispersion\nRatio', 'Range']
    differences = [
        abs(result1['sigma_formant_mean'] - result2['sigma_formant_mean']),
        abs(result1['dispersion_ratio_mean'] - result2['dispersion_ratio_mean']),
        abs(result1['formant_range_mean'] - result2['formant_range_mean']),
    ]
    
    # Normalize for visualization
    norm_diffs = [d / max(differences) if max(differences) > 0 else 0 for d in differences]
    colors = ['#FF6B6B' if d > 0.5 else '#4ECDC4' for d in norm_diffs]
    ax.barh(metric_names, norm_diffs, color=colors, alpha=0.8)
    
    ax.set_xlabel('Normalized Difference', color='#EAEAEA')
    ax.set_title('Metric Differences (Lower = More Similar)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    _style_axis(ax)
    
    # 6. Summary
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    FORMANT DISPERSION ANALYSIS
    ===========================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    σ_formant (std of F1,F2,F3):
    ─────────────────────────────
    {result1['sigma_formant_mean']:.1f} vs {result2['sigma_formant_mean']:.1f} Hz
    (Δ = {abs(result1['sigma_formant_mean']-result2['sigma_formant_mean']):.1f} Hz)
    
    Dispersion Ratio (F3-F1)/F2:
    ─────────────────────────────
    {result1['dispersion_ratio_mean']:.4f} vs {result2['dispersion_ratio_mean']:.4f}
    (Δ = {abs(result1['dispersion_ratio_mean']-result2['dispersion_ratio_mean']):.4f})
    
    Formant Range (F3-F1):
    ─────────────────────────────
    {result1['formant_range_mean']:.1f} vs {result2['formant_range_mean']:.1f} Hz
    (Δ = {abs(result1['formant_range_mean']-result2['formant_range_mean']):.1f} Hz)
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'formant_dispersion_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def _style_axis(ax):
    """Apply dark theme styling to axis."""
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """Compare all audio files in a folder against a reference file."""
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
    
    if not wav_files:
        print(f"Error: No .wav files found in {folder_path}")
        return None
    
    print(f"\n{'='*60}")
    print(f"REFERENCE FILE: {os.path.basename(reference_file)}")
    print(f"{'='*60}")
    
    ref_result = analyze_audio_file(reference_file)
    if ref_result is None:
        print(f"Error: Could not analyze reference file: {reference_file}")
        return None
    
    all_comparisons = []
    successful_results = []
    
    for wav_file in wav_files:
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        print(f"\nAnalyzing: {os.path.basename(wav_file)}")
        result = analyze_audio_file(wav_file)
        
        if result is None:
            print(f"  ⚠ Skipped (could not extract formants)")
            continue
        
        successful_results.append(result)
        
        metrics = [
            'f1_mean', 'f2_mean', 'f3_mean',
            'sigma_formant_mean', 'dispersion_ratio_mean', 'formant_range_mean',
            'sigma_formant_median', 'dispersion_ratio_median', 'formant_range_median',
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
    
    df = pd.DataFrame(all_comparisons)
    
    csv_path = os.path.join(output_dir, 'batch_dispersion_detailed.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    summary_metrics = ['sigma_formant_mean', 'dispersion_ratio_mean', 'formant_range_mean']
    
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
    summary_path = os.path.join(output_dir, 'batch_dispersion_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    create_batch_plots(ref_result, successful_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    ref_color = '#FFD93D'
    other_color = '#4ECDC4'
    
    # 1. σ_formant Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_sigma = ref_result['sigma_formant_mean']
    other_sigmas = [r['sigma_formant_mean'] for r in all_results]
    
    ax.axhline(y=ref_sigma, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_sigmas)), other_sigmas, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('σ_formant (Hz)', color='#EAEAEA')
    ax.set_title('σ_formant vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    _style_axis(ax)
    
    # 2. Dispersion Ratio Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ref_disp = ref_result['dispersion_ratio_mean']
    other_disps = [r['dispersion_ratio_mean'] for r in all_results]
    
    ax.axhline(y=ref_disp, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_disps)), other_disps, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Dispersion Ratio', color='#EAEAEA')
    ax.set_title('Dispersion Ratio (F3-F1)/F2 vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    _style_axis(ax)
    
    # 3. Percent Difference Boxplot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    pct_diff_cols = ['sigma_formant_mean_pct_diff', 'dispersion_ratio_mean_pct_diff', 'formant_range_mean_pct_diff']
    pct_diff_labels = ['σ_formant', 'Disp Ratio', 'Range']
    
    box_data = [comparison_df[col].dropna().values for col in pct_diff_cols if col in comparison_df.columns]
    
    bp = ax.boxplot(box_data, patch_artist=True, labels=pct_diff_labels[:len(box_data)])
    for patch in bp['boxes']:
        patch.set_facecolor(other_color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set_color('#EAEAEA')
    
    ax.set_xlabel('Metric Type', color='#EAEAEA')
    ax.set_ylabel('% Difference from Reference', color='#EAEAEA')
    ax.set_title('Distribution of % Differences', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    _style_axis(ax)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    
    stats_lines = []
    for col, label in zip(pct_diff_cols, pct_diff_labels):
        if col in comparison_df.columns:
            mean_diff = comparison_df[col].mean()
            std_diff = comparison_df[col].std()
            stats_lines.append(f"{label}: {mean_diff:.2f}% ± {std_diff:.2f}%")
    
    summary_text = f"""
    BATCH DISPERSION SUMMARY
    ========================
    
    Reference: {ref_name}
    Files analyzed: {n_files}
    
    AVERAGE % DIFFERENCE FROM REFERENCE:
    ────────────────────────────────────
    {chr(10).join(f'    {line}' for line in stats_lines)}
    
    INTERPRETATION:
    ────────────────────────────────────
    Lower % difference = More similar
    dispersion characteristics.
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_dispersion.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Batch visualization saved to: {plot_path}")


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """Find and compare all golden files across different phonemes."""
    import glob
    
    os.makedirs(output_dir, exist_ok=True)
    
    golden_pattern = os.path.join(cleaned_data_dir, '*', '*_golden_*.wav')
    golden_files = glob.glob(golden_pattern)
    
    if not golden_files:
        print(f"Error: No golden files found in {cleaned_data_dir}")
        return None
    
    print(f"\nFound {len(golden_files)} golden files")
    print("=" * 60)
    
    all_results = []
    
    for golden_file in tqdm(sorted(golden_files), desc="Analyzing golden files"):
        phoneme = os.path.basename(os.path.dirname(golden_file))
        
        result = analyze_audio_file(golden_file)
        
        if result is None:
            continue
        
        result['phoneme'] = phoneme
        all_results.append(result)
    
    if not all_results:
        print("Error: No golden files could be analyzed")
        return None
    
    df = pd.DataFrame(all_results)
    
    csv_path = os.path.join(output_dir, 'golden_dispersion_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_golden_comparison_plots(df, output_dir)
    
    return df


def create_golden_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization comparing all golden files across phonemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # 1. σ_formant by Phoneme
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    phonemes = df['phoneme'].tolist()
    sigmas = df['sigma_formant_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), sigmas, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('σ_formant (Hz)', color='#EAEAEA', fontsize=11)
    ax.set_title('σ_formant by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    _style_axis(ax)
    
    # 2. Dispersion Ratio by Phoneme
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    disp_ratios = df['dispersion_ratio_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), disp_ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('(F3-F1)/F2', color='#EAEAEA', fontsize=11)
    ax.set_title('Dispersion Ratio by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    _style_axis(ax)
    
    # 3. Dispersion Ratio vs σ_formant Scatter
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['sigma_formant_mean'], row['dispersion_ratio_mean'],
                   c=[colors[i]], s=150, alpha=0.8, edgecolors='white', linewidths=1)
        ax.annotate(row['phoneme'], (row['sigma_formant_mean'], row['dispersion_ratio_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, color='#EAEAEA')
    
    ax.set_xlabel('σ_formant (Hz)', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('Dispersion Ratio', color='#EAEAEA', fontsize=11)
    ax.set_title('σ_formant vs Dispersion Ratio', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_color('#333')
    ax.spines['right'].set_color('#333')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_lines = []
    for _, row in df.iterrows():
        summary_lines.append(
            f"{row['phoneme']}: σ={row['sigma_formant_mean']:.0f}Hz, "
            f"DR={row['dispersion_ratio_mean']:.3f}"
        )
    
    summary_text = f"""
    GOLDEN FILES DISPERSION
    =======================
    
    Total phonemes: {len(df)}
    
    DISPERSION SUMMARY:
    ────────────────────────────────────
{chr(10).join(f'    {line}' for line in summary_lines[:20])}
    {'... and more' if len(summary_lines) > 20 else ''}
    
    STATISTICS:
    ────────────────────────────────────
    σ_formant range: {df['sigma_formant_mean'].min():.0f} - {df['sigma_formant_mean'].max():.0f} Hz
    Disp Ratio range: {df['dispersion_ratio_mean'].min():.3f} - {df['dispersion_ratio_mean'].max():.3f}
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_dispersion_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
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
    
    # 1. σ_formant vs Dispersion Ratio
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.scatterplot(data=df, x='sigma_formant_mean', y='dispersion_ratio_mean', 
                    hue='phoneme', style='phoneme', s=250, palette='bright', legend='brief', ax=ax,
                    edgecolor='white', linewidth=0.5)
    ax.set_title('Dispersion Analysis: σ_formant vs Dispersion Ratio', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('σ_formant (Hz)', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('Dispersion Ratio (F3-F1)/F2', fontsize=13, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_sigma_vs_dispersion.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 2. σ_formant by Phoneme (sorted bar)
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sorted_df = df.sort_values('sigma_formant_mean')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    bars = ax.bar(range(n), sorted_df['sigma_formant_mean'].values,
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('σ_formant by Phoneme (sorted)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('σ_formant (Hz)', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_sigma_by_phoneme.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 3. Dispersion Ratio by Phoneme (sorted bar)
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sorted_df = df.sort_values('dispersion_ratio_mean')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    bars = ax.bar(range(n), sorted_df['dispersion_ratio_mean'].values,
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('Dispersion Ratio by Phoneme (sorted)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('(F3-F1)/F2', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.axhline(y=df['dispersion_ratio_mean'].mean(), color=ACCENT_COLOR, linestyle='--', 
               linewidth=2, alpha=0.9, label='Mean')
    ax.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_dispersion_ratio_by_phoneme.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-03_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Formant-Based Invariant Analysis: Formant Dispersion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python formant_dispersion_analysis.py --file1 male.wav --file2 female.wav

  # Custom output directory
  python formant_dispersion_analysis.py --file1 audio1.wav --file2 audio2.wav --output_dir ./results

  # Batch mode: Compare all files in a folder against a reference file
  python formant_dispersion_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python formant_dispersion_analysis.py --golden-compare data/02_cleaned
        """
    )
    
    parser.add_argument('--file1', type=str, help='Path to first audio file')
    parser.add_argument('--file2', type=str, help='Path to second audio file')
    parser.add_argument('--folder', type=str, help='Path to folder containing audio files (batch mode)')
    parser.add_argument('--reference', type=str, help='Path to reference file (batch mode)')
    parser.add_argument('--golden-compare', type=str, dest='golden_compare',
                        help='Path to cleaned data folder (golden mode)')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    parser.add_argument('--no-visual', action='store_true', dest='no_visual',
                        help='Skip generating visualization figures')
    
    args = parser.parse_args()
    
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
    
    if args.output_dir is None:
        base_dir = "results/formant_dispersion_analysis"
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
    print("FORMANT DISPERSION ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: Vowels have characteristic dispersion signatures")
    print(f"Metrics: σ_formant = std(F1,F2,F3), Dispersion Ratio = (F3-F1)/F2")
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
            
            # Generate visualizations for ALL files in batch (parallel processing)
            # Only generate figures relevant to dispersion analysis: 1 (Temporal), 2 (Formant Structure), 3 (Geometry)
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
                
                successful = generate_batch_figures(file_list, visual_base, workers=4, figures=[1, 2, 3])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
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
            print(f"\nσ_formant Range: {results_df['sigma_formant_mean'].min():.0f} - {results_df['sigma_formant_mean'].max():.0f} Hz")
            print(f"Dispersion Ratio Range: {results_df['dispersion_ratio_mean'].min():.3f} - {results_df['dispersion_ratio_mean'].max():.3f}")
            
            # Generate visualizations for ALL golden files (parallel processing)
            # Only generate figures relevant to dispersion analysis: 1 (Temporal), 2 (Formant Structure), 3 (Geometry)
            if HAS_VISUALIZER and not args.no_visual:
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
                
                successful = generate_batch_figures(file_list, visual_base, workers=4, figures=[1, 2, 3])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
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
            
            # Generate visualizations for both files
            if HAS_VISUALIZER and not args.no_visual:
                print("\nGenerating visualization figures...")
                visual_dir = os.path.join(output_dir, 'visual')
                generate_all_figures(args.file1, os.path.join(visual_dir, 'file1'))
                generate_all_figures(args.file2, os.path.join(visual_dir, 'file2'))


if __name__ == "__main__":
    main()
