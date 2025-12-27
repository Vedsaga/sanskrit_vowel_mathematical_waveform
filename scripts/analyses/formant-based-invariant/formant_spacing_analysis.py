#!/usr/bin/env python3
"""
Formant-Based Invariant Analysis: Formant Spacing Patterns

Hypothesis: Relative spacing is invariant across speakers for the same phoneme.

Metrics:
- ΔF21 = F2 - F1, ΔF32 = F3 - F2 (raw spacing)
- Normalized spacing = ΔF / geometric_mean(F1, F2, F3)

IMPORTANT: Same equations applied consistently to ALL files to avoid bias.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob

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

# Import seaborn for enhanced plotting (optional)
if HAS_SEABORN:
    import seaborn as sns

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
    Extract formant frequencies (F1, F2, F3) using Praat with dynamic stability weighting.
    
    This is a thin wrapper around the canonical implementation in common.formant_extraction.
    
    Refined Method (Method 3):
    - Uses Joint Stability-Intensity Weighting (Intensity^2 / Instability)
    - Computes weighted means and standard deviations
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length in seconds
        stability_smoothing: Smoothing constant
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


def compute_formant_spacing(formant_data: dict) -> dict:
    """
    Compute formant spacing metrics - SAME EQUATIONS FOR ALL FILES (no bias).
    
    Metrics:
    - delta_f21 = F2 - F1 (raw spacing)
    - delta_f32 = F3 - F2 (raw spacing)
    - geom_mean = (F1 * F2 * F3)^(1/3)
    - norm_delta_f21 = delta_f21 / geom_mean
    - norm_delta_f32 = delta_f32 / geom_mean
    """
    if formant_data is None:
        return None
    
    # Extract formant values
    f1_mean, f2_mean, f3_mean = formant_data['f1_mean'], formant_data['f2_mean'], formant_data['f3_mean']
    f1_med, f2_med, f3_med = formant_data['f1_median_unweighted'], formant_data['f2_median_unweighted'], formant_data['f3_median_unweighted']
    
    # === CONSISTENT EQUATIONS FOR ALL FILES ===
    
    # Raw spacing (Hz)
    delta_f21_mean = f2_mean - f1_mean
    delta_f32_mean = f3_mean - f2_mean
    delta_f21_median = f2_med - f1_med
    delta_f32_median = f3_med - f2_med
    
    # Geometric mean of formants
    geom_mean_mean = (f1_mean * f2_mean * f3_mean) ** (1/3)
    geom_mean_median = (f1_med * f2_med * f3_med) ** (1/3)
    
    # Normalized spacing (dimensionless - scale invariant)
    norm_delta_f21_mean = delta_f21_mean / geom_mean_mean
    norm_delta_f32_mean = delta_f32_mean / geom_mean_mean
    norm_delta_f21_median = delta_f21_median / geom_mean_median
    norm_delta_f32_median = delta_f32_median / geom_mean_median
    
    # Per-frame analysis for distribution
    f1_vals = formant_data['f1_values']
    f2_vals = formant_data['f2_values']
    f3_vals = formant_data['f3_values']
    
    frame_delta_f21 = f2_vals - f1_vals
    frame_delta_f32 = f3_vals - f2_vals
    frame_geom_mean = (f1_vals * f2_vals * f3_vals) ** (1/3)
    frame_norm_delta_f21 = frame_delta_f21 / frame_geom_mean
    frame_norm_delta_f32 = frame_delta_f32 / frame_geom_mean
    
    return {
        # Raw spacing
        'delta_f21_mean': delta_f21_mean, 'delta_f32_mean': delta_f32_mean,
        'delta_f21_median': delta_f21_median, 'delta_f32_median': delta_f32_median,
        # Geometric mean
        'geom_mean_mean': geom_mean_mean, 'geom_mean_median': geom_mean_median,
        # Normalized spacing (key invariant metrics)
        'norm_delta_f21_mean': norm_delta_f21_mean, 'norm_delta_f32_mean': norm_delta_f32_mean,
        'norm_delta_f21_median': norm_delta_f21_median, 'norm_delta_f32_median': norm_delta_f32_median,
        # Per-frame statistics
        'frame_norm_delta_f21_std': np.std(frame_norm_delta_f21),
        'frame_norm_delta_f32_std': np.std(frame_norm_delta_f32),
    }


def analyze_audio_file(audio_path: str) -> dict:
    """Complete formant spacing analysis for a single audio file."""
    max_freq = 8000.0  # Fixed parameter for all files
    
    formant_data = extract_formants(audio_path, max_formant_freq=max_freq)
    if formant_data is None:
        return None
    
    spacing = compute_formant_spacing(formant_data)
    if spacing is None:
        return None
    
    result = {'filename': os.path.basename(audio_path), 'filepath': audio_path}
    result.update({k: v for k, v in formant_data.items() if not k.endswith('_values')})
    result.update(spacing)
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """Compare formant spacing between two audio files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing: {os.path.basename(file1_path)}")
    result1 = analyze_audio_file(file1_path)
    print(f"Analyzing: {os.path.basename(file2_path)}")
    result2 = analyze_audio_file(file2_path)
    
    if result1 is None or result2 is None:
        print("Error: Could not analyze one or both files")
        return None
    
    # Comparison metrics
    metrics = ['delta_f21_mean', 'delta_f32_mean', 'norm_delta_f21_mean', 'norm_delta_f32_mean',
               'geom_mean_mean', 'f1_mean', 'f2_mean', 'f3_mean']
    
    comparison_data = []
    for m in metrics:
        v1, v2 = result1.get(m, np.nan), result2.get(m, np.nan)
        diff = abs(v1 - v2)
        pct_diff = (diff / ((v1 + v2) / 2)) * 100 if (v1 + v2) != 0 else np.nan
        comparison_data.append({'metric': m, 'file1': v1, 'file2': v2, 
                                'abs_difference': diff, 'percent_difference': pct_diff})
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'spacing_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_two_file_plots(result1, result2, output_dir)
    
    return df


def create_two_file_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization for two-file comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    color1, color2 = '#FF6B6B', '#4ECDC4'
    
    # 1. Raw Spacing Comparison
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(2)
    width = 0.35
    
    spacing1 = [result1['delta_f21_mean'], result1['delta_f32_mean']]
    spacing2 = [result2['delta_f21_mean'], result2['delta_f32_mean']]
    
    ax.bar(x - width/2, spacing1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, spacing2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_ylabel('Spacing (Hz)', color='#EAEAEA')
    ax.set_title('Raw Formant Spacing', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['ΔF21 (F2-F1)', 'ΔF32 (F3-F2)'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Normalized Spacing Comparison
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    norm_spacing1 = [result1['norm_delta_f21_mean'], result1['norm_delta_f32_mean']]
    norm_spacing2 = [result2['norm_delta_f21_mean'], result2['norm_delta_f32_mean']]
    
    ax.bar(x - width/2, norm_spacing1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, norm_spacing2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_ylabel('Normalized Spacing', color='#EAEAEA')
    ax.set_title('Normalized Spacing (ΔF / GeomMean)', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Norm ΔF21', 'Norm ΔF32'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. 2D Spacing Space
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    ax.scatter([result1['norm_delta_f21_mean']], [result1['norm_delta_f32_mean']], 
               s=200, c=color1, label=file1_name, marker='o', edgecolors='white', linewidths=2)
    ax.scatter([result2['norm_delta_f21_mean']], [result2['norm_delta_f32_mean']], 
               s=200, c=color2, label=file2_name, marker='s', edgecolors='white', linewidths=2)
    
    ax.set_xlabel('Norm ΔF21', color='#EAEAEA')
    ax.set_ylabel('Norm ΔF32', color='#EAEAEA')
    ax.set_title('Normalized Spacing Space', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    FORMANT SPACING INVARIANCE ANALYSIS
    ====================================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    RAW SPACING (Hz):
    ─────────────────────────────────
    ΔF21: {result1['delta_f21_mean']:.1f} vs {result2['delta_f21_mean']:.1f}  (Δ={abs(result1['delta_f21_mean']-result2['delta_f21_mean']):.1f})
    ΔF32: {result1['delta_f32_mean']:.1f} vs {result2['delta_f32_mean']:.1f}  (Δ={abs(result1['delta_f32_mean']-result2['delta_f32_mean']):.1f})
    
    NORMALIZED SPACING (Scale-Invariant):
    ─────────────────────────────────
    Norm ΔF21: {result1['norm_delta_f21_mean']:.4f} vs {result2['norm_delta_f21_mean']:.4f}  (Δ={abs(result1['norm_delta_f21_mean']-result2['norm_delta_f21_mean']):.4f})
    Norm ΔF32: {result1['norm_delta_f32_mean']:.4f} vs {result2['norm_delta_f32_mean']:.4f}  (Δ={abs(result1['norm_delta_f32_mean']-result2['norm_delta_f32_mean']):.4f})
    
    GEOMETRIC MEAN (Hz):
    ─────────────────────────────────
    {result1['geom_mean_mean']:.1f} vs {result2['geom_mean_mean']:.1f}
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'spacing_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """Compare all audio files in a folder against a pinned reference file."""
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
        print(f"Error: Could not analyze reference file")
        return None
    
    all_comparisons = []
    successful_results = []
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        if result is None:
            continue
        
        successful_results.append(result)
        
        metrics = ['norm_delta_f21_mean', 'norm_delta_f32_mean', 'delta_f21_mean', 'delta_f32_mean', 'geom_mean_mean']
        comparison = {'filename': os.path.basename(wav_file)}
        
        for m in metrics:
            v_ref, v_file = ref_result.get(m, np.nan), result.get(m, np.nan)
            diff = abs(v_ref - v_file)
            pct_diff = (diff / ((v_ref + v_file) / 2)) * 100 if (v_ref + v_file) != 0 else np.nan
            comparison[m] = v_file
            comparison[f'{m}_diff'] = diff
            comparison[f'{m}_pct_diff'] = pct_diff
        
        all_comparisons.append(comparison)
    
    if not all_comparisons:
        print("Error: No files could be analyzed")
        return None
    
    df = pd.DataFrame(all_comparisons)
    csv_path = os.path.join(output_dir, 'batch_spacing_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_batch_plots(ref_result, successful_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    ref_color, other_color = '#FFD93D', '#4ECDC4'
    
    # 1. Norm ΔF21 Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_val = ref_result['norm_delta_f21_mean']
    other_vals = [r['norm_delta_f21_mean'] for r in all_results]
    
    ax.axhline(y=ref_val, color=ref_color, linestyle='--', linewidth=2, label=f'Ref: {ref_name}')
    ax.scatter(range(len(other_vals)), other_vals, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Norm ΔF21', color='#EAEAEA')
    ax.set_title('Normalized ΔF21 vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 2. Norm ΔF32 Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ref_val = ref_result['norm_delta_f32_mean']
    other_vals = [r['norm_delta_f32_mean'] for r in all_results]
    
    ax.axhline(y=ref_val, color=ref_color, linestyle='--', linewidth=2, label=f'Ref: {ref_name}')
    ax.scatter(range(len(other_vals)), other_vals, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Norm ΔF32', color='#EAEAEA')
    ax.set_title('Normalized ΔF32 vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. % Difference Boxplot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    pct_cols = ['norm_delta_f21_mean_pct_diff', 'norm_delta_f32_mean_pct_diff']
    labels = ['Norm ΔF21', 'Norm ΔF32']
    
    box_data = [comparison_df[c].dropna().values for c in pct_cols if c in comparison_df.columns]
    
    bp = ax.boxplot(box_data, patch_artist=True, labels=labels[:len(box_data)])
    for patch in bp['boxes']:
        patch.set_facecolor(other_color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        for item in bp[element]:
            item.set_color('#EAEAEA')
    
    ax.set_xlabel('Metric', color='#EAEAEA')
    ax.set_ylabel('% Difference from Reference', color='#EAEAEA')
    ax.set_title('Distribution of % Differences', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    stats_lines = []
    for col, label in zip(pct_cols, labels):
        if col in comparison_df.columns:
            mean_diff = comparison_df[col].mean()
            std_diff = comparison_df[col].std()
            stats_lines.append(f"{label}: {mean_diff:.2f}% ± {std_diff:.2f}%")
    
    summary_text = f"""
    BATCH SPACING COMPARISON SUMMARY
    =================================
    
    Reference: {ref_name}
    Files analyzed: {n_files}
    
    AVERAGE % DIFFERENCE FROM REFERENCE:
    ────────────────────────────────────
    {chr(10).join(f'    {line}' for line in stats_lines)}
    
    INTERPRETATION:
    ────────────────────────────────────
    Lower % difference = More invariant
    across different recordings.
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'batch_spacing_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Batch visualization saved to: {plot_path}")


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """Find and compare all golden files across different phonemes."""
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
    csv_path = os.path.join(output_dir, 'golden_spacing_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_golden_plots(df, output_dir)
    
    return df


def create_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization comparing all golden files across phonemes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # 1. Normalized Spacing Space
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['norm_delta_f21_mean'], row['norm_delta_f32_mean'],
                   c=[colors[i]], s=150, alpha=0.8, edgecolors='white', linewidths=1)
        ax.annotate(row['phoneme'], (row['norm_delta_f21_mean'], row['norm_delta_f32_mean']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, color='#EAEAEA')
    
    ax.set_xlabel('Norm ΔF21', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('Norm ΔF32', color='#EAEAEA', fontsize=11)
    ax.set_title('Normalized Spacing Space (Golden Files)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Norm ΔF21 by Phoneme
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    phonemes = df['phoneme'].tolist()
    values = df['norm_delta_f21_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Norm ΔF21', color='#EAEAEA', fontsize=11)
    ax.set_title('Normalized ΔF21 by Phoneme', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 3. Norm ΔF32 by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    values = df['norm_delta_f32_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Norm ΔF32', color='#EAEAEA', fontsize=11)
    ax.set_title('Normalized ΔF32 by Phoneme', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    GOLDEN FILES SPACING COMPARISON
    ================================
    
    Total phonemes: {len(df)}
    
    NORMALIZED SPACING STATISTICS:
    ────────────────────────────────────
    Norm ΔF21 range: {df['norm_delta_f21_mean'].min():.3f} - {df['norm_delta_f21_mean'].max():.3f}
    Norm ΔF32 range: {df['norm_delta_f32_mean'].min():.3f} - {df['norm_delta_f32_mean'].max():.3f}
    
    GEOMETRIC MEAN STATISTICS:
    ────────────────────────────────────
    GeomMean range: {df['geom_mean_mean'].min():.1f} - {df['geom_mean_mean'].max():.1f} Hz
    
    HYPOTHESIS CHECK:
    ────────────────────────────────────
    If normalized spacing is invariant,
    same phoneme should cluster together
    across different speakers.
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'golden_spacing_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")
    
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn-based visualizations for golden files with high contrast."""
    # COLOR SCHEME aligned with sound-topology dark mode
    BG_COLOR = '#111111'
    TEXT_COLOR = '#eaeaea'
    ACCENT_COLOR = '#e17100'  # Orange for attention - use sparingly
    GRID_COLOR = 'rgba(255,255,255,0.1)'  # Subtle 10% white like sound-topology --border
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
    
    # 1. Normalized Spacing Map
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    sns.scatterplot(data=df, x='norm_delta_f21_mean', y='norm_delta_f32_mean', 
                    hue='phoneme', style='phoneme', s=250, palette='bright', 
                    legend='brief', ax=ax, edgecolor='white', linewidth=0.5)
    ax.set_title('Golden Normalized Spacing Space', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Normalized ΔF21 (F2-F1)/GeomMean', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('Normalized ΔF32 (F3-F2)/GeomMean', fontsize=13, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_norm_spacing_map.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 2. Norm ΔF21 by Phoneme - USE BARPLOT with VIBRANT COLORS (each phoneme has 1 value)
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Sort data
    sorted_df = df.sort_values('norm_delta_f21_mean')
    n = len(sorted_df)
    
    # Create vibrant rainbow colors
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    # Use barplot for clear visibility
    bars = ax.bar(range(n), sorted_df['norm_delta_f21_mean'].values, 
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('Normalized ΔF21 by Phoneme (sorted)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('Norm ΔF21', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_norm_delta_f21_by_phoneme.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 3. Norm ΔF32 by Phoneme - USE BARPLOT with VIBRANT COLORS
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sorted_df = df.sort_values('norm_delta_f32_mean')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    bars = ax.bar(range(n), sorted_df['norm_delta_f32_mean'].values,
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('Normalized ΔF32 by Phoneme (sorted)', fontsize=18, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('Norm ΔF32', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_norm_delta_f32_by_phoneme.png", dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-03_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Formant Spacing Invariance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python formant_spacing_analysis.py --file1 male.wav --file2 female.wav

  # Batch mode: Compare all files in a folder against a reference
  python formant_spacing_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python formant_spacing_analysis.py --golden-compare data/02_cleaned
        """
    )
    
    parser.add_argument('--file1', type=str, help='Path to first audio file')
    parser.add_argument('--file2', type=str, help='Path to second audio file')
    parser.add_argument('--folder', type=str, help='Path to folder containing audio files')
    parser.add_argument('--reference', type=str, help='Path to reference file (for batch mode)')
    parser.add_argument('--golden-compare', type=str, dest='golden_compare',
                        help='Path to cleaned data folder (for golden mode)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default: results/formant_spacing_analysis/{mode})')
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
    
    # Generate default output directory based on mode
    if args.output_dir is None:
        base_dir = "results/formant_spacing_analysis"
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
    print("FORMANT SPACING INVARIANCE ANALYSIS")
    print("=" * 60)
    print("\nHypothesis: Normalized formant spacing (ΔF / GeomMean)")
    print("should be invariant across different speakers")
    print("=" * 60)
    
    if batch_mode:
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        if not os.path.exists(args.reference):
            print(f"Error: Reference file not found: {args.reference}")
            return
        
        print(f"\nMode: BATCH COMPARISON")
        results_df = batch_compare_folder(args.folder, args.reference, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("BATCH ANALYSIS COMPLETE")
            print(f"Files compared: {len(results_df)}")
            
            if 'norm_delta_f21_mean_pct_diff' in results_df.columns:
                best_idx = results_df['norm_delta_f21_mean_pct_diff'].idxmin()
                print(f"\nBest match: {results_df.loc[best_idx, 'filename']}")
                print(f"  Norm ΔF21 diff: {results_df.loc[best_idx, 'norm_delta_f21_mean_pct_diff']:.2f}%")
            
            # Generate visualizations (Figures 1, 2, 3 for spacing analysis)
            if HAS_VISUALIZER and not args.no_visual:
                print(f"\nGenerating visualization figures for {len(results_df) + 1} files...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = [(args.reference, os.path.splitext(os.path.basename(args.reference))[0])]
                for _, row in results_df.iterrows():
                    filename = os.path.splitext(row['filename'])[0]
                    file_path = os.path.join(args.folder, row['filename'])
                    if os.path.exists(file_path):
                        file_list.append((file_path, filename))
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 2, 3])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    elif golden_mode:
        if not os.path.isdir(args.golden_compare):
            print(f"Error: Directory not found: {args.golden_compare}")
            return
        
        print(f"\nMode: GOLDEN FILES COMPARISON")
        results_df = compare_all_golden_files(args.golden_compare, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("GOLDEN FILES ANALYSIS COMPLETE")
            print(f"Phonemes analyzed: {len(results_df)}")
            print(f"\nNorm ΔF21 Range: {results_df['norm_delta_f21_mean'].min():.3f} - {results_df['norm_delta_f21_mean'].max():.3f}")
            print(f"Norm ΔF32 Range: {results_df['norm_delta_f32_mean'].min():.3f} - {results_df['norm_delta_f32_mean'].max():.3f}")
            
            # Generate visualizations (Figures 1, 2, 3 for spacing analysis)
            if HAS_VISUALIZER and not args.no_visual:
                print(f"\nGenerating visualization figures for {len(results_df)} files...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = []
                for _, row in results_df.iterrows():
                    phoneme = row['phoneme']
                    filename = os.path.splitext(row['filename'])[0]
                    subfolder = os.path.join(phoneme, filename)
                    file_list.append((row['filepath'], subfolder))
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 2, 3])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    else:
        if not os.path.exists(args.file1):
            print(f"Error: File not found: {args.file1}")
            return
        if not os.path.exists(args.file2):
            print(f"Error: File not found: {args.file2}")
            return
        
        print(f"\nMode: TWO-FILE COMPARISON")
        results_df = compare_two_files(args.file1, args.file2, output_dir)
        
        if results_df is not None:
            print("\n" + "=" * 60)
            print("RESULTS SUMMARY")
            print("=" * 60)
            print(results_df.to_string(index=False))
            
            # Hypothesis evaluation
            norm_metrics = results_df[results_df['metric'].str.contains('norm')]
            raw_metrics = results_df[results_df['metric'].str.contains('delta_f') & ~results_df['metric'].str.contains('norm')]
            
            avg_norm_pct = norm_metrics['percent_difference'].mean()
            avg_raw_pct = raw_metrics['percent_difference'].mean()
            
            print("\n" + "=" * 60)
            print("HYPOTHESIS EVALUATION")
            print("=" * 60)
            print(f"Average % diff in raw spacing: {avg_raw_pct:.2f}%")
            print(f"Average % diff in normalized spacing: {avg_norm_pct:.2f}%")
            
            if avg_norm_pct < avg_raw_pct:
                print("\n✓ HYPOTHESIS SUPPORTED: Normalized spacing shows better invariance")
            else:
                print("\n✗ HYPOTHESIS NOT SUPPORTED: Normalized spacing does not show better invariance")
            
            # Generate visualizations for both files
            if HAS_VISUALIZER and not args.no_visual:
                print("\nGenerating visualization figures...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = [
                    (args.file1, os.path.splitext(os.path.basename(args.file1))[0]),
                    (args.file2, os.path.splitext(os.path.basename(args.file2))[0])
                ]
                generate_batch_figures(file_list, visual_base, figures=[1, 2, 3])


if __name__ == "__main__":
    main()
