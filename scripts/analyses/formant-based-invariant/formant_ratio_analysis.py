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
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import parselmouth
from parselmouth.praat import call
from pathlib import Path

# Configure matplotlib to use Noto Sans Devanagari for proper script rendering
DEVANAGARI_FONT_PATH = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT_PATH):
    fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']



def extract_formants(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                      max_formant_freq: float = 5500.0, window_length: float = 0.025) -> dict:
    """
    Extract formant frequencies (F1, F2, F3) from an audio file using Praat algorithms.
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis in seconds
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz). Use ~5000 for male, ~5500 for female
        window_length: Analysis window length in seconds
    
    Returns:
        Dictionary containing formant statistics:
        - f1_mean, f2_mean, f3_mean: Mean formant frequencies
        - f1_std, f2_std, f3_std: Standard deviations
        - formant_values: Time-series of formant values
    """
    try:
        # Load the audio file with Praat
        sound = parselmouth.Sound(audio_path)
        
        # Create a Formant object
        formant = call(sound, "To Formant (burg)",
                       time_step,       # Time step
                       max_formants,    # Max number of formants
                       max_formant_freq,  # Maximum formant frequency
                       window_length,   # Window length
                       50.0)            # Pre-emphasis from (Hz)
        
        # Get the number of frames
        n_frames = call(formant, "Get number of frames")
        
        # Extract formant values for each frame
        f1_values = []
        f2_values = []
        f3_values = []
        time_values = []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            # Only include valid (non-undefined) values
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    time_values.append(t)
        
        if len(f1_values) == 0:
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        
        return {
            'f1_mean': np.mean(f1_arr),
            'f2_mean': np.mean(f2_arr),
            'f3_mean': np.mean(f3_arr),
            'f1_median': np.median(f1_arr),
            'f2_median': np.median(f2_arr),
            'f3_median': np.median(f3_arr),
            'f1_std': np.std(f1_arr),
            'f2_std': np.std(f2_arr),
            'f3_std': np.std(f3_arr),
            'f1_values': f1_arr,
            'f2_values': f2_arr,
            'f3_values': f3_arr,
            'time_values': np.array(time_values),
            'n_frames': len(f1_values)
        }
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


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
    
    # Use median values (more robust to outliers)
    f1_med = formant_data['f1_median']
    f2_med = formant_data['f2_median']
    f3_med = formant_data['f3_median']
    
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
        """
    )
    
    parser.add_argument('--file1', type=str, required=True, 
                        help='Path to first audio file')
    parser.add_argument('--file2', type=str, required=True, 
                        help='Path to second audio file')
    parser.add_argument('--output_dir', type=str, 
                        default='results/formant_ratio_analysis',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Validate files exist
    if not os.path.exists(args.file1):
        print(f"Error: File not found: {args.file1}")
        return
    if not os.path.exists(args.file2):
        print(f"Error: File not found: {args.file2}")
        return
    
    print("=" * 60)
    print("FORMANT-BASED INVARIANT ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: Frequency ratios (F1/F2, F2/F3, log ratios)")
    print(f"should be scale-invariant across male/female voices")
    print("=" * 60)
    
    # Run comparison
    results_df = compare_two_files(
        args.file1, args.file2, 
        args.output_dir
    )
    
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


if __name__ == "__main__":
    main()
