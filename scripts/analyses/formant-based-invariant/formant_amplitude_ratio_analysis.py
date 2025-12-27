#!/usr/bin/env python3
"""
Formant-Based Invariant Analysis: Amplitude Ratios

Analyzes formant amplitude ratios from audio files:
- A1/A2, A2/A3 (amplitude at each formant peak)
- H1-H2 (first harmonic vs second)

Hypothesis: Energy distribution patterns are invariant across speakers.
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

# Import visualizer for integration
try:
    from formant_visualizer import generate_all_figures
    HAS_VISUALIZER = True
except ImportError:
    HAS_VISUALIZER = False


def extract_amplitude_ratios(audio_path: str, time_step: float = 0.01, max_formants: int = 5,
                              max_formant_freq: float = 5500.0, window_length: float = 0.025,
                              stability_smoothing: float = 0.1, intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant amplitude ratios (A1/A2, A2/A3) and harmonic ratios (H1-H2).
    
    Refined Method (Method 3):
    - Uses Joint Stability-Intensity Weighting (Intensity^2 / Instability)
    - Computes weighted means for amplitudes and spectral tilt
    """
    try:
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        # Create Formant object
        formant = call(sound, "To Formant (burg)",
                       time_step, max_formants, max_formant_freq, window_length, 50.0)
        
        # Create Intensity object
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        
        # Create Spectrum for harmonic analysis
        spectrum = call(sound, "To Spectrum", "yes")
        
        # Create PointProcess for pitch/harmonics
        pitch = call(sound, "To Pitch", 0.0, 75, 600)
        
        n_frames = call(formant, "Get number of frames")
        
        # Collect data for all frames
        f1_values, f2_values, f3_values = [], [], []
        a1_values, a2_values, a3_values = [], [], []
        time_values, intensity_values = [], []
        h1_h2_values = []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            
            # Get formant frequencies
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            # Get formant bandwidths (used for amplitude estimation)
            b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
            b2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
            b3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")
            
            # Get intensity at this time
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens): intens = 0.0
            except:
                intens = 0.0
            
            # Estimate amplitudes at formant frequencies
            # Using inverse bandwidth as amplitude proxy (narrower = stronger)
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3) and \
               not np.isnan(b1) and not np.isnan(b2) and not np.isnan(b3):
                
                if f1 > 0 and f2 > 0 and f3 > 0 and b1 > 0 and b2 > 0 and b3 > 0:
                    # Amplitude estimation: A ∝ 1/B (narrower bandwidth = higher amplitude)
                    a1 = 1.0 / b1
                    a2 = 1.0 / b2
                    a3 = 1.0 / b3
                    
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    a1_values.append(a1)
                    a2_values.append(a2)
                    a3_values.append(a3)
                    time_values.append(t)
                    intensity_values.append(intens)
                    
                    # H1-H2 calculation (spectral tilt)
                    h1_h2_val = np.nan
                    try:
                        f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
                        if not np.isnan(f0) and f0 > 0:
                            # Get spectral amplitudes at H1 (f0) and H2 (2*f0)
                            h1_amp = call(spectrum, "Get real value in bin", int(f0 * duration))
                            h2_amp = call(spectrum, "Get real value in bin", int(2 * f0 * duration))
                            if h1_amp > 0 and h2_amp > 0:
                                h1_h2_val = 20 * np.log10(h1_amp / h2_amp)
                    except:
                        pass
                    h1_h2_values.append(h1_h2_val)
        
        if len(f1_values) < 3:
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        a1_arr = np.array(a1_values)
        a2_arr = np.array(a2_values)
        a3_arr = np.array(a3_values)
        intensity_arr = np.array(intensity_values)
        time_arr = np.array(time_values)
        h1_h2_arr = np.array(h1_h2_values)
        
        # --- Compute Joint Stability-Intensity Weights ---
        
        # 1. Gradient (dF/dt) using np.gradient for proper edge handling
        df1 = np.abs(np.gradient(f1_arr, time_arr))
        df2 = np.abs(np.gradient(f2_arr, time_arr))
        df3 = np.abs(np.gradient(f3_arr, time_arr))
        
        # Normalized Instability: |dF/dt| / F (dimensionless)
        instability = (df1 / f1_arr) + (df2 / f2_arr) + (df3 / f3_arr)
        
        # 2. Weights - Joint Stability-Intensity
        noise_floor = 50.0
        soft_gate_threshold = 30.0
        intensity_clip_db = 30.0  # Clip to prevent burst dominance
        intensity_exponent = 2.0
        
        intensity_above_floor = np.clip(intensity_arr - noise_floor, 0, intensity_clip_db)
        w_intensity = intensity_above_floor ** intensity_exponent
        w_stability = 1.0 / (instability + stability_smoothing)
        gate_mask = intensity_arr >= soft_gate_threshold
        
        weights = w_intensity * w_stability * gate_mask
        
        if np.sum(weights) == 0:
            weights = np.ones_like(intensity_arr)
            
        # Normalize weights
        weights_norm = weights / np.sum(weights)
        
        # Diagnostics
        sum_w = np.sum(weights)
        sum_w_sq = np.sum(weights**2)
        n_eff = (sum_w**2) / sum_w_sq if sum_w_sq > 0 else 0
        confidence = np.clip(n_eff / len(f1_arr), 0, 1)
        
        # Weight entropy (0=concentrated, 1=uniform)
        p = weights_norm
        weight_entropy = -np.sum(p * np.log(p + 1e-12))
        weight_entropy_norm = weight_entropy / np.log(len(p)) if len(p) > 1 else 0
        
        # Compute amplitude ratios
        a1_a2_ratios = a1_arr / a2_arr
        a2_a3_ratios = a2_arr / a3_arr
        a1_a3_ratios = a1_arr / a3_arr
        
        # Weighted mean for H1-H2 (ignoring NaNs)
        h1_h2_mask = ~np.isnan(h1_h2_arr)
        if np.sum(h1_h2_mask) > 0:
            w_h1h2 = weights[h1_h2_mask]
            v_h1h2 = h1_h2_arr[h1_h2_mask]
            if np.sum(w_h1h2) > 0:
                h1_h2_mean = np.average(v_h1h2, weights=w_h1h2)
            else:
                h1_h2_mean = np.mean(v_h1h2)
            h1_h2_std = np.std(v_h1h2)
            h1_h2_median_unweighted = np.median(v_h1h2)
        else:
            h1_h2_mean = np.nan
            h1_h2_std = np.nan
            h1_h2_median_unweighted = np.nan
        
        return {
            'f1_mean': np.average(f1_arr, weights=weights_norm),
            'f2_mean': np.average(f2_arr, weights=weights_norm),
            'f3_mean': np.average(f3_arr, weights=weights_norm),
            'a1_mean': np.average(a1_arr, weights=weights_norm),
            'a2_mean': np.average(a2_arr, weights=weights_norm),
            'a3_mean': np.average(a3_arr, weights=weights_norm),
            'a1_a2_ratio_mean': np.average(a1_a2_ratios, weights=weights_norm),
            'a2_a3_ratio_mean': np.average(a2_a3_ratios, weights=weights_norm),
            'a1_a3_ratio_mean': np.average(a1_a3_ratios, weights=weights_norm),
            'a1_a2_ratio_median_unweighted': np.median(a1_a2_ratios),
            'a2_a3_ratio_median_unweighted': np.median(a2_a3_ratios),
            'a1_a3_ratio_median_unweighted': np.median(a1_a3_ratios),
            'a1_a2_ratio_std': np.std(a1_a2_ratios),
            'a2_a3_ratio_std': np.std(a2_a3_ratios),
            'h1_h2_mean': h1_h2_mean,
            'h1_h2_median_unweighted': h1_h2_median_unweighted,
            'h1_h2_std': h1_h2_std,
            'log_a1_a2_mean': np.average(np.log(a1_a2_ratios), weights=weights_norm),
            'log_a2_a3_mean': np.average(np.log(a2_a3_ratios), weights=weights_norm),
            'n_frames': len(f1_values),
            'n_eff': n_eff,
            'confidence': confidence,
            'weight_entropy': weight_entropy_norm,
            'duration': duration,
            'frame_weights': weights,
            'frame_weights_norm': weights_norm,
            'a1_a2_ratios': a1_a2_ratios,
            'a2_a3_ratios': a2_a3_ratios,
        }
        
    except Exception as e:
        print(f"Error extracting amplitude ratios from {audio_path}: {e}")
        return None


def analyze_audio_file(audio_path: str) -> dict:
    """Complete amplitude ratio analysis for a single audio file."""
    result = extract_amplitude_ratios(audio_path, max_formant_freq=8000.0)
    
    if result is None:
        return None
    
    return {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        **{k: v for k, v in result.items() 
           if not k.endswith('_ratios') and k != 'stability_weights'},
    }


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """Compare amplitude ratios between two audio files."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nAnalyzing: {os.path.basename(file1_path)}")
    result1 = analyze_audio_file(file1_path)
    
    print(f"Analyzing: {os.path.basename(file2_path)}")
    result2 = analyze_audio_file(file2_path)
    
    if result1 is None or result2 is None:
        print("Error: Could not analyze one or both files")
        return None
    
    metrics = [
        'a1_mean', 'a2_mean', 'a3_mean',
        'a1_a2_ratio_mean', 'a2_a3_ratio_mean', 'a1_a3_ratio_mean',
        'log_a1_a2_mean', 'log_a2_a3_mean',
        'h1_h2_mean', 'h1_h2_median',
    ]
    
    comparison_data = []
    for metric in metrics:
        val1 = result1.get(metric, np.nan)
        val2 = result2.get(metric, np.nan)
        if np.isnan(val1) or np.isnan(val2):
            continue
        diff = abs(val1 - val2)
        pct_diff = (diff / ((abs(val1) + abs(val2)) / 2)) * 100 if (val1 + val2) != 0 else np.nan
        
        comparison_data.append({
            'metric': metric,
            f'{os.path.basename(file1_path)}': val1,
            f'{os.path.basename(file2_path)}': val2,
            'absolute_difference': diff,
            'percent_difference': pct_diff,
        })
    
    df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'amplitude_ratio_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_comparison_plots(result1, result2, output_dir)
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing amplitude ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    color1, color2 = '#FF6B6B', '#4ECDC4'
    
    # Plot 1: Amplitude Values
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(3)
    width = 0.35
    amps1 = [result1['a1_mean'], result1['a2_mean'], result1['a3_mean']]
    amps2 = [result2['a1_mean'], result2['a2_mean'], result2['a3_mean']]
    ax.bar(x - width/2, amps1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, amps2, width, label=file2_name, color=color2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['A1', 'A2', 'A3'], color='#EAEAEA')
    ax.set_title('Formant Amplitudes', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 2: Amplitude Ratios
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    ratios1 = [result1['a1_a2_ratio_mean'], result1['a2_a3_ratio_mean'], result1['a1_a3_ratio_mean']]
    ratios2 = [result2['a1_a2_ratio_mean'], result2['a2_a3_ratio_mean'], result2['a1_a3_ratio_mean']]
    ax.bar(x - width/2, ratios1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, ratios2, width, label=file2_name, color=color2, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['A1/A2', 'A2/A3', 'A1/A3'], color='#EAEAEA')
    ax.set_title('Amplitude Ratios (Energy Distribution)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 3: H1-H2 (Spectral Tilt)
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    h1_h2_1 = result1.get('h1_h2_mean', 0)
    h1_h2_2 = result2.get('h1_h2_mean', 0)
    if not np.isnan(h1_h2_1) and not np.isnan(h1_h2_2):
        ax.bar([0, 1], [h1_h2_1, h1_h2_2], color=[color1, color2], alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([file1_name[:15], file2_name[:15]], color='#EAEAEA', rotation=45)
    ax.set_title('H1-H2 (Spectral Tilt)', color='#EAEAEA', fontweight='bold')
    ax.set_ylabel('dB', color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 4: Log Ratios
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    x2 = np.arange(2)
    log1 = [result1['log_a1_a2_mean'], result1['log_a2_a3_mean']]
    log2 = [result2['log_a1_a2_mean'], result2['log_a2_a3_mean']]
    ax.bar(x2 - width/2, log1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x2 + width/2, log2, width, label=file2_name, color=color2, alpha=0.8)
    ax.set_xticks(x2)
    ax.set_xticklabels(['log(A1/A2)', 'log(A2/A3)'], color='#EAEAEA')
    ax.set_title('Log Amplitude Ratios', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 5: Difference Visualization
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ratio_names = ['A1/A2', 'A2/A3', 'A1/A3', 'log(A1/A2)', 'log(A2/A3)']
    differences = [
        abs(result1['a1_a2_ratio_mean'] - result2['a1_a2_ratio_mean']),
        abs(result1['a2_a3_ratio_mean'] - result2['a2_a3_ratio_mean']),
        abs(result1['a1_a3_ratio_mean'] - result2['a1_a3_ratio_mean']),
        abs(result1['log_a1_a2_mean'] - result2['log_a1_a2_mean']),
        abs(result1['log_a2_a3_mean'] - result2['log_a2_a3_mean']),
    ]
    colors = ['#FF6B6B' if d > 0.1 else '#4ECDC4' for d in differences]
    ax.barh(ratio_names, differences, color=colors, alpha=0.8)
    ax.set_title('Ratio Differences (Lower = More Invariant)', color='#EAEAEA', fontweight='bold')
    ax.axvline(x=0.1, color='#FFD93D', linestyle='--', alpha=0.7)
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 6: Summary
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    summary = f"""
    AMPLITUDE RATIO ANALYSIS
    ========================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    AMPLITUDE RATIOS:
    A1/A2: {result1['a1_a2_ratio_mean']:.4f} vs {result2['a1_a2_ratio_mean']:.4f}
    A2/A3: {result1['a2_a3_ratio_mean']:.4f} vs {result2['a2_a3_ratio_mean']:.4f}
    
    H1-H2 (dB): {result1.get('h1_h2_mean', 'N/A'):.2f} vs {result2.get('h1_h2_mean', 'N/A'):.2f}
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'amplitude_ratio_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")


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
        print(f"Error: Could not analyze reference file")
        return None
    
    all_comparisons = []
    metrics = ['a1_a2_ratio_mean', 'a2_a3_ratio_mean', 'a1_a3_ratio_mean',
               'log_a1_a2_mean', 'log_a2_a3_mean', 'h1_h2_mean']
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        if result is None:
            continue
        
        comparison = {'filename': os.path.basename(wav_file)}
        for metric in metrics:
            val_ref = ref_result.get(metric, np.nan)
            val_file = result.get(metric, np.nan)
            if np.isnan(val_ref) or np.isnan(val_file):
                continue
            diff = abs(val_ref - val_file)
            pct_diff = (diff / ((abs(val_ref) + abs(val_file)) / 2)) * 100 if (val_ref + val_file) != 0 else np.nan
            comparison[f'{metric}'] = val_file
            comparison[f'{metric}_diff'] = diff
            comparison[f'{metric}_pct_diff'] = pct_diff
        
        all_comparisons.append(comparison)
    
    if not all_comparisons:
        return None
    
    df = pd.DataFrame(all_comparisons)
    csv_path = os.path.join(output_dir, 'batch_amplitude_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    return df


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """Find and compare all golden files across phonemes."""
    import glob
    os.makedirs(output_dir, exist_ok=True)
    
    golden_pattern = os.path.join(cleaned_data_dir, '*', '*_golden_*.wav')
    golden_files = glob.glob(golden_pattern)
    
    if not golden_files:
        print(f"Error: No golden files found")
        return None
    
    print(f"\nFound {len(golden_files)} golden files")
    
    all_results = []
    for golden_file in tqdm(sorted(golden_files), desc="Analyzing golden files"):
        phoneme = os.path.basename(os.path.dirname(golden_file))
        result = analyze_audio_file(golden_file)
        if result is None:
            continue
        result['phoneme'] = phoneme
        all_results.append(result)
    
    if not all_results:
        return None
    
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'golden_amplitude_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_golden_plots(df, output_dir)
    return df


def create_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization for golden files comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # Plot 1: A1/A2 by Phoneme
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    phonemes = df['phoneme'].tolist()
    ratios = df['a1_a2_ratio_mean'].tolist()
    ax.barh(range(len(phonemes)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('A1/A2 Ratio', color='#EAEAEA')
    ax.set_title('A1/A2 Ratio by Phoneme', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 2: A2/A3 by Phoneme  
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    ratios = df['a2_a3_ratio_mean'].tolist()
    ax.barh(range(len(phonemes)), ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('A2/A3 Ratio', color='#EAEAEA')
    ax.set_title('A2/A3 Ratio by Phoneme', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 3: H1-H2 by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    h1_h2 = df['h1_h2_mean'].fillna(0).tolist()
    ax.barh(range(len(phonemes)), h1_h2, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('H1-H2 (dB)', color='#EAEAEA')
    ax.set_title('H1-H2 (Spectral Tilt) by Phoneme', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    summary = f"""
    GOLDEN FILES AMPLITUDE ANALYSIS
    ================================
    
    Total phonemes: {len(df)}
    
    RATIO STATISTICS:
    A1/A2 range: {df['a1_a2_ratio_mean'].min():.3f} - {df['a1_a2_ratio_mean'].max():.3f}
    A2/A3 range: {df['a2_a3_ratio_mean'].min():.3f} - {df['a2_a3_ratio_mean'].max():.3f}
    
    ENERGY DISTRIBUTION:
    Higher A1/A2 = More energy in lower formants
    Lower A2/A3 = More energy concentration in mid-formants
    """
    ax.text(0.02, 0.98, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'golden_amplitude_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Formant Amplitude Ratio Analysis: A1/A2, A2/A3, H1-H2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two files
  python formant_amplitude_ratio_analysis.py --file1 male.wav --file2 female.wav

  # Batch mode
  python formant_amplitude_ratio_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison
  python formant_amplitude_ratio_analysis.py --golden-compare data/02_cleaned
        """
    )
    
    parser.add_argument('--file1', type=str, help='First audio file')
    parser.add_argument('--file2', type=str, help='Second audio file')
    parser.add_argument('--folder', type=str, help='Folder containing audio files')
    parser.add_argument('--reference', type=str, help='Reference file for batch mode')
    parser.add_argument('--golden-compare', type=str, dest='golden_compare',
                        help='Cleaned data folder for golden comparison')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-visual', action='store_true', dest='no_visual',
                        help='Skip generating visualization figures')
    
    args = parser.parse_args()
    
    batch_mode = args.folder and args.reference
    single_mode = args.file1 and args.file2
    golden_mode = args.golden_compare is not None
    
    if not batch_mode and not single_mode and not golden_mode:
        print("Error: Specify --file1/--file2, --folder/--reference, or --golden-compare")
        parser.print_help()
        return
    
    if args.output_dir is None:
        base = "results/amplitude_ratio_analysis"
        if golden_mode:
            output_dir = f"{base}/golden"
        elif batch_mode:
            output_dir = f"{base}/batch/{os.path.basename(args.folder)}"
        else:
            output_dir = f"{base}/compare"
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("FORMANT AMPLITUDE RATIO ANALYSIS")
    print("=" * 60)
    print("Hypothesis: Energy distribution patterns (A1/A2, A2/A3, H1-H2)")
    print("are invariant across different speakers")
    print("=" * 60)
    
    if batch_mode:
        results_df = batch_compare_folder(args.folder, args.reference, output_dir)
        if results_df is not None and HAS_VISUALIZER and not args.no_visual:
            print(f"\nGenerating visualization figures for {len(results_df) + 1} files...")
            from formant_visualizer import generate_batch_figures
            visual_base = os.path.join(output_dir, 'visual')
            file_list = [(args.reference, os.path.splitext(os.path.basename(args.reference))[0])]
            for _, row in results_df.iterrows():
                filename = os.path.splitext(row['filename'])[0]
                file_path = os.path.join(args.folder, row['filename'])
                if os.path.exists(file_path):
                    file_list.append((file_path, filename))
            successful = generate_batch_figures(file_list, visual_base, figures=[1, 5, 7])
            print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    elif golden_mode:
        results_df = compare_all_golden_files(args.golden_compare, output_dir)
        if results_df is not None and HAS_VISUALIZER and not args.no_visual:
            print(f"\nGenerating visualization figures for {len(results_df)} files...")
            from formant_visualizer import generate_batch_figures
            visual_base = os.path.join(output_dir, 'visual')
            file_list = []
            for _, row in results_df.iterrows():
                phoneme = row['phoneme']
                filename = os.path.splitext(row['filename'])[0]
                subfolder = os.path.join(phoneme, filename)
                file_list.append((row['file_path'], subfolder))
            successful = generate_batch_figures(file_list, visual_base, figures=[1, 5, 7])
            print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    else:
        results_df = compare_two_files(args.file1, args.file2, output_dir)
        if results_df is not None and HAS_VISUALIZER and not args.no_visual:
            print("\nGenerating visualization figures...")
            from formant_visualizer import generate_batch_figures
            visual_base = os.path.join(output_dir, 'visual')
            file_list = [
                (args.file1, os.path.splitext(os.path.basename(args.file1))[0]),
                (args.file2, os.path.splitext(os.path.basename(args.file2))[0])
            ]
            generate_batch_figures(file_list, visual_base, figures=[1, 5, 7])


if __name__ == "__main__":
    main()
