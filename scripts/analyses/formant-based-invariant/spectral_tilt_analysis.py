#!/usr/bin/env python3
"""
Formant-Based Invariant Analysis: Spectral Tilt

This script analyzes spectral tilt (overall slope of spectral envelope) from audio files.
Measures dB/octave from F1 to F3.

Hypothesis: Open vowels (/a/) have flatter tilt than closed vowels (/i/)

Modes:
- Two-file comparison: Compare spectral tilt between two audio files
- Folder batch: Compare all files in a folder against a reference file
- Golden comparison: Compare all golden files across phonemes
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

# Configure matplotlib for Devanagari
DEVANAGARI_FONT_PATH = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT_PATH):
    fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


def extract_formants_with_amplitudes(audio_path: str, time_step: float = 0.01, 
                                      max_formants: int = 5, max_formant_freq: float = 5500.0,
                                      window_length: float = 0.025, stability_smoothing: float = 0.1,
                                      intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant frequencies and their amplitudes from an audio file.
    
    Refined Method (Method 3):
    - Uses Joint Stability-Intensity Weighting (Intensity^2 / Instability)
    - Computes weighted means for formants and bandwidths
    """
    try:
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        # Create Formant object
        formant = call(sound, "To Formant (burg)",
                       time_step, max_formants, max_formant_freq, window_length, 50.0)
        
        # Create Intensity object
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        
        # Create Spectrum for spectral analysis
        spectrum = call(sound, "To Spectrum", "yes")
        
        n_frames = call(formant, "Get number of frames")
        
        # Collect formant frequencies and bandwidths
        f1_values, f2_values, f3_values = [], [], []
        b1_values, b2_values, b3_values = [], [], []
        time_values, intensity_values = [], []
        
        for i in range(1, n_frames + 1):
            t = call(formant, "Get time from frame number", i)
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            
            # Get bandwidths
            b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
            b2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
            b3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")
            
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens): intens = 0.0
            except:
                intens = 0.0
            
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    b1_values.append(b1 if not np.isnan(b1) else 100)
                    b2_values.append(b2 if not np.isnan(b2) else 100)
                    b3_values.append(b3 if not np.isnan(b3) else 100)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        if len(f1_values) < 3:
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        b1_arr = np.array(b1_values)
        b2_arr = np.array(b2_values)
        b3_arr = np.array(b3_values)
        intensity_arr = np.array(intensity_values)
        time_arr = np.array(time_values)
        
        # --- Compute Joint Stability-Intensity Weights ---
        
        # 1. Gradient (dF/dt)
        dt = np.diff(time_arr)
        dt = np.where(dt == 0, 1e-6, dt)
        
        df1 = np.abs(np.diff(f1_arr)) / dt
        df2 = np.abs(np.diff(f2_arr)) / dt
        df3 = np.abs(np.diff(f3_arr)) / dt
        
        # Pad derivatives
        df1 = np.append(df1, df1[-1])
        df2 = np.append(df2, df2[-1])
        df3 = np.append(df3, df3[-1])
        
        # Normalized Instability: |dF/dt| / F
        instability = (df1 / f1_arr) + (df2 / f2_arr) + (df3 / f3_arr)
        
        # 2. Weights
        noise_floor = 50.0
        soft_gate_threshold = 30.0
        
        w_intensity = np.maximum(0, intensity_arr - noise_floor) ** 2
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
        
        # Weighted means
        f1_mean = np.average(f1_arr, weights=weights_norm)
        f2_mean = np.average(f2_arr, weights=weights_norm)
        f3_mean = np.average(f3_arr, weights=weights_norm)
        b1_mean = np.average(b1_arr, weights=weights_norm)
        b2_mean = np.average(b2_arr, weights=weights_norm)
        b3_mean = np.average(b3_arr, weights=weights_norm)
        
        return {
            'f1_mean': f1_mean, 'f2_mean': f2_mean, 'f3_mean': f3_mean,
            'f1_median': np.median(f1_arr), 'f2_median': np.median(f2_arr), 'f3_median': np.median(f3_arr),
            'b1_mean': b1_mean, 'b2_mean': b2_mean, 'b3_mean': b3_mean,
            'f1_values': f1_arr, 'f2_values': f2_arr, 'f3_values': f3_arr,
            'b1_values': b1_arr, 'b2_values': b2_arr, 'b3_values': b3_arr,
            'time_values': time_arr,
            'stability_weights': weights,
            'n_eff': n_eff,
            'confidence': confidence,
            'n_frames': len(f1_values),
            'duration': duration
        }
        
    except Exception as e:
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def compute_spectral_tilt(formant_data: dict) -> dict:
    """
    Compute spectral tilt (dB/octave) from formant data.
    
    Spectral tilt measures how quickly amplitude decreases with frequency.
    Flatter tilt = more energy in higher frequencies (open vowels like /a/)
    Steeper tilt = less energy in higher frequencies (closed vowels like /i/)
    
    Returns:
        Dictionary containing spectral tilt metrics
    """
    if formant_data is None:
        return None
    
    f1 = formant_data['f1_mean']
    f2 = formant_data['f2_mean']
    f3 = formant_data['f3_mean']
    b1 = formant_data['b1_mean']
    b2 = formant_data['b2_mean']
    b3 = formant_data['b3_mean']
    
    # Estimate relative amplitudes from formant frequencies and bandwidths
    # Using simplified model: A ∝ 1/B (narrower bandwidth = higher amplitude)
    a1_est = 1.0 / b1 if b1 > 0 else 0
    a2_est = 1.0 / b2 if b2 > 0 else 0
    a3_est = 1.0 / b3 if b3 > 0 else 0
    
    # Normalize to A1
    if a1_est > 0:
        a2_rel = a2_est / a1_est
        a3_rel = a3_est / a1_est
    else:
        a2_rel = a2_est
        a3_rel = a3_est
    
    # Convert to dB
    a2_db = 20 * np.log10(a2_rel) if a2_rel > 0 else -60
    a3_db = 20 * np.log10(a3_rel) if a3_rel > 0 else -60
    
    # Calculate octaves between formants
    octaves_f1_f2 = np.log2(f2 / f1) if f1 > 0 else 1
    octaves_f2_f3 = np.log2(f3 / f2) if f2 > 0 else 1
    octaves_f1_f3 = np.log2(f3 / f1) if f1 > 0 else 1
    
    # Spectral tilt (dB/octave)
    tilt_f1_f2 = a2_db / octaves_f1_f2 if octaves_f1_f2 != 0 else 0
    tilt_f2_f3 = (a3_db - a2_db) / octaves_f2_f3 if octaves_f2_f3 != 0 else 0
    tilt_f1_f3 = a3_db / octaves_f1_f3 if octaves_f1_f3 != 0 else 0
    
    # Alternative tilt measure using bandwidth ratio (simpler, more robust)
    bw_ratio_f2_f1 = b2 / b1 if b1 > 0 else 1
    bw_ratio_f3_f2 = b3 / b2 if b2 > 0 else 1
    bw_ratio_f3_f1 = b3 / b1 if b1 > 0 else 1
    
    # Per-frame tilt calculations
    f1_vals = formant_data['f1_values']
    f2_vals = formant_data['f2_values']
    f3_vals = formant_data['f3_values']
    b1_vals = formant_data['b1_values']
    b2_vals = formant_data['b2_values']
    b3_vals = formant_data['b3_values']
    
    frame_tilts = []
    for i in range(len(f1_vals)):
        if b1_vals[i] > 0 and b3_vals[i] > 0 and f1_vals[i] > 0 and f3_vals[i] > 0:
            a1_f = 1.0 / b1_vals[i]
            a3_f = 1.0 / b3_vals[i]
            if a1_f > 0:
                a3_rel_f = a3_f / a1_f
                a3_db_f = 20 * np.log10(a3_rel_f) if a3_rel_f > 0 else -60
                oct_f = np.log2(f3_vals[i] / f1_vals[i])
                if oct_f > 0:
                    frame_tilts.append(a3_db_f / oct_f)
    
    frame_tilts = np.array(frame_tilts) if frame_tilts else np.array([0])
    
    return {
        'tilt_f1_f2': tilt_f1_f2,
        'tilt_f2_f3': tilt_f2_f3,
        'tilt_f1_f3': tilt_f1_f3,
        'tilt_mean': tilt_f1_f3,  # Primary metric
        'tilt_std': np.std(frame_tilts),
        'bw_ratio_f2_f1': bw_ratio_f2_f1,
        'bw_ratio_f3_f2': bw_ratio_f3_f2,
        'bw_ratio_f3_f1': bw_ratio_f3_f1,
        'a2_db': a2_db,
        'a3_db': a3_db,
        'octaves_f1_f3': octaves_f1_f3,
        'frame_tilts': frame_tilts
    }


def analyze_audio_file(audio_path: str) -> dict:
    """Complete spectral tilt analysis for a single audio file."""
    max_freq = 8000.0
    
    formant_data = extract_formants_with_amplitudes(audio_path, max_formant_freq=max_freq)
    
    if formant_data is None:
        return None
    
    tilt_data = compute_spectral_tilt(formant_data)
    
    if tilt_data is None:
        return None
    
    result = {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        'f1_mean': formant_data['f1_mean'],
        'f2_mean': formant_data['f2_mean'],
        'f3_mean': formant_data['f3_mean'],
        'b1_mean': formant_data['b1_mean'],
        'b2_mean': formant_data['b2_mean'],
        'b3_mean': formant_data['b3_mean'],
        'duration': formant_data['duration'],
        **{k: v for k, v in tilt_data.items() if not k.startswith('frame_')}
    }
    
    result['tilt_std'] = tilt_data['tilt_std']
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """Compare spectral tilt between two audio files."""
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
        'tilt_f1_f2', 'tilt_f2_f3', 'tilt_f1_f3',
        'bw_ratio_f2_f1', 'bw_ratio_f3_f2', 'bw_ratio_f3_f1',
        'a2_db', 'a3_db'
    ]
    
    comparison_data = []
    for metric in metrics:
        val1 = result1.get(metric, np.nan)
        val2 = result2.get(metric, np.nan)
        diff = val1 - val2
        abs_diff = abs(diff)
        
        comparison_data.append({
            'metric': metric,
            f'{os.path.basename(file1_path)}': val1,
            f'{os.path.basename(file2_path)}': val2,
            'difference': diff,
            'absolute_difference': abs_diff,
        })
    
    df = pd.DataFrame(comparison_data)
    
    csv_path = os.path.join(output_dir, 'spectral_tilt_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization plots comparing spectral tilt between two files."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor('#111111')
    
    file1_name = os.path.basename(result1['filename'])
    file2_name = os.path.basename(result2['filename'])
    
    color1 = '#FF6B6B'
    color2 = '#4ECDC4'
    
    # 1. Spectral Tilt Comparison
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    x = np.arange(3)
    width = 0.35
    
    tilts1 = [result1['tilt_f1_f2'], result1['tilt_f2_f3'], result1['tilt_f1_f3']]
    tilts2 = [result2['tilt_f1_f2'], result2['tilt_f2_f3'], result2['tilt_f1_f3']]
    
    ax.bar(x - width/2, tilts1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, tilts2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Frequency Range', color='#EAEAEA')
    ax.set_ylabel('Tilt (dB/octave)', color='#EAEAEA')
    ax.set_title('Spectral Tilt Comparison', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['F1→F2', 'F2→F3', 'F1→F3'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.axhline(y=0, color='#666', linestyle='--', alpha=0.5)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 2. Bandwidth Ratios
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    bw1 = [result1['bw_ratio_f2_f1'], result1['bw_ratio_f3_f2'], result1['bw_ratio_f3_f1']]
    bw2 = [result2['bw_ratio_f2_f1'], result2['bw_ratio_f3_f2'], result2['bw_ratio_f3_f1']]
    
    ax.bar(x - width/2, bw1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x + width/2, bw2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Bandwidth Ratio', color='#EAEAEA')
    ax.set_ylabel('Ratio Value', color='#EAEAEA')
    ax.set_title('Bandwidth Ratios (Tilt Indicator)', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['B2/B1', 'B3/B2', 'B3/B1'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 3. Relative Amplitudes
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    x2 = np.arange(2)
    
    amps1 = [result1['a2_db'], result1['a3_db']]
    amps2 = [result2['a2_db'], result2['a3_db']]
    
    ax.bar(x2 - width/2, amps1, width, label=file1_name, color=color1, alpha=0.8)
    ax.bar(x2 + width/2, amps2, width, label=file2_name, color=color2, alpha=0.8)
    
    ax.set_xlabel('Formant', color='#EAEAEA')
    ax.set_ylabel('Relative Amplitude (dB re F1)', color='#EAEAEA')
    ax.set_title('Relative Formant Amplitudes', color='#EAEAEA', fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(['A2', 'A3'], color='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 4. Spectral Envelope Visualization
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    # File 1 envelope
    freqs1 = [result1['f1_mean'], result1['f2_mean'], result1['f3_mean']]
    amps1_norm = [0, result1['a2_db'], result1['a3_db']]
    ax.plot(freqs1, amps1_norm, 'o-', color=color1, markersize=10, linewidth=2, label=file1_name)
    
    # File 2 envelope
    freqs2 = [result2['f1_mean'], result2['f2_mean'], result2['f3_mean']]
    amps2_norm = [0, result2['a2_db'], result2['a3_db']]
    ax.plot(freqs2, amps2_norm, 's-', color=color2, markersize=10, linewidth=2, label=file2_name)
    
    ax.set_xlabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_ylabel('Relative Amplitude (dB)', color='#EAEAEA')
    ax.set_title('Spectral Envelope (F1 = 0 dB)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ['bottom', 'left', 'top', 'right']:
        ax.spines[spine].set_color('#333')
    
    # 5. Tilt Difference
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    
    tilt_diff = result1['tilt_f1_f3'] - result2['tilt_f1_f3']
    color = '#FF6B6B' if tilt_diff > 0 else '#4ECDC4'
    ax.barh(['Tilt Difference\n(File1 - File2)'], [tilt_diff], color=color, alpha=0.8)
    ax.axvline(x=0, color='#666', linestyle='--')
    
    ax.set_xlabel('dB/octave', color='#EAEAEA')
    ax.set_title('Spectral Tilt Difference (F1→F3)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # Add interpretation
    if tilt_diff > 0:
        ax.text(0.5, 0.2, f'{file1_name} has STEEPER tilt\n(more closed vowel-like)',
                transform=ax.transAxes, ha='center', color='#EAEAEA', fontsize=9)
    else:
        ax.text(0.5, 0.2, f'{file1_name} has FLATTER tilt\n(more open vowel-like)',
                transform=ax.transAxes, ha='center', color='#EAEAEA', fontsize=9)
    
    # 6. Summary
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    SPECTRAL TILT ANALYSIS
    ======================
    
    File 1: {file1_name}
    File 2: {file2_name}
    
    FORMANTS (Hz):
    ─────────────────────────────
    F1: {result1['f1_mean']:.0f} vs {result2['f1_mean']:.0f}
    F2: {result1['f2_mean']:.0f} vs {result2['f2_mean']:.0f}
    F3: {result1['f3_mean']:.0f} vs {result2['f3_mean']:.0f}
    
    SPECTRAL TILT (dB/octave):
    ─────────────────────────────
    F1→F3: {result1['tilt_f1_f3']:.2f} vs {result2['tilt_f1_f3']:.2f}
    
    INTERPRETATION:
    ─────────────────────────────
    Flatter tilt → Open vowel (/a/)
    Steeper tilt → Closed vowel (/i/)
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'spectral_tilt_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """Compare all audio files in a folder against a pinned reference file."""
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
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        
        if result is None:
            continue
        
        successful_results.append(result)
        
        comparison = {'filename': os.path.basename(wav_file)}
        
        for metric in ['tilt_f1_f3', 'tilt_f1_f2', 'tilt_f2_f3', 'bw_ratio_f3_f1']:
            val_ref = ref_result.get(metric, np.nan)
            val_file = result.get(metric, np.nan)
            diff = val_file - val_ref
            
            comparison[metric] = val_file
            comparison[f'{metric}_diff'] = diff
        
        all_comparisons.append(comparison)
    
    if not all_comparisons:
        print("Error: No files could be analyzed")
        return None
    
    df = pd.DataFrame(all_comparisons)
    
    csv_path = os.path.join(output_dir, 'batch_spectral_tilt.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    create_batch_plots(ref_result, successful_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    ref_color = '#FFD93D'
    other_color = '#4ECDC4'
    
    # 1. Tilt Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_tilt = ref_result['tilt_f1_f3']
    other_tilts = [r['tilt_f1_f3'] for r in all_results]
    
    ax.axhline(y=ref_tilt, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_tilts)), other_tilts, c=other_color, s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Spectral Tilt (dB/octave)', color='#EAEAEA')
    ax.set_title('Spectral Tilt vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 2. Tilt Histogram
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ax.hist(other_tilts, bins=15, color=other_color, alpha=0.7, edgecolor='white')
    ax.axvline(x=ref_tilt, color=ref_color, linestyle='--', linewidth=2, label=f'Reference: {ref_tilt:.2f}')
    
    ax.set_xlabel('Spectral Tilt (dB/octave)', color='#EAEAEA')
    ax.set_ylabel('Count', color='#EAEAEA')
    ax.set_title('Distribution of Spectral Tilts', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 3. Tilt Difference Boxplot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    diff_cols = ['tilt_f1_f3_diff', 'tilt_f1_f2_diff', 'tilt_f2_f3_diff']
    diff_labels = ['F1→F3', 'F1→F2', 'F2→F3']
    
    box_data = [comparison_df[col].dropna().values for col in diff_cols if col in comparison_df.columns]
    
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, labels=diff_labels[:len(box_data)])
        for patch in bp['boxes']:
            patch.set_facecolor(other_color)
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            for item in bp[element]:
                item.set_color('#EAEAEA')
    
    ax.set_xlabel('Frequency Range', color='#EAEAEA')
    ax.set_ylabel('Tilt Difference from Reference', color='#EAEAEA')
    ax.set_title('Distribution of Tilt Differences', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.axhline(y=0, color='#666', linestyle='--', alpha=0.5)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    n_files = len(all_results)
    mean_tilt = np.mean(other_tilts)
    std_tilt = np.std(other_tilts)
    
    summary_text = f"""
    BATCH SPECTRAL TILT ANALYSIS
    ============================
    
    Reference: {ref_name}
    Reference Tilt: {ref_tilt:.2f} dB/octave
    
    Files analyzed: {n_files}
    
    TILT STATISTICS:
    ────────────────────────────
    Mean: {mean_tilt:.2f} dB/octave
    Std Dev: {std_tilt:.2f}
    Min: {min(other_tilts):.2f}
    Max: {max(other_tilts):.2f}
    
    INTERPRETATION:
    ────────────────────────────
    Flatter (less negative) → Open vowel
    Steeper (more negative) → Closed vowel
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_spectral_tilt.png')
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
    
    csv_path = os.path.join(output_dir, 'golden_spectral_tilt.csv')
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
    
    # 1. Spectral Tilt by Phoneme (sorted)
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    df_sorted = df.sort_values('tilt_f1_f3')
    phonemes = df_sorted['phoneme'].tolist()
    tilts = df_sorted['tilt_f1_f3'].tolist()
    
    bars = ax.barh(range(len(phonemes)), tilts, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Spectral Tilt (dB/octave)', color='#EAEAEA', fontsize=11)
    ax.set_title('Spectral Tilt by Phoneme (F1→F3)', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    ax.axvline(x=0, color='#666', linestyle='--', alpha=0.5)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 2. F1 vs Tilt (testing open/closed hypothesis)
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['f1_mean'], row['tilt_f1_f3'],
                   c=[colors[i]], s=150, alpha=0.8,
                   edgecolors='white', linewidths=1)
        ax.annotate(row['phoneme'], (row['f1_mean'], row['tilt_f1_f3']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10, color='#EAEAEA')
    
    ax.set_xlabel('F1 (Hz) - Higher = More Open', color='#EAEAEA', fontsize=11)
    ax.set_ylabel('Spectral Tilt (dB/octave)', color='#EAEAEA', fontsize=11)
    ax.set_title('F1 vs Spectral Tilt\n(Testing: Open vowels have flatter tilt)', 
                 color='#EAEAEA', fontweight='bold', fontsize=11)
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ['bottom', 'left', 'top', 'right']:
        ax.spines[spine].set_color('#333')
    
    # 3. Bandwidth Ratio (B3/B1) by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    df_sorted_bw = df.sort_values('bw_ratio_f3_f1')
    phonemes_bw = df_sorted_bw['phoneme'].tolist()
    bw_ratios = df_sorted_bw['bw_ratio_f3_f1'].tolist()
    
    bars = ax.barh(range(len(phonemes_bw)), bw_ratios, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes_bw)))
    ax.set_yticklabels(phonemes_bw, fontsize=10)
    ax.set_xlabel('Bandwidth Ratio (B3/B1)', color='#EAEAEA', fontsize=11)
    ax.set_title('Bandwidth Ratio by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#333')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    # 4. Summary
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    # Find most open (flattest) and most closed (steepest)
    flattest_idx = df['tilt_f1_f3'].idxmax()
    steepest_idx = df['tilt_f1_f3'].idxmin()
    
    summary_lines = []
    for _, row in df.head(15).iterrows():
        summary_lines.append(f"{row['phoneme']}: {row['tilt_f1_f3']:.2f} dB/oct, F1={row['f1_mean']:.0f}Hz")
    
    summary_text = f"""
    GOLDEN FILES SPECTRAL TILT
    ==========================
    
    Total phonemes: {len(df)}
    
    HYPOTHESIS TEST:
    ────────────────────────────
    Open vowels should have FLATTER tilt
    Closed vowels should have STEEPER tilt
    
    RESULTS:
    ────────────────────────────
    Flattest: {df.loc[flattest_idx, 'phoneme']} ({df.loc[flattest_idx, 'tilt_f1_f3']:.2f} dB/oct)
    Steepest: {df.loc[steepest_idx, 'phoneme']} ({df.loc[steepest_idx, 'tilt_f1_f3']:.2f} dB/oct)
    
    TILT RANGE:
    ────────────────────────────
    Min: {df['tilt_f1_f3'].min():.2f} dB/octave
    Max: {df['tilt_f1_f3'].max():.2f} dB/octave
    Mean: {df['tilt_f1_f3'].mean():.2f} dB/octave
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_spectral_tilt.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn-based visualizations."""
    
    sns.set_style("dark")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#111111')
    
    # 1. Scatter with regression line
    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    
    sns.regplot(x='f1_mean', y='tilt_f1_f3', data=df, ax=ax,
                scatter_kws={'s': 100, 'alpha': 0.7, 'color': '#4ECDC4'},
                line_kws={'color': '#FF6B6B', 'linewidth': 2})
    
    for _, row in df.iterrows():
        ax.annotate(row['phoneme'], (row['f1_mean'], row['tilt_f1_f3']),
                    xytext=(3, 3), textcoords='offset points',
                    fontsize=9, color='#EAEAEA')
    
    ax.set_xlabel('F1 (Hz)', color='#EAEAEA')
    ax.set_ylabel('Spectral Tilt (dB/octave)', color='#EAEAEA')
    ax.set_title('F1 vs Spectral Tilt with Regression', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Correlation heatmap
    ax = axes[1]
    ax.set_facecolor('#1a1a1a')
    
    corr_cols = ['f1_mean', 'f2_mean', 'f3_mean', 'tilt_f1_f3', 'bw_ratio_f3_f1']
    corr_matrix = df[corr_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, cbar_kws={'label': 'Correlation'},
                xticklabels=['F1', 'F2', 'F3', 'Tilt', 'BW Ratio'],
                yticklabels=['F1', 'F2', 'F3', 'Tilt', 'BW Ratio'])
    
    ax.set_title('Correlation Matrix', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_spectral_tilt_seaborn.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualization saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Spectral Tilt Analysis: Measure spectral envelope slope (dB/octave)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python spectral_tilt_analysis.py --file1 open_a.wav --file2 closed_i.wav

  # Batch mode: Compare all files in a folder against a reference file
  python spectral_tilt_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python spectral_tilt_analysis.py --golden-compare data/02_cleaned

Hypothesis: Open vowels (/a/) have flatter tilt than closed vowels (/i/)
        """
    )
    
    parser.add_argument('--file1', type=str,
                        help='Path to first audio file (for single comparison mode)')
    parser.add_argument('--file2', type=str,
                        help='Path to second audio file (for single comparison mode)')
    
    parser.add_argument('--folder', type=str,
                        help='Path to folder containing audio files (for batch mode)')
    parser.add_argument('--reference', type=str,
                        help='Path to reference/pinned file to compare against (for batch mode)')
    
    parser.add_argument('--golden-compare', type=str, dest='golden_compare',
                        help='Path to cleaned data folder to compare all golden files')
    
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
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
        base_dir = "results/spectral_tilt_analysis"
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
    print("SPECTRAL TILT ANALYSIS")
    print("=" * 60)
    print(f"\nHypothesis: Open vowels have FLATTER spectral tilt")
    print(f"            Closed vowels have STEEPER spectral tilt")
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
            
            # Generate visualizations (Figures 1, 5, 6 for spectral tilt)
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
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 5, 6])
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
            
            # Check hypothesis
            corr = results_df['f1_mean'].corr(results_df['tilt_f1_f3'])
            print(f"\nF1 vs Tilt Correlation: {corr:.3f}")
            if corr > 0:
                print("✓ HYPOTHESIS SUPPORTED: Higher F1 (open vowels) correlates with flatter tilt")
            else:
                print("✗ HYPOTHESIS NOT SUPPORTED: Correlation is negative or zero")
            
            # Generate visualizations (Figures 1, 5, 6 for spectral tilt)
            if HAS_VISUALIZER and not args.no_visual and len(results_df) > 0:
                print(f"\nGenerating visualization figures for {len(results_df)} files...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = []
                for _, row in results_df.iterrows():
                    phoneme = row['phoneme']
                    filename = os.path.splitext(row['filename'])[0]
                    subfolder = os.path.join(phoneme, filename)
                    file_list.append((row['file_path'], subfolder))
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 5, 6])
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
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = [
                    (args.file1, os.path.splitext(os.path.basename(args.file1))[0]),
                    (args.file2, os.path.splitext(os.path.basename(args.file2))[0])
                ]
                generate_batch_figures(file_list, visual_base, figures=[1, 5, 6])


if __name__ == "__main__":
    main()
