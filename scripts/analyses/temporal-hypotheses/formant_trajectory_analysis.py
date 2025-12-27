#!/usr/bin/env python3
"""
Temporal/Dynamic Hypothesis Analysis: Formant Trajectory Shape

This script analyzes formant trajectory dynamics to test Hypothesis #6:
"Vowels have characteristic settling patterns"

Key metrics computed:
- Rate of formant change: dF1/dt, dF2/dt (Hz/s)
- Curvature in F1-F2 plane during production
- Settling time estimation
- Trajectory length and smoothness

The script supports three analysis modes:
1. Single comparison: Compare two audio files
2. Batch mode: Compare all files in a folder against a reference
3. Golden mode: Compare all golden files across phonemes
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

# Configure matplotlib to use Noto Sans Devanagari for proper script rendering
DEVANAGARI_FONT_PATH = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT_PATH):
    fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


def extract_formant_trajectory(audio_path: str, time_step: float = 0.005, 
                                max_formants: int = 5, max_formant_freq: float = 5500.0,
                                window_length: float = 0.025, 
                                intensity_threshold: float = 50.0) -> dict:
    """
    Extract formant trajectory data from an audio file.
    
    Unlike static formant extraction, this preserves the full time-series
    for trajectory analysis including derivatives.
    
    Args:
        audio_path: Path to the audio file
        time_step: Time step for analysis (smaller = higher resolution for derivatives)
        max_formants: Maximum number of formants to extract
        max_formant_freq: Maximum formant frequency (Hz)
        window_length: Analysis window length in seconds
        intensity_threshold: Minimum intensity (dB) for frame inclusion
    
    Returns:
        Dictionary containing formant time-series and trajectory metrics
    """
    try:
        # Load the audio file with Praat
        sound = parselmouth.Sound(audio_path)
        duration = sound.get_total_duration()
        
        # Create a Formant object with finer time resolution
        formant = call(sound, "To Formant (burg)",
                       time_step,
                       max_formants,
                       max_formant_freq,
                       window_length,
                       50.0)
        
        # Create Intensity object for low-energy filtering
        intensity = call(sound, "To Intensity", 100, time_step, "yes")
        
        # Get the number of frames
        n_frames = call(formant, "Get number of frames")
        
        # Collect formant values for all frames
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
            
            # Get intensity at this time
            try:
                intens = call(intensity, "Get value at time", t, "Cubic")
                if np.isnan(intens):
                    intens = 0.0
            except:
                intens = 60.0
            
            # Only include valid values
            if not np.isnan(f1) and not np.isnan(f2) and not np.isnan(f3):
                if f1 > 0 and f2 > 0 and f3 > 0:
                    f1_values.append(f1)
                    f2_values.append(f2)
                    f3_values.append(f3)
                    time_values.append(t)
                    intensity_values.append(intens)
        
        if len(f1_values) < 5:  # Need at least 5 frames for trajectory analysis
            return None
        
        f1_arr = np.array(f1_values)
        f2_arr = np.array(f2_values)
        f3_arr = np.array(f3_values)
        time_arr = np.array(time_values)
        intensity_arr = np.array(intensity_values)
        
        # Filter out low-intensity frames
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
        print(f"Error extracting formants from {audio_path}: {e}")
        return None


def compute_trajectory_metrics(formant_data: dict) -> dict:
    """
    Compute trajectory-specific metrics from formant time-series.
    
    Metrics include:
    - First derivatives: dF1/dt, dF2/dt (rate of change)
    - Second derivatives: d²F1/dt², d²F2/dt² (acceleration)
    - Curvature in F1-F2 plane
    - Trajectory length and smoothness
    - Settling time estimation
    
    Args:
        formant_data: Dictionary from extract_formant_trajectory()
    
    Returns:
        Dictionary containing trajectory metrics
    """
    if formant_data is None:
        return None
    
    f1 = formant_data['f1_values']
    f2 = formant_data['f2_values']
    f3 = formant_data['f3_values']
    t = formant_data['time_values']
    
    n = len(f1)
    if n < 5:
        return None
    
    # === First Derivatives (velocity) ===
    # Using numpy gradient for proper edge handling
    df1_dt = np.gradient(f1, t)  # Hz per second
    df2_dt = np.gradient(f2, t)
    df3_dt = np.gradient(f3, t)
    
    # === Second Derivatives (acceleration) ===
    d2f1_dt2 = np.gradient(df1_dt, t)
    d2f2_dt2 = np.gradient(df2_dt, t)
    
    # === Velocity magnitude in F1-F2 space ===
    velocity_magnitude = np.sqrt(df1_dt**2 + df2_dt**2)
    
    # === Curvature in F1-F2 plane ===
    # κ = |x'y'' - y'x''| / (x'² + y'²)^(3/2)
    # where x = F1, y = F2
    numerator = np.abs(df1_dt * d2f2_dt2 - df2_dt * d2f1_dt2)
    denominator = (df1_dt**2 + df2_dt**2)**(3/2)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = np.where(denominator > 1e-10, numerator / denominator, 0)
    
    # === Trajectory Length in F1-F2 space ===
    # Sum of Euclidean distances between consecutive points
    df1_steps = np.diff(f1)
    df2_steps = np.diff(f2)
    step_lengths = np.sqrt(df1_steps**2 + df2_steps**2)
    trajectory_length = np.sum(step_lengths)
    
    # === Smoothness Index ===
    # Lower jerk (derivative of acceleration) = smoother trajectory
    jerk_f1 = np.gradient(d2f1_dt2, t)
    jerk_f2 = np.gradient(d2f2_dt2, t)
    jerk_magnitude = np.sqrt(jerk_f1**2 + jerk_f2**2)
    smoothness_index = 1.0 / (1.0 + np.mean(jerk_magnitude) / 1e6)  # Normalized 0-1
    
    # === Settling Time Estimation ===
    # Find when velocity drops below 10% of max velocity
    max_velocity = np.max(velocity_magnitude)
    settling_threshold = 0.1 * max_velocity
    
    # Find first time velocity stays below threshold
    settled_mask = velocity_magnitude < settling_threshold
    if np.any(settled_mask):
        # Find the first sustained period below threshold
        settling_idx = np.argmax(settled_mask)
        settling_time = t[settling_idx] - t[0]
    else:
        settling_time = t[-1] - t[0]  # Never fully settled
    
    # === Steady-state detection ===
    # Middle 50% of the signal is considered potential steady-state
    mid_start = n // 4
    mid_end = 3 * n // 4
    steady_state_velocity = np.mean(velocity_magnitude[mid_start:mid_end])
    
    # === Direction changes (trajectory complexity) ===
    # Count sign changes in velocity
    f1_direction_changes = np.sum(np.diff(np.sign(df1_dt)) != 0)
    f2_direction_changes = np.sum(np.diff(np.sign(df2_dt)) != 0)
    
    return {
        # First derivatives (velocity)
        'df1_dt_mean': np.mean(np.abs(df1_dt)),
        'df2_dt_mean': np.mean(np.abs(df2_dt)),
        'df3_dt_mean': np.mean(np.abs(df3_dt)),
        'df1_dt_max': np.max(np.abs(df1_dt)),
        'df2_dt_max': np.max(np.abs(df2_dt)),
        'df1_dt_std': np.std(df1_dt),
        'df2_dt_std': np.std(df2_dt),
        
        # Velocity magnitude
        'velocity_mean': np.mean(velocity_magnitude),
        'velocity_max': np.max(velocity_magnitude),
        'velocity_std': np.std(velocity_magnitude),
        
        # Curvature metrics
        'curvature_mean': np.mean(curvature),
        'curvature_max': np.max(curvature),
        'curvature_std': np.std(curvature),
        
        # Trajectory shape metrics
        'trajectory_length': trajectory_length,
        'smoothness_index': smoothness_index,
        'settling_time': settling_time,
        'steady_state_velocity': steady_state_velocity,
        
        # Trajectory complexity
        'f1_direction_changes': f1_direction_changes,
        'f2_direction_changes': f2_direction_changes,
        'total_direction_changes': f1_direction_changes + f2_direction_changes,
        
        # Raw arrays for visualization
        'df1_dt': df1_dt,
        'df2_dt': df2_dt,
        'curvature': curvature,
        'velocity_magnitude': velocity_magnitude,
        
        # Basic formant stats for reference
        'f1_mean': np.mean(f1),
        'f2_mean': np.mean(f2),
        'f3_mean': np.mean(f3),
        'f1_std': np.std(f1),
        'f2_std': np.std(f2),
    }


def analyze_audio_file(audio_path: str) -> dict:
    """
    Complete trajectory analysis for a single audio file.
    
    Args:
        audio_path: Path to the audio file
    
    Returns:
        Dictionary with all trajectory metrics
    """
    # Use higher max_formant_freq to avoid artificial limits
    max_freq = 8000.0
    
    formant_data = extract_formant_trajectory(audio_path, max_formant_freq=max_freq)
    
    if formant_data is None:
        return None
    
    trajectory = compute_trajectory_metrics(formant_data)
    
    if trajectory is None:
        return None
    
    # Combine results, excluding raw arrays for CSV output
    result = {
        'file_path': audio_path,
        'filename': os.path.basename(audio_path),
        'n_frames': formant_data['n_frames'],
        'duration': formant_data['duration'],
    }
    
    # Add trajectory metrics (excluding raw arrays)
    for key, value in trajectory.items():
        if not isinstance(value, np.ndarray):
            result[key] = value
    
    # Store raw data for visualization
    result['_formant_data'] = formant_data
    result['_trajectory_data'] = trajectory
    
    return result


def compare_two_files(file1_path: str, file2_path: str, output_dir: str) -> pd.DataFrame:
    """
    Compare formant trajectories between two audio files.
    
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
    
    # Trajectory metrics to compare
    metrics = [
        'df1_dt_mean', 'df2_dt_mean', 'df1_dt_max', 'df2_dt_max',
        'velocity_mean', 'velocity_max',
        'curvature_mean', 'curvature_max',
        'trajectory_length', 'smoothness_index', 'settling_time',
        'f1_direction_changes', 'f2_direction_changes',
        'f1_mean', 'f2_mean', 'f3_mean'
    ]
    
    # Build comparison table
    comparison_rows = []
    for metric in metrics:
        val1 = result1.get(metric, np.nan)
        val2 = result2.get(metric, np.nan)
        diff = abs(val1 - val2)
        pct_diff = (diff / ((val1 + val2) / 2)) * 100 if (val1 + val2) != 0 else np.nan
        
        comparison_rows.append({
            'metric': metric,
            'file1': val1,
            'file2': val2,
            'difference': diff,
            'percent_difference': pct_diff
        })
    
    df = pd.DataFrame(comparison_rows)
    
    # Save results
    csv_path = os.path.join(output_dir, 'trajectory_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_comparison_plots(result1, result2, output_dir)
    
    return df


def create_comparison_plots(result1: dict, result2: dict, output_dir: str):
    """Create visualization comparing trajectories between two files."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor('#111111')
    
    name1 = os.path.basename(result1['filename'])
    name2 = os.path.basename(result2['filename'])
    
    # Colors
    color1 = '#FF6B6B'  # Coral for file 1
    color2 = '#4ECDC4'  # Teal for file 2
    
    # Get formant data
    data1 = result1['_formant_data']
    data2 = result2['_formant_data']
    traj1 = result1['_trajectory_data']
    traj2 = result2['_trajectory_data']
    
    # 1. F1-F2 Trajectory Plot
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ax.plot(data1['f2_values'], data1['f1_values'], 'o-', color=color1, 
            alpha=0.7, markersize=4, linewidth=1.5, label=name1)
    ax.plot(data2['f2_values'], data2['f1_values'], 's-', color=color2, 
            alpha=0.7, markersize=4, linewidth=1.5, label=name2)
    
    # Mark start and end
    ax.scatter(data1['f2_values'][0], data1['f1_values'][0], 
               color=color1, s=100, marker='>', zorder=5, edgecolors='white')
    ax.scatter(data2['f2_values'][0], data2['f1_values'][0], 
               color=color2, s=100, marker='>', zorder=5, edgecolors='white')
    
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('F2 (Hz)', color='#EAEAEA')
    ax.set_ylabel('F1 (Hz)', color='#EAEAEA')
    ax.set_title('F1-F2 Trajectory', color='#EAEAEA', fontweight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Velocity over time
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    # Normalize time to 0-100% for comparison
    t1_norm = np.linspace(0, 100, len(traj1['velocity_magnitude']))
    t2_norm = np.linspace(0, 100, len(traj2['velocity_magnitude']))
    
    ax.plot(t1_norm, traj1['velocity_magnitude'], color=color1, linewidth=2, label=name1)
    ax.plot(t2_norm, traj2['velocity_magnitude'], color=color2, linewidth=2, label=name2)
    
    ax.set_xlabel('Time (%)', color='#EAEAEA')
    ax.set_ylabel('Velocity (Hz/s)', color='#EAEAEA')
    ax.set_title('Formant Velocity Magnitude', color='#EAEAEA', fontweight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. Curvature over time
    ax = axes[0, 2]
    ax.set_facecolor('#1a1a1a')
    
    ax.plot(t1_norm, traj1['curvature'], color=color1, linewidth=2, label=name1)
    ax.plot(t2_norm, traj2['curvature'], color=color2, linewidth=2, label=name2)
    
    ax.set_xlabel('Time (%)', color='#EAEAEA')
    ax.set_ylabel('Curvature', color='#EAEAEA')
    ax.set_title('F1-F2 Trajectory Curvature', color='#EAEAEA', fontweight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. dF1/dt comparison
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    ax.plot(t1_norm, traj1['df1_dt'], color=color1, linewidth=2, label=name1)
    ax.plot(t2_norm, traj2['df1_dt'], color=color2, linewidth=2, label=name2)
    ax.axhline(y=0, color='#666', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (%)', color='#EAEAEA')
    ax.set_ylabel('dF1/dt (Hz/s)', color='#EAEAEA')
    ax.set_title('F1 Rate of Change', color='#EAEAEA', fontweight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 5. dF2/dt comparison
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    
    ax.plot(t1_norm, traj1['df2_dt'], color=color1, linewidth=2, label=name1)
    ax.plot(t2_norm, traj2['df2_dt'], color=color2, linewidth=2, label=name2)
    ax.axhline(y=0, color='#666', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Time (%)', color='#EAEAEA')
    ax.set_ylabel('dF2/dt (Hz/s)', color='#EAEAEA')
    ax.set_title('F2 Rate of Change', color='#EAEAEA', fontweight='bold')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 6. Summary metrics comparison
    ax = axes[1, 2]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_text = f"""
    TRAJECTORY COMPARISON SUMMARY
    ═════════════════════════════
    
    File 1: {name1}
    File 2: {name2}
    
    VELOCITY (Hz/s):
    ─────────────────────────────
    Mean:  {result1['velocity_mean']:.1f} vs {result2['velocity_mean']:.1f}
    Max:   {result1['velocity_max']:.1f} vs {result2['velocity_max']:.1f}
    
    CURVATURE:
    ─────────────────────────────
    Mean:  {result1['curvature_mean']:.4f} vs {result2['curvature_mean']:.4f}
    Max:   {result1['curvature_max']:.4f} vs {result2['curvature_max']:.4f}
    
    TRAJECTORY SHAPE:
    ─────────────────────────────
    Length:     {result1['trajectory_length']:.1f} vs {result2['trajectory_length']:.1f}
    Smoothness: {result1['smoothness_index']:.3f} vs {result2['smoothness_index']:.3f}
    Settling:   {result1['settling_time']*1000:.1f}ms vs {result2['settling_time']*1000:.1f}ms
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'trajectory_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")


def batch_compare_folder(folder_path: str, reference_file: str, output_dir: str) -> pd.DataFrame:
    """
    Compare all audio files in a folder against a reference file.
    
    Args:
        folder_path: Path to folder containing audio files
        reference_file: Path to the reference (pinned) file
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
    
    # Compare each file against reference
    all_comparisons = []
    successful_results = []
    
    for wav_file in tqdm(wav_files, desc="Analyzing files"):
        # Skip reference file
        if os.path.abspath(wav_file) == os.path.abspath(reference_file):
            continue
        
        result = analyze_audio_file(wav_file)
        
        if result is None:
            continue
        
        successful_results.append(result)
        
        # Compute comparison metrics
        metrics = [
            'df1_dt_mean', 'df2_dt_mean', 'df1_dt_max', 'df2_dt_max',
            'velocity_mean', 'velocity_max',
            'curvature_mean', 'curvature_max',
            'trajectory_length', 'smoothness_index', 'settling_time',
            'f1_mean', 'f2_mean', 'f3_mean'
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
    
    # Save detailed results
    csv_path = os.path.join(output_dir, 'batch_trajectory_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nDetailed results saved to: {csv_path}")
    
    # Create summary statistics
    summary_metrics = ['velocity_mean', 'curvature_mean', 'trajectory_length', 
                       'smoothness_index', 'settling_time']
    
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
    summary_path = os.path.join(output_dir, 'batch_trajectory_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Create batch visualization
    create_batch_plots(ref_result, successful_results, df, output_dir)
    
    return df


def create_batch_plots(ref_result: dict, all_results: list, comparison_df: pd.DataFrame, output_dir: str):
    """Create visualization for batch trajectory comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#111111')
    
    ref_name = os.path.basename(ref_result['filename'])
    
    # Colors
    ref_color = '#FFD93D'  # Gold for reference
    other_color = '#4ECDC4'  # Teal for others
    
    # 1. Velocity Distribution
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    ref_velocity = ref_result['velocity_mean']
    other_velocities = [r['velocity_mean'] for r in all_results]
    
    ax.axhline(y=ref_velocity, color=ref_color, linestyle='--', linewidth=2, 
               label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_velocities)), other_velocities, c=other_color, 
               s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Mean Velocity (Hz/s)', color='#EAEAEA')
    ax.set_title('Trajectory Velocity vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Curvature Distribution
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    ref_curvature = ref_result['curvature_mean']
    other_curvatures = [r['curvature_mean'] for r in all_results]
    
    ax.axhline(y=ref_curvature, color=ref_color, linestyle='--', linewidth=2, 
               label=f'Reference: {ref_name}')
    ax.scatter(range(len(other_curvatures)), other_curvatures, c=other_color, 
               s=50, alpha=0.7, label='Other files')
    
    ax.set_xlabel('File Index', color='#EAEAEA')
    ax.set_ylabel('Mean Curvature', color='#EAEAEA')
    ax.set_title('Trajectory Curvature vs Reference', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.grid(True, alpha=0.2, color='#444')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. Percent Difference Boxplot
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    pct_diff_cols = ['velocity_mean_pct_diff', 'curvature_mean_pct_diff', 
                     'trajectory_length_pct_diff', 'smoothness_index_pct_diff', 
                     'settling_time_pct_diff']
    pct_diff_labels = ['Velocity', 'Curvature', 'Length', 'Smoothness', 'Settling']
    
    box_data = [comparison_df[col].dropna().values for col in pct_diff_cols if col in comparison_df.columns]
    
    if box_data:
        bp = ax.boxplot(box_data, patch_artist=True, labels=pct_diff_labels[:len(box_data)])
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
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary Statistics
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
    BATCH TRAJECTORY COMPARISON
    ═══════════════════════════
    
    Reference: {ref_name}
    Files analyzed: {n_files}
    
    AVERAGE % DIFFERENCE FROM REFERENCE:
    ─────────────────────────────────────
    {chr(10).join(f'    {line}' for line in stats_lines)}
    
    INTERPRETATION:
    ─────────────────────────────────────
    Lower % difference = More similar
    trajectory dynamics across speakers.
    """
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'batch_trajectory_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Batch visualization saved to: {plot_path}")


def compare_all_golden_files(cleaned_data_dir: str, output_dir: str) -> pd.DataFrame:
    """
    Find and compare all golden files across different phonemes.
    
    Args:
        cleaned_data_dir: Path to the cleaned data directory
        output_dir: Directory to save results
    
    Returns:
        DataFrame with all golden file trajectory data
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
    
    # Create DataFrame (exclude internal data)
    df_data = []
    for r in all_results:
        row = {k: v for k, v in r.items() if not k.startswith('_')}
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save results
    csv_path = os.path.join(output_dir, 'golden_trajectory_comparison.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Create visualization
    create_golden_comparison_plots(df, all_results, output_dir)
    
    return df


def create_golden_comparison_plots(df: pd.DataFrame, all_results: list, output_dir: str):
    """Create visualization comparing trajectories of all golden files."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#111111')
    
    # Use colormap for different phonemes
    n_phonemes = len(df)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_phonemes))
    
    # 1. Velocity by Phoneme
    ax = axes[0, 0]
    ax.set_facecolor('#1a1a1a')
    
    phonemes = df['phoneme'].tolist()
    velocities = df['velocity_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), velocities, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Mean Velocity (Hz/s)', color='#EAEAEA', fontsize=11)
    ax.set_title('Formant Velocity by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 2. Curvature by Phoneme
    ax = axes[0, 1]
    ax.set_facecolor('#1a1a1a')
    
    curvatures = df['curvature_mean'].tolist()
    
    bars = ax.barh(range(len(phonemes)), curvatures, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Mean Curvature', color='#EAEAEA', fontsize=11)
    ax.set_title('Trajectory Curvature by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 3. Settling Time by Phoneme
    ax = axes[1, 0]
    ax.set_facecolor('#1a1a1a')
    
    settling_times = [s * 1000 for s in df['settling_time'].tolist()]  # Convert to ms
    
    bars = ax.barh(range(len(phonemes)), settling_times, color=colors, alpha=0.8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=10)
    ax.set_xlabel('Settling Time (ms)', color='#EAEAEA', fontsize=11)
    ax.set_title('Formant Settling Time by Phoneme', color='#EAEAEA', fontweight='bold', fontsize=12)
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.set_facecolor('#1a1a1a')
    ax.axis('off')
    
    summary_lines = []
    for _, row in df.iterrows():
        summary_lines.append(
            f"{row['phoneme']}: vel={row['velocity_mean']:.0f}, "
            f"curv={row['curvature_mean']:.4f}, "
            f"settle={row['settling_time']*1000:.1f}ms"
        )
    
    summary_text = f"""
    GOLDEN FILES TRAJECTORY COMPARISON
    ═══════════════════════════════════
    
    Total phonemes: {len(df)}
    
    TRAJECTORY SUMMARY:
    ─────────────────────────────────────
{chr(10).join(f'    {line}' for line in summary_lines[:15])}
    {'... and more' if len(summary_lines) > 15 else ''}
    
    STATISTICS:
    ─────────────────────────────────────
    Velocity range: {df['velocity_mean'].min():.0f} - {df['velocity_mean'].max():.0f} Hz/s
    Curvature range: {df['curvature_mean'].min():.4f} - {df['curvature_mean'].max():.4f}
    Settling range: {df['settling_time'].min()*1000:.1f} - {df['settling_time'].max()*1000:.1f} ms
    """
    
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='#EAEAEA',
            bbox=dict(boxstyle='round', facecolor='#1a1a1a', edgecolor='#333'))
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'golden_trajectory_comparison.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111', bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {plot_path}")
    
    # Create additional seaborn plots if available
    if HAS_SEABORN:
        create_seaborn_golden_plots(df, output_dir)


def create_seaborn_golden_plots(df: pd.DataFrame, output_dir: str):
    """Create enhanced seaborn visualizations for golden file trajectories."""
    
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
    
    # 1. Velocity vs Curvature scatter
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sns.scatterplot(data=df, x='velocity_mean', y='curvature_mean', 
                    hue='phoneme', style='phoneme', s=250, palette='bright', 
                    legend='brief', ax=ax, edgecolor='white', linewidth=0.5)
    
    ax.set_title('Trajectory Dynamics: Velocity vs Curvature', fontsize=18, 
                 color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Mean Velocity (Hz/s)', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('Mean Curvature', fontsize=13, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_velocity_vs_curvature.png", dpi=300, 
                facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 2. Settling time vs Smoothness
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    df['settling_time_ms'] = df['settling_time'] * 1000
    
    sns.scatterplot(data=df, x='settling_time_ms', y='smoothness_index', 
                    hue='phoneme', style='phoneme', s=250, palette='bright', 
                    legend='brief', ax=ax, edgecolor='white', linewidth=0.5)
    
    ax.set_title('Settling Patterns: Time vs Smoothness', fontsize=18, 
                 color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Settling Time (ms)', fontsize=13, color=TEXT_COLOR)
    ax.set_ylabel('Smoothness Index', fontsize=13, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, alpha=0.15, color='white')
    
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9,
                       facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_settling_vs_smoothness.png", dpi=300, 
                facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    # 3. Trajectory Length by Phoneme (sorted bar)
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    sorted_df = df.sort_values('trajectory_length')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    
    bars = ax.bar(range(n), sorted_df['trajectory_length'].values,
                  color=rainbow_colors, edgecolor='white', linewidth=1.5, alpha=0.9)
    
    ax.set_xticks(range(n))
    ax.set_xticklabels(sorted_df['phoneme'].values, fontsize=11, rotation=45, ha='right')
    ax.set_title('Trajectory Length by Phoneme (F1-F2 path length)', fontsize=18, 
                 color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel('Phoneme', fontsize=14, color=TEXT_COLOR)
    ax.set_ylabel('Trajectory Length (Hz)', fontsize=14, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=11)
    ax.grid(True, axis='y', alpha=0.15, color='white')
    
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_trajectory_length.png", dpi=300, 
                facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Seaborn visualizations saved to: {output_dir}/01-03_*.png")


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Hypothesis Analysis: Formant Trajectory Shape',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two specific files
  python formant_trajectory_analysis.py --file1 male.wav --file2 female.wav

  # Custom output directory
  python formant_trajectory_analysis.py --file1 audio1.wav --file2 audio2.wav --output_dir ./results

  # Batch mode: Compare all files in a folder against a reference file
  python formant_trajectory_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

  # Golden comparison mode: Compare all golden files across phonemes
  python formant_trajectory_analysis.py --golden-compare data/02_cleaned
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
                        help='Path to reference file to compare against (for batch mode)')
    
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
        base_dir = "results/formant_trajectory_analysis"
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
    print("TEMPORAL HYPOTHESIS ANALYSIS: FORMANT TRAJECTORY SHAPE")
    print("=" * 60)
    print(f"\nHypothesis: Vowels have characteristic 'settling' patterns")
    print(f"Metrics: dF1/dt, dF2/dt, curvature, trajectory length")
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
            
            if 'velocity_mean_pct_diff' in results_df.columns:
                best_idx = results_df['velocity_mean_pct_diff'].idxmin()
                worst_idx = results_df['velocity_mean_pct_diff'].idxmax()
                
                print(f"\nMost similar trajectory: {results_df.loc[best_idx, 'filename']}")
                print(f"  Velocity diff: {results_df.loc[best_idx, 'velocity_mean_pct_diff']:.2f}%")
                
                print(f"\nMost different trajectory: {results_df.loc[worst_idx, 'filename']}")
                print(f"  Velocity diff: {results_df.loc[worst_idx, 'velocity_mean_pct_diff']:.2f}%")
    
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
            
            print(f"\nVelocity range: {results_df['velocity_mean'].min():.0f} - {results_df['velocity_mean'].max():.0f} Hz/s")
            print(f"Curvature range: {results_df['curvature_mean'].min():.4f} - {results_df['curvature_mean'].max():.4f}")
            print(f"Settling time range: {results_df['settling_time'].min()*1000:.1f} - {results_df['settling_time'].max()*1000:.1f} ms")
    
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
            
            # Evaluate trajectory similarity
            trajectory_metrics = results_df[results_df['metric'].isin([
                'velocity_mean', 'curvature_mean', 'trajectory_length', 
                'smoothness_index', 'settling_time'
            ])]
            
            avg_pct_diff = trajectory_metrics['percent_difference'].mean()
            
            print(f"\nAverage % difference in trajectory metrics: {avg_pct_diff:.2f}%")
            
            if avg_pct_diff < 20:
                print("\n✓ SIMILAR TRAJECTORIES: Vowels show similar settling patterns")
            elif avg_pct_diff < 40:
                print("\n~ MODERATE DIFFERENCES: Some variation in settling patterns")
            else:
                print("\n✗ DIFFERENT TRAJECTORIES: Vowels have distinct settling patterns")


if __name__ == "__main__":
    main()
