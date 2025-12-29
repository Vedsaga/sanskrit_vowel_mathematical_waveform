#!/usr/bin/env python3
"""
Spectral Trajectory Convergence Analysis

Analyzes whether spectral features (centroid, spread, rolloff) of one signal
converge toward those of another signal.

Uses permutation test for statistical significance.

Usage:
    python spectral_analysis.py --wave1 ka.wav --wave2 a.wav --visualize
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import Tuple, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from common.config import configure_matplotlib
    configure_matplotlib()
except ImportError:
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


@dataclass
class WaveData:
    samples: np.ndarray
    sample_rate: int
    duration: float
    filepath: str
    name: str
    
    @property
    def time_axis(self) -> np.ndarray:
        return np.linspace(0, self.duration, len(self.samples))


@dataclass 
class ConvergenceResult:
    converges: bool
    p_value: float
    distance_reduction: float
    convergence_start: float
    convergence_duration: float
    min_distance: float
    min_distance_time: float
    evidence: Dict[str, Any]
    verdict: str


def load_wave(filepath: str, trim: bool = True) -> WaveData:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    sr, samples = wavfile.read(filepath)
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float64)
    if np.abs(samples).max() > 0:
        samples = samples / np.abs(samples).max()
    
    if trim:
        frame_size, hop_size = int(0.025 * sr), int(0.010 * sr)
        n_frames = max(1, (len(samples) - frame_size) // hop_size + 1)
        energy = np.array([np.sum(samples[i*hop_size:i*hop_size+frame_size]**2) 
                          for i in range(n_frames)])
        if len(energy) > 0 and energy.max() > 0:
            voiced = energy > energy.max() * 0.0003
            if np.any(voiced):
                idx = np.where(voiced)[0]
                start = max(0, idx[0] - 2) * hop_size
                end = min(len(samples), (idx[-1] + 3) * hop_size)
                samples = samples[start:end]
    
    return WaveData(samples, sr, len(samples)/sr, filepath,
                   os.path.splitext(os.path.basename(filepath))[0])


# =============================================================================
# Spectral Feature Computation
# =============================================================================

def compute_spectral_features(samples: np.ndarray, sr: int,
                               n_fft: int = 1024, hop_length: int = 256) -> Dict:
    """Compute spectral centroid, spread, and rolloff over time."""
    n_frames = max(1, 1 + (len(samples) - n_fft) // hop_length)
    
    centroid = np.zeros(n_frames)
    spread = np.zeros(n_frames)
    rolloff = np.zeros(n_frames)
    flatness = np.zeros(n_frames)
    times = np.zeros(n_frames)
    
    window = scipy_signal.windows.hann(n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        if end > len(samples):
            frame = np.pad(samples[start:], (0, end - len(samples)), mode='constant')
        else:
            frame = samples[start:end]
        
        frame = frame * window
        mag = np.abs(np.fft.rfft(frame))
        
        # Avoid division by zero
        mag_sum = mag.sum()
        if mag_sum < 1e-10:
            times[i] = (start + n_fft / 2) / sr
            continue
        
        # Spectral centroid: weighted mean of frequencies
        centroid[i] = np.sum(freqs * mag) / mag_sum
        
        # Spectral spread: weighted std of frequencies
        spread[i] = np.sqrt(np.sum(((freqs - centroid[i]) ** 2) * mag) / mag_sum)
        
        # Spectral rolloff: frequency below which 85% energy is contained
        cumsum = np.cumsum(mag)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff[i] = freqs[min(rolloff_idx, len(freqs) - 1)]
        
        # Spectral flatness: geometric mean / arithmetic mean
        mag_positive = mag[mag > 0]
        if len(mag_positive) > 0:
            geo_mean = np.exp(np.mean(np.log(mag_positive + 1e-10)))
            arith_mean = np.mean(mag_positive)
            flatness[i] = geo_mean / (arith_mean + 1e-10)
        
        times[i] = (start + n_fft / 2) / sr
    
    return {
        'centroid': centroid,
        'spread': spread,
        'rolloff': rolloff,
        'flatness': flatness,
        'times': times,
        'n_frames': n_frames,
    }


def get_spectral_vector(features: Dict) -> np.ndarray:
    """Stack spectral features into a feature matrix."""
    # Normalize each feature to comparable scale
    c = features['centroid'] / 4000  # Normalize by typical max centroid
    s = features['spread'] / 2000     # Normalize by typical max spread
    r = features['rolloff'] / 8000    # Normalize by typical max rolloff
    f = features['flatness']          # Already 0-1
    
    return np.column_stack([c, s, r, f])


def get_reference_spectral(wave: WaveData) -> np.ndarray:
    """Get stable reference spectral features."""
    features = compute_spectral_features(wave.samples, wave.sample_rate)
    vec = get_spectral_vector(features)
    
    n = len(vec)
    start, end = int(n * 0.2), int(n * 0.8)
    if end <= start:
        start, end = 0, n
    
    return vec[start:end].mean(axis=0)


def compute_spectral_distance(vec_traj: np.ndarray, vec_ref: np.ndarray) -> np.ndarray:
    """Compute spectral feature distance trajectory."""
    return np.array([np.linalg.norm(vec_traj[i] - vec_ref) 
                     for i in range(len(vec_traj))])


def compute_reduction_metric(distances: np.ndarray) -> Tuple[float, int, int]:
    if len(distances) < 2:
        return 0.0, 0, 0
    
    d_diff = np.diff(distances)
    decreasing = d_diff < 0
    
    best_start, best_len = 0, 0
    current_start, current_len = 0, 0
    in_decrease = False
    
    for i, is_dec in enumerate(decreasing):
        if is_dec:
            if not in_decrease:
                current_start = i
                in_decrease = True
            current_len = i - current_start + 1
        else:
            if in_decrease and current_len > best_len:
                best_start = current_start
                best_len = current_len
            in_decrease = False
    
    if in_decrease and current_len > best_len:
        best_start = current_start
        best_len = current_len
    
    if best_len > 0:
        d_start = distances[best_start]
        d_end = distances[min(best_start + best_len, len(distances) - 1)]
        reduction = (d_start - d_end) / (d_start + 1e-10)
    else:
        reduction = 0.0
    
    return reduction, best_start, best_len


def permutation_test(vec_traj: np.ndarray, vec_ref: np.ndarray,
                      observed_reduction: float, n_permutations: int = 1000) -> float:
    n_frames = len(vec_traj)
    if n_frames < 3:
        return 1.0
    
    null_reductions = []
    for _ in range(n_permutations):
        shuffled = vec_traj[np.random.permutation(n_frames)]
        distances = compute_spectral_distance(shuffled, vec_ref)
        red, _, _ = compute_reduction_metric(distances)
        null_reductions.append(red)
    
    return np.mean(np.array(null_reductions) >= observed_reduction)


def analyze_convergence(wave_A: WaveData, wave_B: WaveData,
                        n_permutations: int = 1000) -> ConvergenceResult:
    vec_ref = get_reference_spectral(wave_B)
    
    features_A = compute_spectral_features(wave_A.samples, wave_A.sample_rate)
    vec_A = get_spectral_vector(features_A)
    times = features_A['times']
    
    distances = compute_spectral_distance(vec_A, vec_ref)
    
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_time = times[min_idx] if min_idx < len(times) else 0
    
    distance_reduction, best_start, best_len = compute_reduction_metric(distances)
    
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    convergence_start = times[best_start] if best_len > 0 and best_start < len(times) else 0
    convergence_duration = best_len * dt
    
    p_value = permutation_test(vec_A, vec_ref, distance_reduction, n_permutations)
    
    converges = (p_value < 0.05 and distance_reduction > 0.10)
    
    verdict = (f"SIGNIFICANT: reduction={distance_reduction:.1%}, p={p_value:.4f}" 
               if converges else 
               f"NOT SIGNIFICANT: reduction={distance_reduction:.1%}, p={p_value:.4f}")
    
    return ConvergenceResult(
        converges=converges, p_value=p_value,
        distance_reduction=distance_reduction,
        convergence_start=convergence_start,
        convergence_duration=convergence_duration,
        min_distance=min_distance, min_distance_time=min_time,
        evidence={'features_A': features_A, 'vec_ref': vec_ref,
                  'times': times, 'distances': distances},
        verdict=verdict
    )


def create_visualization(wave_A: WaveData, wave_B: WaveData,
                         result: ConvergenceResult, output_dir: str):
    BG, PANEL, TEXT = '#111111', '#1a1a1a', '#eaeaea'
    CONV, NO_CONV, GRID = '#2ECC71', '#FF6B6B', '#333333'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    
    times = result.evidence['times']
    distances = result.evidence['distances']
    features = result.evidence['features_A']
    
    # Panel 1: Spectral features over time
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    ax.plot(times, features['centroid'], 'r-', label='Centroid', alpha=0.8)
    ax.plot(times, features['rolloff'], 'b-', label='Rolloff', alpha=0.8)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Frequency (Hz)', color=TEXT)
    ax.set_title('Spectral Centroid & Rolloff', color=TEXT, fontweight='bold')
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 2: Distance trajectory
    ax = axes[0, 1]
    ax.set_facecolor(PANEL)
    ax.plot(times, distances, color='#4ECDC4', lw=2)
    ax.scatter([result.min_distance_time], [result.min_distance],
               color=CONV if result.converges else NO_CONV,
               s=150, zorder=5, edgecolors='white', lw=2)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Spectral Distance', color=TEXT)
    ax.set_title('Distance to Reference', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 3: Spread and flatness
    ax = axes[1, 0]
    ax.set_facecolor(PANEL)
    ax2 = ax.twinx()
    ax.plot(times, features['spread'], 'g-', label='Spread', alpha=0.8)
    ax2.plot(times, features['flatness'], 'm-', label='Flatness', alpha=0.8)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Spread (Hz)', color='g')
    ax2.set_ylabel('Flatness', color='m')
    ax.set_title('Spectral Spread & Flatness', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    summary = f"""
SPECTRAL TRAJECTORY CONVERGENCE
{'═' * 40}

Wave A: {wave_A.name}
Wave B: {wave_B.name} (reference)

{'─' * 40}
Distance reduction: {result.distance_reduction:.1%}
P-value:            {result.p_value:.4f}

{'═' * 40}
VERDICT: {status}
{'═' * 40}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', color=TEXT, family='monospace')
    
    fig.suptitle(f'Spectral Convergence: {wave_A.name} → {wave_B.name}',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'spectral_convergence.png')
    plt.savefig(path, dpi=300, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Visualization: {path}")


def save_results(wave_A: WaveData, wave_B: WaveData,
                 result: ConvergenceResult, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    metrics = {
        'wave_A': wave_A.name, 'wave_B': wave_B.name,
        'converges': result.converges, 'p_value': result.p_value,
        'distance_reduction': result.distance_reduction,
        'min_distance': result.min_distance,
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'spectral_metrics.csv'), index=False)
    print(f"Metrics: {output_dir}/spectral_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description='Spectral trajectory convergence analysis')
    parser.add_argument('--wave1', required=True)
    parser.add_argument('--wave2', required=True)
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.add_argument('--n_permutations', type=int, default=1000)
    
    args = parser.parse_args()
    
    if not args.output_dir:
        n1 = os.path.splitext(os.path.basename(args.wave1))[0]
        n2 = os.path.splitext(os.path.basename(args.wave2))[0]
        args.output_dir = f'results/spectral_analysis/{n1}_vs_{n2}'
    
    print(f"\n{'='*60}")
    print("SPECTRAL TRAJECTORY CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    wave_A = load_wave(args.wave1)
    wave_B = load_wave(args.wave2)
    print(f"Trajectory: {wave_A.name}, Reference: {wave_B.name}")
    
    result = analyze_convergence(wave_A, wave_B, args.n_permutations)
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    print(f"\nRESULT: {status}")
    print(f"  Reduction: {result.distance_reduction:.1%}, P-value: {result.p_value:.4f}")
    
    save_results(wave_A, wave_B, result, args.output_dir)
    if args.visualize:
        create_visualization(wave_A, wave_B, result, args.output_dir)
    
    return 0 if result.converges else 1


if __name__ == "__main__":
    sys.exit(main())
