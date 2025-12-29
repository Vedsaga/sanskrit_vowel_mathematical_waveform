#!/usr/bin/env python3
"""
MFCC Trajectory Convergence Analysis

Analyzes whether the MFCC trajectory of one signal converges
toward the MFCC pattern of another signal.

Uses scipy for MFCC computation and permutation test for significance.

Usage:
    python mfcc_similarity.py --wave1 ka.wav --wave2 a.wav --visualize
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal as scipy_signal
from scipy.fftpack import dct
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
# MFCC Computation
# =============================================================================

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def create_mel_filterbank(sr: int, n_fft: int, n_mels: int = 26,
                           fmin: float = 0, fmax: float = None) -> np.ndarray:
    """Create mel filterbank."""
    if fmax is None:
        fmax = sr / 2
    
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        for j in range(left, center):
            if center > left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right > center:
                filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


def compute_mfcc(samples: np.ndarray, sr: int, n_mfcc: int = 13,
                  n_fft: int = 512, hop_length: int = None,
                  n_mels: int = 26) -> Tuple[np.ndarray, np.ndarray]:
    """Compute MFCC features."""
    if hop_length is None:
        hop_length = n_fft // 4
    
    # Pre-emphasis
    samples = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    
    # Frame the signal
    n_frames = 1 + (len(samples) - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1
    
    frames = np.zeros((n_frames, n_fft))
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        if end <= len(samples):
            frames[i] = samples[start:end]
        else:
            frames[i, :len(samples)-start] = samples[start:]
    
    # Apply window
    window = scipy_signal.windows.hann(n_fft)
    frames = frames * window
    
    # FFT
    mag_spec = np.abs(np.fft.rfft(frames, n_fft))
    pow_spec = mag_spec ** 2
    
    # Mel filterbank
    mel_fb = create_mel_filterbank(sr, n_fft, n_mels)
    mel_spec = np.dot(pow_spec, mel_fb.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    
    # Log and DCT
    log_mel = np.log(mel_spec)
    mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    
    # Time axis
    times = np.arange(n_frames) * hop_length / sr
    
    return mfcc, times


def get_reference_mfcc(wave: WaveData, n_mfcc: int = 13) -> np.ndarray:
    """Get stable reference MFCC from middle of wave."""
    mfcc, _ = compute_mfcc(wave.samples, wave.sample_rate, n_mfcc)
    n = len(mfcc)
    start, end = int(n * 0.2), int(n * 0.8)
    if end <= start:
        start, end = 0, n
    return mfcc[start:end].mean(axis=0)


def compute_mfcc_distance(mfcc_traj: np.ndarray, mfcc_ref: np.ndarray) -> np.ndarray:
    """Compute MFCC distance trajectory."""
    return np.array([np.linalg.norm(mfcc_traj[i] - mfcc_ref) 
                     for i in range(len(mfcc_traj))])


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


def permutation_test(mfcc_traj: np.ndarray, mfcc_ref: np.ndarray,
                      observed_reduction: float, n_permutations: int = 1000) -> float:
    n_frames = len(mfcc_traj)
    if n_frames < 3:
        return 1.0
    
    null_reductions = []
    for _ in range(n_permutations):
        shuffled = mfcc_traj[np.random.permutation(n_frames)]
        distances = compute_mfcc_distance(shuffled, mfcc_ref)
        red, _, _ = compute_reduction_metric(distances)
        null_reductions.append(red)
    
    return np.mean(np.array(null_reductions) >= observed_reduction)


def analyze_convergence(wave_A: WaveData, wave_B: WaveData,
                        n_permutations: int = 1000) -> ConvergenceResult:
    n_mfcc = 13
    
    mfcc_ref = get_reference_mfcc(wave_B, n_mfcc)
    mfcc_A, times = compute_mfcc(wave_A.samples, wave_A.sample_rate, n_mfcc)
    
    distances = compute_mfcc_distance(mfcc_A, mfcc_ref)
    
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_time = times[min_idx] if min_idx < len(times) else 0
    
    distance_reduction, best_start, best_len = compute_reduction_metric(distances)
    
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    convergence_start = times[best_start] if best_len > 0 and best_start < len(times) else 0
    convergence_duration = best_len * dt
    
    p_value = permutation_test(mfcc_A, mfcc_ref, distance_reduction, n_permutations)
    
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
        evidence={'mfcc_A': mfcc_A, 'mfcc_ref': mfcc_ref, 
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
    mfcc_A = result.evidence['mfcc_A']
    
    # Panel 1: MFCC spectrogram
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    im = ax.imshow(mfcc_A.T, aspect='auto', origin='lower',
                   extent=[times[0], times[-1], 0, mfcc_A.shape[1]],
                   cmap='viridis')
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('MFCC Coefficient', color=TEXT)
    ax.set_title('MFCC Trajectory', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    plt.colorbar(im, ax=ax)
    
    # Panel 2: Distance trajectory
    ax = axes[0, 1]
    ax.set_facecolor(PANEL)
    ax.plot(times, distances, color='#4ECDC4', lw=2)
    ax.scatter([result.min_distance_time], [result.min_distance],
               color=CONV if result.converges else NO_CONV,
               s=150, zorder=5, edgecolors='white', lw=2)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('MFCC Distance', color=TEXT)
    ax.set_title('Distance to Reference MFCC', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 3: First 3 MFCCs over time
    ax = axes[1, 0]
    ax.set_facecolor(PANEL)
    ax.plot(times, mfcc_A[:, 0], 'r-', label='MFCC 0', alpha=0.8)
    ax.plot(times, mfcc_A[:, 1], 'g-', label='MFCC 1', alpha=0.8)
    ax.plot(times, mfcc_A[:, 2], 'b-', label='MFCC 2', alpha=0.8)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Value', color=TEXT)
    ax.set_title('First 3 MFCC Coefficients', color=TEXT, fontweight='bold')
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    summary = f"""
MFCC TRAJECTORY CONVERGENCE
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
    
    fig.suptitle(f'MFCC Convergence: {wave_A.name} → {wave_B.name}',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'mfcc_convergence.png')
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
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'mfcc_metrics.csv'), index=False)
    print(f"Metrics: {output_dir}/mfcc_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description='MFCC trajectory convergence analysis')
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
        args.output_dir = f'results/mfcc_analysis/{n1}_vs_{n2}'
    
    print(f"\n{'='*60}")
    print("MFCC TRAJECTORY CONVERGENCE ANALYSIS")
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
