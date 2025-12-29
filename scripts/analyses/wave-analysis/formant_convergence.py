#!/usr/bin/env python3
"""
Formant Trajectory Convergence Analysis

Analyzes whether the formant trajectory (F1, F2, F3) of one signal
converges toward the formant pattern of another signal.

Uses LPC-based formant extraction and permutation test for significance.

Usage:
    python formant_convergence.py --wave1 ka.wav --wave2 a.wav --visualize
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
from typing import Tuple, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from common.config import configure_matplotlib
    configure_matplotlib()
except ImportError:
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


# =============================================================================
# Data Classes
# =============================================================================

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


# =============================================================================
# Wave Loading
# =============================================================================

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
        # Simple energy-based trimming
        frame_size = int(0.025 * sr)
        hop_size = int(0.010 * sr)
        n_frames = max(1, (len(samples) - frame_size) // hop_size + 1)
        energy = np.array([np.sum(samples[i*hop_size:i*hop_size+frame_size]**2) 
                          for i in range(n_frames)])
        if len(energy) > 0 and energy.max() > 0:
            threshold = energy.max() * 0.0003  # -35dB
            voiced = energy > threshold
            if np.any(voiced):
                idx = np.where(voiced)[0]
                start = max(0, idx[0] - 2) * hop_size
                end = min(len(samples), (idx[-1] + 3) * hop_size)
                samples = samples[start:end]
    
    return WaveData(samples, sr, len(samples)/sr, filepath,
                   os.path.splitext(os.path.basename(filepath))[0])


# =============================================================================
# LPC-based Formant Extraction
# =============================================================================

def apply_preemphasis(samples: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Pre-emphasis filter."""
    return np.append(samples[0], samples[1:] - coeff * samples[:-1])


def compute_lpc_coefficients(frame: np.ndarray, order: int = 12) -> np.ndarray:
    """Compute LPC coefficients using autocorrelation method."""
    frame = frame - np.mean(frame)
    frame = apply_preemphasis(frame)
    
    n = len(frame)
    r = np.correlate(frame, frame, mode='full')[n-1:n+order+1]
    
    if r[0] <= 0:
        return np.zeros(order)
    
    # Levinson-Durbin
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    
    for i in range(1, order + 1):
        lambda_i = sum(a[j] * r[i - j] for j in range(1, i))
        if e <= 1e-10:
            break
        lambda_i = (r[i] - lambda_i) / e
        if abs(lambda_i) >= 1.0:
            lambda_i = 0.99 * np.sign(lambda_i)
        
        a_new = a.copy()
        a_new[i] = lambda_i
        for j in range(1, i):
            a_new[j] = a[j] - lambda_i * a[i - j]
        a = a_new
        e = e * (1 - lambda_i * lambda_i)
        if e <= 0:
            break
    
    return a[1:]


def extract_formants(lpc_coeffs: np.ndarray, sr: int, n_formants: int = 3) -> np.ndarray:
    """Extract formant frequencies from LPC coefficients."""
    poly = np.concatenate([[1], lpc_coeffs])
    roots = np.roots(poly)
    
    formants = []
    for root in roots:
        if np.abs(root) >= 1.0:
            continue
        angle = np.angle(root)
        if angle <= 0:
            continue
        freq = angle * sr / (2 * np.pi)
        if 200 < freq < 5000:
            formants.append(freq)
    
    formants = np.sort(formants)[:n_formants]
    
    # Pad if we don't have enough formants
    while len(formants) < n_formants:
        formants = np.append(formants, 0)
    
    return formants


def compute_formant_trajectory(wave: WaveData, window_ms: float = 25.0,
                                hop_ms: float = 10.0, lpc_order: int = 12) -> Dict:
    """Compute formant trajectory over time."""
    sr = wave.sample_rate
    frame_size = int(window_ms * sr / 1000)
    hop_size = int(hop_ms * sr / 1000)
    window = scipy_signal.windows.hann(frame_size)
    
    n_frames = max(1, (len(wave.samples) - frame_size) // hop_size + 1)
    
    F1 = np.zeros(n_frames)
    F2 = np.zeros(n_frames)
    F3 = np.zeros(n_frames)
    times = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        
        if end > len(wave.samples):
            break
        
        frame = wave.samples[start:end] * window
        
        # Skip low energy frames
        if np.sum(frame**2) < 1e-6:
            continue
        
        lpc = compute_lpc_coefficients(frame, lpc_order)
        formants = extract_formants(lpc, sr, n_formants=3)
        
        F1[i], F2[i], F3[i] = formants
        times[i] = (start + frame_size / 2) / sr
    
    return {
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'formants': np.column_stack([F1, F2, F3]),
        'times': times,
        'n_frames': n_frames,
    }


def get_reference_formants(wave: WaveData) -> np.ndarray:
    """Get stable reference formants from middle of wave."""
    traj = compute_formant_trajectory(wave)
    F = traj['formants']
    
    # Use middle 60% to avoid transients
    n = len(F)
    start, end = int(n * 0.2), int(n * 0.8)
    if end <= start:
        start, end = 0, n
    
    # Average formants in stable region
    F_ref = F[start:end].mean(axis=0)
    
    return F_ref


# =============================================================================
# Convergence Analysis with Permutation Test
# =============================================================================

def compute_formant_distance(F_traj: np.ndarray, F_ref: np.ndarray) -> np.ndarray:
    """Compute distance from each time point to reference formants."""
    # Normalize by typical formant ranges (F1~500Hz, F2~1500Hz, F3~2500Hz)
    scale = np.array([500, 1500, 2500])
    
    distances = np.zeros(len(F_traj))
    for i in range(len(F_traj)):
        diff = (F_traj[i] - F_ref) / scale
        distances[i] = np.linalg.norm(diff)
    
    return distances


def compute_reduction_metric(distances: np.ndarray) -> Tuple[float, int, int]:
    """Compute distance reduction and find best convergence region."""
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


def permutation_test(F_traj: np.ndarray, F_ref: np.ndarray,
                      observed_reduction: float, n_permutations: int = 1000) -> float:
    """Permutation test for significance."""
    n_frames = len(F_traj)
    if n_frames < 3:
        return 1.0
    
    null_reductions = []
    
    for _ in range(n_permutations):
        shuffled_idx = np.random.permutation(n_frames)
        F_shuffled = F_traj[shuffled_idx]
        
        distances = compute_formant_distance(F_shuffled, F_ref)
        reduction, _, _ = compute_reduction_metric(distances)
        null_reductions.append(reduction)
    
    p_value = np.mean(np.array(null_reductions) >= observed_reduction)
    return p_value


def analyze_convergence(wave_A: WaveData, wave_B: WaveData,
                        n_permutations: int = 1000) -> ConvergenceResult:
    """Analyze formant trajectory convergence."""
    # Get reference formants for B
    F_ref = get_reference_formants(wave_B)
    
    # Get formant trajectory for A
    traj_A = compute_formant_trajectory(wave_A)
    F_A = traj_A['formants']
    times = traj_A['times']
    
    # Compute distances
    distances = compute_formant_distance(F_A, F_ref)
    
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_time = times[min_idx] if min_idx < len(times) else 0
    
    # Compute reduction
    distance_reduction, best_start, best_len = compute_reduction_metric(distances)
    
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    convergence_start = times[best_start] if best_len > 0 and best_start < len(times) else 0
    convergence_duration = best_len * dt
    
    # Permutation test
    p_value = permutation_test(F_A, F_ref, distance_reduction, n_permutations)
    
    converges = (p_value < 0.05 and distance_reduction > 0.10)
    
    if converges:
        verdict = f"SIGNIFICANT: reduction={distance_reduction:.1%}, p={p_value:.4f}"
    else:
        verdict = f"NOT SIGNIFICANT: reduction={distance_reduction:.1%}, p={p_value:.4f}"
    
    return ConvergenceResult(
        converges=converges,
        p_value=p_value,
        distance_reduction=distance_reduction,
        convergence_start=convergence_start,
        convergence_duration=convergence_duration,
        min_distance=min_distance,
        min_distance_time=min_time,
        evidence={
            'F_A': F_A,
            'F_ref': F_ref,
            'times': times,
            'distances': distances,
            'traj_A': traj_A,
        },
        verdict=verdict
    )


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(wave_A: WaveData, wave_B: WaveData,
                         result: ConvergenceResult, output_dir: str):
    BG, PANEL, TEXT = '#111111', '#1a1a1a', '#eaeaea'
    CONV, NO_CONV, GRID = '#2ECC71', '#FF6B6B', '#333333'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    
    times = result.evidence['times']
    distances = result.evidence['distances']
    F_A = result.evidence['F_A']
    F_ref = result.evidence['F_ref']
    
    # Panel 1: Formant trajectories
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    ax.plot(times, F_A[:, 0], 'r-', lw=1.5, label='F1', alpha=0.8)
    ax.plot(times, F_A[:, 1], 'g-', lw=1.5, label='F2', alpha=0.8)
    ax.plot(times, F_A[:, 2], 'b-', lw=1.5, label='F3', alpha=0.8)
    ax.axhline(F_ref[0], color='r', ls='--', alpha=0.5)
    ax.axhline(F_ref[1], color='g', ls='--', alpha=0.5)
    ax.axhline(F_ref[2], color='b', ls='--', alpha=0.5)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Frequency (Hz)', color=TEXT)
    ax.set_title('Formant Trajectories (solid) vs Reference (dashed)', 
                color=TEXT, fontweight='bold')
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
    ax.set_ylabel('Formant Distance', color=TEXT)
    ax.set_title('Distance to Reference Formants', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 3: F1-F2 vowel space
    ax = axes[1, 0]
    ax.set_facecolor(PANEL)
    ax.scatter(F_A[:, 1], F_A[:, 0], c=times, cmap='viridis', s=30, alpha=0.7)
    ax.scatter([F_ref[1]], [F_ref[0]], color='#FFD93D', s=200, marker='*',
               edgecolors='white', lw=2, label='Reference', zorder=5)
    ax.set_xlabel('F2 (Hz)', color=TEXT)
    ax.set_ylabel('F1 (Hz)', color=TEXT)
    ax.set_title('F1-F2 Vowel Space (color = time)', color=TEXT, fontweight='bold')
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    ax.invert_xaxis()
    ax.invert_yaxis()
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    
    summary = f"""
FORMANT TRAJECTORY CONVERGENCE
{'═' * 40}

Wave A: {wave_A.name}
Wave B: {wave_B.name} (reference)

Reference: F1={F_ref[0]:.0f}Hz, F2={F_ref[1]:.0f}Hz, F3={F_ref[2]:.0f}Hz

{'─' * 40}
STATISTICAL TEST
{'─' * 40}
  Distance reduction: {result.distance_reduction:.1%}
  P-value:            {result.p_value:.4f}

{'═' * 40}
VERDICT: {status}
{'═' * 40}
"""
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', color=TEXT, family='monospace')
    
    fig.suptitle(f'Formant Convergence: {wave_A.name} → {wave_B.name}',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'formant_convergence.png')
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
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'formant_metrics.csv'), index=False)
    print(f"Metrics: {output_dir}/formant_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description='Formant trajectory convergence analysis')
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
        args.output_dir = f'results/formant_analysis/{n1}_vs_{n2}'
    
    print(f"\n{'='*60}")
    print("FORMANT TRAJECTORY CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    
    wave_A = load_wave(args.wave1)
    wave_B = load_wave(args.wave2)
    
    print(f"Trajectory: {wave_A.name} ({wave_A.duration:.3f}s)")
    print(f"Reference:  {wave_B.name} ({wave_B.duration:.3f}s)")
    
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
