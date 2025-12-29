#!/usr/bin/env python3
"""
Scattering Trajectory Convergence Analysis

Analyzes whether the wavelet scattering trajectory of one signal (e.g., 'ka')
converges toward the reference pattern of another signal (e.g., 'a').

IMPORTANT: This tests ACOUSTIC FEATURE CONVERGENCE, not mathematical containment.
The claim is: "the scattering representation of ka evolves toward that of a over time"

Uses:
- kymatio library for proper wavelet scattering (orders 0, 1, 2)
- Permutation test for statistical significance
- Raw magnitude comparison (no normalization that destroys phonetic info)

Usage:
    python wave_comparison_analysis.py --wave1 ka.wav --wave2 a.wav --visualize
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

from kymatio.numpy import Scattering1D

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
    """Container for wave data."""
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
    """Result of trajectory convergence analysis."""
    converges: bool
    p_value: float  # Statistical significance from permutation test
    distance_reduction: float  # How much distance decreased (ratio)
    convergence_start: float
    convergence_duration: float
    min_distance: float
    min_distance_time: float
    evidence: Dict[str, Any]
    verdict: str


# =============================================================================
# Wave Loading with Silence Trimming
# =============================================================================

def compute_frame_energy(samples: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    """Compute short-term energy."""
    n_frames = max(1, (len(samples) - frame_size) // hop_size + 1)
    energy = np.zeros(n_frames)
    for i in range(n_frames):
        frame = samples[i*hop_size:i*hop_size + frame_size]
        energy[i] = np.sum(frame ** 2) / max(1, len(frame))
    return energy


def trim_silence(samples: np.ndarray, sr: int, threshold_db: float = -35.0) -> np.ndarray:
    """Trim leading and trailing silence."""
    frame_size = int(0.025 * sr)
    hop_size = int(0.010 * sr)
    
    energy = compute_frame_energy(samples, frame_size, hop_size)
    if len(energy) == 0 or energy.max() == 0:
        return samples
    
    threshold = energy.max() * (10 ** (threshold_db / 10))
    voiced = energy > threshold
    
    if not np.any(voiced):
        return samples
    
    voiced_idx = np.where(voiced)[0]
    first, last = voiced_idx[0], voiced_idx[-1]
    first = max(0, first - 2)
    last = min(len(energy) - 1, last + 2)
    
    start = first * hop_size
    end = min(len(samples), (last + 1) * hop_size + frame_size)
    
    return samples[start:end]


def load_wave(filepath: str, trim: bool = True) -> WaveData:
    """Load wave file with optional silence trimming."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    sr, samples = wavfile.read(filepath)
    
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    
    samples = samples.astype(np.float64)
    if np.abs(samples).max() > 0:
        samples = samples / np.abs(samples).max()
    
    if trim:
        samples = trim_silence(samples, sr)
    
    return WaveData(samples, sr, len(samples)/sr, filepath,
                   os.path.splitext(os.path.basename(filepath))[0])


# =============================================================================
# Wavelet Scattering Transform (using kymatio)
# =============================================================================

def compute_time_varying_scattering(samples: np.ndarray, sr: int,
                                     window_ms: float = 50.0,
                                     hop_ms: float = 10.0,
                                     J: int = 6, Q: int = 8) -> Dict:
    """
    Compute time-varying scattering coefficients using sliding windows.
    
    Returns S(t) - the scattering state at each time point.
    """
    window_size = int(window_ms * sr / 1000)
    hop_size = int(hop_ms * sr / 1000)
    
    # Ensure window is power of 2 for FFT efficiency
    window_size = 2 ** int(np.ceil(np.log2(window_size)))
    
    n_frames = max(1, (len(samples) - window_size) // hop_size + 1)
    
    # Initialize scattering for this window size
    scattering = Scattering1D(J=J, shape=(window_size,), Q=Q, max_order=2)
    meta = scattering.meta()
    
    S_list = []
    times = []
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + window_size
        
        if end > len(samples):
            frame = np.pad(samples[start:], (0, end - len(samples)), mode='constant')
        else:
            frame = samples[start:end]
        
        # Compute scattering
        Sx = scattering(frame)
        
        # Average over time dimension to get single feature vector
        S_vec = Sx.mean(axis=1)
        S_list.append(S_vec)
        times.append((start + window_size / 2) / sr)
    
    S_matrix = np.array(S_list)
    
    return {
        'S_matrix': S_matrix,
        'times': np.array(times),
        'meta': meta,
        'n_frames': n_frames,
        'J': J,
        'Q': Q,
    }


def get_reference_scattering(wave: WaveData, J: int = 6, Q: int = 8,
                              window_ms: float = 50.0) -> np.ndarray:
    """
    Compute reference scattering representation for target wave.
    
    Uses full signal averaged to get stable representation.
    Returns a single feature vector S_ref.
    """
    N = len(wave.samples)
    N_padded = 2 ** int(np.ceil(np.log2(N)))
    samples_padded = np.pad(wave.samples, (0, N_padded - N), mode='constant')
    
    scattering = Scattering1D(J=J, shape=(N_padded,), Q=Q, max_order=2)
    Sx = scattering(samples_padded)
    
    # Average over time to get single representation
    S_ref = Sx.mean(axis=1)
    
    return S_ref


# =============================================================================
# Statistical Significance via Permutation Test
# =============================================================================

def compute_distance_trajectory(S_matrix: np.ndarray, S_ref: np.ndarray) -> np.ndarray:
    """
    Compute distance from each time point to reference.
    
    NO NORMALIZATION - preserves magnitude information.
    """
    n_frames = S_matrix.shape[0]
    n_coeffs_A = S_matrix.shape[1]
    n_coeffs_B = len(S_ref)
    
    # Handle dimension mismatch by zero-padding shorter vector
    if n_coeffs_A != n_coeffs_B:
        max_coeffs = max(n_coeffs_A, n_coeffs_B)
        if n_coeffs_A < max_coeffs:
            S_matrix = np.pad(S_matrix, ((0, 0), (0, max_coeffs - n_coeffs_A)), mode='constant')
        if n_coeffs_B < max_coeffs:
            S_ref = np.pad(S_ref, (0, max_coeffs - n_coeffs_B), mode='constant')
    
    # Compute raw distance (no normalization)
    distances = np.array([np.linalg.norm(S_matrix[i] - S_ref) for i in range(n_frames)])
    
    return distances


def compute_reduction_metric(distances: np.ndarray) -> Tuple[float, int, int]:
    """
    Compute distance reduction and find best convergence region.
    
    Returns: (reduction_ratio, best_start_idx, best_length)
    """
    if len(distances) < 2:
        return 0.0, 0, 0
    
    # Find longest decreasing run
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
    
    # Compute reduction during best decreasing run
    if best_len > 0:
        d_start = distances[best_start]
        d_end = distances[min(best_start + best_len, len(distances) - 1)]
        reduction = (d_start - d_end) / (d_start + 1e-10)
    else:
        reduction = 0.0
    
    return reduction, best_start, best_len


def permutation_test(S_matrix: np.ndarray, S_ref: np.ndarray, 
                      observed_reduction: float, n_permutations: int = 1000) -> float:
    """
    Test significance of observed reduction via permutation.
    
    Null hypothesis: temporal structure doesn't matter - 
    shuffling time order should give similar reduction by chance.
    
    Returns p-value.
    """
    n_frames = S_matrix.shape[0]
    
    if n_frames < 3:
        return 1.0  # Not enough data
    
    null_reductions = []
    
    for _ in range(n_permutations):
        # Shuffle temporal order
        shuffled_idx = np.random.permutation(n_frames)
        S_shuffled = S_matrix[shuffled_idx]
        
        # Compute distances and reduction with shuffled order
        distances_shuffled = compute_distance_trajectory(S_shuffled, S_ref)
        reduction_shuffled, _, _ = compute_reduction_metric(distances_shuffled)
        null_reductions.append(reduction_shuffled)
    
    null_reductions = np.array(null_reductions)
    
    # P-value: fraction of null reductions >= observed
    p_value = np.mean(null_reductions >= observed_reduction)
    
    return p_value


# =============================================================================
# Convergence Analysis
# =============================================================================

def analyze_convergence(wave_A: WaveData, wave_B: WaveData,
                        J: int = 6, Q: int = 8,
                        window_ms: float = 50.0,
                        n_permutations: int = 1000) -> ConvergenceResult:
    """
    Analyze if wave A's scattering trajectory converges toward wave B's reference.
    
    Uses permutation test for statistical significance.
    """
    # Get reference scattering for B (target)
    S_B = get_reference_scattering(wave_B, J=J, Q=Q, window_ms=window_ms)
    
    # Get time-varying scattering for A
    scat_A = compute_time_varying_scattering(wave_A.samples, wave_A.sample_rate,
                                              window_ms=window_ms, J=J, Q=Q)
    S_A = scat_A['S_matrix']
    times = scat_A['times']
    
    # Compute distance trajectory (NO NORMALIZATION)
    distances = compute_distance_trajectory(S_A, S_B)
    
    # Find minimum and its location
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    min_time = times[min_idx] if min_idx < len(times) else 0
    
    # Compute reduction metric
    distance_reduction, best_start, best_len = compute_reduction_metric(distances)
    
    dt = times[1] - times[0] if len(times) > 1 else 0.01
    convergence_start = times[best_start] if best_len > 0 and best_start < len(times) else 0
    convergence_duration = best_len * dt
    
    # Statistical significance via permutation test
    p_value = permutation_test(S_A, S_B, distance_reduction, n_permutations)
    
    # Determine convergence: requires statistical significance
    converges = (p_value < 0.05 and distance_reduction > 0.10)
    
    if converges:
        verdict = (f"SIGNIFICANT CONVERGENCE: reduction={distance_reduction:.1%}, "
                   f"p={p_value:.4f} < 0.05")
    else:
        verdict = (f"NOT SIGNIFICANT: reduction={distance_reduction:.1%}, "
                   f"p={p_value:.4f}")
    
    return ConvergenceResult(
        converges=converges,
        p_value=p_value,
        distance_reduction=distance_reduction,
        convergence_start=convergence_start,
        convergence_duration=convergence_duration,
        min_distance=min_distance,
        min_distance_time=min_time,
        evidence={
            'S_A': S_A,
            'S_B': S_B,
            'times': times,
            'distances': distances,
            'scat_A': scat_A,
        },
        verdict=verdict
    )


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(wave_A: WaveData, wave_B: WaveData,
                         result: ConvergenceResult, output_dir: str):
    """Create visualization of trajectory convergence analysis."""
    
    BG = '#111111'
    PANEL = '#1a1a1a'
    TEXT = '#eaeaea'
    A_CLR = '#4ECDC4'
    B_CLR = '#FFD93D'
    CONV = '#2ECC71'
    NO_CONV = '#FF6B6B'
    GRID = '#333333'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    
    times = result.evidence['times']
    distances = result.evidence['distances']
    
    # Panel 1: Waveforms
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    ax.plot(wave_A.time_axis, wave_A.samples, color=A_CLR, alpha=0.8, lw=0.6,
            label=f'{wave_A.name}')
    ax.plot(wave_B.time_axis, wave_B.samples - 2.2, color=B_CLR, alpha=0.8, lw=0.6,
            label=f'{wave_B.name}')
    if result.converges:
        ax.axvspan(result.convergence_start,
                   result.convergence_start + result.convergence_duration,
                   color=CONV, alpha=0.2)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Amplitude', color=TEXT)
    ax.set_title('Waveforms', color=TEXT, fontweight='bold')
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 2: Distance trajectory
    ax = axes[0, 1]
    ax.set_facecolor(PANEL)
    ax.plot(times, distances, color=A_CLR, lw=2)
    ax.scatter([result.min_distance_time], [result.min_distance],
               color=CONV if result.converges else NO_CONV, 
               s=150, zorder=5, edgecolors='white', lw=2)
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Distance to reference', color=TEXT)
    ax.set_title('Scattering Distance d(t) = ||S_A(t) - S_B||', 
                color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 3: Scattering coefficients
    ax = axes[1, 0]
    ax.set_facecolor(PANEL)
    
    S_A = result.evidence['S_A']
    n_show = min(20, S_A.shape[1])
    for j in range(n_show):
        ax.plot(times, S_A[:, j], alpha=0.5, lw=0.8)
    
    ax.set_xlabel('Time (s)', color=TEXT)
    ax.set_ylabel('Scattering coefficients', color=TEXT)
    ax.set_title('Time-varying scattering S_A(t)', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    color = CONV if result.converges else NO_CONV
    
    summary = f"""
SCATTERING TRAJECTORY CONVERGENCE
{'═' * 40}

Wave A: {wave_A.name}
Wave B: {wave_B.name} (reference)

{'─' * 40}
STATISTICAL TEST
{'─' * 40}
  Distance reduction: {result.distance_reduction:.1%}
  P-value:            {result.p_value:.4f}
  Significance:       {"p < 0.05" if result.p_value < 0.05 else "NOT SIGNIFICANT"}

{'─' * 40}
CONVERGENCE METRICS
{'─' * 40}
  Start time:     t = {result.convergence_start:.3f}s
  Duration:       {result.convergence_duration:.3f}s
  Min distance:   {result.min_distance:.3f}

{'═' * 40}
VERDICT: {status}
{'═' * 40}
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            va='top', color=TEXT, family='monospace')
    
    fig.suptitle(f'Trajectory Convergence: {wave_A.name} → {wave_B.name}',
                 color=TEXT, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'scattering_convergence.png')
    plt.savefig(path, dpi=300, facecolor=BG, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization: {path}")
    return path


def save_results(wave_A: WaveData, wave_B: WaveData,
                 result: ConvergenceResult, output_dir: str):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'wave_A': wave_A.name,
        'wave_B': wave_B.name,
        'converges': result.converges,
        'p_value': result.p_value,
        'distance_reduction': result.distance_reduction,
        'convergence_start': result.convergence_start,
        'convergence_duration': result.convergence_duration,
        'min_distance': result.min_distance,
        'min_distance_time': result.min_distance_time,
    }
    
    pd.DataFrame([metrics]).to_csv(os.path.join(output_dir, 'scattering_metrics.csv'), 
                                    index=False)
    print(f"Metrics: {output_dir}/scattering_metrics.csv")
    
    with open(os.path.join(output_dir, 'scattering_verdict.txt'), 'w') as f:
        f.write(f"SCATTERING TRAJECTORY ANALYSIS: {wave_A.name} → {wave_B.name}\n")
        f.write("=" * 50 + "\n")
        f.write(result.verdict + "\n")
        f.write(f"\nNote: Tests acoustic feature convergence, not mathematical containment.\n")
    print(f"Verdict: {output_dir}/scattering_verdict.txt")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scattering trajectory convergence analysis (with permutation test)',
        epilog="""
Tests if the wavelet scattering trajectory of wave1 converges toward wave2's reference.
Uses permutation test for statistical significance (p < 0.05).

NOTE: This tests ACOUSTIC FEATURE CONVERGENCE, not mathematical containment.
"""
    )
    
    parser.add_argument('--wave1', required=True, help='Trajectory wave')
    parser.add_argument('--wave2', required=True, help='Reference wave')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate plots (default: True)')
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.add_argument('--J', type=int, default=6, help='Scattering depth (octaves)')
    parser.add_argument('--Q', type=int, default=8, help='Wavelets per octave')
    parser.add_argument('--window_ms', type=float, default=50.0, help='Window size in ms')
    parser.add_argument('--n_permutations', type=int, default=1000, 
                        help='Number of permutations for significance test')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        n1 = os.path.splitext(os.path.basename(args.wave1))[0]
        n2 = os.path.splitext(os.path.basename(args.wave2))[0]
        args.output_dir = f'results/scattering_analysis/{n1}_vs_{n2}'
    
    print(f"\n{'='*60}")
    print("SCATTERING TRAJECTORY CONVERGENCE ANALYSIS")
    print(f"Parameters: J={args.J}, Q={args.Q}, window={args.window_ms}ms")
    print(f"Permutations: {args.n_permutations}")
    print(f"{'='*60}")
    
    wave_A = load_wave(args.wave1)
    wave_B = load_wave(args.wave2)
    
    print(f"\nTrajectory: {wave_A.name} ({wave_A.duration:.3f}s)")
    print(f"Reference:  {wave_B.name} ({wave_B.duration:.3f}s)")
    
    print(f"\n{'='*60}")
    print("Computing scattering and permutation test...")
    print(f"{'='*60}\n")
    
    result = analyze_convergence(wave_A, wave_B, J=args.J, Q=args.Q,
                                  window_ms=args.window_ms,
                                  n_permutations=args.n_permutations)
    
    status = "✓ SIGNIFICANT" if result.converges else "✗ NOT SIGNIFICANT"
    print(f"\nRESULT: {status}")
    print(f"  Distance reduction: {result.distance_reduction:.1%}")
    print(f"  P-value:            {result.p_value:.4f}")
    print(f"  Min distance:       {result.min_distance:.3f}")
    
    save_results(wave_A, wave_B, result, args.output_dir)
    
    if args.visualize:
        create_visualization(wave_A, wave_B, result, args.output_dir)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}\n")
    
    return 0 if result.converges else 1


if __name__ == "__main__":
    sys.exit(main())
