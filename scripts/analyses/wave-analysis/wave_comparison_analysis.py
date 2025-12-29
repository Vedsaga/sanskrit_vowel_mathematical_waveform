#!/usr/bin/env python3
"""
Transfer Function Containment Analysis

Tests whether one wave's articulatory trajectory converges to another's
using LPC-based transfer function analysis.

The key insight: a ⊂ ka iff ∃ t such that H_ka(t,f) → H_a(f)

This is the physics-correct test for containment in resonant systems.

Usage:
    python wave_comparison_analysis.py --wave1 ka.wav --wave2 a.wav --visualize

Author: Generated for sanskrit_vowel_mathematical_waveform project
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import common configuration for Devanagari font support
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
    """Container for mechanical wave data."""
    samples: np.ndarray
    sample_rate: int
    duration: float
    filepath: str
    name: str
    
    @property
    def time_axis(self) -> np.ndarray:
        return np.linspace(0, self.duration, len(self.samples))
    
    @property
    def n_samples(self) -> int:
        return len(self.samples)


@dataclass
class ContainmentResult:
    """Result of containment analysis."""
    contained: bool
    confidence: float
    best_match_time: float
    best_voiced_corr: float
    evidence: Dict[str, Any]
    verdict: str


# =============================================================================
# Wave Loading
# =============================================================================

def load_wave(filepath: str) -> WaveData:
    """Load a wave file and normalize."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Wave file not found: {filepath}")
    
    sample_rate, samples = wavfile.read(filepath)
    
    # Stereo -> mono
    if len(samples.shape) > 1:
        samples = samples.mean(axis=1)
    
    # Normalize to [-1, 1]
    samples = samples.astype(np.float64)
    max_val = np.abs(samples).max()
    if max_val > 0:
        samples = samples / max_val
    
    duration = len(samples) / sample_rate
    name = os.path.splitext(os.path.basename(filepath))[0]
    
    return WaveData(samples, sample_rate, duration, filepath, name)


# =============================================================================
# LPC Transfer Function Analysis
# =============================================================================

def apply_preemphasis(samples: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]"""
    return np.append(samples[0], samples[1:] - coeff * samples[:-1])


def compute_voicing_strength(frame: np.ndarray, sample_rate: int) -> float:
    """
    Compute voicing strength using normalized autocorrelation peak.
    Vowels have high voicing (periodic), consonants have low voicing.
    """
    n = len(frame)
    autocorr = np.correlate(frame, frame, mode='full')[n-1:]
    
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    else:
        return 0.0
    
    # Pitch range: 80-400 Hz
    min_lag = int(sample_rate / 400)
    max_lag = min(int(sample_rate / 80), len(autocorr) - 1)
    
    if min_lag >= max_lag:
        return 0.0
    
    peak_val = np.max(autocorr[min_lag:max_lag])
    return max(0.0, min(1.0, peak_val))


def compute_lpc_coefficients(samples: np.ndarray, order: int = 12) -> np.ndarray:
    """
    Compute LPC coefficients using Levinson-Durbin with pre-emphasis.
    """
    # DC removal + pre-emphasis
    samples = samples - np.mean(samples)
    samples = apply_preemphasis(samples, 0.97)
    
    n = len(samples)
    r = np.correlate(samples, samples, mode='full')[n-1:n+order+1]
    
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


def compute_transfer_function(lpc_coeffs: np.ndarray, n_freqs: int = 256, 
                               sample_rate: int = 16000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute H(f) = 1 / (1 + Σ a_k z^-k) from LPC coefficients.
    """
    frequencies = np.linspace(0, sample_rate / 2, n_freqs)
    z = np.exp(2j * np.pi * frequencies / sample_rate)
    
    denom = np.ones(n_freqs, dtype=complex)
    for k, a_k in enumerate(lpc_coeffs):
        denom += a_k * (z ** (-(k + 1)))
    
    H = 1.0 / (np.abs(denom) + 1e-10)
    H = H / (H.max() + 1e-10)
    
    return frequencies, H


def compute_time_varying_tf(wave: WaveData, frame_ms: float = 25.0,
                             hop_ms: float = 10.0, lpc_order: int = 12) -> Dict:
    """
    Compute time-varying transfer function H(t, f) with voicing detection.
    """
    sr = wave.sample_rate
    frame_size = int(frame_ms * sr / 1000)
    hop_size = int(hop_ms * sr / 1000)
    window = signal.windows.hann(frame_size)
    
    n_frames = max(1, (len(wave.samples) - frame_size) // hop_size + 1)
    n_freqs = 256
    
    H_matrix = np.zeros((n_frames, n_freqs))
    times = np.zeros(n_frames)
    voicing = np.zeros(n_frames)
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        
        if end > len(wave.samples):
            break
        
        frame_raw = wave.samples[start:end]
        frame = frame_raw * window
        
        if np.sum(frame**2) < 1e-6:
            continue
        
        voicing[i] = compute_voicing_strength(frame_raw, sr)
        lpc = compute_lpc_coefficients(frame, lpc_order)
        freqs, H = compute_transfer_function(lpc, n_freqs, sr)
        
        H_matrix[i] = H
        times[i] = (start + frame_size / 2) / sr
    
    return {'H_matrix': H_matrix, 'times': times, 'frequencies': freqs, 
            'voicing': voicing, 'n_frames': n_frames}


def get_reference_tf(wave: WaveData, lpc_order: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get reference transfer function from most voiced frame.
    """
    sr = wave.sample_rate
    frame_size = int(25 * sr / 1000)
    hop_size = int(10 * sr / 1000)
    window = signal.windows.hann(frame_size)
    
    best_frame, best_voicing = None, 0.0
    
    for i in range((len(wave.samples) - frame_size) // hop_size):
        start = i * hop_size
        frame = wave.samples[start:start + frame_size]
        v = compute_voicing_strength(frame, sr)
        if v > best_voicing:
            best_voicing = v
            best_frame = frame * window
    
    if best_frame is None or best_voicing < 0.3:
        mid = len(wave.samples) // 2
        best_frame = wave.samples[mid-frame_size//2:mid+frame_size//2]
        best_frame = best_frame * signal.windows.hann(len(best_frame))
    
    lpc = compute_lpc_coefficients(best_frame, lpc_order)
    return compute_transfer_function(lpc, 256, sr)


# =============================================================================
# Containment Analysis
# =============================================================================

def analyze_containment(wave_A: WaveData, wave_B: WaveData, 
                        lpc_order: int = 12) -> ContainmentResult:
    """
    Test if wave A contains wave B by checking transfer function convergence.
    
    The articulatory trajectory of A converges toward the target of B
    if H_A(t, f) → H_B(f) for some time t.
    """
    # Get reference TF for B (the target)
    freq_B, H_ref_B = get_reference_tf(wave_B, lpc_order)
    
    # Get time-varying TF for A (the container)
    tf_A = compute_time_varying_tf(wave_A, lpc_order=lpc_order)
    
    # Also get reference for A
    freq_A, H_ref_A = get_reference_tf(wave_A, lpc_order)
    
    # Find best matching voiced frame in A
    best_voiced_corr = 0.0
    best_match_time = 0.0
    convergence_scores = np.zeros(tf_A['n_frames'])
    
    for i in range(len(tf_A['times'])):
        H_frame = tf_A['H_matrix'][i]
        voicing = tf_A['voicing'][i]
        
        if np.sum(H_frame) < 0.1:
            continue
        
        if np.std(H_frame) > 0 and np.std(H_ref_B) > 0:
            corr = np.corrcoef(H_frame, H_ref_B)[0, 1]
            if not np.isnan(corr):
                convergence_scores[i] = corr
                
                # Only consider voiced frames (vowel-like)
                if voicing > 0.3 and corr > best_voiced_corr:
                    best_voiced_corr = corr
                    best_match_time = tf_A['times'][i]
    
    # Reference similarity
    ref_sim = 0.0
    if np.std(H_ref_A) > 0 and np.std(H_ref_B) > 0:
        ref_sim = np.corrcoef(H_ref_A, H_ref_B)[0, 1]
    
    # Containment decision (threshold 0.6)
    contained = best_voiced_corr > 0.6
    
    verdict = (f"Convergence detected at t={best_match_time:.3f}s (r={best_voiced_corr:.3f})"
               if contained else
               f"No convergence detected (best r={best_voiced_corr:.3f})")
    
    return ContainmentResult(
        contained=contained,
        confidence=best_voiced_corr,
        best_match_time=best_match_time,
        best_voiced_corr=best_voiced_corr,
        evidence={
            'H_ref_A': (freq_A, H_ref_A),
            'H_ref_B': (freq_B, H_ref_B),
            'tf_A': tf_A,
            'convergence_scores': convergence_scores,
            'reference_similarity': ref_sim,
        },
        verdict=verdict
    )


# =============================================================================
# Visualization
# =============================================================================

def create_visualization(wave_A: WaveData, wave_B: WaveData,
                         result: ContainmentResult, output_dir: str):
    """Create 4-panel visualization showing transfer function convergence."""
    
    BG_COLOR = '#111111'
    PANEL_COLOR = '#1a1a1a'
    TEXT_COLOR = '#eaeaea'
    ACCENT_A = '#4ECDC4'
    ACCENT_B = '#FFD93D'
    SUCCESS = '#2ECC71'
    GRID = '#333333'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG_COLOR)
    
    tf = result.evidence['tf_A']
    times = tf['times']
    voicing = tf['voicing']
    scores = result.evidence['convergence_scores']
    
    # Panel 1: Waveforms with containment region
    ax = axes[0, 0]
    ax.set_facecolor(PANEL_COLOR)
    ax.plot(wave_A.time_axis, wave_A.samples, color=ACCENT_A, alpha=0.8, lw=0.6,
            label=f'{wave_A.name} (A)')
    ax.plot(wave_B.time_axis, wave_B.samples - 2.2, color=ACCENT_B, alpha=0.8, lw=0.6,
            label=f'{wave_B.name} (B)')
    
    if result.contained:
        ax.axvline(result.best_match_time, color=SUCCESS, lw=2, alpha=0.8)
        ax.axvspan(result.best_match_time - 0.03, result.best_match_time + 0.03,
                   color=SUCCESS, alpha=0.2)
    
    ax.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax.set_ylabel('Amplitude', color=TEXT_COLOR)
    ax.set_title('Waveforms', color=TEXT_COLOR, fontweight='bold')
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 2: Convergence timeline with voicing
    ax = axes[0, 1]
    ax.set_facecolor(PANEL_COLOR)
    
    ax.fill_between(times, 0, voicing, color=ACCENT_A, alpha=0.2, label='Voicing')
    ax.plot(times, scores, color=ACCENT_A, lw=2, label='H(t)↔H_B correlation')
    ax.axhline(0.6, color='#FF6B6B', ls='--', lw=1.5, alpha=0.8, label='Threshold')
    
    if result.contained:
        ax.scatter([result.best_match_time], [result.best_voiced_corr], 
                  color=SUCCESS, s=150, zorder=5, edgecolors='white', lw=2)
        ax.annotate(f'r={result.best_voiced_corr:.3f}', 
                   xy=(result.best_match_time, result.best_voiced_corr),
                   xytext=(10, 10), textcoords='offset points',
                   color=SUCCESS, fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Time in A (s)', color=TEXT_COLOR)
    ax.set_ylabel('Correlation / Voicing', color=TEXT_COLOR)
    ax.set_title('Transfer Function Convergence', color=TEXT_COLOR, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper left', facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 3: H(f) overlay at best match
    ax = axes[1, 0]
    ax.set_facecolor(PANEL_COLOR)
    
    freq_B, H_B = result.evidence['H_ref_B']
    H_matrix = tf['H_matrix']
    freqs = tf['frequencies']
    
    best_idx = np.argmin(np.abs(times - result.best_match_time)) if result.best_match_time > 0 else 0
    H_at_match = H_matrix[best_idx] if best_idx < len(H_matrix) else H_matrix[0]
    
    mask = freqs <= 4000
    ax.plot(freq_B[mask], H_B[mask], color=ACCENT_B, lw=2.5, label=f'H_B ({wave_B.name})')
    ax.plot(freqs[mask], H_at_match[mask], color=ACCENT_A, lw=2, ls='--',
            label=f'H_A(t={result.best_match_time:.2f}s)')
    
    ax.set_xlabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.set_ylabel('H(f)', color=TEXT_COLOR)
    ax.set_title(f'Transfer Function Match (r={result.best_voiced_corr:.3f})',
                color=SUCCESS if result.contained else TEXT_COLOR, fontweight='bold')
    ax.legend(facecolor=PANEL_COLOR, labelcolor=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(alpha=0.2, color=GRID)
    
    # Panel 4: Summary
    ax = axes[1, 1]
    ax.set_facecolor(PANEL_COLOR)
    ax.axis('off')
    
    summary = f"""
ARTICULATORY CONVERGENCE ANALYSIS
{'═' * 35}

Wave A: {wave_A.name}
Wave B: {wave_B.name}

{'─' * 35}
TRANSFER FUNCTION MATCH
{'─' * 35}
  Best voiced correlation: {result.best_voiced_corr:.3f}
  Match time:              t = {result.best_match_time:.3f}s
  Reference similarity:    {result.evidence['reference_similarity']:.3f}

{'═' * 35}
VERDICT: {"✓ CONTAINED" if result.contained else "✗ NOT DETECTED"}
CONFIDENCE: {result.confidence:.1%}
{'═' * 35}
"""
    
    color = SUCCESS if result.contained else '#FF6B6B'
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va='top', color=TEXT_COLOR, family='monospace')
    
    fig.suptitle(f'Transfer Function Containment: {wave_A.name} → {wave_B.name}',
                 color=TEXT_COLOR, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'wave_comparison.png')
    plt.savefig(path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization: {path}")
    return path


def save_results(wave_A: WaveData, wave_B: WaveData, 
                 result: ContainmentResult, output_dir: str):
    """Save analysis results to CSV and text."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        'wave_A': wave_A.name,
        'wave_B': wave_B.name,
        'contained': result.contained,
        'confidence': result.confidence,
        'best_voiced_corr': result.best_voiced_corr,
        'best_match_time': result.best_match_time,
        'reference_similarity': result.evidence['reference_similarity'],
    }
    
    csv_path = os.path.join(output_dir, 'comparison_metrics.csv')
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"Metrics: {csv_path}")
    
    txt_path = os.path.join(output_dir, 'containment_verdict.txt')
    with open(txt_path, 'w') as f:
        f.write(f"CONTAINMENT ANALYSIS: {wave_A.name} vs {wave_B.name}\n")
        f.write("=" * 50 + "\n")
        f.write(result.verdict + "\n")
    print(f"Verdict: {txt_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test transfer function containment (articulatory convergence)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python wave_comparison_analysis.py --wave1 ka.wav --wave2 a.wav --visualize

Theory:
  A contains B if the articulatory trajectory of A converges toward B:
    H_A(t, f) → H_B(f) for some time interval
"""
    )
    
    parser.add_argument('--wave1', required=True, help='Container wave (e.g., ka)')
    parser.add_argument('--wave2', required=True, help='Target wave (e.g., a)')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--visualize', action='store_true', help='Generate plots')
    
    args = parser.parse_args()
    
    # Default output dir
    if not args.output_dir:
        name1 = os.path.splitext(os.path.basename(args.wave1))[0]
        name2 = os.path.splitext(os.path.basename(args.wave2))[0]
        args.output_dir = f'results/wave_analysis/{name1}_vs_{name2}'
    
    print(f"\n{'='*60}")
    print("TRANSFER FUNCTION CONTAINMENT ANALYSIS")
    print(f"{'='*60}")
    
    wave_A = load_wave(args.wave1)
    print(f"\nWave A: {wave_A.name} ({wave_A.duration:.3f}s)")
    
    wave_B = load_wave(args.wave2)
    print(f"Wave B: {wave_B.name} ({wave_B.duration:.3f}s)")
    
    print(f"\n{'='*60}")
    print("Analyzing...")
    print(f"{'='*60}\n")
    
    result = analyze_containment(wave_A, wave_B)
    
    print(f"RESULT: {'✓ CONTAINED' if result.contained else '✗ NOT DETECTED'}")
    print(f"  Voiced correlation: {result.best_voiced_corr:.3f}")
    print(f"  Match time: t = {result.best_match_time:.3f}s")
    
    save_results(wave_A, wave_B, result, args.output_dir)
    
    if args.visualize:
        create_visualization(wave_A, wave_B, result, args.output_dir)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}\n")
    
    return 0 if result.contained else 1


if __name__ == "__main__":
    sys.exit(main())
