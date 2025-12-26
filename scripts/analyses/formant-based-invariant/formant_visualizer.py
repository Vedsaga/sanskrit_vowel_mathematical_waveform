#!/usr/bin/env python3
"""
Formant Visualizer: 8 Intuitive Figures for Understanding Formant Analysis

This script generates educational visualizations explaining formant-based acoustic analysis.
Can be run standalone or called from other analysis scripts.

Figures:
1. Temporal Anchor - Why frames are chosen
2. Formant Structure - Core vowel identity (spectrogram + tracks)
3. Formant Geometry - Spacing & dispersion
4. Scale-Invariant Ratios - Why ratios work
5. Spectral Envelope - A1, A2, A3 made obvious
6. Spectral Tilt - Energy slope intuition
7. Harmonics vs Formants - Pitch independence
8. Residual Dynamics - Gunas layer
"""

import os
import argparse
import numpy as np

# Use non-interactive backend for thread safety in parallel processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import signal
from scipy.io import wavfile

try:
    import librosa
    import librosa.display
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PRAAT = True
except ImportError:
    HAS_PRAAT = False

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
BG_COLOR = '#111111'
PANEL_COLOR = '#1a1a1a'
TEXT_COLOR = '#EAEAEA'
BORDER_COLOR = '#333333'
GRID_COLOR = '#444444'

# ============================================================================
# CONSISTENT COLOR SEMANTICS (used across ALL figures)
# ============================================================================
# Formants - track same concept across all visualizations
F1_COLOR = '#FF6B6B'      # Red - First formant
F2_COLOR = '#4ECDC4'      # Cyan/Teal - Second formant  
F3_COLOR = '#7ED321'      # Green - Third formant

# Spectral features
ENVELOPE_COLOR = '#FFD93D'  # Gold - Spectral envelope, takeaway annotations
HARMONIC_COLOR = '#888888'  # Gray - Raw spectrum, harmonics, waveform

# Configure fonts
DEVANAGARI_FONT = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT):
    fm.fontManager.addfont(DEVANAGARI_FONT)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


def style_axis(ax, title=None):
    """Apply consistent dark theme styling to axis."""
    ax.set_facecolor(PANEL_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
    if title:
        ax.set_title(title, color=TEXT_COLOR, fontweight='bold', fontsize=11)


# ============================================================================
# DATA EXTRACTION (reuses logic from analysis scripts)
# ============================================================================
def extract_all_data(audio_path, time_step=0.01, max_formant_freq=5500.0,
                     stability_smoothing=50.0, intensity_threshold=50.0):
    """
    Extract all data needed for visualizations from an audio file.
    Returns a comprehensive dictionary with formants, amplitudes, and raw audio.
    """
    if not HAS_PRAAT:
        raise ImportError("parselmouth required. Install: pip install praat-parselmouth")
    if not HAS_LIBROSA:
        raise ImportError("librosa required. Install: pip install librosa")
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    # Load with Praat
    sound = parselmouth.Sound(audio_path)
    
    # Create Formant object
    formant = call(sound, "To Formant (burg)", time_step, 5, max_formant_freq, 0.025, 50.0)
    intensity = call(sound, "To Intensity", 100, time_step, "yes")
    pitch = call(sound, "To Pitch", 0.0, 75, 600)
    
    n_frames = call(formant, "Get number of frames")
    
    # Collect frame-by-frame data
    f1_values, f2_values, f3_values = [], [], []
    b1_values, b2_values, b3_values = [], [], []
    time_values, intensity_values = [], []
    f0_values = []
    
    for i in range(1, n_frames + 1):
        t = call(formant, "Get time from frame number", i)
        f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
        b1 = call(formant, "Get bandwidth at time", 1, t, "Hertz", "Linear")
        b2 = call(formant, "Get bandwidth at time", 2, t, "Hertz", "Linear")
        b3 = call(formant, "Get bandwidth at time", 3, t, "Hertz", "Linear")
        
        try:
            intens = call(intensity, "Get value at time", t, "Cubic")
            if np.isnan(intens):
                intens = 0.0
        except:
            intens = 60.0
        
        try:
            f0 = call(pitch, "Get value at time", t, "Hertz", "Linear")
            if np.isnan(f0):
                f0 = 0.0
        except:
            f0 = 0.0
        
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
                f0_values.append(f0)
    
    if len(f1_values) < 3:
        return None
    
    # Convert to arrays
    f1_arr = np.array(f1_values)
    f2_arr = np.array(f2_values)
    f3_arr = np.array(f3_values)
    b1_arr = np.array(b1_values)
    b2_arr = np.array(b2_values)
    b3_arr = np.array(b3_values)
    time_arr = np.array(time_values)
    intensity_arr = np.array(intensity_values)
    f0_arr = np.array(f0_values)
    
    # Calculate stability weights
    n = len(f1_arr)
    instability = np.zeros(n)
    for i in range(n):
        if i == 0:
            delta = abs(f1_arr[1] - f1_arr[0]) + abs(f2_arr[1] - f2_arr[0])
        elif i == n - 1:
            delta = abs(f1_arr[-1] - f1_arr[-2]) + abs(f2_arr[-1] - f2_arr[-2])
        else:
            delta = abs(f1_arr[i+1] - f1_arr[i-1]) + abs(f2_arr[i+1] - f2_arr[i-1])
        instability[i] = delta
    
    weights = 1.0 / (instability + stability_smoothing)
    weights[intensity_arr < intensity_threshold] = 0.0
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(n) / n
    
    # Compute weighted means
    f1_mean = np.average(f1_arr, weights=weights)
    f2_mean = np.average(f2_arr, weights=weights)
    f3_mean = np.average(f3_arr, weights=weights)
    
    # Amplitude estimates (inverse bandwidth)
    a1_arr = 1.0 / b1_arr
    a2_arr = 1.0 / b2_arr
    a3_arr = 1.0 / b3_arr
    
    return {
        'y': y, 'sr': sr, 'duration': duration,
        'time': time_arr,
        'f1': f1_arr, 'f2': f2_arr, 'f3': f3_arr,
        'f1_mean': f1_mean, 'f2_mean': f2_mean, 'f3_mean': f3_mean,
        'b1': b1_arr, 'b2': b2_arr, 'b3': b3_arr,
        'a1': a1_arr, 'a2': a2_arr, 'a3': a3_arr,
        'intensity': intensity_arr,
        'f0': f0_arr,
        'weights': weights,
        'intensity_threshold': intensity_threshold,
    }


# ============================================================================
# FIGURE 1: TEMPORAL ANCHOR
# ============================================================================
def figure_temporal_anchor(data, output_path, quiet=False):
    """Why frames are chosen - stability weighting & RMS thresholds."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)
    
    y, sr = data['y'], data['sr']
    time_audio = np.linspace(0, len(y)/sr, len(y))
    
    # Top: Waveform
    ax = axes[0]
    style_axis(ax, "Waveform")
    ax.plot(time_audio, y, color=HARMONIC_COLOR, linewidth=0.5, alpha=0.8)
    ax.set_ylabel('Amplitude', color=TEXT_COLOR)
    ax.set_xlim(0, data['duration'])
    
    # Bottom: Intensity + Stability Weights
    ax = axes[1]
    style_axis(ax, "Stability Weights & Intensity")
    
    # Intensity curve
    ax.plot(data['time'], data['intensity'], color=ENVELOPE_COLOR, linewidth=1.5, 
            label='Intensity (dB)', alpha=0.8)
    ax.axhline(y=data['intensity_threshold'], color='#FF6B6B', linestyle='--', 
               linewidth=1, label=f'Threshold ({data["intensity_threshold"]} dB)')
    
    # Stability weights (scaled for visibility)
    weights_scaled = data['weights'] / data['weights'].max() * data['intensity'].max() * 0.8
    ax.fill_between(data['time'], 0, weights_scaled, color=F2_COLOR, alpha=0.3, 
                    label='Stability Weight')
    
    ax.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax.set_ylabel('Value', color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.set_xlim(0, data['duration'])
    
    # Takeaway annotation
    fig.text(0.5, 0.02, '→ "These parts of the waveform actually matter for formant analysis"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 1 saved: {output_path}")


# ============================================================================
# FIGURE 2: FORMANT STRUCTURE
# ============================================================================
def figure_formant_structure(data, output_path, quiet=False):
    """Core vowel identity - spectrogram with F1/F2/F3 tracks."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    style_axis(ax, "Formant Structure: Spectrogram with F1/F2/F3 Tracks")
    
    y, sr = data['y'], data['sr']
    
    # Wideband spectrogram (25ms window ~ 400 samples at 16kHz)
    n_fft = int(0.025 * sr)
    hop_length = int(0.005 * sr)
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)
    
    img = librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz',
                                    ax=ax, cmap='magma', vmin=-80, vmax=0)
    
    # Overlay formant tracks
    ax.plot(data['time'], data['f1'], color=F1_COLOR, linewidth=2, label='F1', alpha=0.9)
    ax.plot(data['time'], data['f2'], color=F2_COLOR, linewidth=2, label='F2', alpha=0.9)
    ax.plot(data['time'], data['f3'], color=F3_COLOR, linewidth=2, label='F3', alpha=0.9)
    
    # Mean horizontal lines
    ax.axhline(y=data['f1_mean'], color=F1_COLOR, linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(y=data['f2_mean'], color=F2_COLOR, linestyle='--', linewidth=1, alpha=0.6)
    ax.axhline(y=data['f3_mean'], color=F3_COLOR, linestyle='--', linewidth=1, alpha=0.6)
    
    ax.set_ylim(0, 4000)
    ax.set_ylabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, 
              loc='upper right', fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(BORDER_COLOR)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT_COLOR)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Formants are stable resonant bands, not waveform oscillations"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 2 saved: {output_path}")


# ============================================================================
# FIGURE 3: FORMANT GEOMETRY
# ============================================================================
def figure_formant_geometry(data, output_path, quiet=False):
    """Spacing & dispersion - geometric relationships."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    delta_f21 = data['f2'] - data['f1']
    delta_f32 = data['f3'] - data['f2']
    
    # Left: Vertical spacing bars (sample of frames)
    ax = axes[0]
    style_axis(ax, "F1-F2-F3 Spacing (Sample Frames)")
    
    n_samples = min(10, len(data['f1']))
    indices = np.linspace(0, len(data['f1'])-1, n_samples, dtype=int)
    
    for i, idx in enumerate(indices):
        x = i
        ax.barh(0, data['f1'][idx], left=0, height=0.6, color=F1_COLOR, alpha=0.7)
        ax.barh(1, delta_f21[idx], left=data['f1'][idx], height=0.6, color=F2_COLOR, alpha=0.7)
        ax.barh(2, delta_f32[idx], left=data['f2'][idx], height=0.6, color=F3_COLOR, alpha=0.7)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['F1', 'ΔF21', 'ΔF32'], color=TEXT_COLOR)
    ax.set_xlabel('Frequency (Hz)', color=TEXT_COLOR)
    
    # Center: Spacing over time
    ax = axes[1]
    style_axis(ax, "Spacing Over Time")
    ax.plot(data['time'], delta_f21, color=F2_COLOR, linewidth=1.5, label='ΔF21 (F2-F1)')
    ax.plot(data['time'], delta_f32, color=F3_COLOR, linewidth=1.5, label='ΔF32 (F3-F2)')
    ax.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax.set_ylabel('Spacing (Hz)', color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=8)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Right: Dispersion summary
    ax = axes[2]
    style_axis(ax, "Dispersion Summary")
    
    formant_range = data['f3_mean'] - data['f1_mean']
    sigma = np.std([data['f1_mean'], data['f2_mean'], data['f3_mean']])
    
    metrics = ['F3-F1 Range', 'σ(F1,F2,F3)']
    values = [formant_range, sigma]
    colors = ['#FFD93D', '#FF6B6B']
    
    bars = ax.barh(metrics, values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Hz', color=TEXT_COLOR)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2, 
                f'{val:.0f}', va='center', color=TEXT_COLOR, fontsize=10)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Vowels differ by geometric spacing, not absolute frequency"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 3 saved: {output_path}")


# ============================================================================
# FIGURE 4: SCALE-INVARIANT RATIOS
# ============================================================================
def figure_scale_invariant_ratios(data, output_path, quiet=False):
    """Why ratios work - absolute values vs ratios."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)
    
    # Top: Absolute formants
    ax = axes[0]
    style_axis(ax, "Absolute Formant Frequencies")
    ax.plot(data['time'], data['f1'], color=F1_COLOR, linewidth=1.5, label='F1')
    ax.plot(data['time'], data['f2'], color=F2_COLOR, linewidth=1.5, label='F2')
    ax.plot(data['time'], data['f3'], color=F3_COLOR, linewidth=1.5, label='F3')
    ax.set_ylabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Bottom: Ratios
    ax = axes[1]
    style_axis(ax, "Scale-Invariant Ratios")
    
    f1_f2_ratio = data['f1'] / data['f2']
    f2_f3_ratio = data['f2'] / data['f3']
    log_f2_f1 = np.log(data['f2'] / data['f1'])
    
    ax.plot(data['time'], f1_f2_ratio, color=F1_COLOR, linewidth=1.5, label='F1/F2')
    ax.plot(data['time'], f2_f3_ratio, color=F2_COLOR, linewidth=1.5, label='F2/F3')
    ax.plot(data['time'], log_f2_f1, color=F3_COLOR, linewidth=1.5, label='log(F2/F1)')
    
    ax.set_xlabel('Time (s)', color=TEXT_COLOR)
    ax.set_ylabel('Ratio Value', color=TEXT_COLOR)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Absolute values shift, ratios stay stable"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 4 saved: {output_path}")


# ============================================================================
# FIGURE 5: SPECTRAL ENVELOPE
# ============================================================================
def figure_spectral_envelope(data, output_path, quiet=False):
    """A1, A2, A3 made obvious with FFT and LPC envelope."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    style_axis(ax, "Spectral Envelope with Formant Amplitudes (A1, A2, A3)")
    
    y, sr = data['y'], data['sr']
    
    # Select stable region (middle 50%)
    n = len(y)
    stable_y = y[n//4 : 3*n//4]
    
    # FFT
    n_fft = 2048
    fft_result = np.fft.rfft(stable_y * np.hanning(len(stable_y)), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    
    # Normalize
    magnitude_db = magnitude_db - magnitude_db.max()
    
    # Plot FFT spectrum
    mask = freqs < 4000
    ax.plot(freqs[mask], magnitude_db[mask], color='#888888', linewidth=0.8, alpha=0.7, label='FFT Spectrum')
    
    # LPC envelope (simple smoothing as approximation)
    from scipy.ndimage import gaussian_filter1d
    envelope = gaussian_filter1d(magnitude_db, sigma=20)
    ax.plot(freqs[mask], envelope[mask], color='#FFD93D', linewidth=2, label='Spectral Envelope')
    
    # Mark formant peaks
    for f, a, color, label in [
        (data['f1_mean'], 'A1', F1_COLOR, 'A1 @ F1'),
        (data['f2_mean'], 'A2', F2_COLOR, 'A2 @ F2'),
        (data['f3_mean'], 'A3', F3_COLOR, 'A3 @ F3'),
    ]:
        # Find amplitude at this frequency
        idx = np.argmin(np.abs(freqs - f))
        amp = envelope[idx]
        ax.scatter([f], [amp], s=150, color=color, zorder=5, edgecolors='white', linewidths=2)
        ax.annotate(label, (f, amp), xytext=(10, 10), textcoords='offset points',
                    color=color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.set_ylabel('Magnitude (dB)', color=TEXT_COLOR)
    ax.set_xlim(0, 4000)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Amplitude ratios are envelope geometry, not loudness"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 5 saved: {output_path}")


# ============================================================================
# FIGURE 6: SPECTRAL TILT
# ============================================================================
def figure_spectral_tilt(data, output_path, quiet=False):
    """Energy slope intuition - dB/octave visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    style_axis(ax, "Spectral Tilt: Energy Slope from F1 to F3")
    
    y, sr = data['y'], data['sr']
    n = len(y)
    stable_y = y[n//4 : 3*n//4]
    
    # FFT
    n_fft = 2048
    fft_result = np.fft.rfft(stable_y * np.hanning(len(stable_y)), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    magnitude_db = magnitude_db - magnitude_db.max()
    
    # Envelope
    from scipy.ndimage import gaussian_filter1d
    envelope = gaussian_filter1d(magnitude_db, sigma=20)
    
    mask = freqs < 4000
    ax.plot(freqs[mask], envelope[mask], color='#FFD93D', linewidth=2, label='Spectral Envelope')
    
    # Get amplitudes at F1 and F3
    f1, f3 = data['f1_mean'], data['f3_mean']
    idx_f1 = np.argmin(np.abs(freqs - f1))
    idx_f3 = np.argmin(np.abs(freqs - f3))
    a1_db = envelope[idx_f1]
    a3_db = envelope[idx_f3]
    
    # Calculate tilt (dB/octave)
    octaves = np.log2(f3 / f1)
    tilt = (a3_db - a1_db) / octaves if octaves > 0 else 0
    
    # Plot tilt line
    ax.plot([f1, f3], [a1_db, a3_db], color='#FF6B6B', linewidth=3, linestyle='--',
            label=f'Tilt: {tilt:.1f} dB/octave')
    
    # Mark F1 and F3
    ax.scatter([f1, f3], [a1_db, a3_db], s=150, color='#FF6B6B', zorder=5,
               edgecolors='white', linewidths=2)
    ax.annotate('F1', (f1, a1_db), xytext=(-20, 10), textcoords='offset points',
                color=F1_COLOR, fontsize=11, fontweight='bold')
    ax.annotate('F3', (f3, a3_db), xytext=(10, 10), textcoords='offset points',
                color=F3_COLOR, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.set_ylabel('Magnitude (dB)', color=TEXT_COLOR)
    ax.set_xlim(0, 4000)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=10)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Open vowels have flatter spectral slopes"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 6 saved: {output_path}")


# ============================================================================
# FIGURE 7: HARMONICS VS FORMANTS
# ============================================================================
def figure_harmonics_vs_formants(data, output_path, quiet=False):
    """Pitch independence - H1, H2 vs formants."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG_COLOR)
    style_axis(ax, "Harmonics (H1, H2) vs Formants: Pitch Independence")
    
    y, sr = data['y'], data['sr']
    n = len(y)
    stable_y = y[n//4 : 3*n//4]
    
    # Narrowband FFT (longer window for harmonic resolution)
    n_fft = 8192
    fft_result = np.fft.rfft(stable_y * np.hanning(len(stable_y)), n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    magnitude_db = magnitude_db - magnitude_db.max()
    
    mask = freqs < 2000
    ax.plot(freqs[mask], magnitude_db[mask], color='#888888', linewidth=0.8, alpha=0.8,
            label='Narrowband Spectrum')
    
    # Envelope
    from scipy.ndimage import gaussian_filter1d
    envelope = gaussian_filter1d(magnitude_db, sigma=50)
    ax.plot(freqs[mask], envelope[mask], color='#FFD93D', linewidth=2, label='Envelope', alpha=0.7)
    
    # Get f0 (fundamental frequency)
    f0 = np.mean(data['f0'][data['f0'] > 0]) if np.any(data['f0'] > 0) else 150
    
    # Mark harmonics H1 and H2
    for i, (harmonic, label, color) in enumerate([(f0, 'H1 (f₀)', '#FF6B6B'), (2*f0, 'H2 (2f₀)', '#FF9999')]):
        if harmonic < 2000:
            idx = np.argmin(np.abs(freqs - harmonic))
            amp = magnitude_db[idx]
            ax.axvline(x=harmonic, color=color, linewidth=2, linestyle='-', alpha=0.8)
            ax.annotate(label, (harmonic, -10), xytext=(5, 0), textcoords='offset points',
                        color=color, fontsize=10, fontweight='bold')
    
    # Mark formants (as vertical bands, not following harmonics)
    for f, label, color in [(data['f1_mean'], 'F1', F1_COLOR), (data['f2_mean'], 'F2', F2_COLOR)]:
        if f < 2000:
            ax.axvline(x=f, color=color, linewidth=2, linestyle='--', alpha=0.6)
            ax.annotate(label, (f, -5), xytext=(5, 0), textcoords='offset points',
                        color=color, fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Frequency (Hz)', color=TEXT_COLOR)
    ax.set_ylabel('Magnitude (dB)', color=TEXT_COLOR)
    ax.set_xlim(0, 2000)
    ax.legend(facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
    ax.grid(True, alpha=0.2, color=GRID_COLOR)
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "Pitch moves harmonics, not formants"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 7 saved: {output_path}")


# ============================================================================
# FIGURE 8: RESIDUAL DYNAMICS (GUNAS)
# ============================================================================
def figure_residual_dynamics(data, output_path, quiet=False):
    """Gunas layer - phase space, entropy, Lyapunov."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG_COLOR)
    
    y = data['y']
    # Use stable middle portion
    n = len(y)
    chunk = y[n//4 : 3*n//4]
    
    # Downsample for visualization
    step = max(1, len(chunk) // 5000)
    chunk = chunk[::step]
    
    # 1. Phase-space attractor (time-delay embedding)
    ax = axes[0]
    style_axis(ax, "Phase-Space Attractor")
    delay = 10
    x = chunk[:-delay]
    y_delayed = chunk[delay:]
    ax.scatter(x, y_delayed, s=1, c=np.arange(len(x)), cmap='viridis', alpha=0.5)
    ax.set_xlabel('x(t)', color=TEXT_COLOR)
    ax.set_ylabel('x(t+τ)', color=TEXT_COLOR)
    ax.set_aspect('equal')
    
    # 2. Permutation entropy histogram (pattern distribution)
    ax = axes[1]
    style_axis(ax, "Pattern Distribution (Entropy)")
    
    # Simple pattern counting
    order = 3
    patterns = []
    for i in range(len(chunk) - order):
        window = chunk[i:i+order]
        pattern = tuple(np.argsort(window))
        patterns.append(pattern)
    
    from collections import Counter
    counts = Counter(patterns)
    labels = [str(p) for p in counts.keys()]
    values = list(counts.values())
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(labels)))
    ax.bar(range(len(labels)), values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7, color=TEXT_COLOR)
    ax.set_ylabel('Count', color=TEXT_COLOR)
    
    # Calculate entropy
    total = sum(values)
    probs = [v/total for v in values]
    entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
    max_entropy = np.log2(len(labels))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
    ax.set_title(f'Pattern Distribution\n(Rajas = {norm_entropy:.3f})', color=TEXT_COLOR, fontweight='bold')
    
    # 3. Divergence / summary
    ax = axes[2]
    style_axis(ax, "Gunas Summary")
    ax.axis('off')
    
    # Calculate metrics
    sattva = 0.0
    rajas = norm_entropy
    tamas = 0.0
    
    if HAS_NOLDS:
        try:
            sattva = nolds.corr_dim(chunk, emb_dim=5)
        except:
            pass
        try:
            tamas = nolds.lyap_r(chunk[:2000], emb_dim=10, min_tsep=50)
        except:
            pass
    
    summary = f"""
    GUNAS METRICS
    ═════════════════════
    
    Sattva (Complexity):     {sattva:.3f}
    Fractal/correlation dimension
    
    Rajas (Unpredictability): {rajas:.3f}
    Permutation entropy (0-1)
    
    Tamas (Instability):     {tamas:.3f}
    Lyapunov exponent
    
    ═════════════════════
    These describe dynamics
    AFTER removing spectral
    structure.
    """
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', color=TEXT_COLOR, family='monospace',
            bbox=dict(boxstyle='round', facecolor=PANEL_COLOR, edgecolor=BORDER_COLOR))
    
    # Takeaway
    fig.text(0.5, 0.02, '→ "These describe dynamics after removing spectral structure"',
             ha='center', color=ENVELOPE_COLOR, fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Figure 8 saved: {output_path}")


# ============================================================================
# MAIN API
# ============================================================================
def generate_all_figures(audio_path, output_dir, formant_data=None, figures=None, quiet=False):
    """
    Generate all 8 visualization figures for an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Directory to save figures
        formant_data: Pre-computed data (optional, will extract if None)
        figures: List of figure numbers to generate (default: all 1-8)
        quiet: If True, suppress verbose output (for batch mode)
    
    Returns:
        List of generated file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"FORMANT VISUALIZER")
        print(f"{'='*60}")
        print(f"Audio: {os.path.basename(audio_path)}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")
    
    # Extract data if not provided
    if formant_data is None:
        if not quiet:
            print("Extracting formant data...")
        formant_data = extract_all_data(audio_path)
        if formant_data is None:
            if not quiet:
                print("ERROR: Could not extract formant data from audio file")
            return []
    
    # Default to all figures
    if figures is None:
        figures = [1, 2, 3, 4, 5, 6, 7, 8]
    
    generated = []
    
    figure_funcs = {
        1: ('01_temporal_anchor.png', figure_temporal_anchor),
        2: ('02_formant_structure.png', figure_formant_structure),
        3: ('03_formant_geometry.png', figure_formant_geometry),
        4: ('04_scale_invariant_ratios.png', figure_scale_invariant_ratios),
        5: ('05_spectral_envelope.png', figure_spectral_envelope),
        6: ('06_spectral_tilt.png', figure_spectral_tilt),
        7: ('07_harmonics_vs_formants.png', figure_harmonics_vs_formants),
        8: ('08_residual_dynamics.png', figure_residual_dynamics),
    }
    
    if not quiet:
        print("Generating figures...")
    for num in figures:
        if num in figure_funcs:
            filename, func = figure_funcs[num]
            output_path = os.path.join(output_dir, filename)
            try:
                func(formant_data, output_path, quiet=quiet)
                generated.append(output_path)
            except Exception as e:
                if not quiet:
                    print(f"  ✗ Figure {num} failed: {e}")
    
    if not quiet:
        print(f"\n{'='*60}")
        print(f"Generated {len(generated)}/{len(figures)} figures")
        print(f"{'='*60}\n")
    
    return generated


def generate_batch_figures(file_list, output_base_dir, workers=4, quiet=True, figures=None):
    """
    Generate visualizations for multiple files.
    
    Args:
        file_list: List of tuples (audio_path, subfolder_name)
        output_base_dir: Base directory for all outputs
        workers: (unused, kept for API compatibility)
        quiet: Suppress per-file output
        figures: List of figure numbers to generate (default: all 1-8)
    
    Returns:
        Number of successfully processed files
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    fig_names = {1: '01_temporal_anchor.png', 2: '02_formant_structure.png', 
                 3: '03_formant_geometry.png', 4: '04_scale_invariant_ratios.png',
                 5: '05_spectral_envelope.png', 6: '06_spectral_tilt.png',
                 7: '07_harmonics_vs_formants.png', 8: '08_residual_dynamics.png'}
    
    successful = 0
    
    from tqdm import tqdm
    for item in tqdm(file_list, desc="Creating visualizations"):
        if len(item) == 3:
            audio_path, subfolder, formant_data = item
        else:
            audio_path, subfolder = item
            formant_data = None
        
        output_dir = os.path.join(output_base_dir, subfolder)
        
        # Skip if already exists (check first figure in requested list)
        first_fig = figures[0] if figures else 1
        marker_file = os.path.join(output_dir, fig_names.get(first_fig, '01_temporal_anchor.png'))
        if os.path.exists(marker_file):
            successful += 1
            continue
        
        try:
            generate_all_figures(audio_path, output_dir, formant_data=formant_data, figures=figures, quiet=True)
            successful += 1
        except Exception as e:
            if not quiet:
                print(f"  ✗ Failed: {audio_path}: {e}")
    
    return successful


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Generate 8 intuitive visualization figures for formant analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  python formant_visualizer.py --file audio.wav
  
  # Specify output directory
  python formant_visualizer.py --file audio.wav --output_dir results/viz/
  
  # Generate specific figures only
  python formant_visualizer.py --file audio.wav --figures 1,2,5
        """
    )
    
    parser.add_argument('--file', type=str, required=True, help='Path to audio file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/formant_visualizer/)')
    parser.add_argument('--figures', type=str, default=None,
                        help='Comma-separated figure numbers to generate (default: all)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = 'results/formant_visualizer'
    
    figures = None
    if args.figures:
        figures = [int(x.strip()) for x in args.figures.split(',')]
    
    generate_all_figures(args.file, args.output_dir, figures=figures)


if __name__ == "__main__":
    main()
