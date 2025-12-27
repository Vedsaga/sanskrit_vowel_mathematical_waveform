#!/usr/bin/env python3
"""
GUNAS Analysis: Universal Invariants (Sattva, Rajas, Tamas)

Modes:
1. Single file: --file <path>
2. Folder analysis: --folder <path> (all files in a letter folder)
3. Golden comparison: --golden-compare <cleaned_data_dir> (all golden files)
"""

import os
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path
import argparse

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

try:
    import nolds
    HAS_NOLDS = True
except ImportError:
    HAS_NOLDS = False

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

# Configure fonts for Devanagari
DEVANAGARI_FONT = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
if os.path.exists(DEVANAGARI_FONT):
    fm.fontManager.addfont(DEVANAGARI_FONT)
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


def calculate_permutation_entropy(series, order=5, delay=1):
    """RAJAS: Measures the 'Roughness' or Unpredictability (0-1 normalized)."""
    try:
        n = len(series)
        if n < order * delay:
            return 0
        
        partitions = [series[i : i + order] for i in range(n - order + 1)]
        hash_map = {}
        for p in partitions:
            signature = tuple(np.argsort(p))
            hash_map[signature] = hash_map.get(signature, 0) + 1
        
        counts = np.array(list(hash_map.values()))
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        return entropy / np.log2(math.factorial(order))
    except:
        return 0


def calculate_fractal_dimension(series):
    """SATTVA: Measures the 'Complexity' of the Attractor (Correlation Dimension)."""
    try:
        if not HAS_NOLDS:
            return 0
        return float(nolds.corr_dim(series, emb_dim=5))
    except:
        return 0


def calculate_lyapunov(series):
    """TAMAS: Measures Stability (Lyapunov exponent). Low = stable."""
    try:
        if not HAS_NOLDS:
            return 0
        return float(nolds.lyap_r(series, emb_dim=10, min_tsep=50))
    except:
        return 0


def analyze_gunas(audio_path, stability_smoothing: float = 0.01, rms_threshold: float = 0.01):
    """
    Analyze a single audio file and return Gunas metrics.
    Uses dynamic stability weighting - stable frames contribute more than transitional frames.
    """
    if not HAS_LIBROSA:
        raise ImportError("librosa not installed. Run: pip install librosa")
    
    y, sr = librosa.load(audio_path, sr=16000)
    
    # Compute frame-level features for stability weighting
    frame_length = 512
    hop_length = 256
    
    # Calculate RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Expand y to match frame indices
    n_frames = len(rms)
    if n_frames < 3:
        # Fallback for very short audio
        chunk = y
    else:
        # Calculate instability based on amplitude rate of change
        instability = np.zeros(n_frames)
        for i in range(n_frames):
            if i == 0:
                delta = abs(rms[1] - rms[0])
            elif i == n_frames - 1:
                delta = abs(rms[-1] - rms[-2])
            else:
                delta = abs(rms[i+1] - rms[i-1])
            instability[i] = delta
        
        # Compute stability weights
        weights = 1.0 / (instability + stability_smoothing)
        weights[rms < rms_threshold] = 0.0  # Zero out silent frames
        
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_frames) / n_frames
        
        # Select samples from high-weight frames
        # Get indices of frames sorted by weight (descending)
        top_frame_indices = np.argsort(weights)[::-1]
        
        # Collect samples from top-weighted frames (top 50% by weight)
        cumulative_weight = 0
        selected_samples = []
        for frame_idx in top_frame_indices:
            if cumulative_weight >= 0.9:  # Stop when we have 90% of weight
                break
            start_sample = frame_idx * hop_length
            end_sample = min(start_sample + frame_length, len(y))
            selected_samples.append(y[start_sample:end_sample])
            cumulative_weight += weights[frame_idx]
        
        if selected_samples:
            chunk = np.concatenate(selected_samples)
        else:
            # Fallback to middle 50% if weighting fails
            n_samples = len(y)
            chunk = y[int(n_samples * 0.25):int(n_samples * 0.75)]
    
    # Ensure minimum chunk size
    if len(chunk) < 500:
        chunk = y
    
    return {
        'sattva': calculate_fractal_dimension(chunk),
        'rajas': calculate_permutation_entropy(chunk, order=5, delay=1),
        'tamas': calculate_lyapunov(chunk)
    }


def analyze_folder(folder_path, output_dir):
    """Analyze all audio files in a folder (e.g., all أ files)."""
    os.makedirs(output_dir, exist_ok=True)
    
    files = list(Path(folder_path).glob("*.wav"))
    phoneme = os.path.basename(folder_path)
    
    print(f"\nAnalyzing {len(files)} files in '{phoneme}' folder...")
    
    records = []
    for f in tqdm(files, desc=f"Processing {phoneme}"):
        try:
            result = analyze_gunas(str(f))
            records.append({
                'filename': f.name,
                'phoneme': phoneme,
                **result
            })
        except Exception as e:
            continue
    
    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = os.path.join(output_dir, f'{phoneme}_gunas.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    # Plot
    create_folder_plot(df, phoneme, output_dir)
    
    return df


def analyze_golden_files(cleaned_data_dir, output_dir):
    """Analyze all golden files across all phonemes."""
    os.makedirs(output_dir, exist_ok=True)
    
    golden_pattern = os.path.join(cleaned_data_dir, '*', '*_golden_*.wav')
    import glob
    golden_files = glob.glob(golden_pattern)
    
    print(f"\nFound {len(golden_files)} golden files...")
    
    records = []
    for f in tqdm(golden_files, desc="Analyzing golden files"):
        try:
            phoneme = os.path.basename(os.path.dirname(f))
            result = analyze_gunas(f)
            records.append({
                'filename': os.path.basename(f),
                'phoneme': phoneme,
                **result
            })
        except:
            continue
    
    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'golden_gunas.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to: {csv_path}")
    
    # Plot
    create_golden_plot(df, output_dir)
    
    return df


def create_folder_plot(df, phoneme, output_dir):
    """Create visualization for a single phoneme folder."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a1a')
    fig.suptitle(f"Gunas Analysis: '{phoneme}' ({len(df)} files)", color='white', fontsize=14)
    
    colors = ['#4ECDC4', '#FFD93D', '#FF6B6B']
    metrics = ['sattva', 'rajas', 'tamas']
    labels = ['Sattva (Dimension)', 'Rajas (Entropy)', 'Tamas (Lyapunov)']
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[i]
        ax.set_facecolor('#1a1a1a')
        ax.hist(df[metric].dropna(), bins=20, color=colors[i], alpha=0.8, edgecolor='white')
        ax.set_xlabel(label, color='#EAEAEA')
        ax.set_ylabel('Count', color='#EAEAEA')
        ax.set_title(f'{label}\nMean: {df[metric].mean():.3f}', color='white')
        ax.tick_params(colors='#EAEAEA')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{phoneme}_gunas.png')
    plt.savefig(plot_path, dpi=300, facecolor='#1a1a1a')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def create_golden_plot(df, output_dir):
    """Create visualization for all golden files with high contrast."""
    # COLOR SCHEME aligned with sound-topology dark mode
    BG_COLOR = '#111111'
    TEXT_COLOR = '#eaeaea'
    ACCENT_COLOR = '#e17100'  # Orange - use sparingly
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
    plt.rcParams['grid.color'] = '#2a2a2a'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle(f"Gunas Analysis: All Golden Files ({len(df)} phonemes)", 
                 color=TEXT_COLOR, fontsize=16, fontweight='bold')
    
    # 1. Sattva vs Rajas scatter (Complexity vs Unpredictability)
    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    if HAS_SEABORN:
        sns.scatterplot(data=df, x='rajas', y='sattva', hue='phoneme', 
                        s=200, palette='bright', ax=ax, legend=False,
                        edgecolor='white', linewidth=0.5)
    else:
        ax.scatter(df['rajas'], df['sattva'], c=range(len(df)), cmap='rainbow', s=150, edgecolors='white')
    for _, row in df.iterrows():
        ax.annotate(row['phoneme'], (row['rajas'], row['sattva']), 
                    fontsize=11, color=TEXT_COLOR, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Rajas (Entropy/Unpredictability)', color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel('Sattva (Fractal Dimension)', color=TEXT_COLOR, fontsize=12)
    ax.set_title('Sattva vs Rajas', color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    
    # 2. Sattva vs Tamas (Complexity vs Stability)
    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    if HAS_SEABORN:
        sns.scatterplot(data=df, x='tamas', y='sattva', hue='phoneme',
                        s=200, palette='bright', ax=ax, legend=False,
                        edgecolor='white', linewidth=0.5)
    else:
        ax.scatter(df['tamas'], df['sattva'], c=range(len(df)), cmap='rainbow', s=150, edgecolors='white')
    for _, row in df.iterrows():
        ax.annotate(row['phoneme'], (row['tamas'], row['sattva']),
                    fontsize=11, color=TEXT_COLOR, xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel('Tamas (Lyapunov/Instability)', color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel('Sattva (Fractal Dimension)', color=TEXT_COLOR, fontsize=12)
    ax.set_title('Sattva vs Tamas', color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    
    # 3. Bar chart of Sattva by phoneme - RAINBOW COLORS
    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    sorted_df = df.sort_values('sattva')
    n = len(sorted_df)
    rainbow_colors = plt.cm.rainbow(np.linspace(0, 1, n))
    bars = ax.barh(range(n), sorted_df['sattva'], color=rainbow_colors, alpha=0.9, edgecolor='white', linewidth=1)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_df['phoneme'], fontsize=10, color=TEXT_COLOR)
    ax.set_xlabel('Sattva (Fractal Dimension)', color=TEXT_COLOR, fontsize=12)
    ax.set_title('Sattva by Phoneme', color=TEXT_COLOR, fontweight='bold', fontsize=13)
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(True, axis='x', alpha=0.15, color='white')
    for spine in ax.spines.values():
        spine.set_color(BORDER_COLOR)
        spine.set_linewidth(1)
    
    # 4. Summary stats
    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    ax.axis('off')
    
    summary_text = f"""
    GUNAS ANALYSIS SUMMARY
    ======================
    
    Total phonemes: {len(df)}
    
    SATTVA (Complexity):
      Mean: {df['sattva'].mean():.3f}
      Std:  {df['sattva'].std():.3f}
      Range: {df['sattva'].min():.3f} - {df['sattva'].max():.3f}
    
    RAJAS (Entropy):
      Mean: {df['rajas'].mean():.3f}
      Std:  {df['rajas'].std():.3f}
      Range: {df['rajas'].min():.3f} - {df['rajas'].max():.3f}
    
    TAMAS (Lyapunov):
      Mean: {df['tamas'].mean():.3f}
      Std:  {df['tamas'].std():.3f}
      Range: {df['tamas'].min():.3f} - {df['tamas'].max():.3f}
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', color=TEXT_COLOR,
            bbox=dict(boxstyle='round', facecolor=BG_COLOR, edgecolor=BORDER_COLOR, linewidth=1))
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'golden_gunas.png')
    plt.savefig(plot_path, dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='GUNAS Analysis: Universal Invariants (Sattva, Rajas, Tamas)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file
  python measure_gunas.py --file data/02_cleaned/अ/अ_golden_043.wav
  
  # All files in a folder
  python measure_gunas.py --folder data/02_cleaned/अ
  
  # All golden files
  python measure_gunas.py --golden-compare data/02_cleaned
        """
    )
    
    parser.add_argument('--file', type=str, help='Path to single audio file')
    parser.add_argument('--folder', type=str, help='Path to phoneme folder')
    parser.add_argument('--golden-compare', dest='golden_compare', type=str,
                        help='Path to cleaned data directory for golden files')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/gunas_analysis/{mode})')
    parser.add_argument('--no-visual', action='store_true', dest='no_visual',
                        help='Skip generating visualization figures')
    
    args = parser.parse_args()
    
    if not HAS_LIBROSA or not HAS_NOLDS:
        print("Missing dependencies. Install: pip install librosa nolds")
        return
    
    # Generate default output directory based on mode
    if args.output_dir is None:
        base_dir = "results/gunas_analysis"
        if args.golden_compare:
            output_dir = f"{base_dir}/golden"
        elif args.folder:
            folder_name = os.path.basename(os.path.normpath(args.folder))
            output_dir = f"{base_dir}/batch/{folder_name}"
        elif args.file:
            file_name = os.path.splitext(os.path.basename(args.file))[0]
            output_dir = f"{base_dir}/single/{file_name}"
        else:
            output_dir = base_dir
    else:
        output_dir = args.output_dir
    
    print("=" * 60)
    print("GUNAS ANALYSIS: Universal Invariants")
    print("=" * 60)
    print("Sattva = Complexity (Fractal Dimension)")
    print("Rajas  = Unpredictability (Permutation Entropy)")
    print("Tamas  = Instability (Lyapunov Exponent)")
    print("=" * 60)
    
    if args.golden_compare:
        df = analyze_golden_files(args.golden_compare, output_dir)
        if df is not None:
            print(f"\n--- Summary ---")
            print(df[['phoneme', 'sattva', 'rajas', 'tamas']].to_string(index=False))
            
            # Generate visualizations (Figures 1, 8 for Gunas analysis)
            if HAS_VISUALIZER and not args.no_visual:
                print(f"\nGenerating visualization figures for {len(df)} files...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = []
                for _, row in df.iterrows():
                    phoneme = row['phoneme']
                    filename = os.path.splitext(row['filename'])[0]
                    subfolder = os.path.join(phoneme, filename)
                    file_list.append((row['file_path'], subfolder))
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 8])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    elif args.folder:
        df = analyze_folder(args.folder, output_dir)
        if df is not None:
            print(f"\n--- Summary ---")
            print(f"Sattva: {df['sattva'].mean():.3f} ± {df['sattva'].std():.3f}")
            print(f"Rajas:  {df['rajas'].mean():.3f} ± {df['rajas'].std():.3f}")
            print(f"Tamas:  {df['tamas'].mean():.3f} ± {df['tamas'].std():.3f}")
            
            # Generate visualizations (Figures 1, 8 for Gunas analysis)
            if HAS_VISUALIZER and not args.no_visual:
                print(f"\nGenerating visualization figures for {len(df)} files...")
                from formant_visualizer import generate_batch_figures
                visual_base = os.path.join(output_dir, 'visual')
                file_list = []
                for _, row in df.iterrows():
                    filename = os.path.splitext(row['filename'])[0]
                    file_list.append((row['file_path'], filename))
                successful = generate_batch_figures(file_list, visual_base, figures=[1, 8])
                print(f"Visualizations saved to: {visual_base}/ ({successful}/{len(file_list)} files)")
    
    elif args.file:
        print(f"\nAnalyzing: {args.file}")
        result = analyze_gunas(args.file)
        print(f"\nResults:")
        print(f"  Sattva (Complexity):      {result['sattva']:.4f}")
        print(f"  Rajas (Unpredictability): {result['rajas']:.4f}")
        print(f"  Tamas (Instability):      {result['tamas']:.4f}")
        
        # Generate visualization for single file
        if HAS_VISUALIZER and not args.no_visual:
            print("\nGenerating visualization figures...")
            from formant_visualizer import generate_all_figures
            visual_dir = os.path.join(output_dir, 'visual')
            generate_all_figures(args.file, visual_dir, figures=[1, 8])
    
    else:
        print("Please specify --file, --folder, or --golden-compare")
        parser.print_help()


if __name__ == "__main__":
    main()