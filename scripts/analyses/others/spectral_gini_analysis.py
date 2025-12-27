import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ts2vg import NaturalVG

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def analyze_file(file_path, duration_ms=None):
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # 1. Spectral Centroid
        # We take the mean over time
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_centroid = np.mean(cent)
        
        # 2. Gini Coefficient (from Visibility Graph)
        # Downsample if needed for speed, but Gini needs accurate degree dist
        # If > 5000 samples, downsample to ~5000
        if len(y) > 5000:
            step = len(y) // 5000
            y_vg = y[::step]
        else:
            y_vg = y
            
        vg = NaturalVG()
        vg.build(y_vg)
        degrees = vg.degrees
        gini_coeff = gini(degrees)
        
        phoneme = os.path.basename(os.path.dirname(file_path))
        
        return {
            'file': os.path.basename(file_path),
            'phoneme': phoneme,
            'gini': gini_coeff,
            'spectral_centroid': avg_centroid
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Spectral Centroid vs Gini Analysis')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit files per phoneme')
    parser.add_argument('--duration_ms', type=int, default=100, help='Duration (ms)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all wav files
    search_pattern = os.path.join(args.input_dir, '**', '*.wav')
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} .wav files")
    
    # Group by phoneme to apply limit per phoneme if needed
    files_by_phoneme = {}
    for f in wav_files:
        p = os.path.basename(os.path.dirname(f))
        if p not in files_by_phoneme:
            files_by_phoneme[p] = []
        files_by_phoneme[p].append(f)
        
    selected_files = []
    for p, files in files_by_phoneme.items():
        if args.limit:
            selected_files.extend(files[:args.limit])
        else:
            selected_files.extend(files)
            
    print(f"Processing {len(selected_files)} files...")
    
    data = []
    for i, f in enumerate(selected_files):
        res = analyze_file(f, duration_ms=args.duration_ms)
        if res:
            data.append(res)
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(selected_files)}")
            
    df = pd.DataFrame(data)
    
    if df.empty:
        print("No data collected.")
        return

    # Save CSV
    csv_path = os.path.join(args.output_dir, 'spectral_gini_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")
    
    # Plot
    plt.figure(figsize=(10, 8), facecolor='#111111')
    ax = plt.gca()
    ax.set_facecolor('#111111')
    
    # Use seaborn for easy coloring by phoneme
    sns.scatterplot(data=df, x='gini', y='spectral_centroid', hue='phoneme', palette='bright', s=100, alpha=0.8, ax=ax)
    
    plt.title('Spectral Centroid vs Gini Coefficient', color='#EAEAEA', fontsize=16)
    plt.xlabel('Gini Coefficient (Roughness)', color='#EAEAEA', fontsize=12)
    plt.ylabel('Spectral Centroid (Hz) (Brightness)', color='#EAEAEA', fontsize=12)
    
    ax.tick_params(colors='#EAEAEA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#EAEAEA')
    plt.grid(True, alpha=0.2, color='#EAEAEA')
    
    # Legend
    # Try to use Devanagari font for legend if available
    import matplotlib.font_manager as fm
    font_path = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
    if os.path.exists(font_path):
        prop = fm.FontProperties(fname=font_path)
        plt.legend(title='Phoneme', prop=prop, facecolor='#111111', edgecolor='#EAEAEA', labelcolor='#EAEAEA')
        # Also set title font if needed, but title is English
    else:
        plt.legend(title='Phoneme', facecolor='#111111', edgecolor='#EAEAEA', labelcolor='#EAEAEA')
    
    # Annotate regions (optional)
    # plt.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5)
    # plt.axhline(y=2000, color='gray', linestyle='--', alpha=0.5)
    
    plot_path = os.path.join(args.output_dir, 'spectral_gini_scatter.png')
    plt.savefig(plot_path, dpi=300, facecolor='#111111')
    print(f"Saved plot to {plot_path}")

if __name__ == "__main__":
    main()
