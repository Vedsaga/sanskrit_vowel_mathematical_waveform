import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import librosa
import nolds
from sklearn.preprocessing import MinMaxScaler
import argparse

def analyze_phase_space(file_path, output_dir, duration_ms=None, show_plot=False):
    """
    Analyzes the phase space of a .wav file using 2D reconstruction and Matplotlib.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Normalize signal
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_norm = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Phase Space Reconstruction
        # Calculate tau using autocorrelation
        ac = librosa.autocorrelate(y_norm, max_size=2000)
        zero_crossings = np.where(np.diff(np.sign(ac)))[0]
        if len(zero_crossings) > 0:
            tau = zero_crossings[0]
        else:
            tau = 10 # Fallback
            
        m = 2 # For 2D plot
        
        # Create time-delay embedding
        y_delay = np.roll(y_norm, -tau)
        
        # Trim the end
        y_norm_trimmed = y_norm[:-tau]
        y_delay_trimmed = y_delay[:-tau]
        
        # Plot
        plt.figure(figsize=(10, 8), facecolor='#111111')
        ax = plt.gca()
        ax.set_facecolor('#111111')
        
        # Explicitly set global font to ensure Latin characters render correctly
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        
        # Create a LineCollection for color mapping by time
        points = np.array([y_norm_trimmed, y_delay_trimmed]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(0, len(y_norm_trimmed))
        lc = plt.matplotlib.collections.LineCollection(segments, cmap='viridis', norm=norm)
        
        # Set the array used for color mapping
        lc.set_array(np.arange(len(y_norm_trimmed)))
        lc.set_linewidth(0.5)
        lc.set_alpha(0.7)
        
        ax.add_collection(lc)
        ax.autoscale()
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Time (samples)', color='#EAEAEA')
        cbar.ax.yaxis.set_tick_params(color='#EAEAEA')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#EAEAEA')
        
        # Title in default font (for Latin characters)
        title_text = f'Phase Space Reconstruction: {os.path.basename(file_path)}\n(tau={tau}, m={m})'
        if duration_ms:
            title_text += f'\nFirst {duration_ms}ms'
        plt.title(title_text, color='#EAEAEA')
        
        # Add Devanagari Vowel as a separate annotation
        # Extract vowel from parent directory name
        vowel = os.path.basename(os.path.dirname(file_path))
        font_prop = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf', size=30)
        
        # Place it in the corner or somewhere visible
        plt.text(0.05, 0.95, f"{vowel}", transform=ax.transAxes, 
                 fontproperties=font_prop, verticalalignment='top', color='#EAEAEA',
                 bbox=dict(boxstyle='round', facecolor='#111111', edgecolor='#EAEAEA', alpha=0.8))
        
        plt.xlabel('y(t)', color='#EAEAEA')
        plt.ylabel(f'y(t + {tau})', color='#EAEAEA')
        
        ax.tick_params(axis='x', colors='#EAEAEA')
        ax.tick_params(axis='y', colors='#EAEAEA')
        
        # Dark grid
        plt.grid(True, alpha=0.2, color='#EAEAEA')
        
        # Remove spines or color them
        for spine in ax.spines.values():
            spine.set_edgecolor('#EAEAEA')
        
        # Save plot
        filename = os.path.basename(file_path).replace('.wav', '_phase_space.png')
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, facecolor='#111111')
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Phase Space Analysis of WAV files (2D)')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory containing cleaned .wav files')
    parser.add_argument('--output_dir', type=str, default='results/phase_space_plots_2d', help='Output directory for plots')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of files to process (for trial runs)')
    parser.add_argument('--duration_ms', type=int, default=None, help='Duration in milliseconds to analyze (from start)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all wav files recursively
    search_pattern = os.path.join(args.input_dir, '**', '*.wav')
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} .wav files in {args.input_dir}")
    
    if args.limit:
        print(f"Limiting processing to first {args.limit} files.")
        wav_files = wav_files[:args.limit]
        
    count = 0
    for wav_file in wav_files:
        if analyze_phase_space(wav_file, args.output_dir, duration_ms=args.duration_ms):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
