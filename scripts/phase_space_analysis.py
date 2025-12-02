import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import librosa
import nolds
from sklearn.preprocessing import MinMaxScaler
import argparse

def analyze_phase_space(file_path, output_dir, show_plot=False):
    """
    Analyzes the phase space of a .wav file.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Normalize signal
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y_norm = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Phase Space Reconstruction
        # We need to choose an embedding dimension (m) and time delay (tau)
        # For visualization, we often use m=2 or m=3.
        # nolds has a function to estimate these, but it can be slow.
        # For this initial pass, let's use standard heuristics or fixed values for visualization.
        
        # Heuristic for tau: first zero crossing of autocorrelation or first minimum of mutual information
        # For simplicity and speed in this trial, let's try a fixed small lag or a simple autocorrelation check.
        # librosa.autocorrelate can help.
        
        # Let's use a fixed tau for visualization consistency across vowels for now, 
        # or calculate it per file. Let's calculate it per file using autocorrelation.
        ac = librosa.autocorrelate(y_norm, max_size=2000)
        # Find the first local minimum or zero crossing. 
        # A common heuristic is the first zero crossing.
        zero_crossings = np.where(np.diff(np.sign(ac)))[0]
        if len(zero_crossings) > 0:
            tau = zero_crossings[0]
        else:
            tau = 10 # Fallback
            
        m = 2 # For 2D plot
        
        # Create time-delay embedding
        # X(t) = [y(t), y(t + tau), ..., y(t + (m-1)*tau)]
        # We can use nolds.delay_embedding or just numpy
        
        # Let's use m=2 for a 2D Phase Portrait
        y_delay = np.roll(y_norm, -tau)
        
        # Trim the end
        y_norm_trimmed = y_norm[:-tau]
        y_delay_trimmed = y_delay[:-tau]
        
        # Plot
        plt.figure(figsize=(10, 8))
        
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
        
        ax = plt.gca()
        ax.add_collection(lc)
        ax.autoscale()
        
        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax)
        cbar.set_label('Time (samples)')
        
        # Title in default font (for Latin characters)
        plt.title(f'Phase Space Reconstruction: {os.path.basename(file_path)}\n(tau={tau}, m={m})')
        
        # Add Devanagari Vowel as a separate annotation
        # Extract vowel from parent directory name
        vowel = os.path.basename(os.path.dirname(file_path))
        font_prop = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf', size=30)
        
        # Place it in the corner or somewhere visible
        plt.text(0.05, 0.95, f"{vowel}", transform=ax.transAxes, 
                 fontproperties=font_prop, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.xlabel('y(t)')
        plt.ylabel(f'y(t + {tau})')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        filename = os.path.basename(file_path).replace('.wav', '_phase_space.png')
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Phase Space Analysis of WAV files')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory containing cleaned .wav files')
    parser.add_argument('--output_dir', type=str, default='results/phase_space_plots', help='Output directory for plots')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of files to process (for trial runs)')
    
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
        if analyze_phase_space(wav_file, args.output_dir):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
