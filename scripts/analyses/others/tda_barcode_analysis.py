import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import MinMaxScaler

def analyze_tda_barcode(file_path, output_dir, duration_ms=None, max_points=800, use_phoneme_subdir=True):
    """
    Analyzes the persistent homology of a .wav file (Barcode/Diagram).
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

        # Phase Space Reconstruction (Embedding)
        # Calculate tau using autocorrelation
        ac = librosa.autocorrelate(y_norm, max_size=2000)
        zero_crossings = np.where(np.diff(np.sign(ac)))[0]
        if len(zero_crossings) > 0:
            tau = zero_crossings[0]
        else:
            tau = 10 # Fallback
            
        m = 3 # Dimension for embedding
        
        # Create time-delay embedding (Point Cloud)
        # X(t) = [y(t), y(t + tau), y(t + 2*tau)]
        N = len(y_norm) - (m - 1) * tau
        if N < m:
            print(f"Signal too short for embedding with tau={tau}, m={m}")
            return False
            
        point_cloud = np.zeros((N, m))
        for i in range(m):
            point_cloud[:, i] = y_norm[i*tau : i*tau + N]
            
        # Subsampling (TDA is expensive)
        if len(point_cloud) > max_points:
            # Random subsampling
            indices = np.random.choice(len(point_cloud), max_points, replace=False)
            point_cloud = point_cloud[indices]
            
        # Compute Persistent Homology
        # maxdim=1 computes H0 and H1 (loops)
        dgms = ripser(point_cloud, maxdim=1)['dgms']
        
        # Plotting
        plt.figure(figsize=(12, 6), facecolor='white')
        
        # Persistence Diagram
        plt.subplot(1, 2, 1)
        plot_diagrams(dgms, show=False)
        plt.title("Persistence Diagram")
        
        # Barcode (Lifetime)
        # We need to manually plot barcode or use a library helper if available.
        # persim doesn't have a direct 'plot_barcode' but we can visualize lifetimes.
        # Let's stick to Diagram for now as it's standard, or implement a simple barcode.
        
        plt.subplot(1, 2, 2)
        # Custom Barcode Plot for H1
        if len(dgms) > 1:
            h1 = dgms[1]
            # Sort by lifetime (death - birth)
            lifetimes = h1[:, 1] - h1[:, 0]
            sort_idx = np.argsort(lifetimes)[::-1]
            h1_sorted = h1[sort_idx]
            
            # Plot bars
            for i, (birth, death) in enumerate(h1_sorted):
                if death == np.inf:
                    death = np.max(h1[h1 != np.inf]) * 1.1 # Cap infinity for display
                plt.plot([birth, death], [i, i], color='orange', lw=2)
                
            plt.title(f"H1 Barcode (Top {len(h1)} loops)")
            plt.xlabel("Filtration Value")
            plt.ylabel("Feature Index")
            plt.gca().invert_yaxis() # Top is longest
        else:
            plt.text(0.5, 0.5, "No H1 features found", ha='center')
            
        plt.tight_layout()
        
        # Save Plot
        if use_phoneme_subdir:
            phoneme = os.path.basename(os.path.dirname(file_path))
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            save_dir = phoneme_dir
        else:
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_tda_barcode.png')
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='TDA Barcode Analysis of WAV files')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory containing cleaned .wav files')
    parser.add_argument('--output_dir', type=str, default='results/tda_plots', help='Output directory for plots')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of files to process')
    parser.add_argument('--duration_ms', type=int, default=100, help='Duration in milliseconds to analyze')
    parser.add_argument('--max_points', type=int, default=800, help='Max points for TDA (subsampling)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find wav files
    search_pattern = os.path.join(args.input_dir, '**', '*.wav')
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} .wav files in {args.input_dir}")
    
    if args.limit:
        print(f"Limiting processing to first {args.limit} files.")
        wav_files = wav_files[:args.limit]
        
    count = 0
    for wav_file in wav_files:
        if analyze_tda_barcode(wav_file, args.output_dir, duration_ms=args.duration_ms, max_points=args.max_points):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
