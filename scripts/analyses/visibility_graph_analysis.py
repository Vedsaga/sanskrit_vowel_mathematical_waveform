import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from ts2vg import NaturalVG
import networkx as nx

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def analyze_visibility_graph(file_path, output_dir, duration_ms=None, use_phoneme_subdir=True):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Downsample for Adjacency Matrix visualization (keep it sharp)
        # We need a smaller chunk to see the "Stripes" clearly.
        # 1000 nodes is usually the sweet spot for visual patterns.
        viz_limit = 1000 
        y_viz = y[:viz_limit] if len(y) > viz_limit else y

        # 1. Build Full Graph for Stats
        # Warning: For very long files, this can be slow/memory intensive.
        # Maybe limit to 5000 samples for stats if not specified otherwise?
        # For now, we trust the user's duration_ms setting.
        vg = NaturalVG()
        vg.build(y)
        degrees = vg.degrees
        
        # 2. Build Small Graph for Adjacency Matrix Visualization
        vg_viz = NaturalVG()
        vg_viz.build(y_viz)
        adj_matrix = vg_viz.adjacency_matrix()

        # --- METRICS ---
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        gini_coeff = gini(degrees)  # <--- NEW GUNA METRIC

        # --- PLOTTING ---
        fig = plt.figure(figsize=(15, 5))
        
        # Plot 1: The Smoother Log-Log (CCDF)
        ax1 = plt.subplot(1, 3, 1)
        # Sort degrees and calculate survival function
        sorted_degrees = np.sort(degrees)
        y_vals = np.arange(len(sorted_degrees), 0, -1) / len(sorted_degrees)
        ax1.loglog(sorted_degrees, y_vals, marker='.', linestyle='none', color='teal', alpha=0.5)
        ax1.set_title(f'CCDF (Tail Analysis)\nGini: {gini_coeff:.3f}')
        ax1.set_xlabel('Degree (log)')
        ax1.set_ylabel('P(X >= x) (log)')
        ax1.grid(True, which="both", ls="-", alpha=0.2)

        # Plot 2: The Visual Rhythm (Adjacency Matrix)
        ax2 = plt.subplot(1, 3, 2)
        # We convert to dense just for plotting (spy plot)
        # Note: In ts2vg, adjacency is usually upper triangular or symmetric.
        # We plot the first 200x200 to see the fine detail
        zoom = 200
        
        # Handle sparse matrix conversion safely
        if hasattr(adj_matrix, 'todense'):
            dense_matrix = adj_matrix.todense()
        else:
            dense_matrix = adj_matrix
            
        if dense_matrix.shape[0] > zoom:
            dense_matrix = dense_matrix[:zoom, :zoom]
            
        ax2.imshow(dense_matrix, cmap='binary', interpolation='nearest', aspect='auto')
        ax2.set_title(f'Adjacency Texture (First {zoom} nodes)')
        ax2.set_xlabel('Time (t)')
        ax2.set_ylabel('Time (t)')

        # Plot 3: Standard Degree Dist (Your original, for reference)
        ax3 = plt.subplot(1, 3, 3)
        ax3.hist(degrees, bins=50, color='skyblue', edgecolor='black', log=True)
        ax3.set_title(f'Degree Histogram\nMax Deg: {max_degree}')
        
        plt.tight_layout()

        # Save
        if use_phoneme_subdir:
            phoneme = os.path.basename(os.path.dirname(file_path))
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            save_dir = phoneme_dir
        else:
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
            
        filename = os.path.basename(file_path).replace('.wav', '_vg_enhanced.png')
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path)
        plt.close()

        print(f"Processed {os.path.basename(file_path)} | Gini: {gini_coeff:.3f} | Saved.")
        return {
            'file': os.path.basename(file_path),
            'gini': gini_coeff,
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'output_path': output_path
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visibility Graph Analysis of WAV files (Enhanced)')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory containing cleaned .wav files')
    parser.add_argument('--output_dir', type=str, default='results/vg_plots', help='Output directory for plots')
    parser.add_argument('--limit', type=int, default=None, help='Limit the number of files to process')
    parser.add_argument('--duration_ms', type=int, default=100, help='Duration in milliseconds to analyze (default 100ms)')
    
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
        
    results = []
    for wav_file in wav_files:
        res = analyze_visibility_graph(wav_file, args.output_dir, duration_ms=args.duration_ms)
        if res:
            results.append(res)
            
    # Summary
    print("\n--- Summary ---")
    for r in results:
        print(f"{r['file']}: Gini={r['gini']:.3f}, AvgDeg={r['avg_degree']:.2f}")

if __name__ == "__main__":
    main()
