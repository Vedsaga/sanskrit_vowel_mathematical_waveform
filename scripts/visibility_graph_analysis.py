import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from ts2vg import NaturalVG
import networkx as nx
from collections import Counter

def analyze_visibility_graph(file_path, output_dir, duration_ms=None, show_plot=False):
    """
    Analyzes the visibility graph of a .wav file.
    """
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Downsample if too large for graph analysis (optional, but recommended for speed)
        # For now, we'll keep it raw but maybe limit the max nodes if needed.
        # If > 5000 samples, it might be slow. Let's warn or downsample.
        if len(y) > 5000:
            print(f"Warning: Signal length {len(y)} is large. Downsampling to ~5000 points for performance.")
            step = len(y) // 5000
            y = y[::step]

        # Convert to Visibility Graph
        vg = NaturalVG()
        vg.build(y)
        
        # Get degree distribution directly from ts2vg for efficiency
        degrees = vg.degrees
        
        # Calculate Metrics
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        degree_variance = np.var(degrees)
        
        # Create NetworkX graph for other properties (optional, can be slow for large graphs)
        # nx_graph = vg.as_networkx() 
        
        # Plot Degree Distribution
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', rwidth=0.8, color='skyblue', edgecolor='black')
        plt.title('Degree Distribution')
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        
        # Log-Log Plot
        degree_counts = Counter(degrees)
        deg, count = zip(*sorted(degree_counts.items()))
        
        plt.subplot(1, 2, 2)
        plt.loglog(deg, count, 'o', markersize=5, color='orange')
        plt.title('Log-Log Degree Distribution')
        plt.xlabel('Degree (log)')
        plt.ylabel('Frequency (log)')
        
        plt.tight_layout()
        
        # Save Plot
        # Extract phoneme (parent directory name)
        phoneme = os.path.basename(os.path.dirname(file_path))
        phoneme_dir = os.path.join(output_dir, phoneme)
        os.makedirs(phoneme_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_vg_degree_dist.png')
        output_path = os.path.join(phoneme_dir, filename)
        plt.savefig(output_path)
        plt.close()
        
        # Determine Network Type (Heuristic)
        # Random (Poisson) vs Scale-Free (Power Law) vs Periodic (Regular)
        # This is a simplification.
        
        # Check for Hubs (High max degree relative to average)
        hub_ratio = max_degree / avg_degree
        
        network_type = "Unknown"
        if degree_variance < 1.0: # Very uniform
            network_type = "Periodic/Regular"
        elif hub_ratio > 5.0: # Significant hubs
            network_type = "Scale-Free (Chaos)"
        else:
            network_type = "Random (Noise)"

        print(f"Processed: {file_path}")
        print(f"  Nodes: {len(y)}")
        print(f"  Avg Degree: {avg_degree:.2f}")
        print(f"  Max Degree: {max_degree}")
        print(f"  Hub Ratio: {hub_ratio:.2f}")
        print(f"  Type: {network_type}")
        print(f"  Saved plot to: {output_path}")
        
        return {
            'file': os.path.basename(file_path),
            'nodes': len(y),
            'avg_degree': avg_degree,
            'max_degree': max_degree,
            'variance': degree_variance,
            'type': network_type
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Visibility Graph Analysis of WAV files')
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
        print(f"{r['file']}: {r['type']} (Avg Deg: {r['avg_degree']:.2f})")

if __name__ == "__main__":
    main()
