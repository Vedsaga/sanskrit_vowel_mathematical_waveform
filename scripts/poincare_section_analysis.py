import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist, squareform

def get_poincare_section(trajectory, plane_normal, plane_point):
    """
    Finds intersection points of the trajectory with a plane.
    trajectory: (N, 3) array
    plane_normal: (3,) array
    plane_point: (3,) array
    """
    # Project trajectory onto normal
    dist = np.dot(trajectory - plane_point, plane_normal)
    
    # Find crossings (sign change)
    # We only care about one direction (e.g., negative to positive) to avoid double counting loops
    crossings = np.where((dist[:-1] < 0) & (dist[1:] > 0))[0]
    
    intersections = []
    for i in crossings:
        p1 = trajectory[i]
        p2 = trajectory[i+1]
        d1 = dist[i]
        d2 = dist[i+1]
        
        # Linear interpolation
        alpha = -d1 / (d2 - d1)
        intersection = p1 + alpha * (p2 - p1)
        intersections.append(intersection)
        
    return np.array(intersections)

def correlation_dimension(points, k_neighbors=None):
    """
    Estimates Correlation Dimension (D2) of a point cloud.
    """
    if len(points) < 10:
        return 0.0
        
    # Compute pairwise distances
    dists = pdist(points)
    
    if len(dists) == 0:
        return 0.0
        
    # Log-log plot of Correlation Sum C(r)
    # C(r) = fraction of pairs with distance < r
    
    # We use a range of r values
    r_vals = np.logspace(np.log10(np.min(dists[dists > 0])), np.log10(np.max(dists)), num=20)
    c_r = []
    
    for r in r_vals:
        count = np.sum(dists < r)
        c_r.append(count / len(dists))
        
    # Fit line to linear region of log-log
    # Heuristic: take the middle 50% of points where C(r) is not 0 or 1
    valid = (np.array(c_r) > 0) & (np.array(c_r) < 1)
    if np.sum(valid) < 3:
        return 0.0 # Not enough data
        
    log_r = np.log(r_vals[valid])
    log_c = np.log(np.array(c_r)[valid])
    
    # Linear regression
    slope, _ = np.polyfit(log_r, log_c, 1)
    return slope

def analyze_poincare(file_path, output_dir, duration_ms=None):
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
        ac = librosa.autocorrelate(y_norm, max_size=2000)
        zero_crossings = np.where(np.diff(np.sign(ac)))[0]
        tau = zero_crossings[0] if len(zero_crossings) > 0 else 10
        m = 3
        
        # Create Embedding
        N = len(y_norm) - (m - 1) * tau
        trajectory = np.zeros((N, m))
        for i in range(m):
            trajectory[:, i] = y_norm[i*tau : i*tau + N]
            
        # Define Poincaré Section Plane
        # We'll cut through the origin (mean is approx 0) along the 2nd axis (y-axis in 3D)
        # Plane: y = 0
        plane_normal = np.array([0, 1, 0])
        plane_point = np.array([0, 0, 0])
        
        points_3d = get_poincare_section(trajectory, plane_normal, plane_point)
        
        # Project to 2D (drop the dimension corresponding to the normal)
        # Since normal is y-axis, we keep x and z
        if len(points_3d) > 0:
            points_2d = points_3d[:, [0, 2]]
        else:
            points_2d = np.array([])
            
        # Calculate Fractal Dimension
        fractal_dim = correlation_dimension(points_2d)
        
        # Plot
        plt.figure(figsize=(8, 8), facecolor='#111111')
        ax = plt.gca()
        ax.set_facecolor('#111111')
        
        if len(points_2d) > 0:
            plt.scatter(points_2d[:, 0], points_2d[:, 1], s=10, c='cyan', alpha=0.7, edgecolors='none')
        else:
            plt.text(0, 0, "No intersection points", color='white', ha='center')
            
        plt.title(f"Poincaré Section\nFractal Dim: {fractal_dim:.2f}", color='#EAEAEA')
        plt.xlabel('x(t)', color='#EAEAEA')
        plt.ylabel(f'x(t + {2*tau})', color='#EAEAEA')
        
        ax.tick_params(colors='#EAEAEA')
        for spine in ax.spines.values():
            spine.set_edgecolor('#EAEAEA')
        plt.grid(True, alpha=0.2, color='#EAEAEA')
        
        # Save
        phoneme = os.path.basename(os.path.dirname(file_path))
        phoneme_dir = os.path.join(output_dir, phoneme)
        os.makedirs(phoneme_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_poincare.png')
        output_path = os.path.join(phoneme_dir, filename)
        plt.savefig(output_path, dpi=300, facecolor='#111111')
        plt.close()
        
        print(f"Processed {os.path.basename(file_path)} | Dim: {fractal_dim:.2f} | Points: {len(points_2d)}")
        return {
            'file': os.path.basename(file_path),
            'fractal_dim': fractal_dim,
            'points': len(points_2d)
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Poincaré Section Analysis')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='results/poincare_plots', help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit files')
    parser.add_argument('--duration_ms', type=int, default=100, help='Duration (ms)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    search_pattern = os.path.join(args.input_dir, '**', '*.wav')
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} .wav files")
    
    if args.limit:
        wav_files = wav_files[:args.limit]
        
    results = []
    for wav_file in wav_files:
        res = analyze_poincare(wav_file, args.output_dir, duration_ms=args.duration_ms)
        if res:
            results.append(res)
            
    print("\n--- Summary ---")
    for r in results:
        print(f"{r['file']}: Dim={r['fractal_dim']:.2f} (Pts: {r['points']})")

if __name__ == "__main__":
    main()
