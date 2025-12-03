import os
import glob
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField

def analyze_gaf(file_path, output_dir, duration_ms=None):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Downsample for GAF
        # GAF creates an N x N image. If N is too large (e.g. 4000 samples), the image is huge (16MP).
        # We want to see the texture/pattern. 
        # For a 100ms clip at 44.1kHz, we have 4410 samples.
        # Let's downsample to something manageable like 200-300 points.
        # This preserves the shape but reduces resolution.
        target_length = 256
        if len(y) > target_length:
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=sr * (target_length / len(y)))
            # Ensure exact length (resample might be off by a few samples)
            if len(y_resampled) > target_length:
                y_resampled = y_resampled[:target_length]
            elif len(y_resampled) < target_length:
                y_resampled = librosa.util.fix_length(y_resampled, size=target_length)
        else:
            y_resampled = librosa.util.fix_length(y, size=target_length)
            
        # Reshape for pyts (n_samples, n_timestamps)
        X = y_resampled.reshape(1, -1)
        
        # Compute GAF
        # method='summation' (GASF) or 'difference' (GADF)
        # GASF preserves temporal correlation better along diagonal
        gaf = GramianAngularField(image_size=target_length, method='summation')
        X_gaf = gaf.fit_transform(X)
        
        # Plot
        plt.figure(figsize=(6, 6), facecolor='#111111')
        ax = plt.gca()
        ax.set_facecolor='#111111'
        
        # Plot the first (and only) image
        img = plt.imshow(X_gaf[0], cmap='rainbow', origin='lower', extent=[0, 1, 0, 1])
        
        # Remove axes for pure texture view, or keep them minimal
        plt.axis('off')
        
        # Title
        # Use default font for main title (filename)
        plt.title(f'GAF (GASF): {os.path.basename(file_path)}', color='#EAEAEA', pad=10)
        
        # Add Devanagari Phoneme as a separate text annotation in the corner
        phoneme = os.path.basename(os.path.dirname(file_path))
        import matplotlib.font_manager as fm
        font_path = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path, size=20)
            plt.text(0.02, 0.95, phoneme, transform=ax.transAxes, fontproperties=prop, 
                     color='#EAEAEA', verticalalignment='top', 
                     bbox=dict(facecolor='#111111', edgecolor='#EAEAEA', alpha=0.7))

        # Save
        phoneme_dir = os.path.join(output_dir, phoneme)
        os.makedirs(phoneme_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_gaf.png')
        output_path = os.path.join(phoneme_dir, filename)
        plt.savefig(output_path, dpi=300, facecolor='#111111', bbox_inches='tight')
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Gramian Angular Field (GAF) Analysis')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='results/gaf_plots', help='Output directory')
    parser.add_argument('--limit', type=int, default=None, help='Limit files')
    parser.add_argument('--duration_ms', type=int, default=None, help='Duration (ms)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    search_pattern = os.path.join(args.input_dir, '**', '*.wav')
    wav_files = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(wav_files)} .wav files")
    
    if args.limit:
        wav_files = wav_files[:args.limit]
        
    count = 0
    for wav_file in wav_files:
        if analyze_gaf(wav_file, args.output_dir, duration_ms=args.duration_ms):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
