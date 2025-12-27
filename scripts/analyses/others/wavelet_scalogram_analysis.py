import os
import glob
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def analyze_wavelet_scalogram(file_path, output_dir, duration_ms=None, use_phoneme_subdir=True):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Pad signal if it's too short for CQT
        # CQT needs enough samples for the lowest frequency analysis
        min_samples = 1024 # Heuristic safe margin
        if len(y) < min_samples:
            y = librosa.util.fix_length(y, size=min_samples)
        
        # Compute Constant-Q Transform (CQT)
        # CQT is a wavelet transform with logarithmically spaced frequencies.
        # It provides the "Zoom Lens" effect: good time resolution at high freqs, 
        # good freq resolution at low freqs.
        
        # hop_length=512 is standard, but for short duration we might want smaller
        hop_length = 64 if (duration_ms and duration_ms < 200) else 512
        
        C = librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=72, bins_per_octave=12)
        
        # Convert to dB (Magnitude)
        C_dB = librosa.amplitude_to_db(np.abs(C), ref=np.max)
        
        # Plot
        plt.figure(figsize=(10, 4), facecolor='#111111')
        ax = plt.gca()
        ax.set_facecolor='#111111'
        
        img = librosa.display.specshow(C_dB, sr=sr, x_axis='time', y_axis='cqt_note', 
                                     hop_length=hop_length, ax=ax, cmap='inferno')
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#EAEAEA')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#EAEAEA')
        
        # Labels and Title
        # Use default font for main title (filename)
        plt.title(f'Scalogram (CQT): {os.path.basename(file_path)}', color='#EAEAEA', pad=20)
        
        # Add Devanagari Phoneme as a separate text annotation in the corner
        phoneme = os.path.basename(os.path.dirname(file_path))
        import matplotlib.font_manager as fm
        font_path = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'
        if os.path.exists(font_path):
            prop = fm.FontProperties(fname=font_path, size=20)
            plt.text(0.02, 0.95, phoneme, transform=ax.transAxes, fontproperties=prop, 
                     color='#EAEAEA', verticalalignment='top', 
                     bbox=dict(facecolor='#111111', edgecolor='#EAEAEA', alpha=0.7))

        plt.xlabel('Time', color='#EAEAEA')
        plt.ylabel('Frequency (Note)', color='#EAEAEA')
        
        ax.tick_params(axis='x', colors='#EAEAEA')
        ax.tick_params(axis='y', colors='#EAEAEA')
        
        # Save
        if use_phoneme_subdir:
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            save_dir = phoneme_dir
        else:
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_scalogram.png')
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=300, facecolor='#111111', bbox_inches='tight')
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Wavelet Scalogram Analysis (CQT)')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='results/wavelet_scalograms', help='Output directory')
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
        if analyze_wavelet_scalogram(wav_file, args.output_dir, duration_ms=args.duration_ms):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
