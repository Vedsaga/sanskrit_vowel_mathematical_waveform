import os
import glob
import argparse
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def analyze_mel_spectrogram(file_path, output_dir, duration_ms=None, use_phoneme_subdir=True):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        
        # Slice to specific duration if requested
        if duration_ms is not None:
            samples = int(sr * duration_ms / 1000)
            if len(y) > samples:
                y = y[:samples]
        
        # Compute Mel Spectrogram
        # n_mels=128 is standard for high quality
        # fmax=8000 is good for speech
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        
        # Convert to dB (log scale)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot
        plt.figure(figsize=(10, 4), facecolor='#111111')
        ax = plt.gca()
        ax.set_facecolor='#111111'
        
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax, cmap='magma')
        
        # Colorbar
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#EAEAEA')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#EAEAEA')
        
        # Labels and Title
        # Use default font for main title (filename)
        plt.title(f'Mel-Spectrogram: {os.path.basename(file_path)}', color='#EAEAEA', pad=20)
        
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
        plt.ylabel('Frequency (Hz)', color='#EAEAEA')
        
        ax.tick_params(axis='x', colors='#EAEAEA')
        ax.tick_params(axis='y', colors='#EAEAEA')
        
        # Save
        if use_phoneme_subdir:
            phoneme = os.path.basename(os.path.dirname(file_path))
            phoneme_dir = os.path.join(output_dir, phoneme)
            os.makedirs(phoneme_dir, exist_ok=True)
            save_dir = phoneme_dir
        else:
            save_dir = output_dir
            os.makedirs(save_dir, exist_ok=True)
        
        filename = os.path.basename(file_path).replace('.wav', '_mel_spectrogram.png')
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=300, facecolor='#111111', bbox_inches='tight')
        plt.close()
        
        print(f"Processed: {file_path} -> {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Mel-Spectrogram Analysis')
    parser.add_argument('--input_dir', type=str, default='data/02_cleaned', help='Input directory')
    parser.add_argument('--output_dir', type=str, default='results/mel_spectrograms', help='Output directory')
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
        if analyze_mel_spectrogram(wav_file, args.output_dir, duration_ms=args.duration_ms):
            count += 1
            
    print(f"Completed analysis. Processed {count} files.")

if __name__ == "__main__":
    main()
