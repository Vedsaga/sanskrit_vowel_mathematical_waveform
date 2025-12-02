import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import librosa
import nolds
from sklearn.preprocessing import MinMaxScaler
import argparse

import plotly.graph_objects as go

def analyze_phase_space(file_path, output_dir, duration_ms=None, show_plot=False):
    """
    Analyzes the phase space of a .wav file using 3D reconstruction and Plotly.
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
        # We need to choose an embedding dimension (m) and time delay (tau)
        
        # Calculate tau using autocorrelation
        ac = librosa.autocorrelate(y_norm, max_size=2000)
        zero_crossings = np.where(np.diff(np.sign(ac)))[0]
        if len(zero_crossings) > 0:
            tau = zero_crossings[0]
        else:
            tau = 10 # Fallback
            
        m = 3 # For 3D plot
        
        # Create time-delay embedding
        # X(t) = [y(t), y(t + tau), y(t + 2*tau)]
        
        y_delay1 = np.roll(y_norm, -tau)
        y_delay2 = np.roll(y_norm, -2*tau)
        
        # Trim the end
        trim = 2 * tau
        x_val = y_norm[:-trim]
        y_val = y_delay1[:-trim]
        z_val = y_delay2[:-trim]
        
        # Create Plotly 3D Scatter plot
        
        # Create a color scale based on time
        colors = np.arange(len(x_val))
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_val,
            y=y_val,
            z=z_val,
            mode='lines',
            opacity=0.8,
            line=dict(
                color=colors,
                colorscale='Viridis',
                width=2
            ),
            name='Trajectory'
        )])
        
        # Extract vowel from parent directory name for title
        vowel = os.path.basename(os.path.dirname(file_path))
        
        title_text = f'Phase Space Reconstruction: {os.path.basename(file_path)}<br>(tau={tau}, m={m})'
        if duration_ms:
            title_text += f'<br>First {duration_ms}ms'
            
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(color='#EAEAEA')
            ),
            scene=dict(
                xaxis=dict(title=dict(text='y(t)', font=dict(color='#EAEAEA')), backgroundcolor='#111111', gridcolor='#333333', showbackground=True, zerolinecolor='#333333', tickfont=dict(color='#EAEAEA')),
                yaxis=dict(title=dict(text=f'y(t + {tau})', font=dict(color='#EAEAEA')), backgroundcolor='#111111', gridcolor='#333333', showbackground=True, zerolinecolor='#333333', tickfont=dict(color='#EAEAEA')),
                zaxis=dict(title=dict(text=f'y(t + {2*tau})', font=dict(color='#EAEAEA')), backgroundcolor='#111111', gridcolor='#333333', showbackground=True, zerolinecolor='#333333', tickfont=dict(color='#EAEAEA')),
                bgcolor='#111111'
            ),
            paper_bgcolor='#111111',
            plot_bgcolor='#111111',
            margin=dict(l=0, r=0, b=0, t=50),
            annotations=[
                dict(
                    x=0,
                    y=1,
                    xref='paper',
                    yref='paper',
                    text=f"Vowel: {vowel}",
                    showarrow=False,
                    font=dict(
                        size=20,
                        color='#EAEAEA'
                    ),
                    bgcolor='#111111',
                    bordercolor='#EAEAEA',
                    borderwidth=1,
                    borderpad=4
                )
            ]
        )
        
        # Save plot as HTML
        filename = os.path.basename(file_path).replace('.wav', '_phase_space.html')
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)
        
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
