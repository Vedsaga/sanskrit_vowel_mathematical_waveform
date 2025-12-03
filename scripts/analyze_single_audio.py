import os
import argparse
import sys
import glob

# Add the parent directory to sys.path to allow importing from scripts.analyses
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.analyses import (
    gaf_analysis,
    mel_spectrogram_analysis,
    phase_space_analysis_2d,
    phase_space_analysis_3d,
    poincare_section_analysis,
    tda_barcode_analysis,
    visibility_graph_analysis,
    wavelet_scalogram_analysis
)

def analyze_single_audio(file_path, output_base_dir, duration_ms=None):
    """
    Runs all available analyses on a single audio file.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(filename)[0]
    
    # Create a specific directory for this file's results
    output_dir = os.path.join(output_base_dir, name_without_ext)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing {filename}...")
    print(f"Results will be saved to: {output_dir}")
    
    results = {}
    
    # 1. Mel Spectrogram
    print("\n[1/7] Running Mel Spectrogram Analysis...")
    res = mel_spectrogram_analysis.analyze_mel_spectrogram(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['Mel Spectrogram'] = res

    # 2. Wavelet Scalogram
    print("\n[2/7] Running Wavelet Scalogram Analysis...")
    res = wavelet_scalogram_analysis.analyze_wavelet_scalogram(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['Wavelet Scalogram'] = res

    # 3. GAF Analysis
    print("\n[3/7] Running GAF Analysis...")
    res = gaf_analysis.analyze_gaf(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['GAF'] = res

    # 4. Phase Space 2D
    print("\n[4/7] Running Phase Space Analysis (2D)...")
    res = phase_space_analysis_2d.analyze_phase_space(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['Phase Space 2D'] = res

    # 5. Phase Space 3D
    print("\n[5/7] Running Phase Space Analysis (3D)...")
    res = phase_space_analysis_3d.analyze_phase_space(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['Phase Space 3D'] = res

    # 6. Poincaré Section
    print("\n[6/7] Running Poincaré Section Analysis...")
    res = poincare_section_analysis.analyze_poincare(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    if isinstance(res, dict):
        results['Poincare'] = res.get('output_path')
    else:
        results['Poincare'] = res

    # 7. Visibility Graph
    print("\n[7/7] Running Visibility Graph Analysis...")
    res = visibility_graph_analysis.analyze_visibility_graph(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    if isinstance(res, dict):
        results['Visibility Graph'] = res.get('output_path')
    else:
        results['Visibility Graph'] = res
        
    # TDA Barcode (Optional/Heavy)
    print("\n[Optional] Running TDA Barcode Analysis...")
    res = tda_barcode_analysis.analyze_tda_barcode(
        file_path, output_dir, duration_ms=duration_ms, use_phoneme_subdir=False
    )
    results['TDA Barcode'] = res

    print("\n--- Analysis Complete ---")
    print(f"All results saved in: {output_dir}")
    
    # Generate a simple index.html to view results
    generate_report(output_dir, filename, results)

def generate_report(output_dir, filename, results):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Report: {filename}</title>
        <style>
            body {{ font-family: sans-serif; background-color: #111; color: #eee; padding: 20px; }}
            h1 {{ color: #fff; border-bottom: 1px solid #333; padding-bottom: 10px; }}
            .container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .card {{ background: #222; padding: 15px; border-radius: 8px; width: 45%; min-width: 400px; }}
            .card h2 {{ margin-top: 0; color: #ddd; font-size: 1.2em; }}
            img {{ max_width: 100%; height: auto; border-radius: 4px; }}
            iframe {{ width: 100%; height: 500px; border: none; }}
            a {{ color: #4da6ff; }}
        </style>
    </head>
    <body>
        <h1>Analysis Report: {filename}</h1>
        <div class="container">
    """
    
    for name, path in results.items():
        if path:
            rel_path = os.path.basename(path)
            html_content += f"""
            <div class="card">
                <h2>{name}</h2>
            """
            if path.endswith('.png'):
                html_content += f'<img src="{rel_path}" alt="{name}">'
            elif path.endswith('.html'):
                html_content += f'<iframe src="{rel_path}"></iframe><br><a href="{rel_path}" target="_blank">Open Fullscreen</a>'
            
            html_content += "</div>"
            
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "report.html"), "w") as f:
        f.write(html_content)
    
    print(f"Report generated: {os.path.join(output_dir, 'report.html')}")

def main():
    parser = argparse.ArgumentParser(description='Run all analyses on a single audio file')
    parser.add_argument('input_file', type=str, help='Path to the input .wav file')
    parser.add_argument('--output_dir', type=str, default='results/single_analysis', help='Base output directory')
    parser.add_argument('--duration_ms', type=int, default=None, help='Duration in ms to analyze')
    
    args = parser.parse_args()
    
    analyze_single_audio(args.input_file, args.output_dir, args.duration_ms)

if __name__ == "__main__":
    main()
