#!/usr/bin/env python3
"""
Verification Script: Check for Weighting Bias

This script runs formant analysis on a given audio file and compares the results
of the "Joint Stability-Intensity Weighting" (Method 3) against "Unweighted" (Raw) statistics.

It produces:
1. A text report showing the delta (Weighted - Unweighted).
2. A plot showing the raw trajectories with the weights overlaid, demonstrating
   which parts of the signal strongly influence the weighted result.

Usage:
    python3 check_weighting_bias.py --file <path_to_wav>
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add specific folder to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts/analyses/formant-based-invariant')))

import formant_ratio_analysis

def main():
    parser = argparse.ArgumentParser(description="Check for weighting bias in formant analysis")
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--output_dir", default="verification_results/bias_check", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Analyzing: {args.file}")
    
    # 1. Extract Data (Use existing script logic)
    data = formant_ratio_analysis.extract_formants(args.file)
    
    if data is None:
        print("Error: Could not extract formants.")
        return

    # 2. Get Raw Arrays and Weights
    f1_raw = data['f1_values']
    f2_raw = data['f2_values']
    f3_raw = data['f3_values']
    time = data['time_values']
    weights = data['frame_weights']  # Renamed from stability_weights
    weights_norm = data['frame_weights_norm']
    weight_entropy = data.get('weight_entropy', 0)
    n_eff = data.get('n_eff', 0)
    confidence = data.get('confidence', 0)
    
    # 3. Calculate UNWEIGHTED Metrics (Simple Mean)
    f1_unweighted = np.mean(f1_raw)
    f2_unweighted = np.mean(f2_raw)
    f3_unweighted = np.mean(f3_raw)
    
    # 4. Get WEIGHTED Metrics (From Script)
    f1_weighted = data['f1_mean']
    f2_weighted = data['f2_mean']
    f3_weighted = data['f3_mean']
    
    # 5. Calculate Delta
    f1_delta = f1_weighted - f1_unweighted
    f2_delta = f2_weighted - f2_unweighted
    f3_delta = f3_weighted - f3_unweighted
    
    # 6. Report
    report = f"""
==================================================
WEIGHTING BIAS REPORT
File: {os.path.basename(args.file)}
Method: Joint Stability-Intensity Weighting (Improved)
==================================================

METRIC      | WEIGHTED (New) | UNWEIGHTED (Raw) | DELTA (W - U)
--------------------------------------------------------------
F1 Mean     | {f1_weighted:8.2f} Hz | {f1_unweighted:8.2f} Hz   | {f1_delta:+8.2f} Hz
F2 Mean     | {f2_weighted:8.2f} Hz | {f2_unweighted:8.2f} Hz   | {f2_delta:+8.2f} Hz
F3 Mean     | {f3_weighted:8.2f} Hz | {f3_unweighted:8.2f} Hz   | {f3_delta:+8.2f} Hz

DIAGNOSTICS:
--------------------------------------------------------------
N_eff (Effective Frames) : {n_eff:8.2f} / {len(f1_raw)} total
Confidence               : {confidence:8.4f}
Weight Entropy (Norm)    : {weight_entropy:8.4f}  (0=concentrated, 1=uniform)

INTERPRETATION:
- A POSITIVE delta means weighting favors higher frequencies (likely ignoring low-freq onset).
- A NEGATIVE delta means weighting favors lower frequencies.
- The goal is for the weighted mean to be closer to the "steady state" target.
- Low weight entropy indicates weights are concentrated on few frames.
"""
    print(report)
    
    with open(os.path.join(args.output_dir, "bias_report.txt"), "w") as f:
        f.write(report)
        
    # 7. Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor('#111111')
    
    # Plot F1, F2
    ax = axes[0]
    ax.set_facecolor('#1a1a1a')
    ax.plot(time, f1_raw, color='#FF6B6B', alpha=0.9, label='F1 (Raw)')
    ax.plot(time, f2_raw, color='#4ECDC4', alpha=0.9, label='F2 (Raw)')
    ax.axhline(f1_weighted, color='#FF6B6B', linestyle='--', linewidth=2, label=f'F1 Weighted Mean: {f1_weighted:.0f}')
    ax.axhline(f1_unweighted, color='#FF6B6B', linestyle=':', linewidth=1.5, alpha=0.7, label=f'F1 Unweighted: {f1_unweighted:.0f}')
    ax.legend(facecolor='#1a1a1a', edgecolor='#333', labelcolor='#EAEAEA')
    ax.set_ylabel('Frequency (Hz)', color='#EAEAEA')
    ax.set_title('Formant Trajectories & Means', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    
    # Plot Weights
    ax = axes[1]
    ax.set_facecolor('#1a1a1a')
    
    # Normalize weights for plotting visibility
    w_norm = weights / np.max(weights) if np.max(weights) > 0 else weights
    
    ax.fill_between(time, 0, w_norm, color='#FFE66D', alpha=0.4, label='Weight Magnitude')
    ax.plot(time, w_norm, color='#FFE66D', linewidth=1.5)
    ax.set_ylabel('Normalized Weight', color='#EAEAEA')
    ax.set_xlabel('Time (s)', color='#EAEAEA')
    ax.set_title('Weighting Profile (High = Stable & Intense)', color='#EAEAEA', fontweight='bold')
    ax.tick_params(colors='#EAEAEA')
    ax.grid(True, alpha=0.1, color='#EAEAEA')
    
    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, "weight_overlay.png")
    plt.savefig(plot_path, dpi=150, facecolor='#111111')
    print(f"Visualization saved to: {plot_path}")

if __name__ == "__main__":
    main()
