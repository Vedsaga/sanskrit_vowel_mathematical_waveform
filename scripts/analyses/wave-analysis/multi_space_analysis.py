#!/usr/bin/env python3
"""
Multi-Space Convergence Analysis

Runs all 4 feature space analyses (scattering, formant, MFCC, spectral)
and aggregates results to show consensus across different representations.

Each feature space is an independent test - convergence across multiple
spaces provides stronger evidence than any single analysis.

Usage:
    python multi_space_analysis.py --wave1 ka.wav --wave2 a.wav
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all analysis modules
from wave_comparison_analysis import (
    load_wave as load_wave_scat,
    analyze_convergence as analyze_scattering,
    create_visualization as viz_scattering,
    save_results as save_scattering,
)
from formant_convergence import (
    load_wave as load_wave_formant,
    analyze_convergence as analyze_formant,
    create_visualization as viz_formant,
    save_results as save_formant,
)
from mfcc_similarity import (
    load_wave as load_wave_mfcc,
    analyze_convergence as analyze_mfcc,
    create_visualization as viz_mfcc,
    save_results as save_mfcc,
)
from spectral_analysis import (
    load_wave as load_wave_spectral,
    analyze_convergence as analyze_spectral,
    create_visualization as viz_spectral,
    save_results as save_spectral,
)

# Import font config
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.config import configure_matplotlib
    configure_matplotlib()
except ImportError:
    plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']


@dataclass
class MultiSpaceResult:
    """Aggregated results from all feature spaces."""
    results: Dict[str, any]  # name -> ConvergenceResult
    consensus_count: int
    total_spaces: int
    summary_df: pd.DataFrame


def run_all_analyses(wave1_path: str, wave2_path: str,
                      output_dir: str, n_permutations: int = 1000,
                      visualize: bool = True) -> MultiSpaceResult:
    """Run all 4 feature space analyses."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    print("\n" + "=" * 60)
    print("1/4: SCATTERING ANALYSIS")
    print("=" * 60)
    try:
        wave_A = load_wave_scat(wave1_path)
        wave_B = load_wave_scat(wave2_path)
        result = analyze_scattering(wave_A, wave_B, n_permutations=n_permutations)
        results['Scattering'] = result
        scat_dir = os.path.join(output_dir, 'scattering')
        save_scattering(wave_A, wave_B, result, scat_dir)
        if visualize:
            viz_scattering(wave_A, wave_B, result, scat_dir)
        print(f"  Result: {'✓' if result.converges else '✗'} "
              f"reduction={result.distance_reduction:.1%}, p={result.p_value:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Scattering'] = None
    
    print("\n" + "=" * 60)
    print("2/4: FORMANT ANALYSIS")
    print("=" * 60)
    try:
        wave_A = load_wave_formant(wave1_path)
        wave_B = load_wave_formant(wave2_path)
        result = analyze_formant(wave_A, wave_B, n_permutations=n_permutations)
        results['Formant'] = result
        form_dir = os.path.join(output_dir, 'formant')
        save_formant(wave_A, wave_B, result, form_dir)
        if visualize:
            viz_formant(wave_A, wave_B, result, form_dir)
        print(f"  Result: {'✓' if result.converges else '✗'} "
              f"reduction={result.distance_reduction:.1%}, p={result.p_value:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Formant'] = None
    
    print("\n" + "=" * 60)
    print("3/4: MFCC ANALYSIS")
    print("=" * 60)
    try:
        wave_A = load_wave_mfcc(wave1_path)
        wave_B = load_wave_mfcc(wave2_path)
        result = analyze_mfcc(wave_A, wave_B, n_permutations=n_permutations)
        results['MFCC'] = result
        mfcc_dir = os.path.join(output_dir, 'mfcc')
        save_mfcc(wave_A, wave_B, result, mfcc_dir)
        if visualize:
            viz_mfcc(wave_A, wave_B, result, mfcc_dir)
        print(f"  Result: {'✓' if result.converges else '✗'} "
              f"reduction={result.distance_reduction:.1%}, p={result.p_value:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['MFCC'] = None
    
    print("\n" + "=" * 60)
    print("4/4: SPECTRAL ANALYSIS")
    print("=" * 60)
    try:
        wave_A = load_wave_spectral(wave1_path)
        wave_B = load_wave_spectral(wave2_path)
        result = analyze_spectral(wave_A, wave_B, n_permutations=n_permutations)
        results['Spectral'] = result
        spec_dir = os.path.join(output_dir, 'spectral')
        save_spectral(wave_A, wave_B, result, spec_dir)
        if visualize:
            viz_spectral(wave_A, wave_B, result, spec_dir)
        print(f"  Result: {'✓' if result.converges else '✗'} "
              f"reduction={result.distance_reduction:.1%}, p={result.p_value:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results['Spectral'] = None
    
    # Create summary
    rows = []
    for name, result in results.items():
        if result is not None:
            rows.append({
                'Feature Space': name,
                'Converges': '✓' if result.converges else '✗',
                'P-value': f"{result.p_value:.4f}",
                'Distance Reduction': f"{result.distance_reduction:.1%}",
                'Significant': result.p_value < 0.05,
            })
        else:
            rows.append({
                'Feature Space': name,
                'Converges': 'ERROR',
                'P-value': '-',
                'Distance Reduction': '-',
                'Significant': False,
            })
    
    df = pd.DataFrame(rows)
    
    consensus_count = sum(1 for r in results.values() 
                          if r is not None and r.converges)
    total_spaces = sum(1 for r in results.values() if r is not None)
    
    return MultiSpaceResult(
        results=results,
        consensus_count=consensus_count,
        total_spaces=total_spaces,
        summary_df=df,
    )


def create_summary_visualization(multi_result: MultiSpaceResult,
                                  wave1_name: str, wave2_name: str,
                                  output_dir: str):
    """Create combined summary visualization."""
    
    BG = '#111111'
    PANEL = '#1a1a1a'
    TEXT = '#eaeaea'
    CONV = '#2ECC71'
    NO_CONV = '#FF6B6B'
    GRID = '#333333'
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor(BG)
    
    df = multi_result.summary_df
    
    # Panel 1: Bar chart of p-values
    ax = axes[0]
    ax.set_facecolor(PANEL)
    
    names = df['Feature Space'].tolist()
    p_vals = []
    colors = []
    for _, row in df.iterrows():
        if row['P-value'] != '-':
            p = float(row['P-value'])
            p_vals.append(p)
            colors.append(CONV if p < 0.05 else NO_CONV)
        else:
            p_vals.append(1.0)
            colors.append('#666666')
    
    bars = ax.bar(names, p_vals, color=colors, alpha=0.8, edgecolor='white')
    ax.axhline(0.05, color='#FFD93D', ls='--', lw=2, label='α = 0.05')
    ax.set_ylabel('P-value', color=TEXT)
    ax.set_title('Statistical Significance by Feature Space', 
                color=TEXT, fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    ax.tick_params(colors=TEXT)
    ax.grid(axis='y', alpha=0.2, color=GRID)
    
    # Add significance markers
    for i, bar in enumerate(bars):
        if p_vals[i] < 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   '★', ha='center', va='bottom', color=CONV, fontsize=20)
    
    # Panel 2: Summary table
    ax = axes[1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    consensus = multi_result.consensus_count
    total = multi_result.total_spaces
    
    # Build summary text
    summary = f"""
MULTI-SPACE CONVERGENCE ANALYSIS
{'═' * 50}

Trajectory: {wave1_name}
Reference:  {wave2_name}

{'─' * 50}
RESULTS BY FEATURE SPACE
{'─' * 50}
"""
    
    for _, row in df.iterrows():
        converge = row['Converges']
        summary += f"  {row['Feature Space']:12} | {converge:^8} | "
        summary += f"p={row['P-value']:>6} | Δ={row['Distance Reduction']:>5}\n"
    
    summary += f"""
{'═' * 50}
CONSENSUS: {consensus}/{total} spaces show significant convergence
{'═' * 50}

INTERPRETATION:
"""
    
    if consensus >= 3:
        summary += "Strong evidence of trajectory convergence across\nmultiple independent feature representations."
    elif consensus >= 2:
        summary += "Moderate evidence of convergence in some feature\nspaces. Results are mixed."
    else:
        summary += "Weak or no evidence of consistent convergence\nacross feature spaces."
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            va='top', color=TEXT, family='monospace')
    
    fig.suptitle(f'Multi-Space Analysis: {wave1_name} → {wave2_name}',
                 color=TEXT, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    path = os.path.join(output_dir, 'multi_space_summary.png')
    plt.savefig(path, dpi=300, facecolor=BG, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary visualization: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(
        description='Multi-space convergence analysis',
        epilog="""
Runs 4 independent analyses (scattering, formant, MFCC, spectral)
and reports consensus across feature spaces.
"""
    )
    
    parser.add_argument('--wave1', required=True, help='Trajectory wave')
    parser.add_argument('--wave2', required=True, help='Reference wave')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    
    args = parser.parse_args()
    
    if not args.output_dir:
        n1 = os.path.splitext(os.path.basename(args.wave1))[0]
        n2 = os.path.splitext(os.path.basename(args.wave2))[0]
        args.output_dir = f'results/multi_space/{n1}_vs_{n2}'
    
    wave1_name = os.path.splitext(os.path.basename(args.wave1))[0]
    wave2_name = os.path.splitext(os.path.basename(args.wave2))[0]
    
    print("\n" + "=" * 60)
    print("MULTI-SPACE TRAJECTORY CONVERGENCE ANALYSIS")
    print("=" * 60)
    print(f"\nTrajectory: {wave1_name}")
    print(f"Reference:  {wave2_name}")
    print(f"Output:     {args.output_dir}")
    
    result = run_all_analyses(
        args.wave1, args.wave2, args.output_dir,
        n_permutations=args.n_permutations,
        visualize=args.visualize
    )
    
    # Save summary CSV
    result.summary_df.to_csv(os.path.join(args.output_dir, 'summary.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(result.summary_df.to_string(index=False))
    print()
    print(f"CONSENSUS: {result.consensus_count}/{result.total_spaces} "
          f"spaces show significant convergence")
    
    # Create combined visualization
    if args.visualize:
        create_summary_visualization(result, wave1_name, wave2_name, args.output_dir)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60 + "\n")
    
    # Return success if majority converge
    return 0 if result.consensus_count >= result.total_spaces // 2 else 1


if __name__ == "__main__":
    sys.exit(main())
