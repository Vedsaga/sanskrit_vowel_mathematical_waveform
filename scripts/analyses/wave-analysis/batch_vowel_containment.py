#!/usr/bin/env python3
"""
Batch Scattering Convergence Analysis (using kymatio)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave_comparison_analysis import (
    load_wave, analyze_convergence, create_visualization, save_results
)

try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from common.config import configure_matplotlib
    configure_matplotlib()
except ImportError:
    pass

CONSONANTS = [
    'क', 'ख', 'ग', 'घ', 'ङ',
    'च', 'छ', 'ज', 'झ', 'ञ',
    'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न',
    'प', 'फ', 'ब', 'भ', 'म',
    'य', 'र', 'ल', 'व',
    'श', 'ष', 'स', 'ह',
]


def find_golden(data_dir: str, phoneme: str) -> str:
    phoneme_dir = os.path.join(data_dir, phoneme)
    if not os.path.exists(phoneme_dir):
        return None
    for f in os.listdir(phoneme_dir):
        if 'golden' in f.lower() and f.endswith('.wav'):
            return os.path.join(phoneme_dir, f)
    for f in os.listdir(phoneme_dir):
        if f.endswith('.wav'):
            return os.path.join(phoneme_dir, f)
    return None


def create_summary_plot(df: pd.DataFrame, vowel: str, output_dir: str):
    """Create summary visualization."""
    BG = '#111111'
    PANEL = '#1a1a1a'
    TEXT = '#eaeaea'
    CONV = '#2ECC71'
    NO_CONV = '#FF6B6B'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(BG)
    
    # Panel 1: Distance reduction bar chart
    ax = axes[0, 0]
    ax.set_facecolor(PANEL)
    colors = [CONV if c else NO_CONV for c in df['converges']]
    ax.barh(range(len(df)), df['distance_reduction'] * 100, color=colors, alpha=0.8)
    ax.axvline(15, color=TEXT, ls='--', alpha=0.5, label='Threshold (15%)')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['consonant'], fontsize=9)
    ax.set_xlabel('Distance Reduction (%)', color=TEXT)
    ax.set_title('Scattering Distance Reduction', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    ax.invert_yaxis()
    ax.legend(facecolor=PANEL, labelcolor=TEXT)
    
    # Panel 2: Min distance scatter
    ax = axes[0, 1]
    ax.set_facecolor(PANEL)
    for i, row in df.iterrows():
        color = CONV if row['converges'] else NO_CONV
        ax.scatter(row['min_distance'], row['distance_reduction'] * 100,
                  s=100, color=color, alpha=0.7, edgecolor='white')
    ax.axhline(15, color=TEXT, ls='--', alpha=0.5)
    ax.set_xlabel('Min Distance', color=TEXT)
    ax.set_ylabel('Distance Reduction (%)', color=TEXT)
    ax.set_title('Quality of Convergence', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    
    # Panel 3: Convergence duration
    ax = axes[1, 0]
    ax.set_facecolor(PANEL)
    ax.hist(df['duration'] * 1000, bins=15, color=CONV, alpha=0.8, edgecolor='white')
    ax.set_xlabel('Convergence Duration (ms)', color=TEXT)
    ax.set_ylabel('Count', color=TEXT)
    ax.set_title('Duration Distribution', color=TEXT, fontweight='bold')
    ax.tick_params(colors=TEXT)
    
    # Panel 4: Summary stats
    ax = axes[1, 1]
    ax.set_facecolor(PANEL)
    ax.axis('off')
    
    n_conv = df['converges'].sum()
    summary = f"""
KYMATIO SCATTERING ANALYSIS
{'═' * 35}

Total consonants: {len(df)}
Converging:       {n_conv} ({n_conv/len(df)*100:.1f}%)

{'─' * 35}
METRICS
{'─' * 35}
Avg distance reduction: {df['distance_reduction'].mean()*100:.1f}%
Avg min distance:       {df['min_distance'].mean():.3f}
Avg duration:           {df['duration'].mean()*1000:.1f}ms

{'═' * 35}
"""
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            va='top', color=TEXT, family='monospace')
    
    fig.suptitle(f'Scattering Convergence: All Consonants → {vowel}',
                 color=TEXT, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    path = os.path.join(output_dir, 'summary.png')
    plt.savefig(path, dpi=300, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"Summary: {path}")


def run_batch(vowel: str, data_dir: str, output_dir: str, visualize: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    
    vowel_file = find_golden(data_dir, vowel)
    if not vowel_file:
        print(f"ERROR: No file for vowel '{vowel}'")
        return None
    
    wave_vowel = load_wave(vowel_file)
    print(f"Target: {vowel_file} ({wave_vowel.duration:.3f}s)")
    print("=" * 60)
    
    results = []
    
    for cons in CONSONANTS:
        print(f"\n{cons} → {vowel}?", end=" ")
        
        cons_file = find_golden(data_dir, cons)
        if not cons_file:
            print("SKIP")
            continue
        
        try:
            wave_cons = load_wave(cons_file)
            pair_dir = os.path.join(output_dir, f"{cons}_→_{vowel}")
            
            result = analyze_convergence(wave_cons, wave_vowel)
            save_results(wave_cons, wave_vowel, result, pair_dir)
            
            if visualize:
                create_visualization(wave_cons, wave_vowel, result, pair_dir)
            
            results.append({
                'consonant': cons,
                'converges': result.converges,
                'distance_reduction': result.distance_reduction,
                'min_distance': result.min_distance,
                'min_time': result.min_distance_time,
                'duration': result.convergence_duration,
            })
            
            status = "✓" if result.converges else "✗"
            print(f"{status} red={result.distance_reduction:.1%}, min_d={result.min_distance:.3f}")
            
        except Exception as e:
            print(f"ERROR: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    n = df['converges'].sum()
    print(f"\n{'='*60}")
    print(f"RESULT: {n}/{len(df)} consonants CONVERGE")
    print(f"Avg reduction: {df['distance_reduction'].mean()*100:.1f}%")
    print(f"Avg min dist:  {df['min_distance'].mean():.3f}")
    
    create_summary_plot(df, vowel, output_dir)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vowel', default='अ')
    parser.add_argument('--data_dir', default='data/02_cleaned')
    parser.add_argument('--output_dir', default='results/scattering_analysis')
    parser.add_argument('--no-visualize', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"KYMATIO SCATTERING: consonants → {args.vowel}")
    print(f"{'='*60}")
    
    run_batch(args.vowel, args.data_dir, args.output_dir, not args.no_visualize)


if __name__ == "__main__":
    main()
