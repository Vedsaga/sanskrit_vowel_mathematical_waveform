#!/usr/bin/env python3
"""
Batch Vowel Containment Analysis

Tests whether vowel 'अ' (a) is contained in all consonant sounds using
transfer function analysis (articulatory convergence).

Usage:
    python batch_vowel_containment.py --vowel अ --output_dir results/vowel_containment_analysis
"""

import os
import sys
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from wave_comparison_analysis import (
    load_wave, analyze_containment, create_visualization, save_results
)

# Sanskrit consonants (all pronounced with inherent 'a' vowel)
CONSONANTS = [
    'क', 'ख', 'ग', 'घ', 'ङ',  # Velars
    'च', 'छ', 'ज', 'झ', 'ञ',  # Palatals
    'ट', 'ठ', 'ड', 'ढ', 'ण',  # Retroflexes
    'त', 'थ', 'द', 'ध', 'न',  # Dentals
    'प', 'फ', 'ब', 'भ', 'म',  # Labials
    'य', 'र', 'ल', 'व',       # Semivowels
    'श', 'ष', 'स', 'ह',       # Sibilants & Aspirate
]


def find_golden_file(data_dir: str, phoneme: str) -> str:
    """Find a golden .wav file for a given phoneme."""
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


def run_batch(vowel: str, data_dir: str, output_dir: str, visualize: bool = True):
    """Run containment analysis for all consonants against a vowel."""
    os.makedirs(output_dir, exist_ok=True)
    
    vowel_file = find_golden_file(data_dir, vowel)
    if not vowel_file:
        print(f"ERROR: No file found for vowel '{vowel}'")
        return None
    
    print(f"Reference vowel: {vowel_file}")
    print("=" * 60)
    
    wave_vowel = load_wave(vowel_file)
    results = []
    
    for consonant in CONSONANTS:
        print(f"\n{consonant} → {vowel}?")
        
        consonant_file = find_golden_file(data_dir, consonant)
        if not consonant_file:
            print(f"  SKIP: No file")
            continue
        
        try:
            wave_consonant = load_wave(consonant_file)
            pair_dir = os.path.join(output_dir, f"{consonant}_contains_{vowel}")
            
            result = analyze_containment(wave_consonant, wave_vowel)
            save_results(wave_consonant, wave_vowel, result, pair_dir)
            
            if visualize:
                create_visualization(wave_consonant, wave_vowel, result, pair_dir)
            
            row = {
                'consonant': consonant,
                'vowel': vowel,
                'contained': result.contained,
                'confidence': result.confidence,
                'voiced_corr': result.best_voiced_corr,
                'match_time': result.best_match_time,
                'ref_similarity': result.evidence['reference_similarity'],
            }
            results.append(row)
            
            status = "✓" if result.contained else "✗"
            print(f"  {status} r={result.best_voiced_corr:.3f} @ t={result.best_match_time:.3f}s")
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'summary.csv'), index=False)
    
    n_contained = df['contained'].sum()
    print(f"\n{'='*60}")
    print(f"RESULT: {n_contained}/{len(df)} consonants contain '{vowel}'")
    print(f"Average voiced correlation: {df['voiced_corr'].mean():.3f}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Batch vowel containment test')
    parser.add_argument('--vowel', default='अ')
    parser.add_argument('--data_dir', default='data/02_cleaned')
    parser.add_argument('--output_dir', default='results/vowel_containment_analysis')
    parser.add_argument('--no-visualize', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"BATCH: Testing '{args.vowel}' containment in consonants")
    print(f"{'='*60}")
    
    run_batch(args.vowel, args.data_dir, args.output_dir, not args.no_visualize)


if __name__ == "__main__":
    main()
