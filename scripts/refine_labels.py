#!/usr/bin/env python3
"""
Smart Label Refinement Using Intensity Analysis

This script refines human-labeled boundaries using intensity analysis to avoid
including unwanted audio segments (like neighboring vowels or noise).

Instead of blind padding, it:
1. Loads audio with a generous search buffer around the human label
2. Computes intensity profile to find voiced regions
3. Detects gaps (drops in intensity > min_gap_duration = boundary)
4. Selects the region containing the human-labeled center
5. Outputs clean boundaries that never include disconnected segments

Usage:
    python3 refine_labels.py --input_dir data/01_raw/labels \
                              --audio_dir data/01_raw/normalized \
                              --output_dir data/01_raw/labels_refined

After running, update 03_extract_clean_samples.py to use labels_refined/
"""

import os
import sys
import glob
import argparse
import numpy as np
import parselmouth
from parselmouth.praat import call
from pathlib import Path

# Add analyses directory to path for common package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyses'))

try:
    from common import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def compute_intensity_profile(
    audio_path: str,
    start_time: float,
    end_time: float,
    time_step: float = 0.005
) -> tuple:
    """
    Compute intensity profile for a region of an audio file.
    
    Args:
        audio_path: Path to the audio file
        start_time: Start time in seconds
        end_time: End time in seconds
        time_step: Time step for intensity analysis
    
    Returns:
        (time_array, intensity_array, sound_start, sound_end)
    """
    try:
        sound = parselmouth.Sound(audio_path)
        total_duration = sound.get_total_duration()
        
        # Clip to valid range
        start_time = max(0, start_time)
        end_time = min(total_duration, end_time)
        
        # Extract the portion of sound we care about
        sound_part = sound.extract_part(start_time, end_time, parselmouth.WindowShape.RECTANGULAR, 1, False)
        
        # Compute intensity
        intensity = call(sound_part, "To Intensity", 100, time_step, "yes")
        n_frames = call(intensity, "Get number of frames")
        
        time_values = []
        intensity_values = []
        
        for i in range(1, n_frames + 1):
            t = call(intensity, "Get time from frame number", i)
            intens = call(intensity, "Get value at time", t, "Cubic")
            if not np.isnan(intens):
                time_values.append(start_time + t)  # Convert to absolute time
                intensity_values.append(intens)
        
        return np.array(time_values), np.array(intensity_values), start_time, end_time
        
    except Exception as e:
        print(f"Error computing intensity for {audio_path}: {e}")
        return None, None, start_time, end_time


def find_voiced_regions(
    time_arr: np.ndarray,
    intensity_arr: np.ndarray,
    intensity_threshold: float = 40.0,
    min_gap_duration: float = 0.05
) -> list:
    """
    Find contiguous voiced regions where intensity is above threshold.
    
    Args:
        time_arr: Array of time values
        intensity_arr: Array of intensity values (dB)
        intensity_threshold: Minimum intensity to be considered voiced (dB)
        min_gap_duration: Minimum gap duration to split regions (seconds)
    
    Returns:
        List of (start_time, end_time) tuples for each voiced region
    """
    if len(time_arr) < 2:
        return []
    
    # Find frames above threshold
    above_threshold = intensity_arr >= intensity_threshold
    
    if not np.any(above_threshold):
        return []
    
    # Find transitions
    regions = []
    in_region = False
    region_start = None
    last_above_time = None
    
    dt = np.median(np.diff(time_arr)) if len(time_arr) > 1 else 0.01
    
    for i, (t, is_above) in enumerate(zip(time_arr, above_threshold)):
        if is_above:
            if not in_region:
                # Starting a new region
                region_start = t
                in_region = True
            last_above_time = t
        else:
            if in_region:
                # Check if gap is long enough to end region
                gap_so_far = t - last_above_time
                if gap_so_far >= min_gap_duration:
                    # End this region
                    regions.append((region_start, last_above_time))
                    in_region = False
                    region_start = None
    
    # Close final region if still open
    if in_region and region_start is not None:
        regions.append((region_start, last_above_time))
    
    return regions


def select_region_containing_center(
    regions: list,
    center_time: float
) -> tuple:
    """
    Select the region that contains the specified center time.
    
    Args:
        regions: List of (start, end) tuples
        center_time: The time point that must be contained
    
    Returns:
        (start, end) of the selected region, or None if no region contains center
    """
    for start, end in regions:
        if start <= center_time <= end:
            return (start, end)
    
    # If no region contains center, find the closest one
    if regions:
        closest_region = min(regions, key=lambda r: min(abs(r[0] - center_time), abs(r[1] - center_time)))
        return closest_region
    
    return None


def refine_single_label(
    audio_path: str,
    human_start: float,
    human_end: float,
    search_buffer: float = 0.3,
    intensity_threshold: float = 40.0,
    min_gap_duration: float = 0.05,
    min_duration: float = 0.1
) -> tuple:
    """
    Refine a single label using intensity analysis.
    
    Args:
        audio_path: Path to the audio file
        human_start: Human-labeled start time
        human_end: Human-labeled end time
        search_buffer: How much to search beyond human label (seconds)
        intensity_threshold: Minimum intensity for voiced region (dB)
        min_gap_duration: Minimum gap to split regions (seconds)
        min_duration: Minimum output duration (seconds)
    
    Returns:
        (refined_start, refined_end)
    """
    human_center = (human_start + human_end) / 2
    human_duration = human_end - human_start
    
    # Define search region
    search_start = max(0, human_start - search_buffer)
    search_end = human_end + search_buffer
    
    # Compute intensity profile
    time_arr, intensity_arr, actual_start, actual_end = compute_intensity_profile(
        audio_path, search_start, search_end
    )
    
    if time_arr is None or len(time_arr) < 3:
        # Fallback to original human label
        return (human_start, human_end)
    
    # Find voiced regions
    regions = find_voiced_regions(
        time_arr, intensity_arr,
        intensity_threshold=intensity_threshold,
        min_gap_duration=min_gap_duration
    )
    
    if not regions:
        # No voiced regions found, return original
        return (human_start, human_end)
    
    # Select region containing the human-labeled center
    selected_region = select_region_containing_center(regions, human_center)
    
    if selected_region is None:
        return (human_start, human_end)
    
    refined_start, refined_end = selected_region
    
    # Ensure minimum duration
    if refined_end - refined_start < min_duration:
        # Expand symmetrically around center
        center = (refined_start + refined_end) / 2
        refined_start = center - min_duration / 2
        refined_end = center + min_duration / 2
    
    return (refined_start, refined_end)


def refine_labels(
    input_dir: str,
    audio_dir: str,
    output_dir: str,
    search_buffer: float = 0.3,
    intensity_threshold: float = 40.0,
    min_gap_duration: float = 0.05,
    min_duration: float = 0.1,
    verbose: bool = True
):
    """
    Refine all labels in a directory using intensity analysis.
    
    Args:
        input_dir: Directory containing human-labeled .txt files
        audio_dir: Directory containing corresponding audio files
        output_dir: Output directory for refined labels
        search_buffer: How much to search beyond human label
        intensity_threshold: Minimum intensity for voiced region (dB)
        min_gap_duration: Minimum gap to split regions
        min_duration: Minimum output duration
        verbose: Print progress
    """
    os.makedirs(output_dir, exist_ok=True)
    
    search_pattern = os.path.join(input_dir, '*.txt')
    label_files = glob.glob(search_pattern)
    
    if verbose:
        print(f"Found {len(label_files)} label files in {input_dir}")
        print(f"Settings: search_buffer={search_buffer}s, intensity_threshold={intensity_threshold}dB")
        print(f"          min_gap={min_gap_duration}s, min_duration={min_duration}s")
        print()
    
    stats = {
        'total_labels': 0,
        'refined': 0,
        'unchanged': 0,
        'expanded': 0,
        'contracted': 0
    }
    
    for file_path in tqdm(label_files, desc="Processing labels"):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, filename)
        
        # Find corresponding audio file
        audio_path = os.path.join(audio_dir, f"{base_name}.wav")
        if not os.path.exists(audio_path):
            if verbose:
                print(f"  Warning: No audio file for {filename}, copying as-is")
            # Copy original labels if no audio found
            with open(file_path, 'r') as f_in, open(output_path, 'w') as f_out:
                f_out.write(f_in.read())
            continue
        
        with open(file_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                
                try:
                    human_start = float(parts[0])
                    human_end = float(parts[1])
                    label = parts[2]
                except ValueError:
                    continue
                
                stats['total_labels'] += 1
                human_duration = human_end - human_start
                
                # Refine this label
                refined_start, refined_end = refine_single_label(
                    audio_path, human_start, human_end,
                    search_buffer=search_buffer,
                    intensity_threshold=intensity_threshold,
                    min_gap_duration=min_gap_duration,
                    min_duration=min_duration
                )
                
                refined_duration = refined_end - refined_start
                
                # Track stats
                if abs(refined_start - human_start) < 0.01 and abs(refined_end - human_end) < 0.01:
                    stats['unchanged'] += 1
                else:
                    stats['refined'] += 1
                    if refined_duration > human_duration:
                        stats['expanded'] += 1
                    else:
                        stats['contracted'] += 1
                
                # Write refined label
                f_out.write(f"{refined_start:.6f}\t{refined_end:.6f}\t{label}\n")
        
        if verbose:
            print(f"  Processed: {filename}")
    
    # Print summary
    if verbose:
        print()
        print("=" * 50)
        print("REFINEMENT SUMMARY")
        print("=" * 50)
        print(f"Total labels processed: {stats['total_labels']}")
        print(f"  - Refined: {stats['refined']} ({100*stats['refined']/max(1,stats['total_labels']):.1f}%)")
        print(f"    - Expanded:   {stats['expanded']}")
        print(f"    - Contracted: {stats['contracted']}")
        print(f"  - Unchanged: {stats['unchanged']}")
        print()
        print(f"Output written to: {output_dir}")
        print()
        print("Next step: Update 03_extract_clean_samples.py to use this folder")
        print(f"  Change LABELS_FOLDER to: \"{output_dir}/\"")


def main():
    parser = argparse.ArgumentParser(
        description='Refine human labels using intensity-based boundary detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python3 refine_labels.py

  # Custom thresholds
  python3 refine_labels.py --intensity_threshold 45 --min_gap 0.08

  # Different directories
  python3 refine_labels.py --input_dir my_labels --audio_dir my_audio --output_dir refined
        """
    )
    
    parser.add_argument('--input_dir', type=str, default='data/01_raw/labels',
                        help='Input directory containing human-labeled .txt files')
    parser.add_argument('--audio_dir', type=str, default='data/01_raw/normalized',
                        help='Directory containing corresponding audio files')
    parser.add_argument('--output_dir', type=str, default='data/01_raw/labels_refined',
                        help='Output directory for refined labels')
    
    parser.add_argument('--search_buffer', type=float, default=0.3,
                        help='How much to search beyond human label (seconds)')
    parser.add_argument('--intensity_threshold', type=float, default=40.0,
                        help='Minimum intensity for voiced region (dB)')
    parser.add_argument('--min_gap', type=float, default=0.05,
                        help='Minimum gap duration to split regions (seconds)')
    parser.add_argument('--min_duration', type=float, default=0.1,
                        help='Minimum output label duration (seconds)')
    
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')
    
    args = parser.parse_args()
    
    refine_labels(
        input_dir=args.input_dir,
        audio_dir=args.audio_dir,
        output_dir=args.output_dir,
        search_buffer=args.search_buffer,
        intensity_threshold=args.intensity_threshold,
        min_gap_duration=args.min_gap,
        min_duration=args.min_duration,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
