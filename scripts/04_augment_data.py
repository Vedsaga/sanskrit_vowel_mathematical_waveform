import os
import random
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from collections import defaultdict
import shutil

# ==============================================================================
#                      --- TUNABLE PARAMETERS ---
# ==============================================================================
# --- Input/Output Paths ---
CLEAN_SAMPLES_FOLDER = "data/02_cleaned/"
NOISE_FOLDER = "data/01_raw/noise/"
OUTPUT_ROOT = "data/04_augmented_leakproof/" # Using a new folder for the clean dataset

# --- Augmentation Settings ---
TARGET_SAMPLES_PER_PHONEME = 2000 # Target number of files in the final training set per phoneme
NOISE_SNR_DB_RANGE = (5, 25)
STRETCH_RATE_RANGE = (0.85, 1.15)
PITCH_SHIFT_SEMITONES_RANGE = (-2.5, 2.5)

# --- Speaker-Independent Split Configuration ---
# Define which speakers go into which set. This is the key to preventing data leakage.
# You have 7 female and 9 male speakers. Let's use an ~80/10/10 split.
# 80% train = 13 speakers, 10% val = 2 speakers, 10% test = 1 speaker
TRAIN_SPEAKERS = [f"female-{i}" for i in [1,2,3,4,5]] + [f"male-{i}" for i in [1,2,3,4,5,6,7,8]]
VAL_SPEAKERS = ["female-6", "male-9"]
TEST_SPEAKERS = ["female-7"]

# ==============================================================================
#                      --- AUGMENTATION FUNCTIONS ---
# ==============================================================================

def add_noise(y, noise_files, sr):
    """Adds a random snippet of a random noise file to the audio."""
    noise_path = random.choice(noise_files)
    noise, _ = librosa.load(noise_path, sr=sr)
    if len(noise) < len(y):
        return y # Skip if noise is shorter

    start_idx = random.randint(0, len(noise) - len(y))
    noise_snippet = noise[start_idx : start_idx + len(y)]

    snr_db = random.uniform(*NOISE_SNR_DB_RANGE)
    audio_power = np.sum(y ** 2) / len(y)
    noise_power = np.sum(noise_snippet ** 2) / len(noise_snippet)

    # Required noise power to achieve target SNR
    required_noise_power = audio_power / (10 ** (snr_db / 10))

    # Scale the noise snippet
    scaled_noise = noise_snippet * np.sqrt(required_noise_power / noise_power)

    return y + scaled_noise

def time_stretch(y):
    """Applies time stretching."""
    rate = random.uniform(*STRETCH_RATE_RANGE)
    return librosa.effects.time_stretch(y=y, rate=rate)

def pitch_shift(y, sr):
    """Applies pitch shifting."""
    n_steps = random.uniform(*PITCH_SHIFT_SEMITONES_RANGE)
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

# ==============================================================================
#                      --- MAIN SCRIPT LOGIC ---
# ==============================================================================

def main():
    print("🚀 Starting Leak-Proof Data Augmentation and Splitting...")

    # 1. Discover all clean files and identify speakers
    all_clean_files = glob(os.path.join(CLEAN_SAMPLES_FOLDER, "*", "*.wav"))
    speaker_files = defaultdict(list)
    for f in all_clean_files:
        # Assuming filename format is like "phoneme-speaker-id.wav" e.g. "a-female-1-001.wav"
        # Or that speaker is in the path. Let's assume a simpler case from your `male-9` etc.
        # We can infer speaker from the original source file name. This part may need adjustment
        # based on your exact file naming conventions in `02_cleaned`.
        # For now, let's assume a simple structure or manually created list.
        # This is a placeholder for a more robust speaker identification method.
        # Let's assume files are named like 'अ/female-1_001.wav'
        try:
            speaker = os.path.basename(f).split('_')[0]
            speaker_files[speaker].append(f)
        except IndexError:
            continue

    print(f"Discovered {len(all_clean_files)} files from {len(speaker_files)} speakers.")

    # 2. Prepare output directories and split the original files
    print("\nSplitting original files into train/val/test sets based on speaker...")
    split_counts = defaultdict(int)
    for split_name, speakers in [('train', TRAIN_SPEAKERS), ('val', VAL_SPEAKERS), ('test', TEST_SPEAKERS)]:
        for speaker in speakers:
            if speaker in speaker_files:
                for file_path in speaker_files[speaker]:
                    phoneme = os.path.basename(os.path.dirname(file_path))
                    out_dir = os.path.join(OUTPUT_ROOT, split_name, phoneme)
                    os.makedirs(out_dir, exist_ok=True)
                    shutil.copy(file_path, out_dir)
                    split_counts[split_name] += 1

    print("Original file split complete:")
    print(f"  Training set:   {split_counts['train']} files from {len(TRAIN_SPEAKERS)} speakers")
    print(f"  Validation set: {split_counts['val']} files from {len(VAL_SPEAKERS)} speakers")
    print(f"  Test set:       {split_counts['test']} files from {len(TEST_SPEAKERS)} speakers")

    # 3. Augment ONLY the training data
    print("\n🎤 Augmenting the training set...")
    noise_files = glob(os.path.join(NOISE_FOLDER, "*.wav"))
    if not noise_files:
        print("⚠️ No noise files found. Augmenting without noise.")

    train_phoneme_dirs = glob(os.path.join(OUTPUT_ROOT, "train", "*"))

    for phoneme_dir in tqdm(train_phoneme_dirs, desc="Augmenting Phonemes"):
        phoneme = os.path.basename(phoneme_dir)
        source_files = glob(os.path.join(phoneme_dir, "*.wav"))

        if not source_files:
            continue

        num_to_generate = TARGET_SAMPLES_PER_PHONEME - len(source_files)
        if num_to_generate <= 0:
            continue

        for i in range(num_to_generate):
            # Pick a random source file from this phoneme's training set
            source_path = random.choice(source_files)
            y, sr = librosa.load(source_path, sr=None)

            # Apply a random combination of augmentations
            y_aug = y
            if random.random() < 0.8 and noise_files: y_aug = add_noise(y_aug, noise_files, sr)
            if random.random() < 0.5: y_aug = time_stretch(y_aug)
            if random.random() < 0.5: y_aug = pitch_shift(y_aug, sr)

            # Save the new file
            out_filename = f"{phoneme}_aug_{i:05d}.wav"
            out_path = os.path.join(phoneme_dir, out_filename)
            sf.write(out_path, y_aug, sr)

    print("\n🎉 Augmentation complete!")

if __name__ == "__main__":
    main()
