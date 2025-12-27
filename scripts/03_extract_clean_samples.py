import os
import shutil
import subprocess
from collections import defaultdict

# --- Configuration ---
LABELS_FOLDER = "data/01_raw/labels_refined/"
NORMALIZED_AUDIO_FOLDER = "data/01_raw/normalized/"
CLEANED_SAMPLES_FOLDER = "data/02_cleaned/"

# --- Main Script ---
def get_user_choice():
    """Gets the user's choice for how to handle existing files."""
    print("How should existing files be handled?")
    print("  [1] Default mode (Skip existing files)")
    print("  [2] Override ALL existing files (delete and re-create all)")
    print("  [3] Override a specific phoneme folder")

    choice = input("Enter your choice (1, 2, or 3): ")

    if choice == '2':
        confirm = input(f"Are you sure you want to delete and re-create everything in '{CLEANED_SAMPLES_FOLDER}'? (y/n): ")
        if confirm.lower() == 'y':
            print("Deleting existing cleaned samples...")
            if os.path.exists(CLEANED_SAMPLES_FOLDER):
                shutil.rmtree(CLEANED_SAMPLES_FOLDER)
            return 'override_all', None

    if choice == '3':
        phoneme_to_override = input("Enter the phoneme you want to override (e.g., अ): ").strip()
        if phoneme_to_override:
            folder_to_delete = os.path.join(CLEANED_SAMPLES_FOLDER, phoneme_to_override)
            confirm = input(f"Are you sure you want to delete and re-create everything in '{folder_to_delete}'? (y/n): ")
            if confirm.lower() == 'y':
                print(f"Deleting existing samples for '{phoneme_to_override}'...")
                if os.path.exists(folder_to_delete):
                    shutil.rmtree(folder_to_delete)
                return 'override_specific', phoneme_to_override

    print("Proceeding in default (skip) mode.")
    return 'skip', None

def extract_clean_samples():
    """
    Reads Audacity label files, extracts audio segments, and provides override options.
    """
    override_mode, phoneme_to_override = get_user_choice()

    print("\nStarting the automated clipping process...")
    os.makedirs(CLEANED_SAMPLES_FOLDER, exist_ok=True)

    try:
        label_files = [f for f in os.listdir(LABELS_FOLDER) if f.endswith('.txt')]
    except FileNotFoundError:
        print(f"❌ ERROR: Labels folder not found at '{LABELS_FOLDER}'")
        return

    # --- NEW: Dictionaries to track stats ---
    new_counts = defaultdict(int)
    skipped_counts = defaultdict(int)
    phoneme_counter = defaultdict(int)

    for label_filename in label_files:
        base_name = os.path.splitext(label_filename)[0]
        # ... (rest of the file processing logic is the same)
        label_filepath = os.path.join(LABELS_FOLDER, label_filename)
        audio_filepath = os.path.join(NORMALIZED_AUDIO_FOLDER, f"{base_name}.wav")

        if not os.path.exists(audio_filepath):
            continue

        with open(label_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    parts = line.strip().split('\t')
                    start_time, end_time, phoneme = float(parts[0]), float(parts[1]), parts[2].strip()

                    phoneme_dir = os.path.join(CLEANED_SAMPLES_FOLDER, phoneme)
                    os.makedirs(phoneme_dir, exist_ok=True)

                    phoneme_counter[phoneme] += 1
                    count_str = str(phoneme_counter[phoneme]).zfill(3)
                    output_filename = f"{phoneme}_{base_name}_{count_str}.wav"
                    output_filepath = os.path.join(phoneme_dir, output_filename)

                    # --- NEW: Logic to handle skipping or processing ---
                    if os.path.exists(output_filepath) and override_mode == 'skip':
                        skipped_counts[phoneme] += 1
                        continue

                    # If we are here, we are creating the file (either new or overriding)
                    new_counts[phoneme] += 1

                    command = [
                        'ffmpeg', '-i', audio_filepath, '-ss', str(start_time),
                        '-to', str(end_time), '-c', 'copy', '-y', output_filepath
                    ]

                    subprocess.run(command, check=True, capture_output=True, text=True)

                except (ValueError, IndexError):
                    continue

    # --- NEW: Detailed Final Report ---
    print("\n--- Extraction Report ---")
    all_phonemes = sorted(set(list(new_counts.keys()) + list(skipped_counts.keys())))

    for phoneme in all_phonemes:
        new = new_counts.get(phoneme, 0)
        skipped = skipped_counts.get(phoneme, 0)
        total = new + skipped
        print(f"Phoneme '{phoneme}':")
        print(f"  - Newly Created: {new}")
        print(f"  - Skipped (Existed): {skipped}")
        print(f"  - Total Samples: {total}")

    print("\n✅ Extraction complete.")

if __name__ == "__main__":
    extract_clean_samples()
