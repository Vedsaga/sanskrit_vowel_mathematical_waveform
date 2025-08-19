import os
import random
import subprocess
import multiprocessing
import csv
from tqdm import tqdm
import math

# --- Configuration ---
CLEAN_SAMPLES_FOLDER = "data/02_cleaned/"
NOISE_FOLDER = "data/01_raw/noise/"
AUGMENTED_FOLDER = "data/03_augmented/"
METADATA_LOG_FILE = os.path.join(AUGMENTED_FOLDER, "augmentation_log.csv")

TARGET_COUNT_PER_PHONEME = 2000
SNR_RANGE_DB = (0, 20)
TIME_STRETCH_RANGE = (0.9, 1.1)
# --- OPTIMIZATION: Use 11 of your 12 cores to keep the system responsive ---
CPU_CORES_TO_USE = 11

# --- Helper Function ---
def get_audio_duration(filepath):
    """Gets the duration of an audio file in seconds using ffprobe."""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

# --- Worker Function (runs on each CPU core) ---
def run_augmentation_task(task_params):
    """
    Takes a tuple of parameters and runs a single FFmpeg augmentation command.
    On success, it returns a dictionary of the metadata for logging.
    """
    clean_sample_path, noise_path, output_filepath, target_snr, stretch_factor = task_params

    speech_duration = get_audio_duration(clean_sample_path)
    noise_duration = get_audio_duration(noise_path)

    if speech_duration is None or noise_duration is None or noise_duration < speech_duration / stretch_factor:
        return None

    noise_start = random.uniform(0, noise_duration - (speech_duration / stretch_factor))
    noise_weight = 10**(-target_snr/20)

    command = [
        'ffmpeg', '-y',
        '-i', clean_sample_path,
        '-i', noise_path,
        '-filter_complex',
        f"[0:a]atempo={stretch_factor},aformat=sample_rates=16000:channel_layouts=mono[speech];" +
        f"[1:a]atrim=start={noise_start}:duration={speech_duration / stretch_factor},aformat=sample_rates=16000:channel_layouts=mono[noise];" +
        f"[speech][noise]amix=inputs=2:duration=first:weights='1 {noise_weight}'",
        '-c:a', 'pcm_s16le',
        output_filepath
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {
            "output_file": os.path.basename(output_filepath),
            "source_phoneme": os.path.basename(clean_sample_path),
            "source_noise": os.path.basename(noise_path),
            "snr_db": f"{target_snr:.2f}",
            "time_stretch": f"{stretch_factor:.2f}",
            "noise_start_time": f"{noise_start:.2f}"
        }
    except subprocess.CalledProcessError:
        return None

# --- Main Script ---
def augment_data_parallel():
    """
    Generates a list of all required augmentations and processes them in parallel
    across multiple CPU cores.
    """
    print(f"Starting parallel data augmentation using {CPU_CORES_TO_USE} CPU cores...")
    os.makedirs(AUGMENTED_FOLDER, exist_ok=True)

    noise_files = [os.path.join(NOISE_FOLDER, f) for f in os.listdir(NOISE_FOLDER) if os.path.isfile(os.path.join(NOISE_FOLDER, f))]
    if not noise_files:
        print(f"❌ ERROR: No noise files found in '{NOISE_FOLDER}'.")
        return

    phoneme_dirs = [d for d in os.listdir(CLEAN_SAMPLES_FOLDER) if os.path.isdir(os.path.join(CLEAN_SAMPLES_FOLDER, d))]

    all_results = []

    for phoneme in phoneme_dirs:
        clean_phoneme_dir = os.path.join(CLEAN_SAMPLES_FOLDER, phoneme)
        augmented_phoneme_dir = os.path.join(AUGMENTED_FOLDER, phoneme)
        os.makedirs(augmented_phoneme_dir, exist_ok=True)

        clean_samples = [os.path.join(clean_phoneme_dir, f) for f in os.listdir(clean_phoneme_dir)]
        if not clean_samples:
            continue

        num_existing = len(os.listdir(augmented_phoneme_dir))
        num_to_generate = TARGET_COUNT_PER_PHONEME - num_existing

        if num_to_generate <= 0:
            print(f"\nSkipping phoneme '{phoneme}': Target of {TARGET_COUNT_PER_PHONEME} already met.")
            continue

        print(f"\nGenerating {num_to_generate} new samples for phoneme: '{phoneme}'")

        tasks_to_run = []
        for i in range(num_to_generate):
            clean_sample_path = random.choice(clean_samples)
            noise_path = random.choice(noise_files)
            target_snr = random.uniform(SNR_RANGE_DB[0], SNR_RANGE_DB[1])
            stretch_factor = random.uniform(TIME_STRETCH_RANGE[0], TIME_STRETCH_RANGE[1])
            output_filename = f"{phoneme}_aug_{num_existing + i + 1}.wav"
            output_filepath = os.path.join(augmented_phoneme_dir, output_filename)
            tasks_to_run.append((clean_sample_path, noise_path, output_filepath, target_snr, stretch_factor))

        chunksize = math.ceil(len(tasks_to_run) / (CPU_CORES_TO_USE * 4))
        if chunksize < 1:
            chunksize = 1

        with multiprocessing.Pool(processes=CPU_CORES_TO_USE) as pool:
            results = list(tqdm(pool.imap_unordered(run_augmentation_task, tasks_to_run, chunksize=chunksize), total=len(tasks_to_run), desc=phoneme))
            all_results.extend([res for res in results if res is not None])

    if all_results:
        print(f"\nWriting metadata for {len(all_results)} new files to '{METADATA_LOG_FILE}'...")
        file_exists = os.path.exists(METADATA_LOG_FILE)
        with open(METADATA_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(all_results)

    print("\n✅ Data augmentation complete.")

if __name__ == "__main__":
    augment_data_parallel()
