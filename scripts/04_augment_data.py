import os
import random
import subprocess
import multiprocessing
import csv
from tqdm import tqdm

# --- Config ---
CLEAN_SAMPLES_FOLDER = "data/02_cleaned/"
NOISE_FOLDER = "data/01_raw/noise/"
OUTPUT_ROOT = "data/03_augmented/"
METADATA_LOG_FILE = os.path.join(OUTPUT_ROOT, "augmentation_log.csv")

TARGET_AUG_PER_PHONEME = 20000   # target augmented per phoneme
SAMPLE_RATE = 16000
CPU_CORES = 12

# Random ranges instead of fixed grid
SNR_RANGE = (0, 20)         # dB
STRETCH_RANGE = (0.9, 1.1)  # speed
PITCH_RANGE = (-2, 2)       # semitones (optional)

# Train/Val/Test split
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

def get_audio_duration(filepath):
    """Duration of wav file in seconds."""
    cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', filepath
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except:
        return None

def run_augmentation_task(task_params):
    clean_path, noise_path, output_filepath, snr, stretch, pitch = task_params
    speech_dur = get_audio_duration(clean_path)
    noise_dur = get_audio_duration(noise_path)
    if speech_dur is None or noise_dur is None or noise_dur < (speech_dur / stretch):
        return None

    noise_start = random.uniform(0, noise_dur - (speech_dur / stretch))
    noise_weight = 10**(-snr / 20)

    # build ffmpeg filter chain
    filters = [
        f"atempo={stretch}",
        f"asetrate={SAMPLE_RATE}*pow(2,{pitch}/12),aresample={SAMPLE_RATE}"
        if pitch != 0 else "",
        "aformat=sample_rates=16000:channel_layouts=mono"
    ]
    filters = [f for f in filters if f]  # remove empty

    command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
        '-i', clean_path,
        '-i', noise_path,
        '-filter_complex',
        f"[0:a]{','.join(filters)}[speech];"
        f"[1:a]atrim=start={noise_start}:duration={speech_dur/stretch},aformat=sample_rates=16000:channel_layouts=mono[noise];"
        f"[speech][noise]amix=inputs=2:duration=first:weights='1 {noise_weight}'",
        '-c:a', 'pcm_s16le',
        output_filepath
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        return {
            "output_file": os.path.basename(output_filepath),
            "phoneme": os.path.basename(os.path.dirname(clean_path)),
            "clean_src": os.path.basename(clean_path),
            "noise_src": os.path.basename(noise_path),
            "snr_db": round(snr, 2),
            "stretch": round(stretch, 2),
            "pitch": pitch,
        }
    except:
        return None

def augment_data():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    noise_files = [os.path.join(NOISE_FOLDER, f) for f in os.listdir(NOISE_FOLDER)]
    if not noise_files:
        print("❌ No noise files found!")
        return

    all_results = []
    phoneme_dirs = [d for d in os.listdir(CLEAN_SAMPLES_FOLDER) if os.path.isdir(os.path.join(CLEAN_SAMPLES_FOLDER, d))]

    with multiprocessing.Pool(processes=CPU_CORES) as pool:
        async_results = []

        for phoneme in phoneme_dirs:
            clean_dir = os.path.join(CLEAN_SAMPLES_FOLDER, phoneme)
            clean_samples = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir)]

            if not clean_samples:
                continue

            for i in range(TARGET_AUG_PER_PHONEME):
                clean_path = random.choice(clean_samples)
                noise_path = random.choice(noise_files)
                snr = random.uniform(*SNR_RANGE)
                stretch = random.uniform(*STRETCH_RANGE)
                pitch = random.choice(range(PITCH_RANGE[0], PITCH_RANGE[1] + 1))

                # split folder assignment
                rnd = random.random()
                if rnd < SPLIT_RATIOS["train"]:
                    split = "train"
                elif rnd < SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"]:
                    split = "val"
                else:
                    split = "test"

                out_dir = os.path.join(OUTPUT_ROOT, split, phoneme)
                os.makedirs(out_dir, exist_ok=True)
                output_filename = f"{phoneme}_{i:05d}.wav"
                output_path = os.path.join(out_dir, output_filename)

                task_params = (clean_path, noise_path, output_path, snr, stretch, pitch)
                res = pool.apply_async(run_augmentation_task, (task_params,))
                async_results.append(res)

        for res in tqdm(async_results, desc="Augmenting"):
            result_data = res.get()
            if result_data:
                all_results.append(result_data)

    # write metadata
    if all_results:
        with open(METADATA_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

    print(f"\n✅ Augmentation complete: {len(all_results)} files generated.")

if __name__ == "__main__":
    augment_data()
