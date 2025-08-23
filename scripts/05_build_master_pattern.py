# scripts/05_build_master_pattern.py (FINAL, CLEANED VERSION)

import os
import struct
import numpy as np
import warnings
import toml
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import librosa
from typing import Any, cast

# --- Load Master Configuration ---
try:
    CONFIG = toml.load("config/dhvani_config.toml")
    PROFILE_NAME = CONFIG['active_profile']
    PROFILE = CONFIG['profiles'][PROFILE_NAME]

    PATHS = CONFIG['paths']
    PROC = CONFIG['processing']
    FEAT = CONFIG['features']

    print(f"✅ Configuration loaded for profile: '{PROFILE['description']}'")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load or parse 'config/dhvani_config.toml'. Error: {e}")
    exit()

# --- Assign variables from config ---
AUGMENTED_FOLDER = PATHS['augmented_data']
PATTERNS_FOLDER = PATHS['pattern_output']
FILES_PER_PHONEME_TO_PROCESS = PROC['files_per_phoneme']
SAMPLE_RATE = FEAT['sample_rate']
N_MFCC = PROFILE['n_mfcc']
FRAME_LENGTH_SAMPLES = FEAT['frame_length_samples']
HOP_LENGTH_SAMPLES = FEAT['hop_length_samples']
N_FFT = FEAT['n_fft']
GMM_COMPONENTS = PROFILE['gmm_components']
KMEANS_CLUSTERS = PROFILE['kmeans_clusters']
QUICK_MATCH_TEMPLATE_LENGTH = PROFILE['template_length']
PATTERN_SIZE_KB = PROFILE['pattern_size_kb']

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def extract_features(filepath):
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        if len(y) < FRAME_LENGTH_SAMPLES: return None
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, win_length=FRAME_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        combined_features = np.hstack([mfccs.T, delta_mfccs.T, delta2_mfccs.T])

        features = {
            'mfccs': combined_features, # Use the new combined features
            'energy': librosa.feature.rms(y=y, frame_length=FRAME_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES)[0],
            'zcr': librosa.feature.zero_crossing_rate(y=y, frame_length=FRAME_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES)[0],
            'centroid': librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, win_length=FRAME_LENGTH_SAMPLES, hop_length=HOP_LENGTH_SAMPLES)[0],
            'duration': librosa.get_duration(y=y, sr=sr)
        }
        return features
    except Exception: return None

def interpolate_to_length(sequence, target_length):
    if len(sequence) < 2: return np.full(target_length, sequence[0] if len(sequence) > 0 else 0)
    original_indices = np.linspace(0, 1, len(sequence))
    target_indices = np.linspace(0, 1, target_length)
    interpolator = interp1d(original_indices, sequence, kind="linear", bounds_error=False, fill_value=cast(Any, "extrapolate"))
    return interpolator(target_indices)

def align_and_average(sequences):
    # Simplified alignment for speed.
    aligned_sequences = [interpolate_to_length(seq, QUICK_MATCH_TEMPLATE_LENGTH) for seq in sequences]
    return np.mean(aligned_sequences, axis=0) if aligned_sequences else np.array([])

def create_transition_matrix(all_mfccs_list, kmeans):
    n_clusters = kmeans.n_clusters
    transition_matrix = np.zeros((n_clusters, n_clusters))
    for mfccs in tqdm(all_mfccs_list, desc="Building Transition Matrix", leave=False):
        if len(mfccs) < 2: continue
        mfccs_for_prediction = mfccs[:, 1:]
        cluster_labels = kmeans.predict(mfccs_for_prediction)
        for i in range(len(cluster_labels) - 1):
            transition_matrix[cluster_labels[i], cluster_labels[i + 1]] += 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    return np.divide(transition_matrix, row_sums, where=row_sums!=0).astype(np.float16)

def build_master_patterns():
    print(f"Starting master pattern generation for profile '{PROFILE_NAME}'...")
    os.makedirs(PATTERNS_FOLDER, exist_ok=True)
    phoneme_dirs = [d for d in os.listdir(AUGMENTED_FOLDER) if os.path.isdir(os.path.join(AUGMENTED_FOLDER, d))]

    for phoneme in tqdm(phoneme_dirs, desc="Overall Progress"):
        augmented_phoneme_dir = os.path.join(AUGMENTED_FOLDER, phoneme)
        wav_files = [f for f in os.listdir(augmented_phoneme_dir) if f.endswith('.wav')]
        wav_files.sort(key=lambda f: int(f.split('_')[2].split('.')[0]))
        all_files = [os.path.join(augmented_phoneme_dir, f) for f in wav_files]
        if not all_files:
            continue

        files_to_process = all_files[:FILES_PER_PHONEME_TO_PROCESS]
        with tqdm(total=5, desc=f"Training '{phoneme}'", leave=False) as pbar:
            pbar.set_description(f"'{phoneme}': Stage 1 [Feature Extraction]")
            all_features = [extract_features(f) for f in tqdm(files_to_process, desc="Reading Files", leave=False)]
            all_features = [f for f in all_features if f is not None]
            if not all_features: print(f"Skipping '{phoneme}': No valid features."); continue

            all_mfccs_list = [f['mfccs'] for f in all_features]
            pbar.update(1)

            pbar.set_description(f"'{phoneme}': Stage 2 [Quick Match]")
            energy_template = align_and_average([f['energy'] for f in all_features]).astype(np.float16)
            centroid_template = align_and_average([f['centroid'] for f in all_features]).astype(np.float16)
            zcr_template = np.clip(align_and_average([f['zcr'] for f in all_features]) * 255, 0, 255).astype(np.uint8)
            pbar.update(1)

            pbar.set_description(f"'{phoneme}': Stage 3 [Deep Match Models]")
            all_mfcc_frames = np.vstack(all_mfccs_list)
            all_mfcc_frames = all_mfcc_frames[:, 1:]
            n_clusters = min(KMEANS_CLUSTERS, len(all_mfcc_frames))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(all_mfcc_frames)
            transition_matrix = create_transition_matrix(all_mfccs_list, kmeans)
            pbar.update(1)

            pbar.set_description(f"'{phoneme}': Stage 4 [GMM & Metadata]")
            n_components = min(GMM_COMPONENTS, len(all_mfcc_frames))
            n_components = max(1, n_components)
            mfcc_mean, mfcc_std = np.mean(all_mfcc_frames, axis=0), np.std(all_mfcc_frames, axis=0)
            gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=42, reg_covar=1e-4).fit(all_mfcc_frames)
            all_durations = [f['duration'] for f in all_features]
            min_duration, max_duration = np.min(all_durations), np.max(all_durations)
            pbar.update(1)

            pbar.set_description(f"'{phoneme}': Stage 5 [Assembling Binary]")
            output_pattern_path = os.path.join(PATTERNS_FOLDER, f"{phoneme}.dbp")
            with open(output_pattern_path, "wb") as f:
                # --- Write Quick Match Section ---
                f.write(energy_template.tobytes())
                f.write(centroid_template.tobytes())
                f.write(zcr_template.tobytes())
                target_pos = int(PATTERN_SIZE_KB * 1024 * 0.1)
                f.write(b'\x00' * (target_pos - f.tell()))

                # --- Write Deep Match Section ---
                assert hasattr(gmm, 'weights_') and gmm.weights_ is not None
                assert hasattr(gmm, 'means_') and gmm.means_ is not None
                assert hasattr(gmm, 'covariances_') and gmm.covariances_ is not None

                f.write(gmm.weights_.astype(np.float32).tobytes())
                f.write(gmm.means_.astype(np.float32).tobytes())
                f.write(gmm.covariances_.astype(np.float32).tobytes())
                f.write(kmeans.cluster_centers_.astype(np.float32).tobytes())
                f.write(transition_matrix.tobytes())
                f.write(np.zeros(1000, dtype=np.float16).tobytes())

                target_pos += int(PATTERN_SIZE_KB * 1024 * 0.7)
                padding = target_pos - f.tell()
                if padding > 0: f.write(b'\x00' * padding)

                # --- Write Metadata Section ---
                f.write(struct.pack('ff', min_duration, max_duration))
                f.write(mfcc_mean.astype(np.float32).tobytes())
                f.write(mfcc_std.astype(np.float32).tobytes())
                f.truncate(PATTERN_SIZE_KB * 1024)
            pbar.update(1)

if __name__ == "__main__":
    build_master_patterns()
    print("\n✅ Master pattern generation complete!")
