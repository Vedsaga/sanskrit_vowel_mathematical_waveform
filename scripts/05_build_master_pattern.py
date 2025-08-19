import os
import struct
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d
import librosa
import librosa.feature
from typing import Any, cast

# Optional DTW import with fallback
try:
    import dtw as dtw_module
    DTW_AVAILABLE = True
except ImportError:
    dtw_module = None  # type: ignore
    DTW_AVAILABLE = False
    print("Warning: DTW not available. Using simple averaging for alignment.")

# --- Configuration ---
AUGMENTED_FOLDER = "data/03_augmented/"
PATTERNS_FOLDER = "data/05_patterns/"
FILES_PER_PHONEME_TO_PROCESS = 500
SAMPLE_RATE = 16000
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

# --- Suppress scikit-learn warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Helper Functions ---
def extract_features(filepath):
    """Extract real audio features using librosa."""
    try:
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        if len(y) < FRAME_LENGTH:
            return None, None, None, None, None

        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH, n_fft=FRAME_LENGTH
        ).T

        energy = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)[0]
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
        duration = len(y) / sr

        return mfccs, energy, zcr, centroid, duration

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None, None, None, None, None

def align_sequences_dtw(sequences, reference_idx=None):
    """Align sequences using DTW if available, otherwise use simple interpolation."""
    if not sequences:
        return []

    lengths = [len(seq) for seq in sequences]
    if reference_idx is None:
        reference_idx = np.argsort(lengths)[len(lengths) // 2]

    reference = sequences[reference_idx]
    aligned_sequences = []

    if DTW_AVAILABLE and dtw_module is not None and len(sequences) > 1:
        for seq in tqdm(sequences, desc="DTW Alignment", leave=False):
            if len(seq) == 0:
                continue
            try:
                alignment = dtw_module.dtw(reference, seq, keep_internals=True)
                index2 = getattr(alignment, "index2", None)
                if index2 is not None:
                    warped_seq = seq[index2]
                    aligned_sequences.append(warped_seq)
                else:
                    aligned_sequences.append(interpolate_to_length(seq, len(reference)))
            except Exception:
                aligned_sequences.append(interpolate_to_length(seq, len(reference)))
    else:
        target_length = len(reference)
        for seq in sequences:
            aligned_sequences.append(interpolate_to_length(seq, target_length))

    return aligned_sequences

def interpolate_to_length(sequence, target_length):
    """Interpolate sequence to target length."""
    if len(sequence) < 2:
        return np.zeros(target_length)

    original_indices = np.linspace(0, 1, len(sequence))
    target_indices = np.linspace(0, 1, target_length)

    try:
        interpolator = interp1d(
            original_indices,
            sequence,
            kind="linear",
            bounds_error=False,
            fill_value=cast(Any, "extrapolate")  # now works
        )
        return interpolator(target_indices)
    except Exception:
        if len(sequence) >= target_length:
            return sequence[:target_length]
        else:
            return np.pad(sequence, (0, target_length - len(sequence)), mode="edge")

def create_transition_matrix(kmeans_labels, n_clusters):
    """Create transition matrix from K-means cluster labels."""
    transition_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(len(kmeans_labels) - 1):
        transition_matrix[kmeans_labels[i], kmeans_labels[i + 1]] += 1
    row_sums = transition_matrix.sum(axis=1)
    row_sums[row_sums == 0] = 1
    return transition_matrix / row_sums[:, np.newaxis]

def resample_to_fixed_length(array, target_length=200):
    """Resample array to fixed length for binary storage."""
    if len(array) == 0:
        return np.zeros(target_length)
    return interpolate_to_length(array, target_length)

# --- Main Training Script ---
def build_master_patterns():
    print("Starting master pattern generation...")
    os.makedirs(PATTERNS_FOLDER, exist_ok=True)

    if not os.path.exists(AUGMENTED_FOLDER):
        print(f"Error: Augmented folder '{AUGMENTED_FOLDER}' not found!")
        return

    phoneme_dirs = [
        d for d in os.listdir(AUGMENTED_FOLDER)
        if os.path.isdir(os.path.join(AUGMENTED_FOLDER, d))
    ]
    if not phoneme_dirs:
        print("No phoneme directories found!")
        return

    print(f"Found {len(phoneme_dirs)} phoneme directories")

    for phoneme in tqdm(phoneme_dirs, desc="Overall Progress"):
        augmented_phoneme_dir = os.path.join(AUGMENTED_FOLDER, phoneme)
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        all_files = [
            os.path.join(augmented_phoneme_dir, f)
            for f in os.listdir(augmented_phoneme_dir)
            if any(f.lower().endswith(ext) for ext in audio_extensions)
        ]
        if not all_files:
            print(f"No audio files found for phoneme '{phoneme}'")
            continue

        files_to_process = all_files[:FILES_PER_PHONEME_TO_PROCESS]
        with tqdm(total=6, desc=f"Training '{phoneme}'", leave=False) as pbar:
            all_mfccs_list, all_energies, all_zcrs, all_centroids, all_durations = [], [], [], [], []
            for f in files_to_process:
                mfccs, energy, zcr, centroid, duration = extract_features(f)
                if mfccs is not None and len(mfccs) > 0:
                    all_mfccs_list.append(mfccs)
                    all_energies.append(energy)
                    all_zcrs.append(zcr)
                    all_centroids.append(centroid)
                    all_durations.append(duration)
            if not all_mfccs_list:
                print(f"No valid features extracted for phoneme '{phoneme}'")
                continue
            pbar.update(1)

            aligned_energies = align_sequences_dtw(all_energies)
            aligned_centroids = align_sequences_dtw(all_centroids)
            aligned_zcrs = align_sequences_dtw(all_zcrs)
            if aligned_energies:
                TARGET_LEN = 200
                aligned_energies = [resample_to_fixed_length(seq, TARGET_LEN) for seq in aligned_energies]
                aligned_centroids = [resample_to_fixed_length(seq, TARGET_LEN) for seq in aligned_centroids]
                aligned_zcrs = [resample_to_fixed_length(seq, TARGET_LEN) for seq in aligned_zcrs]

                avg_energy = np.mean(aligned_energies, axis=0)
                avg_centroid = np.mean(aligned_centroids, axis=0)
                avg_zcr = np.mean(aligned_zcrs, axis=0)

                energy_template = avg_energy.astype(np.float16)
                centroid_template = avg_centroid.astype(np.float16)
                zcr_template = (avg_zcr * 255).astype(np.uint8)
            else:
                energy_template = np.zeros(200, dtype=np.float16)
                centroid_template = np.zeros(200, dtype=np.float16)
                zcr_template = np.zeros(200, dtype=np.uint8)

            all_mfcc_frames = np.vstack(all_mfccs_list)
            n_clusters = min(50, len(all_mfcc_frames) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(all_mfcc_frames)
            cluster_labels = kmeans.labels_
            transition_matrix = create_transition_matrix(cluster_labels, n_clusters)
            pbar.update(1)

            n_components = min(5, len(all_mfcc_frames) // 10)
            n_components = max(1, n_components)
            gmm = GaussianMixture(
                n_components=n_components, covariance_type="full",
                random_state=42, max_iter=100
            ).fit(all_mfcc_frames)
            pbar.update(1)

            min_duration, max_duration = np.min(all_durations), np.max(all_durations)
            mean_duration, std_duration = np.mean(all_durations), np.std(all_durations)
            num_samples = len(all_durations)
            pbar.update(1)

            output_pattern_path = os.path.join(PATTERNS_FOLDER, f"{phoneme}.dbp")
            with open(output_pattern_path, "wb") as f:
                f.write(energy_template.tobytes())
                f.write(centroid_template.tobytes())
                f.write(zcr_template.tobytes())
                padding_needed = (10 * 1024) - f.tell()
                if padding_needed > 0:
                    f.write(b"\x00" * padding_needed)
                assert gmm.weights_ is not None
                assert gmm.means_ is not None
                assert gmm.covariances_ is not None
                f.write(gmm.weights_.astype(np.float32).tobytes())
                f.write(gmm.means_.astype(np.float32).tobytes())
                f.write(gmm.covariances_.astype(np.float32).tobytes())
                f.write(kmeans.cluster_centers_.astype(np.float32).tobytes())
                f.write(transition_matrix.astype(np.float32).tobytes())
                padding_needed = (80 * 1024) - f.tell()
                if padding_needed > 0:
                    f.write(b"\x00" * padding_needed)

                f.write(struct.pack("fffff", min_duration, max_duration,
                                    mean_duration, std_duration, float(num_samples)))
                f.write(struct.pack("ii", n_clusters, n_components))
                total_size = 100 * 1024
                if f.tell() < total_size:
                    f.write(b"\x00" * (total_size - f.tell()))
            pbar.update(1)

    print("✅ Master pattern generation complete!")
    print(f"Generated patterns for {len(phoneme_dirs)} phonemes")
    print(f"Pattern files saved to: {PATTERNS_FOLDER}")

if __name__ == "__main__":
    build_master_patterns()
