# scripts/07_evaluate_patterns.py (STABLE VERSION)

import os
import struct
import warnings
from dataclasses import dataclass
from typing import Dict

import librosa
import numpy as np
import toml
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# --- Suppress specific warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Evaluation Mode Control ---
# True:  Run full recognition (match each file against ALL patterns).
# False: Run self-accuracy check (match each file ONLY against its own pattern).
FULL_RECOGNITION_MODE = True


# --- Define a data structure for our patterns (Simplified) ---
@dataclass
class PhonemePattern:
    name: str
    energy_template: np.ndarray
    centroid_template: np.ndarray
    zcr_template: np.ndarray
    gmm_weights: np.ndarray
    gmm_means: np.ndarray
    gmm_covariances: np.ndarray
    kmeans_centers: np.ndarray
    transition_matrix: np.ndarray
    mfcc_mean: np.ndarray
    mfcc_std: np.ndarray
    min_duration: float
    max_duration: float


# --- Load Master Configuration ---
try:
    CONFIG = toml.load("config/dhvani_config.toml")
    PROFILE_NAME = CONFIG["active_profile"]
    PROFILE = CONFIG["profiles"][PROFILE_NAME]
    PATHS = CONFIG["paths"]
    PROC = CONFIG["processing"]
    FEAT = CONFIG["features"]
    print(f"✅ Configuration loaded for profile: '{PROFILE['description']}'")
except Exception as e:
    print(
        f"❌ FATAL ERROR: Could not load or parse 'config/dhvani_config.toml'. Error: {e}"
    )
    exit()

# --- Assign variables from config ---
N_MFCC_TOTAL = PROFILE["n_mfcc"]
N_FEATURES_TOTAL = N_MFCC_TOTAL * 3
N_FEATURES_MODEL = N_FEATURES_TOTAL - 1
GMM_COMPONENTS = PROFILE["gmm_components"]
KMEANS_CLUSTERS = PROFILE["kmeans_clusters"]
TEMPLATE_LENGTH = PROFILE["template_length"]

# --- Core Functions ---


def load_patterns(patterns_folder: str) -> Dict[str, PhonemePattern]:
    """Loads all .dbp pattern files from a directory into a dictionary."""
    patterns = {}
    print("🔎 Loading phoneme patterns from disk...")
    for filename in tqdm(os.listdir(patterns_folder), desc="Loading Patterns"):
        if not filename.endswith(".dbp"):
            continue

        phoneme_name = filename.split(".")[0]
        filepath = os.path.join(patterns_folder, filename)

        with open(filepath, "rb") as f:

            def read_np_array(dtype, shape):
                num_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
                buffer = f.read(num_bytes)
                return np.frombuffer(buffer, dtype=dtype).reshape(shape)

            energy = read_np_array(np.float16, (TEMPLATE_LENGTH,))
            centroid = read_np_array(np.float16, (TEMPLATE_LENGTH,))
            zcr = read_np_array(np.uint8, (TEMPLATE_LENGTH,))
            f.seek(int(PROFILE["pattern_size_kb"] * 1024 * 0.1))

            weights = read_np_array(np.float32, (GMM_COMPONENTS,))
            means = read_np_array(np.float32, (GMM_COMPONENTS, N_FEATURES_MODEL))
            covars = read_np_array(
                np.float32, (GMM_COMPONENTS, N_FEATURES_MODEL, N_FEATURES_MODEL)
            )
            kmeans_centers = read_np_array(
                np.float32, (KMEANS_CLUSTERS, N_FEATURES_MODEL)
            )
            trans_matrix = read_np_array(np.float16, (KMEANS_CLUSTERS, KMEANS_CLUSTERS))

            f.seek(int(PROFILE["pattern_size_kb"] * 1024 * 0.8))

            min_dur, max_dur = struct.unpack("ff", f.read(8))
            mfcc_mean = read_np_array(np.float32, (N_FEATURES_MODEL,))
            mfcc_std = read_np_array(np.float32, (N_FEATURES_MODEL,))

            patterns[phoneme_name] = PhonemePattern(
                name=phoneme_name,
                energy_template=energy,
                centroid_template=centroid,
                zcr_template=zcr,
                gmm_weights=weights,
                gmm_means=means,
                gmm_covariances=covars,
                kmeans_centers=kmeans_centers,
                transition_matrix=trans_matrix,
                mfcc_mean=mfcc_mean,
                mfcc_std=mfcc_std,
                min_duration=min_dur,
                max_duration=max_dur,
            )
    return patterns


def extract_features_for_test(filepath: str):
    try:
        y, sr = librosa.load(filepath, sr=FEAT["sample_rate"], mono=True)
        if len(y) < FEAT["frame_length_samples"]:
            return None
        mfccs = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC_TOTAL,
            n_fft=FEAT["n_fft"],
            win_length=FEAT["frame_length_samples"],
            hop_length=FEAT["hop_length_samples"],
        )
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        combined_features = np.hstack([mfccs.T, delta_mfccs.T, delta2_mfccs.T])
        energy = librosa.feature.rms(
            y=y,
            frame_length=FEAT["frame_length_samples"],
            hop_length=FEAT["hop_length_samples"],
        )[0]
        return {"mfccs": combined_features, "energy": energy}
    except Exception:
        return None


def recognize_phoneme(
    wav_path: str, all_patterns: Dict[str, PhonemePattern], config: Dict
) -> tuple[str, float]:
    features = extract_features_for_test(wav_path)
    if features is None:
        return "error", 0.0

    input_mfccs = features["mfccs"]
    candidates = list(all_patterns.keys())
    best_score = -np.inf
    best_phoneme = "unknown"

    for phoneme_name in candidates:
        pattern = all_patterns[phoneme_name]
        input_mfccs_model = input_mfccs[:, 1:]
        input_mfccs_normalized = (input_mfccs_model - pattern.mfcc_mean) / (
            pattern.mfcc_std + 1e-8
        )

        gmm = GaussianMixture(
            n_components=len(pattern.gmm_weights),
            covariance_type="full",
            reg_covar=1e-4,
        )
        gmm.weights_ = pattern.gmm_weights
        gmm.means_ = pattern.gmm_means
        gmm.covariances_ = pattern.gmm_covariances

        try:
            # Final check needed after manual loading for the .score() method
            covars_float64 = pattern.gmm_covariances.astype(np.float64)
            gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covars_float64))
        except np.linalg.LinAlgError:
            continue  # If a model is unstable, we can't score it, so we skip it.

        score = gmm.score(input_mfccs_normalized)
        if score > best_score:
            best_score = score
            best_phoneme = phoneme_name

    return best_phoneme, best_score


# --- Main Execution ---
if __name__ == "__main__":
    patterns = load_patterns(PATHS["pattern_output"])
    if not patterns:
        print("❌ No patterns found. Please run the training script first.")
        exit()

    augmented_folder = PATHS["augmented_data"]
    test_files = []
    for phoneme_dir in os.listdir(augmented_folder):
        dir_path = os.path.join(augmented_folder, phoneme_dir)
        if not os.path.isdir(dir_path) or not phoneme_dir in patterns:
            continue

        all_files_in_dir = sorted(
            os.listdir(dir_path), key=lambda f: int(f.split("_")[2].split(".")[0])
        )

        # Select exactly 100 files with indices from 1901 to 2000
        files_to_test = [
            f
            for f in all_files_in_dir
            if f.endswith(".wav") and 501 <= int(f.split("_")[2].split(".")[0]) <= 2000
        ]

        for f in files_to_test:
            test_files.append((os.path.join(dir_path, f), phoneme_dir))

    print(f"\n✅ Found {len(test_files)} files for testing (100 per phoneme).")

    if FULL_RECOGNITION_MODE:
        # ... (The rest of your main execution block for both modes remains the same)
        # ... I've omitted it here for brevity but you should keep it as it was ...
        print("🚀 Running in FULL RECOGNITION mode (matching against all patterns)...")
        results_file = "evaluation_results_full.csv"
        with open(results_file, "w") as f:
            f.write(
                "file_path,actual_phoneme,predicted_phoneme,confidence_score,is_correct\n"
            )

            correct_predictions = 0
            for file_path, actual_phoneme in tqdm(test_files, desc="Evaluating"):
                predicted_phoneme, score = recognize_phoneme(
                    file_path, patterns, CONFIG
                )
                is_correct = actual_phoneme == predicted_phoneme
                if is_correct:
                    correct_predictions += 1
                f.write(
                    f"{file_path},{actual_phoneme},{predicted_phoneme},{score},{is_correct}\n"
                )

        accuracy = (correct_predictions / len(test_files)) * 100 if test_files else 0
        print("\n--- Full Recognition Complete ---")
        print(f"📊 Results saved to: {results_file}")
        print(
            f"🎯 Overall Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(test_files)})"
        )
        print("---------------------------------")

    else:  # Self-Accuracy Mode
        print(
            "🎯 Running in SELF-ACCURACY mode (matching each file against its own pattern)..."
        )
        results_file = "evaluation_results_self_accuracy.csv"
        with open(results_file, "w") as f:
            f.write("file_path,target_phoneme,self_match_score\n")

            for file_path, actual_phoneme in tqdm(
                test_files, desc="Evaluating Self-Accuracy"
            ):
                if actual_phoneme not in patterns:
                    continue
                # Create a dictionary with only the one correct pattern to test against
                target_pattern = {actual_phoneme: patterns[actual_phoneme]}

                # The "predicted" phoneme will be the actual one; we care about the score
                _, score = recognize_phoneme(file_path, target_pattern, CONFIG)

                f.write(f"{file_path},{actual_phoneme},{score}\n")

        print("\n--- Self-Accuracy Evaluation Complete ---")
        print(f"📊 Results saved to: {results_file}")
        print(
            "📈 You can now analyze the scores in this file to see how well each pattern matches its own samples."
        )
        print("-----------------------------------------")
