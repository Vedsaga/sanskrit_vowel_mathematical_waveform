# scripts/06_probe_patterns.py (Final Version with Improved Layout)

import os
import struct
import numpy as np
import toml
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import argparse
import math

# --- Load Master Configuration ---
try:
    CONFIG = toml.load("config/dhvani_config.toml")
    PROFILE_NAME = CONFIG['active_profile']
    PROFILE = CONFIG['profiles'][PROFILE_NAME]

    N_MFCC = PROFILE['n_mfcc']
    GMM_COMPONENTS = PROFILE['gmm_components']
    KMEANS_CLUSTERS = PROFILE['kmeans_clusters']
    QUICK_MATCH_TEMPLATE_LENGTH = PROFILE['template_length']
    print(f"✅ Probe configuration loaded for profile: '{PROFILE['description']}'")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load or parse 'config/dhvani_config.toml'. Error: {e}")
    exit()

# --- Font Handling for Devanagari ---
try:
    font_path = next(font for font in fm.findSystemFonts() if 'NotoSansDevanagari' in font)
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = [prop.get_name(), 'sans-serif']
except StopIteration:
    print("Warning: Noto Sans Devanagari font not found.")

def read_and_parse_dbp(filepath):
    """Reads a .dbp file and parses it completely using the configured parameters."""
    data = {}
    with open(filepath, 'rb') as f:
        total_size = os.path.getsize(filepath)
        f.seek(0)

        data['energy_template'] = np.fromfile(f, dtype=np.float16, count=QUICK_MATCH_TEMPLATE_LENGTH)
        data['centroid_template'] = np.fromfile(f, dtype=np.float16, count=QUICK_MATCH_TEMPLATE_LENGTH)
        data['zcr_template'] = np.fromfile(f, dtype=np.uint8, count=QUICK_MATCH_TEMPLATE_LENGTH)

        f.seek(int(total_size * 0.1))
        data['gmm_weights'] = np.fromfile(f, dtype=np.float32, count=GMM_COMPONENTS)
        data['gmm_means'] = np.fromfile(f, dtype=np.float32, count=GMM_COMPONENTS * N_MFCC).reshape(GMM_COMPONENTS, N_MFCC)
        data['gmm_covariances'] = np.fromfile(f, dtype=np.float32, count=GMM_COMPONENTS * N_MFCC * N_MFCC).reshape(GMM_COMPONENTS, N_MFCC, N_MFCC)
        data['transition_matrix'] = np.fromfile(f, dtype=np.float16, count=KMEANS_CLUSTERS * KMEANS_CLUSTERS).reshape(KMEANS_CLUSTERS, KMEANS_CLUSTERS)

        f.seek(int(total_size * 0.8))
        data['min_duration'], data['max_duration'] = struct.unpack('ff', f.read(8))
        data['mfcc_mean'] = np.fromfile(f, dtype=np.float32, count=N_MFCC)
        data['mfcc_std'] = np.fromfile(f, dtype=np.float32, count=N_MFCC)

    return data

def visualize_pattern(pattern_data, phoneme_name):
    """Creates a comprehensive dashboard of plots for the master pattern."""

    # ---  LAYOUT CONTROL ---
    NUM_COLS = 1 # Change to 2 for a side-by-side grid
    FIG_HEIGHT_PER_ROW = 6 # Increased for more breathing room
    # ----------------------

    plots = {
        "Quick Match Templates": plot_quick_match,
        "GMM Component Weights": plot_gmm_weights,
        "GMM Mean Vectors (The 'Flavors')": plot_gmm_means,
        "GMM Variance (Diagonal of Covariance)": plot_gmm_variance,
        "Temporal Transition Matrix": plot_transition_matrix,
    }

    num_plots = len(plots)
    num_rows = math.ceil(num_plots / NUM_COLS)

    # --- FIX: Use `constrained_layout=True` for better automatic spacing ---
    fig, axs = plt.subplots(
        num_rows, NUM_COLS,
        figsize=(8 * NUM_COLS, FIG_HEIGHT_PER_ROW * num_rows),
        squeeze=False,
        constrained_layout=True
    )
    fig.suptitle(f"Master Pattern Dashboard for Phoneme: '{phoneme_name}'", fontsize=24)
    axs = axs.flatten()

    for ax, (title, plot_func) in zip(axs, plots.items()):
        ax.set_title(title, fontsize=16)
        plot_func(ax, fig, pattern_data)

    for i in range(num_plots, len(axs)):
        axs[i].set_visible(False)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nPlot window closed.")

# --- Individual Plotting Functions (Unchanged) ---
def plot_quick_match(ax, fig, data):
    ax.plot(data['energy_template'], label='Energy Envelope', color='red', linewidth=2)
    if np.any(data['centroid_template']):
        norm_centroid = data['centroid_template'] / np.max(data['centroid_template'])
        ax.plot(norm_centroid, label='Spectral Centroid (Normalized)', color='blue', linestyle='--')
    ax.plot(data['zcr_template'] / 255.0, label='Zero Crossing Rate (Normalized)', color='green', linestyle=':')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_gmm_weights(ax, fig, data):
    ax.bar(range(GMM_COMPONENTS), data['gmm_weights'])
    ax.set_xlabel("GMM Component Index")
    ax.set_ylabel("Weight (Importance)")
    ax.grid(True, axis='y', alpha=0.3)

def plot_gmm_means(ax, fig, data):
    im = ax.imshow(data['gmm_means'], cmap='viridis', aspect='auto')
    fig.colorbar(im, ax=ax, label="Coefficient Value")
    ax.set_xlabel(f"MFCC Coefficients (0-{N_MFCC-1})")
    ax.set_ylabel("GMM Component Index")

def plot_gmm_variance(ax, fig, data):
    variances = np.array([np.diag(cov) for cov in data['gmm_covariances']])
    im = ax.imshow(variances, cmap='magma', aspect='auto')
    fig.colorbar(im, ax=ax, label="Variance ('Fuzziness')")
    ax.set_xlabel(f"MFCC Coefficients (0-{N_MFCC-1})")
    ax.set_ylabel("GMM Component Index")

def plot_transition_matrix(ax, fig, data):
    im = ax.imshow(data['transition_matrix'], cmap='hot', aspect='auto')
    fig.colorbar(im, ax=ax, label="Transition Probability")
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe a Dhvani .dbp master pattern file.")
    parser.add_argument("filepath", type=str, help="Path to the .dbp file.")
    args = parser.parse_args()
    if os.path.exists(args.filepath):
        phoneme_name = os.path.basename(args.filepath).split('.')[0]
        print(f"--- Probing pattern file for phoneme: '{phoneme_name}' ---")
        pattern_data = read_and_parse_dbp(args.filepath)
        visualize_pattern(pattern_data, phoneme_name)
    else:
        print(f"Error: File not found at '{args.filepath}'")
