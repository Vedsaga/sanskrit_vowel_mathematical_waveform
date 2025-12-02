import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from google.colab import drive
import soundfile as sf
import tempfile
import shutil

# ==============================================================================
#                      --- 1. SETUP AND CONFIGURATION ---
# ==============================================================================
print("🔧 Initializing the FINAL, DEFINITIVE Analysis Script...")

# --- Font Setup ---
!sudo apt-get -qq install fonts-indic > /dev/null 2>&1
!fc-cache -fv > /dev/null 2>&1
from matplotlib.font_manager import FontProperties, findfont
try:
    devanagari_font = FontProperties(fname=findfont('Noto Sans Devanagari'))
    print("✅ Devanagari font configured.")
except:
    devanagari_font = FontProperties()
    print("⚠️ Devanagari font not found.")

plt.rcParams['font.family'] = 'DejaVu Sans'; plt.rcParams['axes.unicode_minus'] = False

# --- Path and Model Configuration ---
drive.mount('/content/drive', force_remount=True)
SR = 16000; N_MELS = 40; HOP_SIZE = 160; MAX_LENGTH = 70; DEVICE = "cuda" if torch.cuda.is_available() else "cpu"; BATCH_SIZE = 64

DRIVE_BASE_PATH = "/content/drive/MyDrive/Sanskrit_Phoneme_Recognition"
MODEL_PATH = f"{DRIVE_BASE_PATH}/models/best_model.pth"
RESULTS_DIR = f"{DRIVE_BASE_PATH}/analysis_results_FINAL_CORRECT"
LOCAL_DATA_PATH = "/content/data"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================================================================
#              --- 2. MODEL & DATASET (VERIFIED FROM TRAINING) ---
# ==============================================================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(input_channels, 32, (3,3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Dropout2d(0.1), nn.Conv2d(32, 64, (3,3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Dropout2d(0.1), nn.Conv2d(64, 128, (3,3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)), nn.Dropout2d(0.2))
        self.feature_size = 128 * 4 * 4
        self.classifier = nn.Sequential(nn.Linear(self.feature_size, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.conv_layers(x); x = x.view(x.size(0), -1); x = self.classifier(x)
        return x

# Using the EXACT PhonemeDataset class from your training script
class PhonemeDataset(Dataset):
    def __init__(self, file_paths, labels, max_length):
        self.file_paths = file_paths; self.labels = labels; self.max_length = max_length
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        path = self.file_paths[idx]; label = self.labels[idx]
        y, _ = librosa.load(path, sr=SR)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP_SIZE, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        if self.max_length:
            if mel_db.shape[1] < self.max_length:
                mel_db = np.pad(mel_db, ((0, 0), (0, self.max_length - mel_db.shape[1])), mode='constant')
            else:
                mel_db = mel_db[:, :self.max_length]
        return torch.tensor(mel_db).unsqueeze(0).float(), label

def load_trained_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    phonemes = checkpoint['phonemes']
    model = ImprovedCNN(num_classes=len(phonemes)).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Best model loaded from epoch {checkpoint.get('epoch', 'N/A')} with verified accuracy {checkpoint.get('best_val_acc', 0):.2%}")
    return model, phonemes

# ==============================================================================
#           --- 3. CORRECTED CONTINUOUS ANALYSIS PIPELINE ---
# ==============================================================================

def analyze_continuous_audio(model, phonemes, audio_path):
    print(f"\n🔬 Analyzing with 100% correct pipeline: {os.path.basename(audio_path)}...")
    y, _ = librosa.load(audio_path, sr=SR, mono=True) # Load the full audio once

    window_s = 0.4; step_s = 0.05
    slice_samples = int(SR * window_s)

    # Create a temporary directory to store each slice as an independent file
    temp_dir = tempfile.mkdtemp()
    slice_paths = []

    print(f"Creating temporary audio slices in {temp_dir}...")
    try:
        for i in range(0, len(y) - slice_samples + 1, int(SR * step_s)):
            audio_slice = y[i : i + slice_samples]
            slice_path = os.path.join(temp_dir, f"slice_{i}.wav")
            sf.write(slice_path, audio_slice, SR)
            slice_paths.append(slice_path)

        # Now, use the EXACT same Dataset and DataLoader from training
        # We pass dummy labels because they aren't used for inference
        dummy_labels = [0] * len(slice_paths)
        slice_dataset = PhonemeDataset(slice_paths, dummy_labels, max_length=MAX_LENGTH)
        slice_loader = DataLoader(slice_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        raw_predictions = []
        with torch.no_grad():
            for i, (batch_tensors, _) in enumerate(tqdm(slice_loader, desc="Analyzing Slices")):
                batch_tensors = batch_tensors.to(DEVICE)
                logits = model(batch_tensors)
                probabilities = torch.softmax(logits, dim=-1)

                for j, p in enumerate(probabilities):
                    pred_id = torch.argmax(p).item()
                    slice_index = i * BATCH_SIZE + j
                    raw_predictions.append({
                        "time": slice_index * step_s,
                        "prediction": phonemes[pred_id],
                        "confidence": p[pred_id].item(),
                        "all_probs": p.cpu().numpy()
                    })
    finally:
        # Clean up the temporary directory
        print(f"Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)

    print(f"✅ Analysis complete. Generated {len(raw_predictions)} predictions.")
    return raw_predictions


# ==============================================================================
#                      --- 4. ADVANCED PROBING & MAIN ---
# ==============================================================================
def parse_label_file(file_path):
    #... (This function is unchanged)
    labels = [];
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            start, end, label = line.strip().split(); labels.append({'start': float(start), 'end': float(end), 'label': label, 'duration': float(end) - float(start)})
    return labels

def perform_advanced_probing(raw_predictions, ground_truth, phonemes, audio_name):
    """Runs and prints all advanced analysis reports and plots."""

    stability_results, y_true_focused, y_pred_focused = [], [], []

    for i, seg in enumerate(ground_truth):
        preds_in_segment = [p for p in raw_predictions if seg['start'] <= p['time'] < seg['end']]

        if not preds_in_segment:
            stability_results.append({'segment_index': i, 'true_label': seg['label'], 'start_time': f"{seg['start']:.2f}s", 'duration_ms': f"{seg['duration']*1000:.0f}", 'stability_%': 0.0})
            continue

        correct_preds = [p for p in preds_in_segment if p['prediction'] == seg['label']]
        stability_score = (len(correct_preds) / len(preds_in_segment)) * 100

        # --- BUGFIX ---
        # Store the stability score as a raw number (float) for calculations.
        stability_results.append({
            'segment_index': i,
            'true_label': seg['label'],
            'start_time': f"{seg['start']:.2f}s",
            'duration_ms': f"{seg['duration']*1000:.0f}",
            'stability_%': stability_score
        })

        pred_labels = [p['prediction'] for p in preds_in_segment]
        most_common_pred = max(set(pred_labels), key=pred_labels.count)
        y_true_focused.append(seg['label'])
        y_pred_focused.append(most_common_pred)

    print("\n" + "="*80)
    print("              FINAL ANALYSIS 1: PREDICTION STABILITY")
    print("="*80)

    stability_df = pd.DataFrame(stability_results)

    # Create a separate copy for pretty printing, keeping original numbers for calculation
    printable_df = stability_df.copy()
    printable_df['stability_%'] = printable_df['stability_%'].map('{:.1f}'.format)
    print(printable_df.to_string())

    # Now the groupby().mean() will work on the original DataFrame with numbers
    avg_stability = stability_df.groupby('true_label')['stability_%'].mean().sort_values(ascending=False)
    print("\n--- Average Stability per Phoneme ---")
    print(avg_stability.to_string())

    print("\n\n" + "="*80)
    print("              FINAL ANALYSIS 2: FOCUSED ACCURACY")
    print("="*80)
    print(classification_report(y_true_focused, y_pred_focused, zero_division=0))

    print("\n\n" + "="*80)
    print("              FINAL ANALYSIS 3: ACTIVATION PLOT")
    print("="*80)

    fig, ax = plt.subplots(figsize=(20, 8))
    phoneme_indices = {name: i for i, name in enumerate(phonemes)}

    for seg in ground_truth:
        ax.axvspan(seg['start'], seg['end'], alpha=0.2, color='gray')
        preds_in_segment = [p for p in raw_predictions if seg['start'] <= p['time'] <= seg['end']]
        if not preds_in_segment: continue
        times = [p['time'] for p in preds_in_segment]
        true_label_index = phoneme_indices.get(seg['label'])
        if true_label_index is not None:
            activation_strengths = [p['all_probs'][true_label_index] for p in preds_in_segment]
            ax.plot(times, activation_strengths, marker='.', linestyle='-')
        ax.text(seg['start'] + seg['duration']/2, 1.01, seg['label'], ha='center', va='bottom', fontproperties=devanagari_font, fontsize=14)

    ax.set_ylim(0, 1.1)
    ax.set_xlim(ground_truth[0]['start'] - 1, ground_truth[-1]['end'] + 1)
    ax.set_title(f'Activation Strength of Correct Phoneme ({audio_name})', fontsize=16)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Model Probability')
    ax.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(RESULTS_DIR, f"activation_plot_{audio_name.replace('.wav', '')}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✅ Activation plot saved to {plot_path}")
    plt.show()

def main():
    #... (This function is unchanged)
    model, phonemes = load_trained_model()
    if model is None: return
    audio_file_to_analyze = os.path.join(LOCAL_DATA_PATH, "01_raw/normalized/male-2.wav"); label_file_to_analyze = os.path.join(LOCAL_DATA_PATH, "01_raw/labels/male-2.txt")
    if not os.path.exists(audio_file_to_analyze): print(f"❌ Analysis audio file not found: {audio_file_to_analyze}"); return
    raw_predictions = analyze_continuous_audio(model, phonemes, audio_file_to_analyze); ground_truth = parse_label_file(label_file_to_analyze)
    perform_advanced_probing(raw_predictions, ground_truth, phonemes, os.path.basename(audio_file_to_analyze)); print("\n🎉 Final analysis complete!")

if __name__ == "__main__":
    main()
