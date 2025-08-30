# ================================
#  STEP 0: INSTALL REQUIREMENTS & MOUNT DRIVE
# ================================
# !pip install torch torchaudio torchvision matplotlib seaborn scikit-learn pandas tqdm librosa soundfile

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os, time, random, csv, glob, shutil, tarfile # ### MODIFIED ### - Added shutil and tarfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

# ================================
#  STEP 1: CONFIG
# ================================
SR = 16000
FRAME_SIZE = 400
HOP_SIZE = 160
N_MELS = 40
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- PERMANENT paths on Google Drive ---
DRIVE_BASE_PATH = "/content/drive/MyDrive/Sanskrit_Phoneme_Recognition"
DRIVE_ARCHIVE_PATH = f"{DRIVE_BASE_PATH}/03_augmented.tar.gz"
RESULTS_PATH = f"{DRIVE_BASE_PATH}/results"
MODELS_PATH = f"{DRIVE_BASE_PATH}/models"

# --- TEMPORARY paths on the fast Colab local disk ---
### MODIFIED ### - This is the key fix to match your archive's structure
LOCAL_DATASET_PATH = "/content/data/03_augmented"

# Create permanent directories on Drive
for path in [DRIVE_BASE_PATH, RESULTS_PATH, MODELS_PATH]:
    os.makedirs(path, exist_ok=True)

PHONEMES = ["अ","आ","इ","ई","उ","ऊ","ऋ","ॠ","ऌ","ॡ","ए","ऐ","ओ","औ","अं","अः"]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEMES)}

print(f"🚀 Using device: {DEVICE}")
print(f"📁 Dataset will be processed locally at: {LOCAL_DATASET_PATH}")
print(f"📊 Results will be saved to: {RESULTS_PATH}")

# ================================
#  STEP 2: DATA LOADING FUNCTIONS
# ================================

### MODIFIED ### - This function is rewritten for local extraction
def extract_dataset():
    """Copies archive from Drive and extracts it on the fast local disk."""
    if os.path.exists(LOCAL_DATASET_PATH):
        print(f"✅ Dataset already exists locally at {LOCAL_DATASET_PATH}")
        return True

    if not os.path.exists(DRIVE_ARCHIVE_PATH):
        print(f"❌ Archive not found on Google Drive: {DRIVE_ARCHIVE_PATH}")
        return False

    print("🚀 Copying archive from Drive to local Colab disk...")
    local_archive = "/content/03_augmented.tar.gz"
    shutil.copy(DRIVE_ARCHIVE_PATH, local_archive)
    print("✅ Copy complete.")

    print(f"📦 Extracting archive locally to /content/ ...")
    with tarfile.open(local_archive, 'r:gz') as tar:
        tar.extractall(path="/content/", filter='data')
    print("✅ Local extraction complete.")

    os.remove(local_archive)
    print("🧹 Removed temporary local archive.")
    return True

### MODIFIED ### - This function now reads from the LOCAL path
def load_pre_split_dataset():
    """Load from the pre-split dataset structure on the LOCAL disk."""
    train_files, train_labels = [], []
    val_files, val_labels = [], []
    test_files, test_labels = [], []

    splits = {'train': (train_files, train_labels), 'val': (val_files, val_labels), 'test': (test_files, test_labels)}

    for split_name, (files_list, labels_list) in splits.items():
        split_path = os.path.join(LOCAL_DATASET_PATH, split_name)
        split_total = 0
        print(f"\n📂 Loading {split_name} data from local disk...")
        for phoneme in PHONEMES:
            phoneme_path = os.path.join(split_path, phoneme)
            if os.path.exists(phoneme_path):
                audio_files = glob.glob(os.path.join(phoneme_path, "*.wav"))
                audio_files.extend(glob.glob(os.path.join(phoneme_path, "*.mp3")))
                files_list.extend(audio_files)
                labels_list.extend([PHONEME_TO_IDX[phoneme]] * len(audio_files))
                split_total += len(audio_files)
        print(f"📊 Total {split_name}: {split_total} files")
    return (train_files, train_labels), (val_files, val_labels), (test_files, test_labels)


### MODIFIED ### - This function also reads from the LOCAL path
def load_augmentation_log():
    """Load augmentation log if available from the LOCAL path."""
    log_path = os.path.join(LOCAL_DATASET_PATH, "augmentation_log.csv")
    if os.path.exists(log_path):
        try:
            aug_log = pd.read_csv(log_path)
            print(f"📋 Augmentation log loaded: {len(aug_log)} entries")
            return aug_log
        except Exception as e:
            print(f"⚠️  Could not load augmentation log: {e}")
    return None

# ================================
#  STEP 3: DATASET + AUGMENTATION (No changes needed)
# ================================
class PhonemeDataset(Dataset):
    def __init__(self, file_paths, labels, augment=False, max_length=None):
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.max_length = max_length
    def __len__(self):
        return len(self.file_paths)
    def augment_audio(self, y):
        if random.random() > 0.3: y = librosa.effects.time_stretch(y, rate=random.uniform(0.9, 1.1))
        if random.random() > 0.5: y = y + np.random.normal(0, random.uniform(0.001, 0.01), len(y))
        if random.random() > 0.7: y = librosa.effects.pitch_shift(y, sr=SR, n_steps=random.uniform(-2, 2))
        return y
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            y, sr = librosa.load(path, sr=SR)
            if len(y) < SR * 0.1: y = np.pad(y, (0, int(SR * 0.1) - len(y)), mode='constant')
            if self.augment: y = self.augment_audio(y)
            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP_SIZE, n_mels=N_MELS)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
            if self.max_length:
                if mel_db.shape[1] < self.max_length:
                    pad_width = self.max_length - mel_db.shape[1]
                    mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    mel_db = mel_db[:, :self.max_length]
            mel_db = torch.tensor(mel_db).unsqueeze(0)
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
            mel_db = torch.zeros(1, N_MELS, 100)
        return mel_db.float(), label

# ================================
#  STEP 4: IMPROVED MODEL (No changes needed)
# ================================
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(ImprovedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d((2, 2)), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Dropout2d(0.2)
        )
        self.feature_size = 128 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




# ================================
#  STEP 5: UTILS & VISUALIZATION
# ================================
def plot_confusion_matrix(y_true, y_pred, labels, epoch, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    acc = np.trace(cm) / np.sum(cm)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Class (भविष्यवाणी वर्ग)", fontsize=12)
    plt.ylabel("Actual Class (वास्तविक वर्ग)", fontsize=12)
    plt.title(f"Sanskrit Phoneme Recognition Confusion Matrix (Epoch {epoch})\nAccuracy: {acc*100:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_training_history(history_df, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    train_loss = history_df[history_df['phase'] == 'train']['loss']
    val_loss = history_df[history_df['phase'] == 'val']['loss']
    epochs = history_df[history_df['phase'] == 'train']['epoch']
    
    axes[0,0].plot(epochs, train_loss, 'b-', label='Training Loss')
    axes[0,0].plot(epochs, val_loss, 'r-', label='Validation Loss')
    axes[0,0].set_title('Model Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Accuracy
    train_acc = history_df[history_df['phase'] == 'train']['accuracy']
    val_acc = history_df[history_df['phase'] == 'val']['accuracy']
    
    axes[0,1].plot(epochs, train_acc, 'b-', label='Training Accuracy')
    axes[0,1].plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    axes[0,1].set_title('Model Accuracy')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Accuracy')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # F1 Score
    train_f1 = history_df[history_df['phase'] == 'train']['f1']
    val_f1 = history_df[history_df['phase'] == 'val']['f1']
    
    axes[1,0].plot(epochs, train_f1, 'b-', label='Training F1')
    axes[1,0].plot(epochs, val_f1, 'r-', label='Validation F1')
    axes[1,0].set_title('Model F1 Score')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('F1 Score')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Learning curve comparison
    axes[1,1].plot(epochs, train_loss, 'b--', alpha=0.7, label='Train Loss')
    axes[1,1].plot(epochs, val_loss, 'r--', alpha=0.7, label='Val Loss')
    ax2 = axes[1,1].twinx()
    ax2.plot(epochs, train_acc, 'b-', alpha=0.7, label='Train Acc')
    ax2.plot(epochs, val_acc, 'r-', alpha=0.7, label='Val Acc')
    axes[1,1].set_title('Combined Loss & Accuracy')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    axes[1,1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# ================================
#  STEP 6: TRAINING LOOP
# ================================
def train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, epochs):
    history_file = os.path.join(RESULTS_PATH, "training_log.csv")
    
    # Initialize CSV
    with open(history_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","phase","loss","accuracy","precision","recall","f1"])

    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(1, epochs+1):
        print(f"\n🚀 Epoch {epoch}/{epochs}")
        print("-" * 50)

        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            if phase == "train":
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_labels = []

            start_time = time.time()
            progress_bar = tqdm(loader, desc=f"{phase.upper():5s}")
            
            for batch_idx, (xb, yb) in enumerate(progress_bar):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(xb)
                    loss = criterion(outputs, yb)
                    
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * xb.size(0)
                preds = outputs.argmax(1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{correct/total:.4f}'
                })

            # Calculate metrics
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="weighted", zero_division=0
            )

            # Log to CSV
            with open(history_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, phase, epoch_loss, epoch_acc, precision, recall, f1])

            elapsed = time.time() - start_time
            print(f"✅ {phase:5s} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
                  f"F1: {f1:.4f} | Time: {elapsed:.1f}s")

            # Save best model and early stopping
            if phase == "val":
                if scheduler:
                    scheduler.step(epoch_loss)
                
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    patience_counter = 0
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_acc': best_val_acc,
                        'phonemes': PHONEMES
                    }, os.path.join(MODELS_PATH, "best_model.pth"))
                    print(f"🎯 New best model saved! Accuracy: {best_val_acc:.4f}")
                else:
                    patience_counter += 1

                # Plot confusion matrix every 5 epochs or at the end
                if epoch % 5 == 0 or epoch == epochs:
                    cm_path = os.path.join(RESULTS_PATH, f"confusion_epoch_{epoch}.png")
                    plot_confusion_matrix(all_labels, all_preds, PHONEMES, epoch, cm_path)

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered after {epoch} epochs")
            break

    print(f"\n🏆 Training completed! Best validation accuracy: {best_val_acc:.4f}")
    return model

# ================================
#  STEP 7: INFERENCE & PROBING
# ================================
def load_best_model():
    model_path = os.path.join(MODELS_PATH, "best_model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model = ImprovedCNN(len(PHONEMES)).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Best model loaded from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_val_acc']:.4f}")
        return model
    else:
        print("❌ No saved model found!")
        return None

def inference_single_file(model, file_path, max_length=None):
    """Inference on a single audio file"""
    model.eval()
    
    try:
        y, sr = librosa.load(file_path, sr=SR)
        
        # Process the same way as training data
        mel = librosa.feature.melspectrogram(
            y, sr=SR, n_fft=512, hop_length=HOP_SIZE, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
        
        if max_length:
            if mel_db.shape[1] < max_length:
                pad_width = max_length - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mel_db = mel_db[:, :max_length]
        
        mel_tensor = torch.tensor(mel_db).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
        
        with torch.no_grad():
            outputs = model(mel_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities.max().item()
        
        return predicted_idx, confidence, probabilities.cpu().numpy()[0]
    
    except Exception as e:
        print(f"❌ Error in inference: {e}")
        return None, 0.0, None

def generate_inference_report(model, test_loader):
    """Generate comprehensive inference report"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("🔍 Running inference on test set...")
    
    with torch.no_grad():
        for xb, yb in tqdm(test_loader, desc="Inference"):
            xb = xb.to(DEVICE)
            outputs = model(xb)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    
    # Classification report
    class_report = classification_report(
        all_labels, all_preds, target_names=PHONEMES, output_dict=True
    )
    
    print(f"\n📊 INFERENCE RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save detailed report
    report_path = os.path.join(RESULTS_PATH, "inference_report.txt")
    with open(report_path, "w") as f:
        f.write("Sanskrit Phoneme Recognition - Inference Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Overall Precision: {precision:.4f}\n")
        f.write(f"Overall Recall: {recall:.4f}\n")
        f.write(f"Overall F1 Score: {f1:.4f}\n\n")
        f.write("Per-class Results:\n")
        f.write("-" * 20 + "\n")
        
        for i, phoneme in enumerate(PHONEMES):
            if str(i) in class_report:
                p = class_report[str(i)]['precision']
                r = class_report[str(i)]['recall']
                f1_class = class_report[str(i)]['f1-score']
                support = class_report[str(i)]['support']
                f.write(f"{phoneme}: P={p:.3f}, R={r:.3f}, F1={f1_class:.3f}, Support={support}\n")
    
    return all_preds, all_labels, all_probs

# ================================
#  STEP 8: MAIN EXECUTION
# ================================
def main():
    print("🕉️  Sanskrit Phoneme Recognition System")
    print("=" * 60)

    if not extract_dataset():
        return

    print("\n📂 Loading pre-split dataset...")
    (train_files, train_labels), (val_files, val_labels), (test_files, test_labels) = load_pre_split_dataset()

    if len(train_files) == 0:
        print("❌ No training data found! Please check your dataset path.")
        ### MODIFIED ### - Corrected the misleading error message
        print(f"Expected structure: {LOCAL_DATASET_PATH}/train/[phoneme]/audio_files.wav")
        return
    
    # Load augmentation log for reference
    aug_log = load_augmentation_log()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Training: {len(train_files)} files")
    print(f"   Validation: {len(val_files)} files")
    print(f"   Testing: {len(test_files)} files")
    print(f"   Total phonemes: {len(PHONEMES)}")
    
    # Check class distribution
    print(f"\n📈 Training set class distribution:")
    train_counts = np.bincount(train_labels)
    for i, count in enumerate(train_counts):
        if count > 0:
            print(f"   {PHONEMES[i]}: {count} samples ({count/len(train_files)*100:.1f}%)")
    
    # Step 2: Create datasets and dataloaders
    # Determine max length from training samples
    print(f"\n🔍 Analyzing audio lengths...")
    sample_lengths = []
    for i in range(0, min(50, len(train_files)), 10):  # Sample every 10th file from first 50
        try:
            y, _ = librosa.load(train_files[i], sr=SR)
            mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=512, hop_length=HOP_SIZE, n_mels=N_MELS)
            sample_lengths.append(mel.shape[1])
        except:
            continue
    
    if sample_lengths:
        max_length = int(np.percentile(sample_lengths, 95))  # Use 95th percentile
        print(f"📏 Using max length: {max_length} frames (95th percentile)")
    else:
        max_length = 200  # Default fallback
        print(f"⚠️  Using default max length: {max_length} frames")
    
    # Create datasets with appropriate augmentation
    train_dataset = PhonemeDataset(train_files, train_labels, augment=True, max_length=max_length)
    val_dataset = PhonemeDataset(val_files, val_labels, augment=False, max_length=max_length)
    test_dataset = PhonemeDataset(test_files, test_labels, augment=False, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Step 3: Initialize model, criterion, optimizer
    model = ImprovedCNN(len(PHONEMES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    print(f"\n🏗️  Model Architecture:")
    print(f"   Input shape: (batch, 1, {N_MELS}, {max_length})")
    print(f"   Output classes: {len(PHONEMES)}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Device: {DEVICE}")
    
    # Step 4: Train the model
    print(f"\n🚀 Starting training for {EPOCHS} epochs...")
    trained_model = train_model(train_loader, val_loader, model, criterion, optimizer, scheduler, EPOCHS)
    
    # Step 5: Load best model and run inference
    print(f"\n🔍 Running final evaluation on test set...")
    best_model = load_best_model()
    if best_model:
        all_preds, all_labels, all_probs = generate_inference_report(best_model, test_loader)
        
        # Final confusion matrix
        final_cm_path = os.path.join(RESULTS_PATH, "final_confusion_matrix.png")
        plot_confusion_matrix(all_labels, all_preds, PHONEMES, "Final Test", final_cm_path)
        
        # Plot training history
        history_df = pd.read_csv(os.path.join(RESULTS_PATH, "training_log.csv"))
        history_plot_path = os.path.join(RESULTS_PATH, "training_history.png")
        plot_training_history(history_df, history_plot_path)
        
        # Save model info
        model_info = {
            'phonemes': PHONEMES,
            'model_architecture': 'ImprovedCNN',
            'sample_rate': SR,
            'n_mels': N_MELS,
            'max_length': max_length,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'epochs_trained': EPOCHS,
            'train_samples': len(train_files),
            'val_samples': len(val_files),
            'test_samples': len(test_files)
        }
        
        with open(os.path.join(RESULTS_PATH, "model_info.txt"), "w") as f:
            f.write("Sanskrit Phoneme Recognition Model Information\n")
            f.write("=" * 50 + "\n\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✅ Training and evaluation completed!")
        print(f"📁 All results saved to: {RESULTS_PATH}")
        print(f"📝 Training log: training_log.csv")
        print(f"🎯 Best model: {MODELS_PATH}/best_model.pth")
        print(f"📊 Inference report: inference_report.txt")
        print(f"📈 Training plots: training_history.png")
        print(f"ℹ️  Model info: model_info.txt")
        
        return best_model
    else:
        print("❌ Could not load best model for final evaluation")

# Run the main function
if __name__ == "__main__":
    main()

# ================================
#  STEP 9: ADDITIONAL UTILITIES
# ================================

# Function to test on individual files from any split
def test_individual_file(file_path):
    """Test inference on a single file"""
    model = load_best_model()
    if model:
        pred_idx, confidence, probs = inference_single_file(model, file_path)
        if pred_idx is not None:
            predicted_phoneme = PHONEMES[pred_idx]
            print(f"🎵 File: {os.path.basename(file_path)}")
            print(f"🔮 Predicted: {predicted_phoneme} (confidence: {confidence:.3f})")
            print(f"📊 Top 3 predictions:")
            top_3_indices = np.argsort(probs)[-3:][::-1]
            for i, idx in enumerate(top_3_indices):
                print(f"   {i+1}. {PHONEMES[idx]}: {probs[idx]:.3f}")

def test_random_samples():
    """Test on random samples from each split"""
    # Check the LOCAL path now
    if not os.path.exists(LOCAL_DATASET_PATH):
        print("❌ Local dataset not found. Please run main() first.")
        return

    model = load_best_model() # This correctly loads from Drive
    if not model:
        return

    print("🎲 Testing random samples from each split:")
    print("=" * 50)

    for split in ['train', 'val', 'test']:
        split_path = os.path.join(LOCAL_DATASET_PATH, split) # Use local path
        if os.path.exists(split_path):
            print(f"\n📂 {split.upper()} SET:")
            available_phonemes = [p for p in PHONEMES if os.path.exists(os.path.join(split_path, p))]
            if not available_phonemes: continue
            sample_phonemes = np.random.choice(available_phonemes, size=min(3, len(available_phonemes)), replace=False)

            for phoneme in sample_phonemes:
                phoneme_path = os.path.join(split_path, phoneme)
                audio_files = glob.glob(os.path.join(phoneme_path, "*.wav"))
                if audio_files:
                    random_file = np.random.choice(audio_files)
                    print(f"\n🎯 Actual: {phoneme}")
                    test_individual_file(random_file) # This function is fine as is

# Example usage functions:
def quick_test_setup():
    """Quick setup for testing individual files"""
    print("🚀 Quick Test Setup")
    print("Use these functions after running main():")
    print("1. test_individual_file('/path/to/file.wav')")
    print("2. test_random_samples()")
    print("3. Test specific phoneme:")
    print("   phoneme_path = f'{DATASET_PATH}/test/अ'")
    print("   files = glob.glob(f'{phoneme_path}/*.wav')")
    print("   test_individual_file(files[0])")

# Example usage (uncomment after training):
# test_random_samples()
# test_individual_file(f"{DATASET_PATH}/test/अ/some_file.wav")