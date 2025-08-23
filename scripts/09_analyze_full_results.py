# scripts/09_analyze_full_results.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Configuration ---
RESULTS_FILE = "evaluation_results_full.csv"
OUTPUT_PLOT_FILE = "full_recognition_confusion_matrix.png"

# Use a font that supports Devanagari characters for the plot
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans Devanagari'] # Make sure you have this font installed

def analyze_full_results(filepath: str):
    """
    Loads the full recognition results and generates a detailed report and confusion matrix.
    """
    print(f"🔎 Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{filepath}' was not found. Please make sure it exists.")
        return

    # Extract the actual and predicted labels
    y_true = df['actual_phoneme']
    y_pred = df['predicted_phoneme']

    # Get the unique list of phonemes in the order they appear
    labels = sorted(y_true.unique())

    print("\n--- 1. Overall Accuracy ---")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"🎯 Overall Accuracy: {accuracy:.2%}")

    print("\n--- 2. Detailed Classification Report ---")
    # This report gives you precision, recall, and f1-score for each phoneme
    print("Precision: How many selected items are relevant?")
    print("Recall: How many relevant items are selected?")
    print("F1-Score: A balance between Precision and Recall.")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0)) # type: ignore

    print("\n--- 3. Confusion Matrix Analysis ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Create a DataFrame for better visualization with Seaborn
    cm_df = pd.DataFrame(cm, index=pd.Index(labels), columns=pd.Index(labels))

    # Plotting the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')

    plt.title('Confusion Matrix', fontsize=20)
    plt.ylabel('Actual Phoneme', fontsize=14)
    plt.xlabel('Predicted Phoneme', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"\n✅ Visual analysis complete. Confusion matrix saved to '{OUTPUT_PLOT_FILE}'.")
    print("Read the matrix: Rows are what it ACTUALLY was, Columns are what the model PREDICTED.")
    print("High numbers on the diagonal are good. Off-diagonal numbers show specific errors.")

if __name__ == "__main__":
    analyze_full_results(RESULTS_FILE)
