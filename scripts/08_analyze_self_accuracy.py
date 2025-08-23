# scripts/08_analyze_self_accuracy.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
RESULTS_FILE = "evaluation_results_self_accuracy.csv"
OUTPUT_PLOT_FILE = "self_accuracy_boxplot.png"

# ADD THESE LINES to set the font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Noto Sans Devanagari']

def analyze_scores(filepath: str):
    """
    Loads the self-accuracy results and performs statistical and visual analysis.
    """
    print(f"🔎 Loading data from '{filepath}'...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{filepath}' was not found. Please make sure it exists.")
        return

    print("\n--- 1. Statistical Summary ---")
    # Group by each phoneme and get descriptive statistics for the scores
    # We focus on the mean, std, min, and max scores
    summary = df.groupby('target_phoneme')['self_match_score'].describe()
    print("Key metrics to watch: 'mean' (higher is better) and 'std' (lower is better).")
    print(summary[['mean', 'std', 'min', 'max']])

    print("\n--- 2. Visual Analysis ---")

    # Set the style and figure size for the plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 9))

    # Create the box plot
    sns.boxplot(data=df, x='target_phoneme', y='self_match_score', palette="viridis")

    # Improve plot readability
    plt.title('Self-Match Score Distribution per Phoneme', fontsize=20)
    plt.xlabel('Phoneme', fontsize=14)
    plt.ylabel('Log-Likelihood Score (Higher is Better)', fontsize=14)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better fit
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    # Save the plot to a file
    plt.savefig(OUTPUT_PLOT_FILE)
    print(f"\n✅ Visual analysis complete. Box plot saved to '{OUTPUT_PLOT_FILE}'.")
    print("Look for patterns with high scores (boxes are high up) and low variance (boxes are short).")

if __name__ == "__main__":
    analyze_scores(RESULTS_FILE)
