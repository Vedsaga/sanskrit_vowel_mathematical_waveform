#!/bin/bash

# run_analyses.sh - Unified runner for formant analysis scripts
# Usage: ./run_analyses.sh <script_path>

if [ -z "$1" ]; then
    echo "Usage: $0 <script_path>"
    exit 1
fi

SCRIPT_PATH=$1
GOLDEN_A="data/02_cleaned/अ/अ_golden_043.wav"
GOLDEN_I="data/02_cleaned/इ/इ_golden_036.wav"
FOLDER_A="data/02_cleaned/अ"
DATA_ROOT="data/02_cleaned"

echo "================================================================================"
echo "Running analyses for: $SCRIPT_PATH"
echo "================================================================================"

# 1. Single-file Compare (Golden 'a' vs 'i')
echo "[1/3] Mode: Single-file Compare (Golden 'a' vs 'i')"
python "$SCRIPT_PATH" --file1 "$GOLDEN_A" --file2 "$GOLDEN_I"
echo "--------------------------------------------------------------------------------"

# 2. Batch Mode (Folder 'अ')
echo "[2/3] Mode: Batch (Folder 'अ')"
python "$SCRIPT_PATH" --folder "$FOLDER_A" --reference "$GOLDEN_A"
echo "--------------------------------------------------------------------------------"

# 3. Golden Mode
echo "[3/3] Mode: Golden (All golden files)"
python "$SCRIPT_PATH" --golden-compare "$DATA_ROOT"
echo "================================================================================"
echo "Done!"
