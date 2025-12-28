#!/bin/bash

# Configuration
# =================================================================================================
# Standard test files for verification
# Using 043 (Akara) and 036 (Ikara) as standard golden files based on project conventions
FILE1="data/02_cleaned/अ/अ_golden_043.wav"
FILE2="data/02_cleaned/इ/इ_golden_036.wav"
FOLDER="data/02_cleaned/अ"
REFERENCE="data/02_cleaned/अ/अ_golden_043.wav"
GOLDEN_DIR="data/02_cleaned"

# List of 8 Analysis Scripts
# =================================================================================================
SCRIPTS=(
    # Formant-based Invariant Hypotheses
    "scripts/analyses/formant-based-invariant/formant_ratio_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_spacing_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_amplitude_ratio_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_dispersion_analysis.py"
    "scripts/analyses/formant-based-invariant/spectral_tilt_analysis.py"

    # Temporal Hypotheses
    "scripts/analyses/temporal-hypotheses/steady_state_stability_analysis.py"
    "scripts/analyses/temporal-hypotheses/formant_trajectory_analysis.py"
    "scripts/analyses/temporal-hypotheses/formant_convergence_analysis.py"
)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}   SANSKRIT VOWEL ANALYSIS RUNNER   ${NC}"
echo -e "${BLUE}=================================================================${NC}"
echo -e "Start time: $(date)"
echo -e "Scripts to run: ${#SCRIPTS[@]}"
echo

# Check if data exists
if [ ! -d "$GOLDEN_DIR" ]; then
    echo -e "${RED}Error: Data directory $GOLDEN_DIR not found.${NC}"
    echo "Please ensure data is cleaned and available."
    exit 1
fi

if [ ! -f "$FILE1" ]; then
    echo -e "${YELLOW}Warning: Test file $FILE1 not found. Single comparison mode might fail.${NC}"
fi

# Main Loop
# =================================================================================================
for SCRIPT in "${SCRIPTS[@]}"; do
    SCRIPT_NAME=$(basename "$SCRIPT")
    echo -e "${YELLOW}-----------------------------------------------------------------${NC}"
    echo -e "${GREEN}Running: $SCRIPT_NAME${NC}"
    echo -e "${YELLOW}-----------------------------------------------------------------${NC}"

    # 1. Single Comparison Mode
    echo -e "${BLUE}[Mode 1] Single File Comparison${NC}"
    if [ -f "$FILE1" ] && [ -f "$FILE2" ]; then
        python3 "$SCRIPT" --file1 "$FILE1" --file2 "$FILE2"
    else
        echo "Skipping Single Mode (Files not found)"
    fi
    echo

    # 2. Batch Mode
    echo -e "${BLUE}[Mode 2] Batch Folder Analysis${NC}"
    if [ -d "$FOLDER" ] && [ -f "$REFERENCE" ]; then
        python3 "$SCRIPT" --folder "$FOLDER" --reference "$REFERENCE"
    else
        echo "Skipping Batch Mode (Folder or Reference not found)"
    fi
    echo

    # 3. Golden Mode
    echo -e "${BLUE}[Mode 3] Golden Files Analysis${NC}"
    if [ -d "$GOLDEN_DIR" ]; then
        python3 "$SCRIPT" --golden-compare "$GOLDEN_DIR"
    else
        echo "Skipping Golden Mode (Directory not found)"
    fi
    echo 
done

echo -e "${BLUE}=================================================================${NC}"
echo -e "${GREEN}All analyses completed.${NC}"
echo -e "${BLUE}=================================================================${NC}"
