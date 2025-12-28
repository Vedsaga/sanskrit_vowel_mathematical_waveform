#!/bin/bash
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPTS=(
    "scripts/analyses/formant-based-invariant/formant_ratio_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_spacing_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_amplitude_ratio_analysis.py"
    "scripts/analyses/formant-based-invariant/formant_dispersion_analysis.py"
    "scripts/analyses/formant-based-invariant/spectral_tilt_analysis.py"
    "scripts/analyses/formant-based-invariant/measure_gunas.py"
    "scripts/analyses/temporal-hypotheses/steady_state_stability_analysis.py"
    "scripts/analyses/temporal-hypotheses/formant_trajectory_analysis.py"
    "scripts/analyses/temporal-hypotheses/formant_convergence_analysis.py"
)

PHONEME_CONFIGS=(
    "इ|data/02_cleaned/इ/इ_golden_036.wav"
    "उ|data/02_cleaned/उ/उ_golden_034.wav"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    SCRIPT_NAME=$(basename "$SCRIPT")
    echo -e "${YELLOW}-----------------------------------------------------------------${NC}"
    echo -e "${GREEN}Running Batch (i, u): $SCRIPT_NAME${NC}"
    echo -e "${YELLOW}-----------------------------------------------------------------${NC}"

    for CONFIG in "${PHONEME_CONFIGS[@]}"; do
        IFS='|' read -r PHONEME REF_FILE <<< "$CONFIG"
        TARGET_FOLDER="data/02_cleaned/$PHONEME"
        
        echo -e "${GREEN}  > Analyzing: $PHONEME${NC}"
        if [ -d "$TARGET_FOLDER" ] && [ -f "$REF_FILE" ]; then
            # Use venv python explicitly to avoid ModuleNotFoundError
            ./venv/bin/python "$SCRIPT" --folder "$TARGET_FOLDER" --reference "$REF_FILE"
        else
            echo "Skipping $PHONEME (Folder or Ref not found)"
        fi
    done
    echo
done
