#!/bin/bash

SAMPLE_WAV="data/02_cleaned/ऐ/ऐ_golden_031.wav"
OUTPUT_DIR="verification_results"
mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "VERIFICATION: Joint Stability-Intensity Weighting"
echo "Sample File: $SAMPLE_WAV"
echo "=============================================="

pass_count=0
fail_count=0

function check_status() {
    local ret_code=$1
    local log_file=$2
    local expected=$3
    local test_name=$4

    if [ $ret_code -eq 0 ]; then
        if grep -q "$expected" "$log_file"; then
            echo "  [PASS] $test_name"
            pass_count=$((pass_count+1))
        else
            echo "  [FAIL] $test_name (Missing output: '$expected')"
            echo "  See: $log_file"
            fail_count=$((fail_count+1))
        fi
    else
        echo "  [FAIL] $test_name (Exit Code $ret_code)"
        echo "  See: $log_file"
        head -n 20 "$log_file" | tail -n 5
        fail_count=$((fail_count+1))
    fi
}

# 1. TEMPORAL HYPOTHESES =======================================================

echo ""
echo "Testing formant_convergence..."
log="$OUTPUT_DIR/formant_convergence.log"
python3 "scripts/analyses/temporal-hypotheses/formant_convergence_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_convergence"

echo "Testing formant_trajectory..."
log="$OUTPUT_DIR/formant_trajectory.log"
python3 "scripts/analyses/temporal-hypotheses/formant_trajectory_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_trajectory"

echo "Testing steady_state_stability..."
log="$OUTPUT_DIR/steady_state_stability.log"
python3 "scripts/analyses/temporal-hypotheses/steady_state_stability_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "steady_state_stability"

# 2. FORMANT BASED INVARIANT ===================================================

echo "Testing formant_ratio..."
log="$OUTPUT_DIR/formant_ratio.log"
python3 "scripts/analyses/formant-based-invariant/formant_ratio_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_ratio"

echo "Testing formant_spacing..."
log="$OUTPUT_DIR/formant_spacing.log"
python3 "scripts/analyses/formant-based-invariant/formant_spacing_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_spacing"

echo "Testing measure_gunas..."
log="$OUTPUT_DIR/measure_gunas.log"
python3 "scripts/analyses/formant-based-invariant/measure_gunas.py" --file "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Sattva" "measure_gunas"

echo "Testing formant_dispersion..."
log="$OUTPUT_DIR/formant_dispersion.log"
python3 "scripts/analyses/formant-based-invariant/formant_dispersion_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_dispersion"

echo "Testing formant_amplitude..."
log="$OUTPUT_DIR/formant_amplitude.log"
python3 "scripts/analyses/formant-based-invariant/formant_amplitude_ratio_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "formant_amplitude"

echo "Testing spectral_tilt..."
log="$OUTPUT_DIR/spectral_tilt.log"
python3 "scripts/analyses/formant-based-invariant/spectral_tilt_analysis.py" --file1 "$SAMPLE_WAV" --file2 "$SAMPLE_WAV" > "$log" 2>&1
check_status $? "$log" "Effective Frames (N_eff)" "spectral_tilt"


echo ""
echo "=============================================="
echo "SUMMARY: Passed $pass_count / $((pass_count+fail_count))"
echo "=============================================="

if [ $fail_count -eq 0 ]; then
    exit 0
else
    exit 1
fi
