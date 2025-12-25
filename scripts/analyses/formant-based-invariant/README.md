# Formant Analysis Scripts

Quick reference for running formant analysis scripts.

## Output Structure

By default, results are saved to:
```
results/{script_name}/{mode}/
```

| Mode | Default Path |
|------|--------------|
| Golden | `results/{script}/golden/` |
| Batch | `results/{script}/batch/{phoneme}/` |
| Compare | `results/{script}/compare/{file1}_vs_{file2}/` |

Use `--output_dir path` to override.

---

## 1. formant_ratio_analysis.py

Analyzes scale-invariant frequency ratios (F1/F2, F2/F3).

```bash
# Two files (→ .../compare/file1_vs_file2/)
python formant_ratio_analysis.py --file1 f1.wav --file2 f2.wav

# Batch (→ .../batch/अ/)
python formant_ratio_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

# Golden (→ .../golden/)
python formant_ratio_analysis.py --golden-compare data/02_cleaned
```

---

## 2. formant_spacing_analysis.py

Analyzes normalized formant spacing (ΔF / GeomMean).

```bash
# Two files
python formant_spacing_analysis.py --file1 f1.wav --file2 f2.wav

# Batch
python formant_spacing_analysis.py --folder data/02_cleaned/अ --reference data/02_cleaned/अ/अ_golden_043.wav

# Golden
python formant_spacing_analysis.py --golden-compare data/02_cleaned
```

---

## 3. measure_gunas.py

Analyzes Gunas metrics (Sattva, Rajas, Tamas).

```bash
# Single file (→ .../single/filename/)
python measure_gunas.py --file data/02_cleaned/अ/अ_golden_043.wav

# Batch (→ .../batch/अ/)
python measure_gunas.py --folder data/02_cleaned/अ

# Golden (→ .../golden/)
python measure_gunas.py --golden-compare data/02_cleaned
```

---

## Output Files

Each analysis generates:
- `*.csv` - Detailed metrics data
- `*.png` - Visualization plots
