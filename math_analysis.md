# Mathematical Analysis of Sanskrit Vowel Waveforms

This document details the mathematical algorithms, metrics, and hypotheses implementation for the 9 analysis scripts used in the Sanskrit Vowel project.

## 1. Shared Mathematical Frameworks

All analysis scripts share a common foundation for signal processing and statistical weighting to ensure robustness against noise and transient states.

### 1.1 Joint Stability-Intensity Weighting (Method 3)

To prioritize stable, high-energy vowel segments and downweight transitions or silence, a joint weighting scheme is applied to every frame $t$.

The total weight $W_t$ for a frame is defined as:

$$ W_t = W_{\text{intensity}, t} \cdot W_{\text{stability}, t} \cdot \mathbb{I}_{\text{gate}, t} $$

Where:

1.  **Intensity Weight ($W_{\text{intensity}}$)**:
    Prioritizes louder segments (vowels) over quieter ones (consonants/background).
    $$ W_{\text{intensity}} = \min( (\max(0, I_t - I_{\text{floor}}))^{2}, I_{\text{max\_clip}} ) $$
    *   $I_t$: Intensity in dB at time $t$.
    *   $I_{\text{floor}}$: Noise floor threshold (default: 50 dB).
    *   Exponent: 2.0 (quadratic weighting).

2.  **Stability Weight ($W_{\text{stability}}$)**:
    Prioritizes steady-state regions where formants are constant (derivatives are zero).
    $$ W_{\text{stability}} = \frac{1}{\text{Instability}_t + \epsilon} $$
    *   $\text{Instability}_t = \sum_{i=1}^{3} \frac{|dF_i/dt|_t}{F_{i,t}}$ (Frequency-normalized sum of gradients).
    *   $\epsilon$: Smoothing factor (default: 0.1).

3.  **Soft Gate ($\mathbb{I}_{\text{gate}}$)**:
    Binary mask to hard-reject extremely low-energy frames.
    $$ \mathbb{I}_{\text{gate}} = \begin{cases} 1 & \text{if } I_t \ge 30\text{dB} \\ 0 & \text{otherwise} \end{cases} $$

### 1.2 Weighted Statistics
All aggregated metrics (means, standard deviations, variances) use these weights. For a sequence of values $x$ with weights $w$:

$$ \mu_w = \frac{\sum w_i x_i}{\sum w_i} $$
$$ \sigma_w^2 = \frac{\sum w_i (x_i - \mu_w)^2}{\sum w_i} $$

---

## 2. Invariant Hypotheses (Static/Structural)

These analyses test for properties that remain constant across speakers, scales, or pitches.

### 2.1 Formant Ratio Analysis
**Script**: `formant_ratio_analysis.py`
**Hypothesis**: Vowel identity is defined by the relative ratios of formants, not absolute frequencies (Scale Invariance).

*   **Linear Ratios**:
    $$ R_{12} = \frac{F1}{F2}, \quad R_{23} = \frac{F2}{F3} $$
*   **Logarithmic Ratios**:
    $$ R_{\log} = \ln(F1) - \ln(F2) = \ln\left(\frac{F1}{F2}\right) $$
    (Equivalent to differences in musical intervals/octaves).

### 2.2 Formant Spacing Analysis
**Script**: `formant_spacing_analysis.py`
**Hypothesis**: The spacing between formants holds invariant properties.

*   **Raw Spacing**:
    $$ \Delta F_{21} = F2 - F1, \quad \Delta F_{32} = F3 - F2 $$
*   **Normalized Spacing (Scale Invariant)**:
    Normalized by the geometric mean of the first three formants to account for vocal tract length differences.
    $$ \bar{F}_g = \sqrt[3]{F1 \cdot F2 \cdot F3} $$
    $$ \Delta F_{norm} = \frac{\Delta F}{\bar{F}_g} $$

### 2.3 Formant Amplitude Ratio Analysis
**Script**: `formant_amplitude_ratio_analysis.py`
**Hypothesis**: Relative energy distribution between formants is invariant.

*   **Amplitude Estimation**:
    Amplitudes $A_i$ are estimated from formant bandwidths $B_i$ using the inverse relationship (assuming constant pole-zero gain for simplified modeling):
    $$ A_i \propto \frac{1}{B_i} $$
*   **Amplitude Ratios**:
    $$ R_{A12} = \frac{A1}{A2} $$

### 2.4 Spectral Tilt Analysis
**Script**: `spectral_tilt_analysis.py`
**Hypothesis**: The slope of the spectral envelope (spectral tilt) characterizes openness/effort.

Calculated as the slope (dB/octave) between two formant peaks.
$$ \text{Slope}_{ij} = \frac{A_j(\text{dB}) - A_i(\text{dB})}{\log_2(F_j / F_i)} $$
where $A(\text{dB}) = 20 \log_{10}(A_{\text{linear}})$.

### 2.5 Formant Dispersion Analysis
**Script**: `formant_dispersion_analysis.py`
**Hypothesis**: Vowels minimize or maximize the dispersion (spread) of formants in frequency space.

*   **Average Formant**: $\bar{F} = \frac{F1+F2+F3}{3}$
*   **Dispersion (Standard Deviation)**:
    $$ \sigma_F = \sqrt{\frac{1}{3} \sum_{i=1}^3 (F_i - \bar{F})^2} $$

### 2.6 Gunas Metrics (Complexity Analysis)
**Script**: `measure_gunas.py`
**Hypothesis**: Audio textures map to the three Gunas (Sattva-Balance, Rajas-Activity, Tamas-Inertia) via chaos theory metrics.

*   **Sattva (Complexity/Balance)**:
    Calculated using **Fractal Dimension** (specifically Correlation Dimension $D_2$).
    Measures the self-similarity of the signal trace.
*   **Rajas (Activity/Randomness)**:
    Calculated using **Permutation Entropy**.
    Measures the complexity of the ordering of values in the time series.
    $$ H(n) = -\sum p(\pi) \log_2 p(\pi) $$
*   **Tamas (Stability/Inertia)**:
    Calculated using **Lyapunov Exponent** ($\lambda$).
    Measures the rate of divergence of close trajectories in phase space.
    Lower $\lambda$ (or negative) implies stability (Tamas).

---

## 3. Temporal Hypotheses (Dynamic)

These analyses focus on how the sound evolves over time.

### 3.1 Formant Convergence Analysis
**Script**: `formant_convergence_analysis.py`
**Hypothesis**: /a/ is convergent (F1 and F2 move closer), /i/ is divergent.

*   **Metric**: Rate of change of the distance between F1 and F2.
    $$ D(t) = |F2(t) - F1(t)| $$
*   **Convergence Rate**:
    Slope $\beta$ of the weighted linear regression of $D(t)$ over time $t$.
    $$ D(t) = \beta t + c $$
    *   $\beta < -50 \text{ Hz/s}$: **Convergent**
    *   $\beta > 50 \text{ Hz/s}$: **Divergent**
    *   Otherwise: **Stable**

### 3.2 Formant Trajectory Analysis
**Script**: `formant_trajectory_analysis.py`
**Hypothesis**: Vowels have characteristic settling patterns and curvature in the F1-F2 plane.

*   **Velocity**: Vector magnitude of change in formant space.
    $$ v(t) = \sqrt{\left(\frac{dF1}{dt}\right)^2 + \left(\frac{dF2}{dt}\right)^2} $$
*   **Curvature ($\kappa$)**:
    Measures how sharply the trajectory turns in the F1-F2 plane.
    $$ \kappa(t) = \frac{|F1' F2'' - F2' F1''|}{(F1'^2 + F2'^2)^{3/2}} $$
*   **Smoothness**: Inverse of the weighted mean jerk ($d^3F/dt^3$).

### 3.3 Steady-State Stability Analysis
**Script**: `steady_state_stability_analysis.py`
**Hypothesis**: The "true" vowel sound is found in the region of maximum stability (minimum variance).

*   **Sliding Window Analysis**: The script analyzes multiple time windows (0-20%, 20-40%, ..., 0-100%).
*   **Coefficient of Variation (CV)**:
    $$ CV_i = \frac{\sigma_{F_i}}{\mu_{F_i}} \times 100\% $$
*   **Combined Stability Metric**:
    $$ CV_{\text{combined}} = \frac{CV_{F1} + CV_{F2} + CV_{F3}}{3} $$
    The window with the lowest $CV_{\text{combined}}$ is best representative of the steady-state vowel.
