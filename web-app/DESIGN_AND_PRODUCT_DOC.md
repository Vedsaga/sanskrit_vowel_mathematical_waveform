# Vak – Guna-Oriented Audio Analysis Platform

## UI/UX Design, Feature, and Product Document

**Version:** 2.0
**Date:** 2025-12-20
**Purpose:** Bridge the philosophical/mathematical requirements to a concrete, implementable, and user-friendly interface design.

---

## Part 1: Executive Summary & Guiding Philosophy

### 1.1. The Core Problem with the Current UI

The existing "Vak" web application was built as a **shape generator**, not an **observational analysis instrument**. This fundamental mismatch manifests as:

| Current State Problem | Root Cause |
| :--- | :--- |
| **Clutter & Information Overload** | UI shows raw data (frequency lists) instead of synthesized insights. |
| **No Individual Control** | Cannot select a single frequency and apply rotation, loop, transparency, or delete. |
| **No Relational Analysis** | Cannot identify harmonics, primes, odd/even, or golden ratio relationships. |
| **Memory-less Interaction** | No way to save an "observation state" and compare it to another. |
| **Visualization is an Afterthought** | Canvas is boxed into a corner; data lists dominate the screen. |

### 1.2. The Redesign Goal: An Observatory, Not a Generator

The new interface must embody the principle:
> **"Vak is an observatory for discovering Gunas—stable, invariant, relational qualities of sound."**

This means:
1.  **Visualization is the Hero**: The geometry canvas is the primary workspace. Everything else supports it.
2.  **Controls Expose Analysis, Not Just Data**: Instead of showing a list of 20 frequencies, the UI should show "5 Harmonic Groups" or "3 Persistent Components".
3.  **State is Persistent**: Users can save, recall, and overlay observation states to discover convergence.
4.  **Relationships are First-Class**: Harmonic series, prime factors, and golden ratios are surfaced, not hidden.

---

## Part 2: Current State Analysis (Screenshots Review)

### 2.1. Home Page
- **Assessment**: Acceptable. Clean introduction to features.
- **Action**: Minor polish. No structural changes needed.

### 2.2. Visualizer Page (`/visualizer`)
- **Problem: Fragmented Controls**: "Add Shape", "Shape List", "Rotation" are three separate, vertically stacked cards.
- **Problem: No Selection on Canvas**: Cannot click a shape on the canvas to select it.
- **Problem: Global-Only Controls**: Rotation and Amplitude apply globally, not per-shape.
- **Action**: Major refactor. Introduce per-shape controls and a unified control panel.

### 2.3. Audio Analysis Page (`/audio-analysis`)
- **Problem: Wall of Frequencies**: A long, scrollable, checkbox-based list of "Frequency Components" dominates the left side. This is raw data, not insight.
- **Problem: No Grouping**: Frequencies are listed individually. There's no indication of which are harmonically related.
- **Problem: No Temporal Analysis**: No way to select a time window, slide through the audio, or observe stability.
- **Action**: Complete redesign. Replace list with a spectral graph. Add temporal controls.

### 2.4. Comparison Page (`/comparison`)
- **Problem: Duplicate UI**: Two massive, identical frequency lists side-by-side. This is overwhelming.
- **Problem: Visualization is Secondary**: The two canvases are small or pushed off-screen by the lists.
- **Problem: No Convergence Tools**: No way to lock scale, overlay saved states, or compare structures.
- **Action**: Complete redesign. Canvases are primary. Lists are collapsible.

---

## Part 3: New Information Architecture (IA)

The application will be restructured around the *type of observation* the user is performing.

### 3.1. Proposed Navigation

| Current | New | Purpose |
| :--- | :--- | :--- |
| Home | **Home** | Introduction, philosophy, quick links. |
| Visualizer | **Compose Lab** | *Manual* shape generation from typed frequencies. For learning the geometry system. |
| Audio Analysis | **Analysis Observatory** | *Single audio* analysis. Time windowing, frequency grouping, stability analysis. |
| Comparison | **Convergence Studio** | *Dual audio* comparison. Overlay, intersection, divergence analysis. |

### 3.2. New Global Features

- **State Cards**: A persistent drawer/sidebar to save and recall observation states.
- **Harmonic Palette**: A UI element that, given a fundamental frequency, shows its harmonic series, prime factors, and golden-ratio related frequencies.

---

## Part 4: Feature-by-Feature UI/UX Specification

This section maps the mathematical requirements to UI elements.

---

### 4.1. Individual Frequency/Shape Control (Critical Gap)

**Requirement**: "Select one or more and apply rotation, speed, loop, transparency, amplitude, delete it."

**Current State**: The `ShapeList.svelte` component shows shapes but only allows global selection and a single color picker/opacity slider per row. Rotation is a global control.

**Proposed UI**:

1.  **Canvas Selection**: Clicking on a shape directly selects it. `Shift+Click` for multi-select. Selected shapes get a visible highlight (e.g., a bright ring around them).
2.  **Contextual Popover**: When one or more shapes are selected, a small floating popover appears near the selection with controls:
    - Color picker
    - Opacity slider (`α`)
    - Rotation speed slider (per-selection)
    - Direction toggle (CW/CCW)
    - Loop mode toggle (Continuous / Bounce)
    - Delete button
3.  **Properties Panel (Alternative)**: A collapsible right-hand panel shows detailed properties when a shape is selected, allowing for more precise numeric input.

---

### 4.2. Harmonic & Number-Theoretic Analysis (New Feature)

**Requirement**: "Show frequencies which are prime, even, odd, or related by golden ratio."

**Proposed UI: "Frequency Analyzer" Panel**

When viewing a list of detected frequencies (e.g., from audio analysis), each frequency entry will have **badges** indicating its properties:

| Badge | Meaning | Visual |
| :--- | :--- | :--- |
| `H1`, `H2`, `H3`... | Harmonic order relative to a detected fundamental. | Blue badge. |
| `P` | The frequency value (when normalized) is a prime number. | Amber badge. |
| `E` | Even wiggle count. | Light grey badge. |
| `O` | Odd wiggle count. | Light grey badge. |
| `φ` | Frequency is approximately in golden ratio (1.618) with another detected frequency. | Gold badge. |

**Filtering**: A toggle bar allows users to filter the list/view:
`[All] [Harmonics Only] [Primes] [Golden Ratio (φ)]`

---

### 4.3. Time Window Control (New Feature)

**Requirement**: "All analysis operates only within selected window."
**Mathematical Basis**: `x_w(t) = x(t) * w(t - t_0)`

**Proposed UI: "Temporal Navigator"**

This is a horizontal bar placed *above* or *below* the main canvas.

```
┌────────────────────────────────────────────────────────────────────┐
│ [▶] ───[====WINDOW====]───────────────────────────────── 00:02.5s  │
│       ↑ (draggable handles)                                       │
└────────────────────────────────────────────────────────────────────┘
     [Start: 0.5s] [Width: 0.3s] [Step: 0.1s] [Mode: Hann | Rect]
```

- **Waveform Display**: The bar shows a miniature waveform of the entire audio.
- **Draggable Window**: A highlighted region (with draggable start/end handles) indicates the current analysis window.
- **Playhead**: When playing, a playhead moves across.
- **Controls**: Input fields for Start Time, Window Width, and Step Size (for sliding analysis).

---

### 4.4. Frequency Grouping (Core Guna Feature)

**Requirement**: "Frequencies belong to same group if `corr > γ`."

**Proposed UI: "Component Groups"**

Instead of a flat list of 20+ frequencies, the UI presents:

```
┌─────────────────────────────────────────────┐
│ Detected Components                   [⚙]  │
├─────────────────────────────────────────────┤
│ ▼ Group 1 (Fundamental + Harmonics)        │ <-- Collapsible
│   ◉ 210 Hz (f₀) ████████ 100%              │
│   ○ 420 Hz (H2) ████░░░░  52%              │
│   ○ 630 Hz (H3) ██░░░░░░  28%              │
├─────────────────────────────────────────────┤
│ ▼ Group 2 (Secondary Cluster)              │
│   ◉ 890 Hz     █████░░░  65%               │
│   ○ 1.1 kHz    ███░░░░░  38%               │
├─────────────────────────────────────────────┤
│ ▶ Outliers (Low Correlation)               │ <-- Collapsed by default
│   ...                                       │
└─────────────────────────────────────────────┘
```

- Each **Group** is a collapsible section.
- Selection is per-group or per-component.
- Selecting a group renders it as a single geometric layer on the canvas.

---

### 4.5. Analysis State Management (Memory Feature)

**Requirement**: "Save, Recall, Overlay, Compare."

**Proposed UI: "Observation Log" (Drawer)**

A slide-out drawer (triggered by a button in the header or sidebar) that holds "State Cards".

```
┌─────────────────────────────────────────────┐
│ Observation Log                       [+]  │
├─────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────┐ │
│ │ State 1: "Vowel 'आ' - Speaker A"       │ │
│ │ Audio: golden_043.wav                   │ │
│ │ Window: 0.3s - 0.6s                     │ │
│ │ Groups: 1, 2                            │ │
│ │ [Load] [Overlay] [Delete]               │ │
│ └─────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────┐ │
│ │ State 2: "Vowel 'आ' - Speaker B"       │ │
│ │ ...                                      │ │
│ └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

- **Save**: Creates a new card with the current analysis parameters and geometry snapshot.
- **Load**: Restores the canvas and controls to that state.
- **Overlay**: Adds the geometry of that state *on top of* the current view (for convergence analysis).

---

### 4.6. Stability & Invariance Indicators (New Feature)

**Requirement**: "High stability => vowel-like Guna."
**Mathematical Basis**: `Stability = 1 - (1/M) * Σ D(S_i, S_{i+1})`

**Proposed UI: "Guna Strength" Indicator**

When temporal analysis is active (sliding window mode), the UI displays:

```
┌───────────────────────────────────────────┐
│ Guna Strength                             │
│ ████████████████░░░░ 82% (Stable)         │
│ Energy Invariant: ✓ Yes                   │
│ Transient Score: 15% (Carrier Dominant)   │
└───────────────────────────────────────────┘
```

This provides an at-a-glance summary of whether the current selection exhibits Guna-like properties.

---

## Part 5: Page-by-Page Wireframe Descriptions

### 5.1. Compose Lab (formerly Visualizer)

**Layout**: Full-screen canvas. Floating glass panel on the right.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│                                                                     │
│                         ┌───────────────────┐                       │
│                         │ [FULL SCREEN      │                       │
│                         │  CANVAS WITH      │   ┌─────────────────┐ │
│                         │  SHAPES]          │   │ Control Panel   │ │
│                         │                   │   │ ─────────────── │ │
│                         │                   │   │ [Compose|Layers │ │
│                         │                   │   │  |Animate]      │ │
│                         │                   │   │ (Tabbed Content)│ │
│                         │                   │   └─────────────────┘ │
│                         └───────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

- **Control Panel Tabs**:
    - `Compose`: Frequency input, Amplitude slider, Add button.
    - `Layers`: List of added shapes with per-shape controls.
    - `Animate`: Global rotation presets, play/pause.

---

### 5.2. Analysis Observatory (formerly Audio Analysis)

**Layout**: Temporal Navigator at top. Split view: Spectrum/Groups on left, Canvas on right.

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Audio: file.wav ▼]  ───[====]─────────────────────────── 00:03.2s  │  <-- Temporal Navigator
├──────────────────────────┬──────────────────────────────────────────┤
│                          │                                          │
│  [Spectrum Graph]        │      [LARGE CANVAS]                      │
│  ~~~~~~~~~~~~~~~~        │                                          │
│         ^                │      (Geometry from selected groups)     │
│    (clickable peaks)     │                                          │
│                          │                                          │
├──────────────────────────┤                                          │
│  Component Groups        │      ┌────────────────────────────────┐  │
│  ──────────────────      │      │ Guna Strength: 78% (Stable)    │  │
│  ▼ Group 1 (Harmonics)   │      │ Energy Invariant: Yes          │  │
│    ◉ 210 Hz ...          │      └────────────────────────────────┘  │
│  ▼ Group 2               │                                          │
│    ...                   │                                          │
└──────────────────────────┴──────────────────────────────────────────┘
```

---

### 5.3. Convergence Studio (formerly Comparison)

**Layout**: Two canvases dominate. Central spine for shared controls. Collapsible data panels.

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Audio A: file_a.wav ▼]                   [Audio B: file_b.wav ▼]   │
├─────────────────────┬───────────────────────┬───────────────────────┤
│                     │                       │                       │
│   [CANVAS A]        │   ┌───────────────┐   │   [CANVAS B]          │
│                     │   │ Comparison    │   │                       │
│                     │   │ Mode: [Sync▼] │   │                       │
│                     │   │ [Overlay]     │   │                       │
│                     │   │ [Highlight ∩] │   │                       │
│                     │   │ [Highlight Δ] │   │                       │
│                     │   └───────────────┘   │                       │
│                     │                       │                       │
├─────────────────────┴───────────────────────┴───────────────────────┤
│ [▼ Show Frequency Details]  (Collapsible panel for raw data)        │
└─────────────────────────────────────────────────────────────────────┘
```

- The frequency lists are **hidden by default** behind a collapsible panel at the bottom.
- **Comparison Modes**:
    - `Overlay`: Draws A's geometry over B's (with different opacity/color).
    - `Highlight Intersection (∩)`: Only regions where A and B overlap are highlighted.
    - `Highlight Difference (Δ)`: Regions unique to A or B are highlighted.

---

## Part 6: Implementation Roadmap (Phased)

### Phase 1: Foundation (De-clutter & Core Refactors)
- [ ] Refactor layout to "Canvas-First" architecture.
- [ ] Merge `ShapeControls`, `ShapeList`, `RotationControls` into a single `ControlPanel.svelte` with tabs.
- [ ] Implement per-shape selection on canvas (click-to-select).
- [ ] Create `StateCard` component and `Observation Log` drawer.

### Phase 2: Advanced Analysis (New Features)
- [ ] Add Time Window Control (`TemporalNavigator.svelte`).
- [ ] Implement Frequency Grouping logic and UI (`ComponentGroups.svelte`).
- [ ] Calculate and display Harmonic/Prime/Golden Ratio badges.
- [ ] Implement Stability Score calculation and `GunaStrength` indicator.

### Phase 3: Convergence & Comparison (New Features)
- [ ] Redesign Comparison page layout.
- [ ] Implement Overlay, Intersection, and Difference rendering modes.
- [ ] Implement State loading and overlay for cross-state comparison.

### Phase 4: Polish & DX
- [ ] Micro-animations and transitions.
- [ ] Keyboard shortcuts (e.g., `Space` for rotate, `Delete` for remove).
- [ ] Onboarding walkthrough for new users.

---

## Part 7: Appendix: Component Mapping

| New Component | Purpose | Replaces |
| :--- | :--- | :--- |
| `ControlPanel.svelte` | Unified tabbed panel for Compose Lab. | `ShapeControls`, `ShapeList`, `RotationControls` |
| `TemporalNavigator.svelte` | Time window selection bar. | New |
| `ComponentGroups.svelte` | Collapsible, grouped frequency list. | Flat frequency checkbox list |
| `SpectrumGraph.svelte` | Visual frequency plot with clickable peaks. | New |
| `GunaStrengthIndicator.svelte` | Stability and invariance display. | New |
| `StateCard.svelte` | Saved observation state. | New |
| `ObservationLog.svelte` | Drawer containing all State Cards. | New |
| `ShapePopover.svelte` | Per-shape contextual controls on canvas. | New |

---

This document should now serve as the complete design specification for the UI/UX team to implement the Guna-Oriented Analysis Platform as envisioned.


# Vak – Guna-Oriented Audio Analysis Platform

## UI/UX Design, Feature, and Product Document

**Version:** 3.0
**Date:** 2025-12-20
**Purpose:** Comprehensive analysis of the existing implementation and a detailed specification for the new UI/UX, incorporating user-provided wireframe sketches and global/local control synchronization requirements.

---

# Part A: Existing Implementation Analysis

This section details what **already exists** in the codebase, identifying reusable components and gaps.

---

## A.1. Technology Stack & Architecture

| Layer | Technology |
|---|---|
| Framework | SvelteKit (Svelte 5 with Runes) |
| State Management | Svelte 5 `$state` rune (reactive stores) |
| Styling | Tailwind CSS v4 + CSS Custom Properties |
| UI Components | shadcn-svelte (Card, Button, Slider, Input, Checkbox, etc.) |
| Audio Processing | Web Audio API (`OfflineAudioContext`, `AnalyserNode`) |
| FFT | Custom DFT implementation in `fftProcessor.ts` |

---

## A.2. Core Modules (What's Built)

### A.2.1. `shapeEngine.ts` — Shape Generation
**Status: ✅ Complete and Correct**

```
r(θ) = R + A·sin((fq-1)·θ + φ)
```

| Function | Purpose |
|---|---|
| `generateShapePoints()` | Generates `Point[]` from `fq, R, A, phi, resolution`. |
| `validateShapeParams()` | Validates `fq ≥ 1`, `A < R`, `resolution ∈ [360, 2048]`. |
| `countWiggles()` | Detects local maxima in radius function (verification). |
| `validateFrequencyInput()` | Validates user input for frequency. |

**Gap**: No support for temporal stability analysis, energy normalization, or transient suppression (these are new requirements).

---

### A.2.2. `fftProcessor.ts` — Audio Analysis
**Status: ✅ Functional, but limited**

| Function | Purpose |
|---|---|
| `computeFFT()` | Async FFT using `OfflineAudioContext`. |
| `computeFFTSync()` | Sync DFT with Hanning window. Operates on **middle segment only**. |
| `mapFrequencyToFq()` | Linear or Logarithmic mapping from Hz to `fq`. |
| `extractTopFrequencies()` | Extracts top N peaks by magnitude. |

**Gaps**:
-   **No Time Windowing**: `computeFFTSync` uses a hardcoded middle segment. No user control over start time or window width.
-   **No Frequency Grouping**: Frequencies are returned as a flat list. No harmonic clustering or co-persistence analysis.
-   **No Harmonic/Prime/Golden Ratio Tags**: No metadata about the nature of the frequency.

---

### A.2.3. `shapeStore.svelte.ts` — Shape State
**Status: ✅ Functional, but limited**

| Feature | Status |
|---|---|
| Add/Remove Shapes | ✅ Works |
| Multi-Selection | ✅ Works (`selectedIds: Set<string>`) |
| Per-Shape Color | ✅ Works |
| Per-Shape Opacity | ✅ Works |
| Global Rotation (selected shapes) | ✅ Works (`updateSelectedShapesPhi`) |
| Global Amplitude (`A`) | ✅ Works (via `config.A`) |

**Gaps**:
-   **No Per-Shape Rotation Speed/Direction**: Rotation is applied globally to selected shapes. Cannot set different speeds for different shapes.
-   **No Loop Mode per Shape**: Loop mode is global (`rotation.mode`).

---

### A.2.4. `comparisonStore.svelte.ts` — Comparison State
**Status: ✅ Functional**

| Feature | Status |
|---|---|
| Dual Panel State (`leftPanel`, `rightPanel`) | ✅ Works |
| Sync Mode (`independent` / `synchronized`) | ✅ Works (toggle exists, but limited functionality) |
| Per-Panel Shape Management | ✅ Works |
| Shared Frequency Scale | ✅ Calculated |

**Gaps**:
-   **Sync Mode Underutilized**: When `syncMode === 'synchronized'`, nothing actually synchronizes globally. It's just a flag.
-   **No Global Control Binding**: There's no "Master Control Panel" that applies changes to **both** panels simultaneously.
-   **No Overlay/Intersection Rendering**: The canvases are independent; no visual overlay mode.

---

### A.2.5. Key UI Components

| Component | Purpose | Location |
|---|---|---|
| `ShapeCanvas.svelte` | Renders shapes on `<canvas>`. | `lib/components/` |
| `ShapeControls.svelte` | Frequency input + Amplitude slider. | `lib/components/` |
| `ShapeList.svelte` | Displays shapes with checkboxes, color picker, opacity, delete. | `lib/components/` |
| `RotationControls.svelte` | Direction, Mode, Speed controls. | `lib/components/` |
| `AudioUploader.svelte` | Drag-and-drop file upload. | `lib/components/` |
| `FFTDisplay.svelte` | Displays frequency components as a list with checkboxes. | `lib/components/` |
| `ComparisonPanel.svelte` | Combines uploader, FFT display, canvas, and shape list for one side. | `lib/components/` |
| `SyncControls.svelte` | Toggle for `Independent` vs `Synchronized` mode. | `lib/components/` |

---

## A.3. Current UI Issues (Summary from Screenshots)

| Issue | Page | Cause |
|---|---|---|
| Clutter / Information Overload | Audio Analysis, Comparison | `FFTDisplay` shows a raw list of 20 frequencies. |
| Canvas is Secondary | All | Rigid grid layout places canvas alongside, not as primary. |
| No Individual Shape Control | Visualizer, Audio Analysis | `RotationControls` applies globally. No contextual controls. |
| No Grouping | Audio Analysis | Frequencies are flat, not grouped by harmonic relationship. |
| Sync Mode Does Nothing Visible | Comparison | `syncMode` flag exists but doesn't bind controls. |

---

# Part B: New UI/UX Specification

This section defines the **target state** based on user requirements and wireframe sketches.

---

## B.1. The "Multi-Analysis Grid" Concept (from User Sketches)

Your wireframes introduce a powerful new paradigm:

> **A single audio file generates MULTIPLE analysis views simultaneously, displayed in a grid.**

Each "Analysis Tile" in the grid represents a **different analysis configuration** (e.g., different time window, different frequency range, different grouping).

### B.1.1. Wireframe Interpretation

**Sketch 1: Upload State**
```
┌──────────────────────────────────────────────────┐
│                                                  │
│           [ Upload audio (dropzone) ]            │
│                                                  │
└──────────────────────────────────────────────────┘
```
*Initial state before audio is loaded.*

**Sketch 2: Multi-Analysis Grid (Audio Loaded)**
```
┌──────────────────────────────────────────────────────────────────────┐
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│ │Analysis1│ │Analysis2│ │Analysis3│ │Analysis4│ │Analysis5│         │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐ │
│ │Analysis6│ │Analysis7│ │Analysis8│ │Analysis9│ │Analysi10│ │Ctrl │ │  <-- Right Panel: Shared (Global) Control
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │Panel│ │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │     │ │
│ │Analysi11│ │Analysi12│ │Analysi13│ │Analysi14│ │Analysi15│ │     │ │
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────┘ │
│ ┌───────────────────────────────────────────────────────────────────┤
│ │ Bottom Panel: Control panel (Global)                              │
│ └───────────────────────────────────────────────────────────────────┘
└──────────────────────────────────────────────────────────────────────┘
```
*Multiple analysis views in a grid. Control Panels on the right and bottom.*

**Sketch 3: Focused Analysis Mode**
```
┌──────────────────────────────────────────────────────────────────────┐
│ ┌─────────────────────────────────────────────────────────┐ ┌──────┐ │
│ │                                                         │ │ Ctrl │ │
│ │      [SELECTED ANALYSIS - FULL VIEW]                    │ │Panel │ │
│ │                                                         │ │ for  │ │
│ │      "Select Analysis" prompt or canvas                 │ │select│ │
│ │                                                         │ │  ed  │ │
│ └─────────────────────────────────────────────────────────┘ └──────┘ │
│ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────────────┐  │
│ │Analysis1│ │Analysis2│ │Analysis3│ │Analysi14│ │ scroll for more │  │  <-- Scrollable row of analysis tiles
│ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```
*When an analysis is selected, it expands. Others become a thumbnail row. Right panel shows LOCAL controls.*

**Sketch 4 & 5: Comparison Mode (Dual Audio)**
```
┌─────────────────────────────────────────────────────────────────────────┐
│       [ Upload audio-1 ]        │         [ Upload audio-2 ]            │
└─────────────────────────────────┴───────────────────────────────────────┘
```
*Initial: Side-by-side upload zones.*

```
┌─────────────────────────────────────────────────────────────────────────┐
│ ┌─────┐                                                        ┌─────┐ │
│ │Ctrl │  ┌─────┐ ┌─────┐    │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│Ctrl │ │
│ │Panel│  │ A-1 │ │ A-2 │    │    │ B-1 │ │ B-2 │ │ B-6 │ │ B-7 ││Panel│ │
│ │ (L) │  └─────┘ └─────┘    │    └─────┘ └─────┘ └─────┘ └─────┘│ (R) │ │
│ │     │  ┌─────┐ ┌─────┐    │    ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐│     │ │
│ │     │  │ A-6 │ │ A-7 │    │    │B-11 │ │B-12 │ │B-11 │ │B-12 ││     │ │
│ │     │  └─────┘ └─────┘    │    └─────┘ └─────┘ └─────┘ └─────┘│     │ │
│ │     │  ┌─────┐ ┌─────┐    │    ...                            │     │ │
│ │     │  │A-11 │ │A-12 │    │                                   │     │ │
│ └─────┘  └─────┘ └─────┘    │                                   └─────┘ │
└─────────────────────────────┴───────────────────────────────────────────┘
```
*Loaded: Each audio has its own grid of analyses. Left/Right control panels.*

---

## B.2. Global vs. Local Control Synchronization

This is a **critical architectural requirement**.

### B.2.1. The Principle

> **Global Panel**: Controls that, when changed, update **ALL** analysis tiles simultaneously.
> **Local Panel**: Controls that, when an analysis is selected, affect **ONLY** that analysis.

### B.2.2. Control Binding Matrix

| Control | Global Effect (when no selection) | Local Effect (when analysis selected) |
|---|---|---|
| **Time Window (Start, Width)** | Sets default for all new analyses. | Overrides for selected analysis only. |
| **Frequency Range (Min, Max)** | Filters all analyses. | Overrides for selected analysis only. |
| **Amplitude (A)** | Applies to all shapes. | Applies only to shapes in selected analysis. |
| **Rotation (Speed, Direction, Loop)** | Applies to all shapes. | Applies only to shapes in selected analysis. |
| **Opacity** | Sets default. | Per-shape override. |
| **Grouping Strategy** | Applies to all. | Per-analysis override. |
| **Normalize Energy Toggle** | Applies to all. | Per-analysis override. |
| **Suppress Transients Toggle** | Applies to all. | Per-analysis override. |

### B.2.3. State Architecture (Proposed)

```typescript
// Global Settings (single instance)
interface GlobalSettings {
  timeWindow: { start: number; width: number };
  frequencyRange: { min: number; max: number };
  amplitude: number;
  rotation: RotationState;
  normalize: boolean;
  suppressTransients: boolean;
}

// Analysis Instance (one per tile)
interface AnalysisState {
  id: string;
  // Overrides - if undefined, inherit from GlobalSettings
  timeWindow?: { start: number; width: number }; 
  frequencyRange?: { min: number; max: number };
  amplitude?: number;
  rotation?: RotationState;
  normalize?: boolean;
  suppressTransients?: boolean;
  // Computed state
  frequencyComponents: FrequencyComponent[];
  shapes: Shape[];
  stabilityScore?: number;
}

// Store Structure
interface AnalysisStore {
  audioBuffer: AudioBuffer | null;
  globalSettings: GlobalSettings;
  analyses: AnalysisState[]; // The grid of analysis tiles
  selectedAnalysisId: string | null; // For local control binding
}
```

When `selectedAnalysisId === null`:
-   Changes to controls update `globalSettings`.
-   All `analyses` that don't have local overrides re-compute.

When `selectedAnalysisId !== null`:
-   Changes to controls update `analyses[selectedIndex].localOverrides`.
-   Only the selected analysis re-computes.

---

## B.3. Individual Frequency/Shape Control (Gap Fill)

**Requirement**: "Select one or more and apply rotation, speed, loop, transparency, amplitude, delete it."

### B.3.1. Per-Shape Properties (Extend `Shape` Interface)

```typescript
interface Shape {
  id: string;
  fq: number;
  R: number;
  phi: number;
  color: string;
  opacity: number;
  strokeWidth: number;
  selected: boolean;
  // NEW: Per-shape animation overrides
  animationOverride?: {
    speed?: number;         // rad/s, if undefined, use global
    direction?: 'cw' | 'ccw';
    mode?: 'loop' | 'fixed' | 'none';
  };
}
```

### B.3.2. UI: Shape Popover

When a shape is clicked on the canvas, a popover appears with:
-   Color Picker
-   Opacity Slider
-   Rotation Speed Slider
-   Direction Toggle (CW/CCW/None)
-   Loop Toggle
-   Delete Button

---

## B.4. Harmonic & Number-Theoretic Badges (New Feature)

**Requirement**: "Show frequencies which are prime, even, odd, or related by golden ratio."

### B.4.1. Badge Definitions

| Badge | Condition | Symbol |
|---|---|---|
| Harmonic | `freq % fundamental === 0` where fundamental is the lowest detected freq in the group. | `H2`, `H3`, `H4`... |
| Prime | `fq` is a prime number. | `P` |
| Even | `fq % 2 === 0` | `E` |
| Odd | `fq % 2 !== 0` | `O` |
| Golden Ratio | `freq_a / freq_b ≈ 1.618` (within 2% tolerance). | `φ` |

### B.4.2. Implementation

Add a utility function:
```typescript
function analyzeFrequencyRelationships(components: FrequencyComponent[]): FrequencyComponent[] {
  // Add 'badges' property to each component
}
```

Extend `FrequencyComponent`:
```typescript
interface FrequencyComponent {
  // ... existing
  badges: ('P' | 'E' | 'O' | `H${number}` | 'φ')[];
}
```

---

## B.5. Feature Roadmap (Updated)

### Phase 0: State Refactor (Foundation)
- [ ] Create `GlobalSettings` type and store.
- [ ] Create `AnalysisState` type supporting local overrides.
- [ ] Refactor `shapeStore` to support per-shape animation overrides.
- [ ] Implement `analysisStore` with global/local binding logic.

### Phase 1: Analysis Observatory (Single Audio)
- [ ] Implement "Multi-Analysis Grid" layout.
- [ ] Add "Time Window" controls (`TemporalNavigator`).
- [ ] Implement Frequency Grouping (`ComponentGroups`).
- [ ] Add Harmonic/Prime/Golden badges.
- [ ] Implement `GunaStrengthIndicator` (Stability Score).

### Phase 2: Convergence Studio (Dual Audio)
- [ ] Implement dual-grid layout per user sketch.
- [ ] Add Overlay/Intersection/Difference rendering modes.
- [ ] Implement State Cards and Observation Log.

### Phase 3: Per-Shape Controls
- [ ] Implement `ShapePopover` for canvas selection.
- [ ] Add per-shape animation overrides.
- [ ] Implement click-to-select on canvas.

### Phase 4: Polish
- [ ] Transitions & Animations.
- [ ] Keyboard shortcuts.
- [ ] Onboarding walkthrough.

---

## B.6. Component Tree (New)

```
src/lib/
├── stores/
│   ├── analysisStore.svelte.ts      # NEW: Single-audio multi-analysis state
│   ├── comparisonStore.svelte.ts    # REFACTOR: Dual-grid state
│   ├── globalSettingsStore.svelte.ts # NEW: Global control settings
│   └── shapeStore.svelte.ts         # REFACTOR: Per-shape overrides
├── components/
│   ├── layout/
│   │   ├── AnalysisGrid.svelte      # NEW: Renders grid of AnalysisTile
│   │   └── AnalysisTile.svelte      # NEW: Single analysis canvas + mini-controls
│   ├── controls/
│   │   ├── GlobalControlPanel.svelte  # NEW: The "master" panel
│   │   ├── LocalControlPanel.svelte   # NEW: Context-aware panel
│   │   ├── TemporalNavigator.svelte   # NEW: Time window slider with waveform
│   │   └── ShapePopover.svelte        # NEW: Per-shape controls
│   ├── analysis/
│   │   ├── ComponentGroups.svelte     # NEW: Grouped frequency list
│   │   ├── FrequencyBadges.svelte     # NEW: Harmonic/Prime/Golden badges
│   │   └── GunaStrengthIndicator.svelte # NEW: Stability score display
│   ├── ShapeCanvas.svelte             # ENHANCE: Add click-to-select
│   └── ... (existing components)
└── routes/
    ├── +page.svelte                   # Home (minor changes)
    ├── compose-lab/+page.svelte       # RENAME from /visualizer
    ├── analysis-observatory/+page.svelte # RENAME from /audio-analysis
    └── convergence-studio/+page.svelte   # RENAME from /comparison
```

---

## B.7. Data Flow Diagram (Proposed)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Interaction                           │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
        ▼                                                       ▼
┌───────────────────┐                               ┌───────────────────┐
│ Global Panel      │                               │ Local Panel       │
│ (No Analysis Sel) │                               │ (Analysis Sel)    │
└────────┬──────────┘                               └────────┬──────────┘
         │                                                   │
         ▼                                                   ▼
┌───────────────────┐                               ┌───────────────────┐
│ globalSettingsStore│                               │ analysisStore     │
│   .setGlobal(...)  │                               │  .setLocal(id,...)|
└────────┬──────────┘                               └────────┬──────────┘
         │                                                   │
         └───────────────────────┬───────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   $derived: Merge      │
                    │   Global + LocalOverride│
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Re-compute FFT,      │
                    │   Grouping, Shapes     │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Render AnalysisGrid  │
                    └────────────────────────┘
```

---

This document now provides a complete specification for the UI/UX team, including:
1.  **What exists** (reusable code).
2.  **What's missing** (gaps to fill).
3.  **How the new layout works** (from your sketches).
4.  **How global/local controls sync** (the key architectural pattern).
5.  **A phased roadmap** for implementation.

Let me know if you'd like me to start implementing **Phase 0 (State Refactor)** or generate task tickets from this document!
