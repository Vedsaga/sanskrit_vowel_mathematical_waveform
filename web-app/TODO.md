# Vak Platform – Implementation Task List

This document tracks all tasks required to transform the current "shape generator" into a **Guna-Oriented Audio Analysis Platform**.

> **Reference**: See `DESIGN_AND_PRODUCT_DOC.md` for detailed specifications.

---

## Requirements Traceability Matrix

| Requirement (from Guna Doc) | Phase | Task |
|---|---|---|
| Geometry invariant to signal energy | Phase 0 | 0.1 (GlobalSettings), existing `shapeEngine.ts` ✓ |
| Frequency controls topology only | - | Already implemented in `shapeEngine.ts` ✓ |
| Time Window Selection | Phase 1 | 1.1 |
| Frequency Range Gating | Phase 1 | 1.5 (GlobalControlPanel) |
| Frequency Grouping (co-persistence) | Phase 1 | 1.2 |
| Harmonic/Prime/Golden badges | Phase 1 | 1.3 |
| Geometry Modes (Single/Overlay/Accumulation) | Phase 1 | 1.4 |
| Temporal Stability Analysis | Phase 1 | 1.5 |
| Energy Invariance Test | Phase 1 | 1.5 |
| Transient Suppression | Phase 1 | 1.5 |
| Audio Playback (play/pause) | Phase 1 | 1.7 |
| Audio Metadata Display | Phase 1 | 1.7 |
| Shared Geometry Canvas (Comparison) | Phase 2 | 2.2 |
| Convergence Analysis | Phase 2 | 2.3 |
| State Cards / Observation Log | Phase 2 | 2.4 |
| Overlay / Intersection / Difference | Phase 2 | 2.3 |
| Per-Shape Controls | Phase 3 | 3.1, 3.2, 3.3 |
| STFT Engine (upgrade from DFT) | Phase 0 | 0.5 |

---

## Phase 0: State Architecture & Core Engine (Foundation)

> **Goal**: Establish the global/local control binding architecture and upgrade core analysis engines.

### 0.1 Global Settings Store ✅
- [x] Create `src/lib/stores/globalSettingsStore.svelte.ts`
  - [x] Define `GlobalSettings` interface:
    ```typescript
    interface GlobalSettings {
      timeWindow: { start: number; width: number; step: number; type: 'hann' | 'rectangular' };
      frequencyRange: { min: number; max: number };
      amplitude: number; // Wiggle amplitude (A)
      rotation: RotationState;
      normalize: boolean; // Energy normalization toggle
      suppressTransients: boolean;
      transientThreshold: number; // δ for spectral flux
      geometryMode: 'single' | 'overlay' | 'accumulation';
    }
    ```
  - [x] Implement `setGlobal()` method for bulk updates
  - [x] Implement individual setters (e.g., `setTimeWindow()`, `setFrequencyRange()`)

### 0.2 Analysis Store (Single Audio) ✅
- [x] Create `src/lib/stores/analysisStore.svelte.ts`
  - [x] Define `AnalysisState` interface with optional local overrides
  - [x] Implement `analyses: AnalysisState[]` reactive state
  - [x] Implement `selectedAnalysisId` for tracking focused analysis
  - [x] Implement `addAnalysis()`, `removeAnalysis()`, `duplicateAnalysis()` methods
  - [x] Implement `setLocalOverride(id, key, value)` for per-analysis settings
  - [x] Implement `getEffectiveSettings(analysisId)` - merges global + local overrides

### 0.3 Shape Store Enhancements ✅
- [x] Extend `Shape` interface in `src/lib/types.ts`
  - [x] Add `animationOverride?: { speed?: number; direction?: 'cw' | 'ccw'; mode?: 'loop' | 'fixed' | 'none' }`
  - [x] Add `sourceFrequencyHz?: number` (for traceability to original audio frequency)
  - [x] Add `groupId?: string` (for grouping shapes by frequency cluster)
- [x] Update `shapeStore.svelte.ts`
  - [x] Implement `setShapeAnimationOverride(shapeId, override)` method
  - [x] Update `updateSelectedShapesPhi()` to respect per-shape overrides

### 0.4 Comparison Store Enhancements ✅
- [x] Refactor `comparisonStore.svelte.ts` to use the new global/local pattern
  - [x] Each panel (`leftPanel`, `rightPanel`) has `analyses[]` array
  - [x] Implement sync mode support
  - [x] Add `sharedCanvasShapes: Shape[]` - shapes from both panels merged for shared canvas

### 0.5 FFT Processor Upgrade (STFT) ✅
- [x] Update `src/lib/fftProcessor.ts`
  - [x] Replace `computeFFTSync()` with proper STFT implementation
  - [x] Accept parameters: `startTime`, `windowWidth`, `windowType`
  - [x] Return `FFTResult[]` array for sliding window analysis (not just single result)
  - [x] Add `computeSpectralFlux(fftResults)` for transient detection

### 0.6 Geometry Accumulation Engine ✅
- [x] Create `src/lib/utils/geometryAccumulator.ts`
  - [x] Implement `accumulateGeometries(shapes[], weights[])` - weighted sum of geometries
  - [x] Implement `computeAccumulatedShape(fftResults[], config)` - sliding window accumulation

---

## Phase 1: Analysis Observatory (Single Audio Page)

> **Goal**: Redesign `/audio-analysis` into a multi-analysis grid with advanced controls.

### 1.1 Time Window Control ✅
- [x] Create `src/lib/components/controls/TemporalNavigator.svelte`
  - [x] Display miniature waveform of full audio
  - [x] Draggable window region (start/end handles)
  - [x] Inputs for Start Time (seconds), Window Width (ms), Step Size (ms)
  - [x] Window type selector (Hann / Rectangular)
  - [x] "Slide" button to animate window across audio
- [x] Update `fftProcessor.ts` (done in Phase 0.5)

### 1.2 Frequency Grouping ✅
- [x] Create `src/lib/utils/frequencyGrouping.ts`
  - [x] Implement `detectHarmonics(components, fundamentalThreshold)` - returns groups based on harmonic series (fₙ = n × f₀)
  - [x] Implement `clusterByCorrelation(fftResults, γ)` - groups by temporal co-persistence (corr > γ)
  - [x] Implement `splitGroups(components, { harmonics: boolean, correlation: boolean })` - combined grouping
- [x] Create `src/lib/components/analysis/SpectrumGraph.svelte`
  - [x] Visual frequency plot showing magnitude vs frequency
  - [x] Clickable peaks to select/deselect frequency components
  - [x] Highlight selected frequencies
  - [x] Sync with ComponentGroups selection
- [x] Create `src/lib/components/analysis/ComponentGroups.svelte`
  - [x] Collapsible group sections (Group 1: Fundamental + Harmonics, Group 2: Secondary, etc.)
  - [x] Per-group selection toggle (render as single geometric layer)
  - [x] Expand/collapse all
  - [x] Group color assignment

### 1.3 Harmonic & Number-Theoretic Badges ✅
- [x] Create `src/lib/utils/frequencyAnalysis.ts`
  - [x] Implement `isPrime(n)` utility
  - [x] Implement `isGoldenRatioRelated(freqA, freqB, tolerance = 0.02)` utility (ratio ≈ 1.618)
  - [x] Implement `getHarmonicOrder(freq, fundamental)` - returns H2, H3, etc. or null
  - [x] Implement `analyzeFrequencyRelationships(components)` - adds `badges` property
- [x] Extend `FrequencyComponent` interface in `types.ts`
  - [x] Add `badges: ('P' | 'E' | 'O' | 'H2' | 'H3' | ... | 'φ')[]`
- [x] Create `src/lib/components/analysis/FrequencyBadges.svelte`
  - [x] Render badges with appropriate colors (Harmonic: Blue, Prime: Amber, Golden: Gold, E/O: Grey)
- [x] Create `src/lib/components/analysis/FrequencyBadgeFilter.svelte`
  - [x] Add filter toggles: `[All] [Harmonics Only] [Primes] [Golden Ratio (φ)]`

### 1.4 Geometry Modes ✅
- [x] Create `src/lib/components/controls/GeometryModeSelector.svelte`
  - [x] Radio/Toggle group: `Single-Group` | `Overlay` | `Accumulation`
  - [x] **Single-Group**: Render only selected group in isolation
  - [x] **Overlay**: Render all selected groups overlaid on same canvas
  - [x] **Accumulation**: Render accumulated geometry from sliding window
- [x] Update `ShapeCanvas.svelte`
  - [x] Accept `mode: 'single' | 'overlay' | 'accumulation'` prop
  - [x] Accept `overlayShapes` and `accumulationWeights` props
  - [x] In `accumulation` mode, blend geometries with increasing opacity for persistent structures

### 1.5 Guna Strength Indicator (Analysis Metrics) ✅
- [x] Create `src/lib/utils/gunaAnalysis.ts`
  - [x] Implement `calculateStabilityScore(geometrySequence)`:
    ```
    D(Sᵢ, Sⱼ) = (1/Nθ) × Σ|rᵢ(θ) - rⱼ(θ)|
    Stability = 1 - (1/M) × Σ D(Sᵢ, Sᵢ₊₁)
    ```
  - [x] Implement `checkEnergyInvariance(rawShapes, normalizedShapes, ε)`:
    - Returns `true` if `D(Sraw, Snorm) < ε`
  - [x] Implement `calculateTransientScore(fftResults)`:
    ```
    Flux(t) = Σf (|X(t,f)| - |X(t-1,f)|)²
    ```
- [x] Create `src/lib/components/analysis/GunaStrengthIndicator.svelte`
  - [x] Display Stability Score: `████████░░ 78% (Stable)` or `███░░░░░░░ 32% (Transient)`
  - [x] Display Energy Invariant: `✓ Yes` or `✗ No`
  - [x] Display Transient Score: percentage + interpretation

### 1.6 Multi-Analysis Grid Layout ✅
- [x] Create `src/lib/components/layout/AnalysisTile.svelte`
  - [x] Mini-canvas for geometry preview (150x150px)
  - [x] Label: "Analysis 1", "Analysis 2", etc.
  - [x] Click to select (emits event)
  - [x] Visual highlight when selected (ring border)
  - [x] Hover tooltip: Time window, Frequency range, Guna score
- [x] Create `src/lib/components/layout/AnalysisGrid.svelte`
  - [x] Responsive grid of `AnalysisTile` components
  - [x] "Add Analysis" tile with `+` icon (creates new analysis with current global settings)
  - [ ] Drag-to-reorder (optional, nice-to-have)
- [x] Create `src/lib/components/controls/GlobalControlPanel.svelte`
  - [x] Contains: TemporalNavigator, Frequency Range sliders, Amplitude slider, Rotation controls
  - [x] Toggles: Normalize Energy, Suppress Transients
  - [x] Geometry Mode selector
  - [ ] Position: Right sidebar or bottom bar (based on screen size) *(page integration)*
- [x] Create `src/lib/components/controls/LocalControlPanel.svelte`
  - [x] Same controls as Global, but values are per-analysis overrides
  - [x] Shows "Inherited from Global" indicator if no local override is set
  - [x] "Reset to Global" button per control section

### 1.7 Audio Playback & Metadata ✅
- [x] Create `src/lib/components/audio/AudioPlayer.svelte`
  - [x] Play/Pause button
  - [x] Playhead position indicator
  - [x] Playback restricted to current time window (start → start + width)
  - [x] Visual sync: playhead position updates TemporalNavigator highlight
- [x] Create `src/lib/components/audio/AudioMetadata.svelte`
  - [x] Display: Filename, Duration, Sample Rate, Channels
  - [x] Compact inline display (e.g., "sample.wav | 3.2s | 44.1kHz")
- [x] Integrate into `AudioUploader.svelte` ✅
  - [x] After upload, show `AudioMetadata` and `AudioPlayer`

### 1.8 Page Assembly (Analysis Observatory) ✅
- [x] Rename `/routes/audio-analysis/` to `/routes/analysis-observatory/` (or keep path, update title)
- [x] Rewrite `+page.svelte` to use:
  - [x] `AnalysisGrid` as primary content
  - [x] `GlobalControlPanel` in sidebar (when no analysis selected)
  - [x] `LocalControlPanel` in sidebar (when analysis selected)
  - [x] `AudioPlayer` and `AudioMetadata` in header area
  - [x] Focused mode: selected analysis expands, others become thumbnail row at bottom

> **Phase 1 Complete! ✅** All utilities, components, and page assembly done.

---

## Phase 2: Convergence Studio (Dual Audio Comparison) ✅

> **Goal**: Redesign `/comparison` with dual grids, **shared canvas**, and overlay capabilities.

### 2.1 Dual-Grid Layout ✅
- [x] Update `comparisonStore.svelte.ts` to support `analyses[]` per panel
- [x] Create `src/lib/components/layout/DualAnalysisGrid.svelte`
  - [x] Side-by-side grids for Audio A and Audio B
  - [x] Each side is an `AnalysisGrid` instance
  - [x] Central "Spine" with shared controls

### 2.2 Shared Geometry Canvas ✅
- [x] Create `src/lib/components/canvas/SharedCanvas.svelte`
  - [x] Positioned in center, visually prominent
  - [x] Receives shapes from **both** panels (Audio A and Audio B)
  - [x] Shapes from A rendered in warm colors, B in cool colors
- [x] Update `comparisonStore` with linkControls and showSharedCanvas toggles

### 2.3 Overlay & Comparison Rendering ✅
- [x] Enhanced `SharedCanvas.svelte` with comparison modes
  - [x] `comparisonMode: 'overlay' | 'intersection' | 'difference'`
  - [x] **Overlay**: Draw both layers with distinct colors/opacity
  - [x] **Intersection**: Highlight overlapping regions
  - [x] **Difference**: Highlight unique regions
- [x] Create `src/lib/utils/shapeComparison.ts`
  - [x] `computeIntersection(shapesA, shapesB, resolution)`
  - [x] `computeDifference(shapesA, shapesB)`
  - [x] `computeSimilarityScore(shapesA, shapesB)`

### 2.4 State Cards & Observation Log ✅
- [x] Create `src/lib/components/analysis/StateCard.svelte`
  - [x] Displays: Label, Audio filename, Time window, Frequency range, Thumbnail
  - [x] Buttons: `Load`, `Overlay`, `Delete`
- [x] Create `src/lib/components/analysis/ObservationLog.svelte`
  - [x] Expandable dropdown panel
  - [x] List of saved `StateCard` components
  - [x] "Save Current State" button (prompts for label)
  - [x] Persist to localStorage

### 2.5 Convergence Detection ✅
- [x] Create `src/lib/utils/convergenceAnalysis.ts`
  - [x] `computeConvergenceScore(statesA[], statesB[])` → 0-1 score
  - [x] `computeQuickConvergence()` for real-time updates
  - [x] `identifyMatchingShapes()` for pairs
- [x] Create `src/lib/components/analysis/ConvergenceIndicator.svelte`
  - [x] Display: "Convergence: X%" with label
  - [x] Visual: Progress bar, matching pairs, common frequencies

### 2.6 Page Assembly (Convergence Studio) ✅
- [x] Rewrite `/routes/comparison/+page.svelte` to use:
  - [x] `DualAnalysisGrid` (left/right)
  - [x] `SharedCanvas` in center (toggleable)
  - [x] `SyncControls` in central spine
  - [x] `ConvergenceIndicator` in header and footer
  - [x] `ObservationLog` dropdown (button in header)

> **Phase 2 Complete! ✅** All components, utilities, and page assembly done.

---

## Phase 3: Per-Shape Interaction (All Pages) ✅

> **Goal**: Enable direct manipulation of shapes on the canvas.

### 3.1 Canvas Click-to-Select ✅
- [x] Update `ShapeCanvas.svelte`
  - [x] Add `onclick` handler with canvas coordinate conversion
  - [x] Implement hit-testing via `hitTesting.ts` utilities
  - [x] Emit `onShapeClick(shapeId, event)` callback
  - [x] Support `Shift+Click` for multi-select
  - [x] Support `Ctrl/Cmd+Click` for toggle-select

### 3.2 Shape Popover ✅
- [x] Create `src/lib/components/controls/ShapePopover.svelte`
  - [x] Positioned near selected shape
  - [x] Color picker (9 preset colors)
  - [x] Opacity slider (0-100%)
  - [x] Rotation speed slider (0-5 rad/s)
  - [x] Direction toggle: CW / CCW / None
  - [x] Loop toggle: Continuous / Once / Off
  - [x] Delete button
  - [x] Multi-select: "X shapes selected" display
  - [x] Dismisses on Escape

### 3.3 Per-Shape Animation System ✅
- [x] Update `animationLoop.ts`
  - [x] `updateShapePhiWithOverride()` for per-shape animation
  - [x] `updateShapesWithAnimation()` batch update
  - [x] Skip shapes with `mode: 'off'` or `direction: 'none'`
- [x] Update `shapeStore.svelte.ts` for per-shape overrides

> **Phase 3 Complete! ✅** All per-shape interaction features implemented.

---

## Phase 4: Compose Lab (Manual Frequency Entry)

> **Goal**: Polish the `/visualizer` page with the new architecture.

### 4.1 Page Rename & Refactor
- [ ] Rename `/routes/visualizer/` to `/routes/compose-lab/` (or keep path, update title)
- [ ] Update `+page.svelte` to use:
  - [ ] Full-screen canvas (edge-to-edge, no card boxing)
  - [ ] Floating glass `ControlPanel` on right side

### 4.2 Unified Control Panel
- [ ] Create `src/lib/components/controls/ControlPanel.svelte`
  - [ ] Use shadcn Tabs component
  - [ ] **Tab 1 (Compose)**: Frequency input + Amplitude slider + Add button
  - [ ] **Tab 2 (Layers)**: Compact shape list with inline controls (color, opacity, visibility toggle, delete)
  - [ ] **Tab 3 (Animate)**: Global rotation controls (speed, direction, loop) + Presets (slow, medium, fast)

---

## Phase 5: Navigation & Polish

> **Goal**: Final integration and UX polish.

### 5.1 Navigation Updates
- [ ] Update `Sidebar.svelte` with new page names/routes:
  - [ ] Home
  - [ ] Compose Lab (was Visualizer)
  - [ ] Analysis Observatory (was Audio Analysis)
  - [ ] Convergence Studio (was Comparison)
- [ ] Update Home page (`+page.svelte`) feature cards with new names and descriptions

### 5.2 Animations & Transitions
- [ ] Add enter/exit animations for analysis tiles (fade + scale)
- [ ] Add smooth transitions when switching between grid/focused mode
- [ ] Add loading states for FFT processing (skeleton or spinner)
- [ ] Add subtle hover effects on all interactive elements

### 5.3 Keyboard Shortcuts
- [ ] `Space` - Toggle rotation play/pause
- [ ] `Delete` / `Backspace` - Remove selected shape(s)
- [ ] `Escape` - Deselect all / Close popover / Exit focused mode
- [ ] `A` - Add new analysis tile (when in Analysis Observatory)
- [ ] `S` - Save current state (when in Convergence Studio)
- [ ] `1-9` - Select analysis tile by number

### 5.4 Responsive Design
- [ ] Test and fix layouts for tablet (768px - 1024px)
- [ ] Test and fix layouts for mobile (<768px)
- [ ] Collapse sidebar to icon-only mode on smaller screens
- [ ] Stack grids vertically on mobile (A above B)
- [ ] Bottom sheet for controls on mobile

### 5.5 Accessibility
- [ ] Ensure all controls have proper ARIA labels
- [ ] Keyboard navigation for grid tiles (arrow keys, Enter to select)
- [ ] Screen reader announcements for state changes
- [ ] High contrast mode support (using CSS variables)
- [ ] Focus visible indicators on all interactive elements

---

## Backlog (Future Enhancements)

These items are out of scope for the initial redesign but captured for future work.

- [ ] **Waveform Overlay on Canvas**: Show time-domain waveform behind shapes
- [ ] **Export Features**: Export canvas as SVG/PNG, export analysis state as JSON
- [ ] **Theme Customizer**: User-selectable color palettes for shapes (Neon, Pastel, Monochrome)
- [ ] **Onboarding Walkthrough**: Guided tour for first-time users
- [ ] **Real-time Microphone Input**: Live audio analysis from mic
- [ ] **Snap-to-Silence**: Auto-detect and snap time window to silence boundaries
- [ ] **Phoneme Library**: Pre-loaded reference shapes for Sanskrit vowels
- [ ] **Collaboration**: Share states via URL or export/import

---

## Notes

- **Dependencies**: Each phase builds on the previous. Phase 0 is the foundation.
- **Parallel Work**: Within a phase, tasks marked with independent components can be parallelized.
- **Testing**: Unit tests should be written alongside new utility functions (`frequencyGrouping.ts`, `gunaAnalysis.ts`, `convergenceAnalysis.ts`).
- **Core Philosophy**: Geometry is INVARIANT to amplitude. This must be maintained throughout.
