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

### 0.1 Global Settings Store
- [ ] Create `src/lib/stores/globalSettingsStore.svelte.ts`
  - [ ] Define `GlobalSettings` interface:
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
  - [ ] Implement `setGlobal()` method for bulk updates
  - [ ] Implement individual setters (e.g., `setTimeWindow()`, `setFrequencyRange()`)

### 0.2 Analysis Store (Single Audio)
- [ ] Create `src/lib/stores/analysisStore.svelte.ts`
  - [ ] Define `AnalysisState` interface with optional local overrides
  - [ ] Implement `analyses: AnalysisState[]` reactive state
  - [ ] Implement `selectedAnalysisId` for tracking focused analysis
  - [ ] Implement `addAnalysis()`, `removeAnalysis()`, `duplicateAnalysis()` methods
  - [ ] Implement `setLocalOverride(id, key, value)` for per-analysis settings
  - [ ] Implement `getEffectiveSettings(analysisId)` - merges global + local overrides

### 0.3 Shape Store Enhancements
- [ ] Extend `Shape` interface in `src/lib/types.ts`
  - [ ] Add `animationOverride?: { speed?: number; direction?: 'cw' | 'ccw'; mode?: 'loop' | 'fixed' | 'none' }`
  - [ ] Add `sourceFrequencyHz?: number` (for traceability to original audio frequency)
  - [ ] Add `groupId?: string` (for grouping shapes by frequency cluster)
- [ ] Update `shapeStore.svelte.ts`
  - [ ] Implement `setShapeAnimationOverride(shapeId, override)` method
  - [ ] Update `updateSelectedShapesPhi()` to respect per-shape overrides

### 0.4 Comparison Store Enhancements
- [ ] Refactor `comparisonStore.svelte.ts` to use the new global/local pattern
  - [ ] Each panel (`leftPanel`, `rightPanel`) should have its own `analyses[]` array
  - [ ] Implement true sync mode: when `syncMode === 'synchronized'`, global changes apply to both panels
  - [ ] Add `sharedCanvasShapes: Shape[]` - shapes from both panels merged for shared canvas

### 0.5 FFT Processor Upgrade (STFT)
- [ ] Update `src/lib/fftProcessor.ts`
  - [ ] Replace `computeFFTSync()` with proper STFT implementation
  - [ ] Accept parameters: `startTime`, `windowWidth`, `windowType`
  - [ ] Return `FFTResult[]` array for sliding window analysis (not just single result)
  - [ ] Add `computeSpectralFlux(fftResults)` for transient detection

### 0.6 Geometry Accumulation Engine
- [ ] Create `src/lib/utils/geometryAccumulator.ts`
  - [ ] Implement `accumulateGeometries(shapes[], weights[])` - weighted sum of geometries
  - [ ] Implement `computeAccumulatedShape(fftResults[], config)` - sliding window accumulation

---

## Phase 1: Analysis Observatory (Single Audio Page)

> **Goal**: Redesign `/audio-analysis` into a multi-analysis grid with advanced controls.

### 1.1 Time Window Control
- [ ] Create `src/lib/components/controls/TemporalNavigator.svelte`
  - [ ] Display miniature waveform of full audio
  - [ ] Draggable window region (start/end handles)
  - [ ] Inputs for Start Time (seconds), Window Width (ms), Step Size (ms)
  - [ ] Window type selector (Hann / Rectangular)
  - [ ] "Slide" button to animate window across audio
- [ ] Update `fftProcessor.ts` (done in Phase 0.5)

### 1.2 Frequency Grouping
- [ ] Create `src/lib/utils/frequencyGrouping.ts`
  - [ ] Implement `detectHarmonics(components, fundamentalThreshold)` - returns groups based on harmonic series (fₙ = n × f₀)
  - [ ] Implement `clusterByCorrelation(fftResults, γ)` - groups by temporal co-persistence (corr > γ)
  - [ ] Implement `splitGroups(components, { harmonics: boolean, correlation: boolean })` - combined grouping
- [ ] Create `src/lib/components/analysis/SpectrumGraph.svelte`
  - [ ] Visual frequency plot showing magnitude vs frequency
  - [ ] Clickable peaks to select/deselect frequency components
  - [ ] Highlight selected frequencies
  - [ ] Sync with ComponentGroups selection
- [ ] Create `src/lib/components/analysis/ComponentGroups.svelte`
  - [ ] Collapsible group sections (Group 1: Fundamental + Harmonics, Group 2: Secondary, etc.)
  - [ ] Per-group selection toggle (render as single geometric layer)
  - [ ] Expand/collapse all
  - [ ] Group color assignment

### 1.3 Harmonic & Number-Theoretic Badges
- [ ] Create `src/lib/utils/frequencyAnalysis.ts`
  - [ ] Implement `isPrime(n)` utility
  - [ ] Implement `isGoldenRatioRelated(freqA, freqB, tolerance = 0.02)` utility (ratio ≈ 1.618)
  - [ ] Implement `getHarmonicOrder(freq, fundamental)` - returns H2, H3, etc. or null
  - [ ] Implement `analyzeFrequencyRelationships(components)` - adds `badges` property
- [ ] Extend `FrequencyComponent` interface in `types.ts`
  - [ ] Add `badges: ('P' | 'E' | 'O' | 'H2' | 'H3' | ... | 'φ')[]`
- [ ] Create `src/lib/components/analysis/FrequencyBadges.svelte`
  - [ ] Render badges with appropriate colors (Harmonic: Blue, Prime: Amber, Golden: Gold, E/O: Grey)
  - [ ] Add filter toggles: `[All] [Harmonics Only] [Primes] [Golden Ratio (φ)]`

### 1.4 Geometry Modes
- [ ] Create `src/lib/components/controls/GeometryModeSelector.svelte`
  - [ ] Radio/Toggle group: `Single-Group` | `Overlay` | `Accumulation`
  - [ ] **Single-Group**: Render only selected group in isolation
  - [ ] **Overlay**: Render all selected groups overlaid on same canvas
  - [ ] **Accumulation**: Render accumulated geometry from sliding window
- [ ] Update `ShapeCanvas.svelte`
  - [ ] Accept `mode: 'single' | 'overlay' | 'accumulation'` prop
  - [ ] In `accumulation` mode, blend geometries with increasing opacity for persistent structures

### 1.5 Guna Strength Indicator (Analysis Metrics)
- [ ] Create `src/lib/utils/gunaAnalysis.ts`
  - [ ] Implement `calculateStabilityScore(geometrySequence)`:
    ```
    D(Sᵢ, Sⱼ) = (1/Nθ) × Σ|rᵢ(θ) - rⱼ(θ)|
    Stability = 1 - (1/M) × Σ D(Sᵢ, Sᵢ₊₁)
    ```
  - [ ] Implement `checkEnergyInvariance(rawShapes, normalizedShapes, ε)`:
    - Returns `true` if `D(Sraw, Snorm) < ε`
  - [ ] Implement `calculateTransientScore(fftResults)`:
    ```
    Flux(t) = Σf (|X(t,f)| - |X(t-1,f)|)²
    ```
- [ ] Create `src/lib/components/analysis/GunaStrengthIndicator.svelte`
  - [ ] Display Stability Score: `████████░░ 78% (Stable)` or `███░░░░░░░ 32% (Transient)`
  - [ ] Display Energy Invariant: `✓ Yes` or `✗ No`
  - [ ] Display Transient Score: percentage + interpretation

### 1.6 Multi-Analysis Grid Layout
- [ ] Create `src/lib/components/layout/AnalysisTile.svelte`
  - [ ] Mini-canvas for geometry preview (150x150px)
  - [ ] Label: "Analysis 1", "Analysis 2", etc.
  - [ ] Click to select (emits event)
  - [ ] Visual highlight when selected (ring border)
  - [ ] Hover tooltip: Time window, Frequency range, Guna score
- [ ] Create `src/lib/components/layout/AnalysisGrid.svelte`
  - [ ] Responsive grid of `AnalysisTile` components
  - [ ] "Add Analysis" tile with `+` icon (creates new analysis with current global settings)
  - [ ] Drag-to-reorder (optional, nice-to-have)
- [ ] Create `src/lib/components/controls/GlobalControlPanel.svelte`
  - [ ] Contains: TemporalNavigator, Frequency Range sliders, Amplitude slider, Rotation controls
  - [ ] Toggles: Normalize Energy, Suppress Transients
  - [ ] Geometry Mode selector
  - [ ] Position: Right sidebar or bottom bar (based on screen size)
- [ ] Create `src/lib/components/controls/LocalControlPanel.svelte`
  - [ ] Same controls as Global, but values are per-analysis overrides
  - [ ] Shows "Inherited from Global" indicator if no local override is set
  - [ ] "Reset to Global" button per control section

### 1.7 Audio Playback & Metadata (NEW)
- [ ] Create `src/lib/components/audio/AudioPlayer.svelte`
  - [ ] Play/Pause button
  - [ ] Playhead position indicator
  - [ ] Playback restricted to current time window (start → start + width)
  - [ ] Visual sync: playhead position updates TemporalNavigator highlight
- [ ] Create `src/lib/components/audio/AudioMetadata.svelte`
  - [ ] Display: Filename, Duration, Sample Rate, Channels
  - [ ] Compact inline display (e.g., "sample.wav | 3.2s | 44.1kHz")
- [ ] Integrate into `AudioUploader.svelte`
  - [ ] After upload, show `AudioMetadata` and `AudioPlayer`

### 1.8 Page Assembly (Analysis Observatory)
- [ ] Rename `/routes/audio-analysis/` to `/routes/analysis-observatory/` (or keep path, update title)
- [ ] Rewrite `+page.svelte` to use:
  - [ ] `AnalysisGrid` as primary content
  - [ ] `GlobalControlPanel` in sidebar (when no analysis selected)
  - [ ] `LocalControlPanel` in sidebar (when analysis selected)
  - [ ] `AudioPlayer` and `AudioMetadata` in header area
  - [ ] Focused mode: selected analysis expands, others become thumbnail row at bottom

---

## Phase 2: Convergence Studio (Dual Audio Comparison)

> **Goal**: Redesign `/comparison` with dual grids, **shared canvas**, and overlay capabilities.

### 2.1 Dual-Grid Layout
- [ ] Update `comparisonStore.svelte.ts` to support `analyses[]` per panel
- [ ] Create `src/lib/components/layout/DualAnalysisGrid.svelte`
  - [ ] Side-by-side grids for Audio A and Audio B
  - [ ] Each side is an `AnalysisGrid` instance
  - [ ] Central "Spine" with shared controls

### 2.2 Shared Geometry Canvas (NEW)
- [ ] Create `src/lib/components/canvas/SharedCanvas.svelte`
  - [ ] Positioned in center, visually prominent
  - [ ] Receives shapes from **both** panels (Audio A and Audio B)
  - [ ] Shapes from A rendered in one color family, B in another
- [ ] Update `SyncControls.svelte`
  - [ ] "Link Controls" toggle: when ON, changing A's settings applies to B
  - [ ] "Show Shared Canvas" toggle: when ON, center canvas shows merged geometry
  - [ ] When OFF, each side has its own canvas

### 2.3 Overlay & Comparison Rendering
- [ ] Enhance `ShapeCanvas.svelte`
  - [ ] Accept `overlayShapes?: Shape[]` prop for secondary layer
  - [ ] Accept `comparisonMode?: 'none' | 'overlay' | 'intersection' | 'difference'`
  - [ ] **Overlay**: Draw both layers with distinct colors/opacity
  - [ ] **Intersection**: Highlight only regions where shapes overlap (requires custom rendering)
  - [ ] **Difference**: Highlight regions unique to each source
- [ ] Create `src/lib/utils/shapeComparison.ts`
  - [ ] Implement `computeIntersection(shapesA, shapesB, resolution)` - returns intersection mask
  - [ ] Implement `computeDifference(shapesA, shapesB)` - returns difference regions

### 2.4 State Cards & Observation Log
- [ ] Create `src/lib/components/analysis/StateCard.svelte`
  - [ ] Displays: User-defined label, Audio filename, Time window, Frequency range, Groups, Geometry thumbnail
  - [ ] Buttons: `Load` (restore state), `Overlay` (add to current view), `Delete`
- [ ] Create `src/lib/components/analysis/ObservationLog.svelte`
  - [ ] Slide-out drawer (triggered by button in header)
  - [ ] List of saved `StateCard` components
  - [ ] "Save Current State" button (prompts for label)
  - [ ] Persist to localStorage (or future: backend)
- [ ] Update stores to support state serialization/deserialization
  - [ ] `serializeState()` → returns JSON-serializable object (audio NOT included, just reference)
  - [ ] `deserializeState(stateObject)` → restores store (requires audio to be re-uploaded or cached)

### 2.5 Convergence Detection (NEW)
- [ ] Create `src/lib/utils/convergenceAnalysis.ts`
  - [ ] Implement `computeConvergenceScore(statesA[], statesB[])`:
    ```
    min(D(SᵢA, SⱼB)) for all i, j
    ```
  - [ ] Return: score (0-1), converging pairs
- [ ] Create `src/lib/components/analysis/ConvergenceIndicator.svelte`
  - [ ] Display: "Convergence: 85% (States 2 & 5 match)"
  - [ ] Visual: Highlight matching tiles in both grids

### 2.6 Page Assembly (Convergence Studio)
- [ ] Rename `/routes/comparison/` to `/routes/convergence-studio/` (or keep path, update title)
- [ ] Rewrite `+page.svelte` to use:
  - [ ] `DualAnalysisGrid` (left/right)
  - [ ] `SharedCanvas` in center (toggleable)
  - [ ] `SyncControls` in central spine
  - [ ] `ConvergenceIndicator` below shared canvas
  - [ ] `ObservationLog` drawer (button in header)

---

## Phase 3: Per-Shape Interaction (All Pages)

> **Goal**: Enable direct manipulation of shapes on the canvas.

### 3.1 Canvas Click-to-Select
- [ ] Update `ShapeCanvas.svelte`
  - [ ] Add `onclick` handler with canvas coordinate conversion
  - [ ] Implement hit-testing: for each shape, check if clicked point is within stroke tolerance of path
  - [ ] Emit `shapeSelected(shapeId, event)` event
  - [ ] Support `Shift+Click` for multi-select (add to selection)
  - [ ] Support `Ctrl/Cmd+Click` for toggle-select

### 3.2 Shape Popover
- [ ] Create `src/lib/components/controls/ShapePopover.svelte`
  - [ ] Positioned near selected shape(s) on canvas (use Popover from shadcn)
  - [ ] Contains:
    - [ ] Color picker
    - [ ] Opacity slider (0-100%)
    - [ ] Rotation speed slider (0-5 rad/s)
    - [ ] Direction toggle: CW / CCW / None
    - [ ] Loop toggle: Continuous / Once / Off
    - [ ] Delete button
  - [ ] Multi-select: shows "X shapes selected", controls apply to all
  - [ ] Dismisses on outside click or Escape
- [ ] Integrate `ShapePopover` into canvas component

### 3.3 Per-Shape Animation System
- [ ] Update `animationLoop.ts`
  - [ ] When updating phi, iterate each shape individually
  - [ ] Check `shape.animationOverride` for custom speed/direction/mode
  - [ ] Apply per-shape delta: `deltaPhi = shapeSpeed * direction * dt`
  - [ ] Skip shapes with `mode: 'none'`
  - [ ] Handle `mode: 'once'` - stop at 2π and mark complete

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
