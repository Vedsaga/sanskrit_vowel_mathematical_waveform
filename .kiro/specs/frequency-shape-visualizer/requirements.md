# Requirements Document

## Introduction

This document specifies the requirements for a Frequency Shape Visualizer feature in the Svelte web application. The system enables users to visualize frequencies as 2D geometric shapes (circles with wiggles), overlay and manipulate multiple shapes, analyze audio files using Fourier transformation, and compare audio files side-by-side. The application uses shadcn-svelte components and follows established UI/UX principles for consistent component states and stable layouts.

## Glossary

- **Frequency (fq)**: An integer value (≥ 1) that determines the number of wiggles in the shape (wiggleCount = fq - 1)
- **Wiggle**: A smooth inward-and-outward undulation of the circle's radius, created by sinusoidal modulation
- **In-Phase Radial Modulation**: The mathematical technique where a circle's radius is modulated by a sine wave, producing symmetric undulations
- **Shape**: A 2D geometric curve defined by `r(θ) = R + A·sin((fq-1)·θ)` where R is base radius, A is wiggle amplitude, and θ ∈ [0, 2π]
- **Base Radius (R)**: The unmodulated radius of the circle
- **Wiggle Amplitude (A)**: The magnitude of the radial modulation (how far wiggles extend inward/outward from base radius)
- **Fourier Transformation (FFT)**: A mathematical operation that decomposes an audio signal into its constituent frequency components
- **Discrete Wave**: An individual frequency component extracted from audio via FFT, characterized by frequency (Hz) and amplitude
- **Rotation**: Angular movement of a shape around its center point, implemented as phase offset φ in `sin((fq-1)·θ + φ)`
- **Overlay**: The visual stacking of multiple shapes on a single canvas, each with independent (fq, R, φ) parameters
- **Canvas**: The 2D rendering surface where shapes are drawn using Cartesian coordinates: `x(θ) = r(θ)·cos(θ)`, `y(θ) = r(θ)·sin(θ)`
- **Structural Invariance**: The property that the wiggle geometry (radius modulation amplitude and base radius) does not change based on FFT energy or signal amplitude; only the number of wiggles varies with frequency
- **Sampling Resolution**: The number of points used to render the curve (e.g., 360–2048 points per revolution) to ensure smooth rendering

## Requirements

### Requirement 1: Application Routing and Layout Structure

**User Story:** As a developer, I want a well-organized routing and layout structure, so that new features can be added easily and the application remains maintainable.

#### Acceptance Criteria

1. WHEN the application initializes THEN the System SHALL provide a root layout component that wraps all pages with consistent navigation and styling
2. WHEN a user navigates to a feature page THEN the System SHALL render the appropriate page component within the shared layout
3. WHEN new features are added THEN the System SHALL support nested routing under feature-specific directories
4. WHILE the application is running THEN the System SHALL maintain a sidebar navigation component that displays links to all major features
5. WHEN a navigation link is clicked THEN the System SHALL update the URL and render the corresponding page without full page reload

### Requirement 2: Frequency-to-Shape Visualization

**User Story:** As a user, I want to enter a frequency number and see it represented as a 2D shape, so that I can visualize the mathematical relationship between frequency and geometric form.

#### Shape Definition (In-Phase Radial Modulation)

The shape is a circle whose radius is modulated by a sine wave:
- `r(θ) = R + A·sin((fq-1)·θ)` (polar form)
- `x(θ) = r(θ)·cos(θ)`, `y(θ) = r(θ)·sin(θ)` (Cartesian form)
- Where: R = base radius, A = wiggle amplitude (fixed constant), θ ∈ [0, 2π], fq = frequency input (integer ≥ 1)

This produces smooth, continuous curves with wiggles going both inward and outward from the base circle, evenly spaced and perfectly symmetric.

#### Shape Invariance Constraint

The frequency visualization SHALL be structurally invariant with respect to signal energy or amplitude:
- The wiggle amplitude (A) SHALL be a global constant or user-controlled parameter, but SHALL remain independent of FFT amplitude
- FFT magnitude or signal energy SHALL NOT influence: wiggle depth, base radius, or overall shape geometry
- Frequency (fq) SHALL affect only the number of wiggles via (fq − 1)
- The System SHALL constrain A such that A < R to prevent radius inversion

This ensures that all shapes are directly comparable based solely on frequency topology.

#### Rendering Constraints

- All shapes SHALL be rendered centered at the canvas origin (0, 0)
- The curve SHALL be sampled at a configurable resolution (default: 360–2048 points per revolution) to ensure smooth rendering

#### Acceptance Criteria

1. WHEN a user enters frequency value 1 THEN the System SHALL render a pure circle with no wiggles using the formula `r(θ) = R` (since sin(0·θ) = 0)
2. WHEN a user enters frequency value 2 THEN the System SHALL render a circle with 1 symmetric wiggle using the formula `r(θ) = R + A·sin(1·θ)`
3. WHEN a user enters frequency value n (where n > 1) THEN the System SHALL render a circle with (n-1) wiggles using the formula `r(θ) = R + A·sin((n-1)·θ)` with wiggles evenly distributed at equal angular distances
4. WHEN the frequency input changes THEN the System SHALL update the shape visualization within 100 milliseconds
5. WHEN a user enters a non-positive or non-numeric value THEN the System SHALL display a validation error and maintain the previous valid shape
6. WHILE a shape is displayed THEN the System SHALL render the shape on a canvas with anti-aliased edges and consistent stroke width
7. WHEN shapes are generated from FFT analysis THEN the System SHALL use a fixed wiggle amplitude (A) independent of FFT magnitude or signal energy
8. WHEN two shapes have the same frequency (fq) THEN their geometric structure SHALL be identical regardless of audio amplitude differences
9. WHEN FFT magnitude varies THEN the System SHALL reflect this variation only through non-geometric properties (opacity, color, stroke width) and not through shape geometry

### Requirement 3: Shape Overlay and Manipulation

**User Story:** As a user, I want to overlay multiple frequency shapes and rotate them, so that I can explore visual patterns created by combining different frequencies.

#### Rotation Implementation

Rotation is implemented as a phase offset φ in the shape formula:
- `r(θ) = R + A·sin((fq-1)·θ + φ)`
- Clockwise rotation: decrease φ over time
- Counter-clockwise rotation: increase φ over time
- Animation: φ(t) = φ₀ + ω·t where ω is angular velocity

#### Acceptance Criteria

1. WHEN a user adds a new shape THEN the System SHALL render the shape on the overlay canvas while preserving existing shapes, each with independent (fq, R, φ) parameters and shared constant A
2. WHEN a user selects one or more shapes THEN the System SHALL highlight the selected shapes with a distinct visual indicator
3. WHEN a user initiates clockwise rotation on selected shapes THEN the System SHALL decrease the phase offset φ of those shapes over time
4. WHEN a user initiates counter-clockwise rotation on selected shapes THEN the System SHALL increase the phase offset φ of those shapes over time
5. WHEN a user sets rotation to loop mode THEN the System SHALL continuously animate the phase offset φ(t) until the user stops the rotation
6. WHEN a user specifies a rotation angle THEN the System SHALL rotate selected shapes by the specified degrees (converting to radians for φ) and stop
7. WHEN a user removes a shape THEN the System SHALL remove only that shape from the overlay while preserving other shapes
8. WHILE shapes are overlaid THEN the System SHALL render shapes with configurable opacity and stroke color to allow visual distinction
9. WHEN a user adjusts rotation speed THEN the System SHALL update the angular velocity ω for the selected shapes

### Requirement 4: Audio File Fourier Analysis

**User Story:** As a user, I want to upload an audio file and apply Fourier transformation, so that I can see the discrete frequency components and visualize them as shapes.

#### FFT-to-Shape Mapping

- The System SHALL provide a configurable normalization strategy to map FFT frequency values (Hz) into integer fq values suitable for visualization
- FFT frequency bins SHALL be mapped to integer fq values using a configurable scaling or rounding strategy
- FFT amplitude MAY be optionally mapped to non-geometric properties: stroke opacity, stroke thickness, color intensity, or z-ordering
- FFT amplitude SHALL NOT modify the geometric equation of the shape (structural invariance)

#### Acceptance Criteria

1. WHEN a user drops an audio file onto the designated area THEN the System SHALL accept and load the audio file for processing
2. WHEN a user clicks the Fourier transformation button THEN the System SHALL compute the FFT and display a list of discrete frequency components
3. WHEN displaying frequency components THEN the System SHALL show each component with its frequency value (Hz) and amplitude
4. WHEN a user selects one or more frequency components THEN the System SHALL generate corresponding shapes on the visualization canvas using the frequency-to-wiggle mapping (fq → fq-1 wiggles)
5. WHEN shapes are generated from audio analysis THEN the System SHALL enable all manipulation operations defined in Requirement 3
6. IF the audio file format is unsupported THEN the System SHALL display an error message specifying supported formats (WAV, MP3, OGG)
7. WHILE processing audio THEN the System SHALL display a loading indicator until the transformation completes
8. WHEN displaying shapes from FFT THEN the System SHALL optionally allow users to map FFT amplitude to visual properties (opacity, color, stroke width) without affecting shape geometry

### Requirement 5: Side-by-Side Audio Comparison

**User Story:** As a user, I want to compare two audio files side by side, so that I can analyze differences in their frequency compositions.

#### Acceptance Criteria

1. WHEN a user enters comparison mode THEN the System SHALL display two parallel analysis panels
2. WHEN a user uploads audio to either panel THEN the System SHALL process and display that audio's frequency analysis independently
3. WHEN both panels contain audio data THEN the System SHALL allow synchronized or independent manipulation of shapes
4. WHEN displaying comparison results THEN the System SHALL align frequency scales across both panels for accurate visual comparison
5. WHILE in comparison mode THEN the System SHALL maintain consistent layout dimensions for both panels regardless of content

### Requirement 6: UI Component Standards

**User Story:** As a user, I want consistent and accessible UI components, so that I can interact with the application predictably and efficiently.

#### Acceptance Criteria

1. WHEN a component is loading data THEN the System SHALL display a skeleton or spinner placeholder
2. WHEN a component encounters an error THEN the System SHALL display an error state with a retry option
3. WHEN a component is rendered THEN the System SHALL maintain its layout position without shifting other elements
4. WHEN an action is unavailable THEN the System SHALL display the control in a disabled state rather than hiding the control
5. WHILE the application is running THEN the System SHALL use shadcn-svelte components for all standard UI elements (buttons, inputs, cards, dialogs)
