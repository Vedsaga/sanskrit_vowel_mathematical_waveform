/**
 * Core type definitions for the Frequency Shape Visualizer
 * 
 * These interfaces define the data structures used throughout the application
 * for shape generation, configuration, and state management.
 */

/**
 * Represents a 2D point in Cartesian coordinates
 */
export interface Point {
  x: number;
  y: number;
}

// Shape interface is defined at the end of this file with extended properties

/**
 * Global configuration for shape rendering
 * 
 * These values are shared across all shapes to ensure
 * structural invariance (same fq = same geometry)
 */
export interface ShapeConfig {
  /** Global wiggle amplitude (must be < R for any shape) */
  A: number;
  /** Number of sampling points for curve rendering (360-2048) */
  resolution: number;
  /** Canvas dimensions in pixels */
  canvasSize: number;
}

/**
 * State for rotation animation
 */
export interface RotationState {
  /** Whether rotation animation is currently running */
  isAnimating: boolean;
  /** Direction of rotation */
  direction: 'clockwise' | 'counterclockwise';
  /** Rotation mode: loop continuously or rotate to fixed angle */
  mode: 'loop' | 'fixed';
  /** Target angle in degrees (for fixed mode only) */
  targetAngle?: number;
  /** Angular velocity in radians per second */
  speed: number;
}

/**
 * Result of shape parameter validation
 */
export interface ValidationResult {
  /** Whether the parameters are valid */
  valid: boolean;
  /** Error messages if validation failed */
  errors: string[];
}

/**
 * Result of FFT analysis
 */
export interface FFTResult {
  /** Frequency bins in Hz */
  frequencies: number[];
  /** Amplitude values for each frequency bin */
  magnitudes: number[];
  /** Sample rate of the audio */
  sampleRate: number;
}

/**
 * A frequency component extracted from FFT analysis
 */
export interface FrequencyComponent {
  /** Unique identifier */
  id: string;
  /** Frequency in Hz */
  frequencyHz: number;
  /** Amplitude/magnitude value */
  magnitude: number;
  /** Mapped integer frequency for shape generation */
  fq: number;
  /** Whether this component is selected for visualization */
  selected: boolean;
  /** Badges indicating properties (harmonic, prime, golden ratio, etc.) */
  badges?: ('P' | 'E' | 'O' | 'H2' | 'H3' | 'H4' | 'H5' | 'H6' | 'H7' | 'H8' | 'φ')[];
}

/**
 * Animation override for per-shape animation control
 */
export interface AnimationOverride {
  /** Angular velocity in radians per second */
  speed?: number;
  /** Rotation direction */
  direction?: AnimationDirection;
  /** Animation mode */
  mode?: AnimationMode;
}

/**
 * Animation direction type
 */
export type AnimationDirection = 'cw' | 'ccw' | 'none';

/**
 * Animation mode type
 */
export type AnimationMode = 'continuous' | 'once' | 'off';

/**
 * Time window configuration for STFT analysis
 */
export interface TimeWindow {
  /** Start time in seconds */
  start: number;
  /** Window width in milliseconds */
  width: number;
  /** Step size in milliseconds for sliding window */
  step: number;
  /** Window function type */
  type: WindowType;
}

/**
 * Window function types for FFT
 */
export type WindowType = 'hann' | 'rectangular' | 'hamming' | 'blackman';

/**
 * Frequency range filter
 */
export interface FrequencyRange {
  /** Minimum frequency in Hz */
  min: number;
  /** Maximum frequency in Hz */
  max: number;
}

/**
 * Geometry rendering mode
 */
export type GeometryMode = 'single' | 'overlay' | 'accumulation';

/**
 * Global settings that apply to all analyses
 */
export interface GlobalSettings {
  /** Time window for STFT analysis */
  timeWindow: TimeWindow;
  /** Frequency range filter */
  frequencyRange: FrequencyRange;
  /** Wiggle amplitude (A) */
  amplitude: number;
  /** Rotation state */
  rotation: RotationState;
  /** Whether to normalize energy (amplitude-invariant geometry) */
  normalize: boolean;
  /** Whether to suppress transient components */
  suppressTransients: boolean;
  /** Threshold for spectral flux (transient detection) */
  transientThreshold: number;
  /** Geometry rendering mode */
  geometryMode: GeometryMode;
}

/**
 * Analysis state representing a single analysis tile
 * Properties that are undefined inherit from GlobalSettings
 */
export interface AnalysisState {
  /** Unique identifier */
  id: string;
  /** User-defined label for the analysis */
  label: string;
  /** Time of creation */
  createdAt: number;
  // --- Local overrides (undefined = inherit from global) ---
  /** Override time window */
  timeWindow?: TimeWindow;
  /** Override frequency range */
  frequencyRange?: FrequencyRange;
  /** Override amplitude */
  amplitude?: number;
  /** Override rotation state */
  rotation?: RotationState;
  /** Override normalize setting */
  normalize?: boolean;
  /** Override transient suppression */
  suppressTransients?: boolean;
  // --- Computed/derived state ---
  /** Extracted frequency components */
  frequencyComponents: FrequencyComponent[];
  /** Generated shapes from selected components */
  shapes: Shape[];
  /** Calculated stability score (0-1) */
  stabilityScore?: number;
  /** Whether energy invariance holds */
  energyInvariant?: boolean;
  /** Transient score (0-1) */
  transientScore?: number;
}

/**
 * Extended Shape interface with animation overrides and traceability
 */
export interface Shape {
  /** Unique identifier for the shape */
  id: string;
  /** Frequency value (integer ≥ 1), determines wiggle count = fq - 1 */
  fq: number;
  /** Base radius of the shape */
  R: number;
  /** Phase offset for rotation (radians) */
  phi: number;
  /** Stroke color (CSS color string) */
  color: string;
  /** Stroke opacity (0-1) */
  opacity: number;
  /** Stroke width in pixels */
  strokeWidth: number;
  /** Whether the shape is currently selected */
  selected: boolean;
  /** Per-shape animation override (if undefined, uses global rotation) */
  animationOverride?: AnimationOverride;
  /** Original frequency in Hz (for traceability to audio) */
  sourceFrequencyHz?: number;
  /** Group ID for frequency clustering */
  groupId?: string;
}
