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

/**
 * Represents a frequency shape with all its visual and geometric properties
 * 
 * The shape is defined by the formula: r(θ) = R + A·sin((fq-1)·θ + φ)
 * where:
 * - R is the base radius
 * - A is the wiggle amplitude (from ShapeConfig)
 * - fq is the frequency (determines number of wiggles = fq - 1)
 * - φ (phi) is the phase offset for rotation
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
}

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
}
