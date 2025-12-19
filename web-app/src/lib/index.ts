// place files you want to import through the `$lib` alias in this folder.
export { cn } from "./utils.js";

// Component exports
export { default as ShapeCanvas } from "./components/ShapeCanvas.svelte";

// Shape engine exports
export {
  generateShapePoints,
  validateShapeParams,
  countWiggles,
  validateFrequencyInput
} from "./shapeEngine.js";

// Store exports
export { shapeStore, type ShapeStore } from "./stores/index.js";

// Animation loop exports
export {
  animationLoop,
  createAnimationLoop,
  type AnimationLoop,
  type PhiUpdateCallback,
  type StopCallback
} from "./animationLoop.js";

// FFT processor exports
export {
  computeFFT,
  computeFFTSync,
  mapFrequencyToFq,
  extractTopFrequencies,
  validateFFTParams,
  type NormalizationStrategy,
  type NormalizationOptions
} from "./fftProcessor.js";

// Type exports
export type {
  Point,
  Shape,
  ShapeConfig,
  RotationState,
  ValidationResult,
  FFTResult,
  FrequencyComponent
} from "./types.js";
