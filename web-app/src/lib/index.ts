// place files you want to import through the `$lib` alias in this folder.
export { cn } from "./utils.js";

// Shape engine exports
export {
  generateShapePoints,
  validateShapeParams,
  countWiggles,
  validateFrequencyInput
} from "./shapeEngine.js";

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
