/**
 * Animation Loop - Rotation control for frequency shapes
 * 
 * This module implements a requestAnimationFrame-based animation loop
 * for rotating shapes. It handles:
 * - Continuous loop rotation (clockwise/counter-clockwise)
 * - Fixed angle rotation with automatic stop
 * - Smooth animation with delta-time based updates
 * 
 * Requirements: 3.3, 3.4, 3.5, 3.6
 */

import type { RotationState } from './types';

/**
 * Callback function type for updating shape phi values
 */
export type PhiUpdateCallback = (deltaPhi: number) => void;

/**
 * Callback function type for stopping rotation
 */
export type StopCallback = () => void;

/**
 * Animation loop state
 */
interface AnimationState {
  /** requestAnimationFrame ID for cancellation */
  animationFrameId: number | null;
  /** Timestamp of the last frame */
  lastTimestamp: number | null;
  /** Accumulated rotation for fixed mode (radians) */
  accumulatedRotation: number;
  /** Target rotation for fixed mode (radians) */
  targetRotation: number;
  /** Whether animation is currently running */
  isRunning: boolean;
}

/**
 * Creates an animation loop controller for shape rotation
 * 
 * The controller manages the animation lifecycle and provides
 * functions to start and stop rotation with different modes.
 */
function createAnimationLoop() {
  // Internal state
  const state: AnimationState = {
    animationFrameId: null,
    lastTimestamp: null,
    accumulatedRotation: 0,
    targetRotation: 0,
    isRunning: false
  };

  // Callbacks set by the consumer
  let onPhiUpdate: PhiUpdateCallback | null = null;
  let onStop: StopCallback | null = null;

  // Current rotation configuration
  let currentDirection: 'clockwise' | 'counterclockwise' = 'clockwise';
  let currentMode: 'loop' | 'fixed' = 'loop';
  let currentSpeed: number = 1.0; // radians per second

  /**
   * The main animation frame callback
   * 
   * @param timestamp - High-resolution timestamp from requestAnimationFrame
   */
  function animationFrame(timestamp: number): void {
    if (!state.isRunning) {
      return;
    }

    // Calculate delta time
    if (state.lastTimestamp === null) {
      state.lastTimestamp = timestamp;
      state.animationFrameId = requestAnimationFrame(animationFrame);
      return;
    }

    const deltaTime = (timestamp - state.lastTimestamp) / 1000; // Convert to seconds
    state.lastTimestamp = timestamp;

    // Calculate delta phi based on direction and speed
    // Clockwise: decrease phi (negative delta)
    // Counter-clockwise: increase phi (positive delta)
    let deltaPhi = currentSpeed * deltaTime;
    if (currentDirection === 'clockwise') {
      deltaPhi = -deltaPhi;
    }

    // Handle fixed mode - check if we've reached the target
    if (currentMode === 'fixed') {
      const absDelta = Math.abs(deltaPhi);
      const remaining = state.targetRotation - state.accumulatedRotation;

      if (remaining <= absDelta) {
        // We've reached or would overshoot the target
        // Apply only the remaining rotation
        const finalDelta = currentDirection === 'clockwise'
          ? -remaining
          : remaining;

        if (onPhiUpdate) {
          onPhiUpdate(finalDelta);
        }

        // Stop the animation
        stopRotation();
        return;
      }

      // Track accumulated rotation
      state.accumulatedRotation += absDelta;
    }

    // Apply the rotation
    if (onPhiUpdate) {
      onPhiUpdate(deltaPhi);
    }

    // Continue the animation loop
    state.animationFrameId = requestAnimationFrame(animationFrame);
  }

  /**
   * Starts the rotation animation
   * 
   * @param direction - 'clockwise' decreases phi, 'counterclockwise' increases phi
   * @param mode - 'loop' for continuous rotation, 'fixed' for specific angle
   * @param targetAngle - Target angle in degrees (required for fixed mode)
   * @param speed - Angular velocity in radians per second
   * 
   * Requirements: 3.3, 3.4, 3.5
   */
  function startRotation(
    direction: 'clockwise' | 'counterclockwise',
    mode: 'loop' | 'fixed',
    targetAngle?: number,
    speed: number = 1.0
  ): void {
    // Stop any existing animation
    if (state.isRunning) {
      cancelAnimationFrame(state.animationFrameId!);
    }

    // Reset state
    state.lastTimestamp = null;
    state.accumulatedRotation = 0;
    state.isRunning = true;

    // Set configuration
    currentDirection = direction;
    currentMode = mode;
    currentSpeed = Math.max(0, speed);

    // For fixed mode, convert degrees to radians
    // Requirement 3.6: Convert specified degrees to radians for φ
    if (mode === 'fixed' && targetAngle !== undefined) {
      state.targetRotation = Math.abs(targetAngle) * (Math.PI / 180);
    } else {
      state.targetRotation = 0;
    }

    // Start the animation loop
    state.animationFrameId = requestAnimationFrame(animationFrame);
  }

  /**
   * Stops the rotation animation
   * 
   * Requirements: 3.5
   */
  function stopRotation(): void {
    if (state.animationFrameId !== null) {
      cancelAnimationFrame(state.animationFrameId);
      state.animationFrameId = null;
    }

    state.isRunning = false;
    state.lastTimestamp = null;
    state.accumulatedRotation = 0;

    // Notify consumer that rotation has stopped
    if (onStop) {
      onStop();
    }
  }

  /**
   * Updates the rotation speed during animation
   * 
   * @param speed - New angular velocity in radians per second
   * 
   * Requirements: 3.9
   */
  function setSpeed(speed: number): void {
    currentSpeed = Math.max(0, speed);
  }

  /**
   * Updates the rotation direction during animation
   * 
   * @param direction - New rotation direction
   */
  function setDirection(direction: 'clockwise' | 'counterclockwise'): void {
    currentDirection = direction;
  }

  /**
   * Sets the callback for phi updates
   * 
   * @param callback - Function called with delta phi on each frame
   */
  function setPhiUpdateCallback(callback: PhiUpdateCallback): void {
    onPhiUpdate = callback;
  }

  /**
   * Sets the callback for when rotation stops
   * 
   * @param callback - Function called when rotation stops
   */
  function setStopCallback(callback: StopCallback): void {
    onStop = callback;
  }

  /**
   * Returns whether the animation is currently running
   */
  function isAnimating(): boolean {
    return state.isRunning;
  }

  /**
   * Gets the current accumulated rotation (for fixed mode)
   */
  function getAccumulatedRotation(): number {
    return state.accumulatedRotation;
  }

  /**
   * Gets the target rotation (for fixed mode)
   */
  function getTargetRotation(): number {
    return state.targetRotation;
  }

  return {
    startRotation,
    stopRotation,
    setSpeed,
    setDirection,
    setPhiUpdateCallback,
    setStopCallback,
    isAnimating,
    getAccumulatedRotation,
    getTargetRotation
  };
}

/**
 * Singleton instance of the animation loop controller
 */
export const animationLoop = createAnimationLoop();

/**
 * Export the factory function for testing or multiple instances
 */
export { createAnimationLoop };

/**
 * Export type for the animation loop controller
 */
export type AnimationLoop = ReturnType<typeof createAnimationLoop>;

/**
 * Per-shape animation update function
 * 
 * Updates a shape's phi value based on its AnimationOverride settings.
 * This allows each shape to have independent animation parameters.
 * 
 * Phase 3: Task 3.3
 * 
 * @param shape - The shape to update
 * @param deltaTime - Time elapsed since last frame in seconds
 * @param globalRotation - Global rotation settings (fallback)
 * @returns Updated phi value
 */
export function updateShapePhiWithOverride(
  shape: { phi: number; animationOverride?: import('./types').AnimationOverride },
  deltaTime: number,
  globalRotation: import('./types').RotationState
): number {
  const override = shape.animationOverride;

  // Get animation parameters (shape-specific or global)
  const mode = override?.mode ?? (globalRotation.isAnimating ? 'continuous' : 'off');
  const direction = override?.direction ?? (globalRotation.direction === 'clockwise' ? 'cw' : 'ccw');
  const speed = override?.speed ?? globalRotation.speed;

  // Skip if animation is off or direction is none
  if (mode === 'off') {
    return shape.phi;
  }

  if (direction === 'none') {
    return shape.phi;
  }

  // Calculate delta phi
  let deltaPhi = speed * deltaTime;

  // Apply direction
  if (direction === 'cw') {
    deltaPhi = -deltaPhi;
  }

  // For 'once' mode, we'd need to track completion state per-shape
  // This is a simplified implementation - full implementation would
  // require tracking completed state on the shape itself

  // Return updated phi (wrapped to 0..2π range)
  const newPhi = (shape.phi + deltaPhi) % (2 * Math.PI);
  return newPhi < 0 ? newPhi + 2 * Math.PI : newPhi;
}

/**
 * Updates an array of shapes with per-shape animation
 * 
 * @param shapes - Array of shapes to update
 * @param deltaTime - Time elapsed since last frame in seconds  
 * @param globalRotation - Global rotation settings (fallback)
 * @returns Updated shapes array with new phi values
 */
export function updateShapesWithAnimation<T extends { phi: number; animationOverride?: import('./types').AnimationOverride }>(
  shapes: T[],
  deltaTime: number,
  globalRotation: import('./types').RotationState
): T[] {
  return shapes.map(shape => ({
    ...shape,
    phi: updateShapePhiWithOverride(shape, deltaTime, globalRotation)
  }));
}
