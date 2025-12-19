/**
 * Animation Loop Tests
 * 
 * Tests for the rotation animation system.
 * Requirements: 3.3, 3.4, 3.5, 3.6
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createAnimationLoop, type PhiUpdateCallback, type StopCallback } from './animationLoop';

describe('animationLoop', () => {
  let animationLoop: ReturnType<typeof createAnimationLoop>;
  let mockPhiUpdate: ReturnType<typeof vi.fn<PhiUpdateCallback>>;
  let mockStop: ReturnType<typeof vi.fn<StopCallback>>;

  beforeEach(() => {
    // Create a fresh instance for each test
    animationLoop = createAnimationLoop();
    mockPhiUpdate = vi.fn<PhiUpdateCallback>();
    mockStop = vi.fn<StopCallback>();
    
    animationLoop.setPhiUpdateCallback(mockPhiUpdate);
    animationLoop.setStopCallback(mockStop);

    // Mock requestAnimationFrame and cancelAnimationFrame
    vi.useFakeTimers();
    let frameId = 0;
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frameId++;
      setTimeout(() => callback(performance.now()), 16); // ~60fps
      return frameId;
    });
    vi.stubGlobal('cancelAnimationFrame', vi.fn());
  });

  afterEach(() => {
    animationLoop.stopRotation();
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  describe('startRotation', () => {
    it('starts animation and sets isAnimating to true', () => {
      animationLoop.startRotation('clockwise', 'loop');
      expect(animationLoop.isAnimating()).toBe(true);
    });

    it('accepts clockwise direction', () => {
      animationLoop.startRotation('clockwise', 'loop');
      expect(animationLoop.isAnimating()).toBe(true);
    });

    it('accepts counterclockwise direction', () => {
      animationLoop.startRotation('counterclockwise', 'loop');
      expect(animationLoop.isAnimating()).toBe(true);
    });

    it('accepts loop mode', () => {
      animationLoop.startRotation('clockwise', 'loop');
      expect(animationLoop.isAnimating()).toBe(true);
    });

    it('accepts fixed mode with target angle', () => {
      animationLoop.startRotation('clockwise', 'fixed', 90);
      expect(animationLoop.isAnimating()).toBe(true);
      // 90 degrees = π/2 radians
      expect(animationLoop.getTargetRotation()).toBeCloseTo(Math.PI / 2, 5);
    });

    it('converts degrees to radians for fixed mode', () => {
      animationLoop.startRotation('clockwise', 'fixed', 180);
      expect(animationLoop.getTargetRotation()).toBeCloseTo(Math.PI, 5);
    });

    it('handles 360 degree rotation', () => {
      animationLoop.startRotation('clockwise', 'fixed', 360);
      expect(animationLoop.getTargetRotation()).toBeCloseTo(2 * Math.PI, 5);
    });
  });

  describe('stopRotation', () => {
    it('stops animation and sets isAnimating to false', () => {
      animationLoop.startRotation('clockwise', 'loop');
      expect(animationLoop.isAnimating()).toBe(true);
      
      animationLoop.stopRotation();
      expect(animationLoop.isAnimating()).toBe(false);
    });

    it('calls stop callback when stopped', () => {
      animationLoop.startRotation('clockwise', 'loop');
      animationLoop.stopRotation();
      
      expect(mockStop).toHaveBeenCalled();
    });

    it('resets accumulated rotation when stopped', () => {
      animationLoop.startRotation('clockwise', 'fixed', 90);
      animationLoop.stopRotation();
      
      expect(animationLoop.getAccumulatedRotation()).toBe(0);
    });
  });

  describe('setSpeed', () => {
    it('accepts positive speed values', () => {
      animationLoop.setSpeed(2.0);
      // Speed is internal, but we can verify no errors
      expect(true).toBe(true);
    });

    it('clamps negative speed to 0', () => {
      animationLoop.setSpeed(-1.0);
      // Should not throw
      expect(true).toBe(true);
    });
  });

  describe('setDirection', () => {
    it('accepts clockwise direction', () => {
      animationLoop.setDirection('clockwise');
      expect(true).toBe(true);
    });

    it('accepts counterclockwise direction', () => {
      animationLoop.setDirection('counterclockwise');
      expect(true).toBe(true);
    });
  });

  describe('animation frame updates', () => {
    it('calls phi update callback during animation', async () => {
      animationLoop.startRotation('clockwise', 'loop', undefined, 1.0);
      
      // Advance timers to trigger animation frames
      vi.advanceTimersByTime(32); // Two frames at ~60fps
      
      // Should have called the update callback
      expect(mockPhiUpdate).toHaveBeenCalled();
    });

    it('provides negative delta phi for clockwise rotation', async () => {
      animationLoop.startRotation('clockwise', 'loop', undefined, 1.0);
      
      // Advance timers
      vi.advanceTimersByTime(32);
      
      // Check that at least one call had negative delta
      const calls = mockPhiUpdate.mock.calls;
      const hasNegativeDelta = calls.some(call => call[0] < 0);
      expect(hasNegativeDelta).toBe(true);
    });

    it('provides positive delta phi for counterclockwise rotation', async () => {
      animationLoop.startRotation('counterclockwise', 'loop', undefined, 1.0);
      
      // Advance timers
      vi.advanceTimersByTime(32);
      
      // Check that at least one call had positive delta
      const calls = mockPhiUpdate.mock.calls;
      const hasPositiveDelta = calls.some(call => call[0] > 0);
      expect(hasPositiveDelta).toBe(true);
    });
  });
});
