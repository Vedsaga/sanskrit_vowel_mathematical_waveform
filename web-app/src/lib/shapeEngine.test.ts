/**
 * Unit tests for the Shape Engine
 */

import { describe, it, expect } from 'vitest';
import {
  generateShapePoints,
  validateShapeParams,
  countWiggles,
  validateFrequencyInput
} from './shapeEngine';

describe('generateShapePoints', () => {
  it('generates a pure circle for fq=1 (no wiggles)', () => {
    const points = generateShapePoints(1, 100, 20, 0, 360);
    
    // For fq=1, wiggleCount = 0, so r(θ) = R + A·sin(0) = R
    // All points should be at distance R from origin
    for (const point of points) {
      const radius = Math.sqrt(point.x * point.x + point.y * point.y);
      expect(radius).toBeCloseTo(100, 5);
    }
  });

  it('generates correct number of points', () => {
    const resolution = 360;
    const points = generateShapePoints(3, 100, 20, 0, resolution);
    
    // Should have resolution + 1 points (0 to resolution inclusive)
    expect(points.length).toBe(resolution + 1);
  });

  it('generates closed shape (first and last points match)', () => {
    const points = generateShapePoints(5, 100, 20, 0, 360);
    
    const first = points[0];
    const last = points[points.length - 1];
    
    expect(first.x).toBeCloseTo(last.x, 5);
    expect(first.y).toBeCloseTo(last.y, 5);
  });

  it('applies phase offset correctly', () => {
    const pointsNoPhase = generateShapePoints(3, 100, 20, 0, 360);
    const pointsWithPhase = generateShapePoints(3, 100, 20, Math.PI / 2, 360);
    
    // Points should be different due to phase offset
    expect(pointsNoPhase[0].x).not.toBeCloseTo(pointsWithPhase[0].x, 1);
  });
});

describe('validateShapeParams', () => {
  it('accepts valid parameters', () => {
    const result = validateShapeParams(5, 100, 20, 360);
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('rejects non-integer frequency', () => {
    const result = validateShapeParams(3.5, 100, 20, 360);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be an integer');
  });

  it('rejects frequency less than 1', () => {
    const result = validateShapeParams(0, 100, 20, 360);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be at least 1');
  });

  it('rejects negative frequency', () => {
    const result = validateShapeParams(-5, 100, 20, 360);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be at least 1');
  });

  it('rejects amplitude >= radius', () => {
    const result = validateShapeParams(5, 100, 100, 360);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Wiggle amplitude must be less than base radius');
  });

  it('rejects resolution outside valid range', () => {
    const resultLow = validateShapeParams(5, 100, 20, 100);
    expect(resultLow.valid).toBe(false);
    expect(resultLow.errors).toContain('Resolution must be between 360 and 2048');

    const resultHigh = validateShapeParams(5, 100, 20, 3000);
    expect(resultHigh.valid).toBe(false);
    expect(resultHigh.errors).toContain('Resolution must be between 360 and 2048');
  });

  it('rejects non-positive radius', () => {
    const result = validateShapeParams(5, 0, 20, 360);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Base radius must be positive');
  });
});

describe('countWiggles', () => {
  it('counts 0 wiggles for fq=1', () => {
    const points = generateShapePoints(1, 100, 20, 0, 720);
    const wiggles = countWiggles(points);
    expect(wiggles).toBe(0);
  });

  it('counts 1 wiggle for fq=2', () => {
    const points = generateShapePoints(2, 100, 20, 0, 720);
    const wiggles = countWiggles(points);
    expect(wiggles).toBe(1);
  });

  it('counts 2 wiggles for fq=3', () => {
    const points = generateShapePoints(3, 100, 20, 0, 720);
    const wiggles = countWiggles(points);
    expect(wiggles).toBe(2);
  });

  it('counts (fq-1) wiggles for various frequencies', () => {
    for (const fq of [4, 5, 6, 7, 8]) {
      const points = generateShapePoints(fq, 100, 20, 0, 720);
      const wiggles = countWiggles(points);
      expect(wiggles).toBe(fq - 1);
    }
  });
});

describe('validateFrequencyInput', () => {
  it('accepts valid integer frequency', () => {
    const result = validateFrequencyInput(5);
    expect(result.valid).toBe(true);
  });

  it('accepts string that parses to valid integer', () => {
    const result = validateFrequencyInput('5');
    expect(result.valid).toBe(true);
  });

  it('rejects empty string', () => {
    const result = validateFrequencyInput('');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency is required');
  });

  it('rejects null', () => {
    const result = validateFrequencyInput(null);
    expect(result.valid).toBe(false);
  });

  it('rejects undefined', () => {
    const result = validateFrequencyInput(undefined);
    expect(result.valid).toBe(false);
  });

  it('rejects non-numeric string', () => {
    const result = validateFrequencyInput('abc');
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be a valid number');
  });

  it('rejects decimal numbers', () => {
    const result = validateFrequencyInput(3.5);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be an integer');
  });

  it('rejects zero', () => {
    const result = validateFrequencyInput(0);
    expect(result.valid).toBe(false);
    expect(result.errors).toContain('Frequency must be at least 1');
  });
});
