/**
 * Geometry Accumulator - Utilities for combining and accumulating shapes
 * 
 * This module provides functions for:
 * - Weighted accumulation of multiple geometries
 * - Computing accumulated shapes from STFT results
 * - Blending shapes for temporal stability analysis
 * 
 * Phase 0: Task 0.6
 */

import type { Shape, FFTResult, ShapeConfig, FrequencyComponent } from '../types';
import { generateShapePoints } from '../shapeEngine';

/**
 * Configuration for geometry accumulation
 */
export interface AccumulationConfig {
    /** Shape configuration (A, resolution, canvasSize) */
    shapeConfig: ShapeConfig;
    /** Base radius for shapes */
    baseRadius: number;
    /** How to combine shapes: 'weighted-average' | 'additive' | 'max' */
    blendMode: 'weighted-average' | 'additive' | 'max';
}

/**
 * Accumulated geometry result
 */
export interface AccumulatedGeometry {
    /** The blended/accumulated shape points (polar radius values) */
    radii: number[];
    /** Resolution (number of theta samples) */
    resolution: number;
    /** Weights used for each input shape */
    weights: number[];
    /** Persistence score (0-1) indicating temporal stability */
    persistenceScore: number;
}

/**
 * Computes weighted radii at each theta for a collection of shapes
 * 
 * @param shapes - Array of shapes to blend
 * @param weights - Weights for each shape (should sum to 1 for weighted-average)
 * @param config - Accumulation configuration
 * @returns AccumulatedGeometry with blended radii
 */
export function accumulateGeometries(
    shapes: Shape[],
    weights: number[],
    config: AccumulationConfig
): AccumulatedGeometry {
    const { shapeConfig, baseRadius, blendMode } = config;
    const resolution = shapeConfig.resolution;

    // Initialize result array
    const accumulatedRadii = new Float64Array(resolution);

    if (shapes.length === 0) {
        return {
            radii: Array.from(accumulatedRadii),
            resolution,
            weights: [],
            persistenceScore: 0
        };
    }

    // Normalize weights if needed
    const normalizedWeights = normalizeWeights(weights, shapes.length);

    // Calculate radius for each shape at each theta
    for (let i = 0; i < shapes.length; i++) {
        const shape = shapes[i];
        const weight = normalizedWeights[i];
        const A = shapeConfig.A;

        for (let thetaIdx = 0; thetaIdx < resolution; thetaIdx++) {
            const theta = (2 * Math.PI * thetaIdx) / resolution;
            // r(θ) = R + A·sin((fq-1)·θ + φ)
            const radius = baseRadius + A * Math.sin((shape.fq - 1) * theta + shape.phi);

            switch (blendMode) {
                case 'weighted-average':
                    accumulatedRadii[thetaIdx] += weight * radius;
                    break;
                case 'additive':
                    accumulatedRadii[thetaIdx] += radius;
                    break;
                case 'max':
                    accumulatedRadii[thetaIdx] = Math.max(accumulatedRadii[thetaIdx], radius);
                    break;
            }
        }
    }

    // For additive mode, normalize by count
    if (blendMode === 'additive' && shapes.length > 0) {
        for (let i = 0; i < resolution; i++) {
            accumulatedRadii[i] /= shapes.length;
        }
    }

    // Calculate persistence score (how similar shapes are to each other)
    const persistenceScore = calculatePersistenceScore(shapes, shapeConfig, baseRadius);

    return {
        radii: Array.from(accumulatedRadii),
        resolution,
        weights: normalizedWeights,
        persistenceScore
    };
}

/**
 * Normalizes weights to sum to 1
 */
function normalizeWeights(weights: number[], count: number): number[] {
    if (weights.length === 0 || count === 0) {
        return [];
    }

    // If no weights provided, use equal weights
    if (weights.length < count) {
        const equalWeight = 1 / count;
        return Array(count).fill(equalWeight);
    }

    const sum = weights.reduce((a, b) => a + b, 0);
    if (sum === 0) {
        const equalWeight = 1 / count;
        return weights.slice(0, count).map(() => equalWeight);
    }

    return weights.slice(0, count).map(w => w / sum);
}

/**
 * Calculates a persistence score based on shape similarity
 * Higher score = more similar shapes = more stable/persistent geometry
 * 
 * @param shapes - Array of shapes to compare
 * @param config - Shape configuration
 * @param baseRadius - Base radius for shapes
 * @returns Persistence score (0-1)
 */
function calculatePersistenceScore(
    shapes: Shape[],
    config: ShapeConfig,
    baseRadius: number
): number {
    if (shapes.length < 2) {
        return 1; // Single shape is perfectly persistent
    }

    const resolution = config.resolution;
    let totalDistance = 0;
    let pairCount = 0;

    // Compare consecutive pairs
    for (let i = 1; i < shapes.length; i++) {
        const shapeA = shapes[i - 1];
        const shapeB = shapes[i];

        let distance = 0;
        for (let thetaIdx = 0; thetaIdx < resolution; thetaIdx++) {
            const theta = (2 * Math.PI * thetaIdx) / resolution;
            const radiusA = baseRadius + config.A * Math.sin((shapeA.fq - 1) * theta + shapeA.phi);
            const radiusB = baseRadius + config.A * Math.sin((shapeB.fq - 1) * theta + shapeB.phi);
            distance += Math.abs(radiusA - radiusB);
        }

        totalDistance += distance / resolution;
        pairCount++;
    }

    if (pairCount === 0) return 1;

    // Normalize by expected maximum distance (2 * A)
    const avgDistance = totalDistance / pairCount;
    const maxDistance = 2 * config.A;

    return Math.max(0, 1 - avgDistance / maxDistance);
}

/**
 * Computes accumulated shape from STFT results
 * 
 * Takes FFT results from multiple time windows and creates
 * shapes for persistent frequency components, then accumulates them.
 * 
 * @param fftResults - Array of FFT results from STFT
 * @param config - Accumulation configuration
 * @param frequencyThreshold - Minimum magnitude to consider (0-1)
 * @returns AccumulatedGeometry representing the stable pattern
 */
export function computeAccumulatedShape(
    fftResults: FFTResult[],
    config: AccumulationConfig,
    frequencyThreshold: number = 0.3
): AccumulatedGeometry {
    if (fftResults.length === 0) {
        return {
            radii: [],
            resolution: config.shapeConfig.resolution,
            weights: [],
            persistenceScore: 0
        };
    }

    // Track which frequency bins are persistent across windows
    const binCount = fftResults[0].magnitudes.length;
    const persistenceCounts = new Array(binCount).fill(0);
    const magnitudeSums = new Array(binCount).fill(0);

    // Count how often each frequency bin exceeds threshold
    for (const result of fftResults) {
        for (let i = 0; i < Math.min(binCount, result.magnitudes.length); i++) {
            if (result.magnitudes[i] >= frequencyThreshold) {
                persistenceCounts[i]++;
                magnitudeSums[i] += result.magnitudes[i];
            }
        }
    }

    // Find persistent frequencies (appear in > 50% of windows)
    const persistenceThreshold = fftResults.length * 0.5;
    const persistentBins: { binIndex: number; avgMagnitude: number }[] = [];

    for (let i = 0; i < binCount; i++) {
        if (persistenceCounts[i] >= persistenceThreshold) {
            persistentBins.push({
                binIndex: i,
                avgMagnitude: magnitudeSums[i] / persistenceCounts[i]
            });
        }
    }

    // Create shapes from persistent frequencies
    // Map frequency bins to fq values (simplified linear mapping)
    const shapes: Shape[] = [];
    const weights: number[] = [];

    for (const { binIndex, avgMagnitude } of persistentBins) {
        // Map bin index to fq (1 to 50 range)
        const fq = Math.max(1, Math.min(50, Math.round(binIndex / 10) + 1));

        shapes.push({
            id: `acc_${binIndex}`,
            fq,
            R: config.baseRadius,
            phi: 0,
            color: '#ffffff',
            opacity: avgMagnitude,
            strokeWidth: 2,
            selected: false
        });

        weights.push(avgMagnitude);
    }

    return accumulateGeometries(shapes, weights, config);
}

/**
 * Blends two accumulated geometries together
 * 
 * @param geomA - First accumulated geometry
 * @param geomB - Second accumulated geometry
 * @param blendFactor - How much of B to blend in (0 = all A, 1 = all B)
 * @returns Blended accumulated geometry
 */
export function blendAccumulatedGeometries(
    geomA: AccumulatedGeometry,
    geomB: AccumulatedGeometry,
    blendFactor: number
): AccumulatedGeometry {
    const factor = Math.max(0, Math.min(1, blendFactor));

    if (geomA.radii.length !== geomB.radii.length) {
        // Can't blend different resolutions
        return factor < 0.5 ? geomA : geomB;
    }

    const blendedRadii = geomA.radii.map((rA, i) => {
        const rB = geomB.radii[i];
        return rA * (1 - factor) + rB * factor;
    });

    return {
        radii: blendedRadii,
        resolution: geomA.resolution,
        weights: [],
        persistenceScore: geomA.persistenceScore * (1 - factor) + geomB.persistenceScore * factor
    };
}

/**
 * Calculates the stability score between two geometry snapshots
 * 
 * Uses the formula: D(Si, Sj) = (1/Nθ) × Σ|ri(θ) - rj(θ)|
 * Stability = 1 - D
 * 
 * @param radiiA - First geometry radii
 * @param radiiB - Second geometry radii
 * @param maxRadius - Maximum expected radius for normalization
 * @returns Stability score (0-1)
 */
export function calculateStabilityScore(
    radiiA: number[],
    radiiB: number[],
    maxRadius: number
): number {
    if (radiiA.length !== radiiB.length || radiiA.length === 0) {
        return 0;
    }

    let totalDiff = 0;
    for (let i = 0; i < radiiA.length; i++) {
        totalDiff += Math.abs(radiiA[i] - radiiB[i]);
    }

    const avgDiff = totalDiff / radiiA.length;
    const normalizedDiff = avgDiff / maxRadius;

    return Math.max(0, 1 - normalizedDiff);
}
