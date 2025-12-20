/**
 * Guna Analysis - Stability and quality metrics for audio analysis
 * 
 * This module provides functions for:
 * - Calculating temporal stability of geometries
 * - Checking energy invariance
 * - Calculating transient scores
 * 
 * Based on the Guna philosophy: stable, invariant patterns indicate
 * fundamental qualities (Gunas) of the sound.
 * 
 * Phase 1: Task 1.5
 */

import type { Shape, FFTResult, ShapeConfig } from '../types';

/**
 * Guna analysis results
 */
export interface GunaMetrics {
    /** Temporal stability score (0-1, higher = more stable) */
    stabilityScore: number;
    /** Stability interpretation */
    stabilityLabel: 'Transient' | 'Variable' | 'Stable' | 'Very Stable';
    /** Whether geometry is invariant to amplitude changes */
    energyInvariant: boolean;
    /** Transient score (0-1, higher = more transient) */
    transientScore: number;
    /** Transient interpretation */
    transientLabel: 'Carrier Dominant' | 'Mixed' | 'Transient Heavy';
}

/**
 * Calculates the distance between two geometry snapshots.
 * 
 * D(Si, Sj) = (1/Nθ) × Σ|ri(θ) - rj(θ)|
 * 
 * @param radiiA - Radii array for first geometry
 * @param radiiB - Radii array for second geometry
 * @returns Average absolute difference (normalized)
 */
function geometryDistance(radiiA: number[], radiiB: number[]): number {
    const n = Math.min(radiiA.length, radiiB.length);
    if (n === 0) return 1;

    let totalDiff = 0;
    for (let i = 0; i < n; i++) {
        totalDiff += Math.abs(radiiA[i] - radiiB[i]);
    }

    return totalDiff / n;
}

/**
 * Generates radii array for a shape at given resolution.
 * 
 * r(θ) = R + A·sin((fq-1)·θ + φ)
 */
function generateRadii(shape: Shape, config: ShapeConfig): number[] {
    const radii: number[] = [];
    const { fq, R, phi } = shape;
    const { A, resolution } = config;

    for (let i = 0; i < resolution; i++) {
        const theta = (2 * Math.PI * i) / resolution;
        const radius = R + A * Math.sin((fq - 1) * theta + phi);
        radii.push(radius);
    }

    return radii;
}

/**
 * Calculates stability score for a sequence of geometries.
 * 
 * Stability = 1 - (1/M) × Σ D(Si, Si+1)
 * 
 * High stability indicates the geometry is consistent over time,
 * which is characteristic of vowel-like sounds.
 * 
 * @param shapes - Array of shapes representing temporal sequence
 * @param config - Shape configuration
 * @returns Stability score (0-1)
 */
export function calculateStabilityScore(
    shapes: Shape[],
    config: ShapeConfig
): number {
    if (shapes.length < 2) {
        return 1; // Single shape is perfectly stable
    }

    // Generate radii for all shapes
    const radiiSequence = shapes.map(s => generateRadii(s, config));

    // Calculate distances between consecutive shapes
    let totalDistance = 0;
    for (let i = 1; i < radiiSequence.length; i++) {
        totalDistance += geometryDistance(radiiSequence[i - 1], radiiSequence[i]);
    }

    const avgDistance = totalDistance / (radiiSequence.length - 1);

    // Normalize by expected maximum distance (2 * A)
    const maxDistance = 2 * config.A;
    const normalizedDistance = Math.min(1, avgDistance / maxDistance);

    return Math.max(0, 1 - normalizedDistance);
}

/**
 * Gets stability label from score
 */
function getStabilityLabel(score: number): GunaMetrics['stabilityLabel'] {
    if (score >= 0.85) return 'Very Stable';
    if (score >= 0.65) return 'Stable';
    if (score >= 0.4) return 'Variable';
    return 'Transient';
}

/**
 * Checks if geometry is invariant to energy (amplitude) changes.
 * 
 * Compares raw shapes to normalized shapes. If they're similar,
 * the geometry is energy-invariant.
 * 
 * @param rawShapes - Shapes with original amplitudes
 * @param normalizedShapes - Shapes with normalized amplitudes
 * @param epsilon - Tolerance threshold (default: 0.1)
 * @returns true if geometry is energy-invariant
 */
export function checkEnergyInvariance(
    rawShapes: Shape[],
    normalizedShapes: Shape[],
    config: ShapeConfig,
    epsilon: number = 0.1
): boolean {
    if (rawShapes.length === 0 || normalizedShapes.length === 0) {
        return true; // No data to compare
    }

    // Compare corresponding shapes
    const pairCount = Math.min(rawShapes.length, normalizedShapes.length);
    let totalDistance = 0;

    for (let i = 0; i < pairCount; i++) {
        const rawRadii = generateRadii(rawShapes[i], config);
        const normRadii = generateRadii(normalizedShapes[i], config);
        totalDistance += geometryDistance(rawRadii, normRadii);
    }

    const avgDistance = totalDistance / pairCount;
    const maxDistance = 2 * config.A;
    const normalizedDistance = avgDistance / maxDistance;

    return normalizedDistance < epsilon;
}

/**
 * Calculates transient score from spectral flux.
 * 
 * Uses pre-computed spectral flux values from fftProcessor.
 * 
 * Flux(t) = Σf (|X(t,f)| - |X(t-1,f)|)²
 * 
 * @param fluxValues - Normalized spectral flux values (from computeSpectralFlux)
 * @returns Transient score (0-1)
 */
export function calculateTransientScore(fluxValues: number[]): number {
    if (fluxValues.length === 0) return 0;

    // Average flux is the transient score
    const sum = fluxValues.reduce((a, b) => a + b, 0);
    return sum / fluxValues.length;
}

/**
 * Gets transient label from score
 */
function getTransientLabel(score: number): GunaMetrics['transientLabel'] {
    if (score >= 0.6) return 'Transient Heavy';
    if (score >= 0.3) return 'Mixed';
    return 'Carrier Dominant';
}

/**
 * Performs full Guna analysis on audio/shape data.
 * 
 * @param shapes - Shapes from temporal analysis
 * @param config - Shape configuration
 * @param fluxValues - Optional spectral flux values
 * @returns Complete Guna metrics
 */
export function analyzeGuna(
    shapes: Shape[],
    config: ShapeConfig,
    fluxValues: number[] = []
): GunaMetrics {
    const stabilityScore = calculateStabilityScore(shapes, config);
    const transientScore = calculateTransientScore(fluxValues);

    // For energy invariance, we'd need both raw and normalized shapes
    // For now, assume energy invariant if stability is high
    const energyInvariant = stabilityScore >= 0.7;

    return {
        stabilityScore,
        stabilityLabel: getStabilityLabel(stabilityScore),
        energyInvariant,
        transientScore,
        transientLabel: getTransientLabel(transientScore)
    };
}

/**
 * Formats stability score for display
 */
export function formatStabilityScore(score: number): string {
    return `${Math.round(score * 100)}%`;
}

/**
 * Gets color for stability display
 */
export function getStabilityColor(score: number): string {
    if (score >= 0.85) return '#22c55e'; // Green
    if (score >= 0.65) return '#84cc16'; // Lime
    if (score >= 0.4) return '#f59e0b'; // Amber
    return '#ef4444'; // Red
}

/**
 * Gets color for transient display
 */
export function getTransientColor(score: number): string {
    if (score >= 0.6) return '#ef4444'; // Red (high transients)
    if (score >= 0.3) return '#f59e0b'; // Amber
    return '#22c55e'; // Green (carrier dominant)
}
