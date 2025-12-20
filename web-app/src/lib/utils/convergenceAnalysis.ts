/**
 * Convergence Analysis Utilities
 * 
 * Computes convergence scores between analysis states from two audio sources.
 * Identifies matching geometric patterns and frequency structures.
 * 
 * Phase 2: Task 2.5
 */

import type { Shape, ShapeConfig, AnalysisState } from '$lib/types';
import { computeSimilarityScore, computeFrequencyOverlap } from './shapeComparison';

/**
 * Result of convergence analysis between two state sets
 */
export interface ConvergenceResult {
    /** Overall convergence score (0-1) */
    score: number;
    /** Label describing the convergence level */
    label: 'Low' | 'Moderate' | 'High' | 'Very High';
    /** Best matching pairs of states */
    matchingPairs: Array<{
        stateA: AnalysisState;
        stateB: AnalysisState;
        similarity: number;
    }>;
    /** Frequency patterns that appear in both sources */
    commonFrequencies: number[];
}

/**
 * Computes convergence score between two sets of analysis states
 * 
 * Convergence is defined as: min(D(SᵢA, SⱼB)) for all i, j
 * Where D is a geometric distance/similarity metric
 * 
 * @param statesA - Analysis states from source A
 * @param statesB - Analysis states from source B
 * @param config - Shape configuration for comparison
 * @returns Convergence result with score and matching pairs
 */
export function computeConvergenceScore(
    statesA: AnalysisState[],
    statesB: AnalysisState[],
    config: ShapeConfig
): ConvergenceResult {
    if (statesA.length === 0 || statesB.length === 0) {
        return {
            score: 0,
            label: 'Low',
            matchingPairs: [],
            commonFrequencies: []
        };
    }

    // Compute pairwise similarities
    const pairs: Array<{
        stateA: AnalysisState;
        stateB: AnalysisState;
        similarity: number;
    }> = [];

    for (const stateA of statesA) {
        for (const stateB of statesB) {
            const geometricSimilarity = computeSimilarityScore(
                stateA.shapes,
                stateB.shapes,
                config,
                100 // Lower resolution for performance
            );

            const frequencyOverlap = computeFrequencyOverlap(
                stateA.shapes,
                stateB.shapes,
                20 // 20Hz tolerance
            );

            // Combined similarity (weighted average)
            const similarity = geometricSimilarity * 0.6 + frequencyOverlap.overlapScore * 0.4;

            pairs.push({ stateA, stateB, similarity });
        }
    }

    // Sort by similarity descending
    pairs.sort((a, b) => b.similarity - a.similarity);

    // Take best matches (one per state, greedy)
    const usedA = new Set<string>();
    const usedB = new Set<string>();
    const matchingPairs: typeof pairs = [];

    for (const pair of pairs) {
        if (!usedA.has(pair.stateA.id) && !usedB.has(pair.stateB.id)) {
            matchingPairs.push(pair);
            usedA.add(pair.stateA.id);
            usedB.add(pair.stateB.id);
        }
    }

    // Overall score is the average of best matches
    const score = matchingPairs.length > 0
        ? matchingPairs.reduce((sum, p) => sum + p.similarity, 0) / matchingPairs.length
        : 0;

    // Find common frequencies
    const allFreqsA = statesA.flatMap(s => s.shapes.map(shape => shape.fq));
    const allFreqsB = statesB.flatMap(s => s.shapes.map(shape => shape.fq));
    const commonFrequencies = findCommonFrequencies(allFreqsA, allFreqsB, 20);

    return {
        score,
        label: getConvergenceLabel(score),
        matchingPairs,
        commonFrequencies
    };
}

/**
 * Gets a human-readable label for a convergence score
 */
function getConvergenceLabel(score: number): 'Low' | 'Moderate' | 'High' | 'Very High' {
    if (score >= 0.8) return 'Very High';
    if (score >= 0.6) return 'High';
    if (score >= 0.4) return 'Moderate';
    return 'Low';
}

/**
 * Finds frequencies that appear in both sets (within tolerance)
 */
function findCommonFrequencies(
    freqsA: number[],
    freqsB: number[],
    tolerance: number
): number[] {
    const common: number[] = [];
    const matched = new Set<number>();

    for (const freqA of freqsA) {
        for (const freqB of freqsB) {
            if (Math.abs(freqA - freqB) <= tolerance && !matched.has(freqA)) {
                common.push(Math.round((freqA + freqB) / 2)); // Average
                matched.add(freqA);
            }
        }
    }

    return [...new Set(common)].sort((a, b) => a - b);
}

/**
 * Computes real-time convergence between current shapes
 * 
 * @param shapesA - Current shapes from source A
 * @param shapesB - Current shapes from source B
 * @param config - Shape configuration
 * @returns Quick convergence score (0-1)
 */
export function computeQuickConvergence(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig
): {
    score: number;
    label: string;
} {
    if (shapesA.length === 0 || shapesB.length === 0) {
        return { score: 0, label: 'No Data' };
    }

    const similarity = computeSimilarityScore(shapesA, shapesB, config, 80);

    return {
        score: similarity,
        label: `${Math.round(similarity * 100)}%`
    };
}

/**
 * Identifies which shapes from A have counterparts in B
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param config - Shape configuration
 * @param threshold - Minimum similarity to consider a match
 * @returns Matching shape IDs
 */
export function identifyMatchingShapes(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig,
    threshold: number = 0.5
): {
    matchedA: Set<string>;
    matchedB: Set<string>;
} {
    const matchedA = new Set<string>();
    const matchedB = new Set<string>();

    for (const shapeA of shapesA) {
        for (const shapeB of shapesB) {
            const similarity = computeSimilarityScore([shapeA], [shapeB], config, 50);

            if (similarity >= threshold) {
                matchedA.add(shapeA.id);
                matchedB.add(shapeB.id);
            }
        }
    }

    return { matchedA, matchedB };
}
