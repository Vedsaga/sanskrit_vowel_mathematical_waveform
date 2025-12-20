/**
 * Frequency Analysis - Number-theoretic analysis of frequency components
 * 
 * This module provides functions for:
 * - Prime number detection
 * - Golden ratio relationship detection
 * - Harmonic order identification
 * - Badge assignment for frequency components
 * 
 * Phase 1: Task 1.3
 */

import type { FrequencyComponent } from '../types';

/**
 * Badge types for frequency components
 */
export type FrequencyBadge =
    | 'P'    // Prime fq
    | 'E'    // Even fq
    | 'O'    // Odd fq (non-prime)
    | 'H2' | 'H3' | 'H4' | 'H5' | 'H6' | 'H7' | 'H8'  // Harmonic orders
    | 'φ';   // Golden ratio related

/**
 * Checks if a number is prime.
 * 
 * @param n - Number to check
 * @returns true if n is prime
 */
export function isPrime(n: number): boolean {
    if (!Number.isInteger(n) || n < 2) return false;
    if (n === 2) return true;
    if (n % 2 === 0) return false;

    const sqrt = Math.sqrt(n);
    for (let i = 3; i <= sqrt; i += 2) {
        if (n % i === 0) return false;
    }

    return true;
}

/**
 * The golden ratio (φ)
 */
export const PHI = (1 + Math.sqrt(5)) / 2; // ≈ 1.618

/**
 * Checks if two frequencies are related by the golden ratio.
 * 
 * @param freqA - First frequency
 * @param freqB - Second frequency
 * @param tolerance - Tolerance as decimal (default: 0.02 = 2%)
 * @returns true if ratio is approximately φ or 1/φ
 */
export function isGoldenRatioRelated(
    freqA: number,
    freqB: number,
    tolerance: number = 0.02
): boolean {
    if (freqA <= 0 || freqB <= 0) return false;

    const ratio = freqA > freqB ? freqA / freqB : freqB / freqA;

    // Check if ratio is close to φ (1.618...)
    const diff = Math.abs(ratio - PHI);
    return diff <= PHI * tolerance;
}

/**
 * Gets the harmonic order of a frequency relative to a fundamental.
 * 
 * @param freq - Frequency to check
 * @param fundamental - Fundamental frequency
 * @param tolerance - Tolerance as decimal (default: 0.05 = 5%)
 * @returns Harmonic order (2-8) or null if not a harmonic
 */
export function getHarmonicOrder(
    freq: number,
    fundamental: number,
    tolerance: number = 0.05
): number | null {
    if (fundamental <= 0 || freq <= 0) return null;

    const ratio = freq / fundamental;

    // Check harmonics 2-8
    for (let n = 2; n <= 8; n++) {
        if (Math.abs(ratio - n) <= n * tolerance) {
            return n;
        }
    }

    return null;
}

/**
 * Finds all golden ratio pairs in a set of components
 */
export function findGoldenRatioPairs(
    components: FrequencyComponent[],
    tolerance: number = 0.02
): Array<[string, string]> {
    const pairs: Array<[string, string]> = [];

    for (let i = 0; i < components.length; i++) {
        for (let j = i + 1; j < components.length; j++) {
            if (isGoldenRatioRelated(
                components[i].frequencyHz,
                components[j].frequencyHz,
                tolerance
            )) {
                pairs.push([components[i].id, components[j].id]);
            }
        }
    }

    return pairs;
}

/**
 * Analyzes frequency relationships and assigns badges to components.
 * 
 * Badges:
 * - P: fq is prime
 * - E: fq is even
 * - O: fq is odd (non-prime)
 * - H2-H8: Harmonic of lower frequency
 * - φ: Golden ratio relationship with another frequency
 * 
 * @param components - Frequency components to analyze
 * @returns Components with badges property populated
 */
export function analyzeFrequencyRelationships(
    components: FrequencyComponent[]
): FrequencyComponent[] {
    if (components.length === 0) return [];

    // Find the fundamental (lowest significant frequency)
    const sortedByMagnitude = [...components].sort((a, b) => b.magnitude - a.magnitude);
    const sortedByFrequency = [...components].sort((a, b) => a.frequencyHz - b.frequencyHz);

    // Use the lowest frequency as potential fundamental
    const fundamental = sortedByFrequency[0]?.frequencyHz || 0;

    // Find golden ratio pairs
    const goldenPairs = findGoldenRatioPairs(components);
    const inGoldenPair = new Set<string>();
    goldenPairs.forEach(([a, b]) => {
        inGoldenPair.add(a);
        inGoldenPair.add(b);
    });

    return components.map(component => {
        const badges: FrequencyBadge[] = [];
        const fq = component.fq;

        // Prime check
        if (isPrime(fq)) {
            badges.push('P');
        } else if (fq % 2 === 0) {
            badges.push('E');
        } else {
            badges.push('O');
        }

        // Harmonic check (relative to fundamental)
        if (fundamental > 0 && component.frequencyHz > fundamental) {
            const harmonicOrder = getHarmonicOrder(component.frequencyHz, fundamental);
            if (harmonicOrder !== null && harmonicOrder >= 2 && harmonicOrder <= 8) {
                badges.push(`H${harmonicOrder}` as FrequencyBadge);
            }
        }

        // Golden ratio check
        if (inGoldenPair.has(component.id)) {
            badges.push('φ');
        }

        return {
            ...component,
            badges
        };
    });
}

/**
 * Filters components by badge
 */
export function filterByBadge(
    components: FrequencyComponent[],
    badge: FrequencyBadge | 'all'
): FrequencyComponent[] {
    if (badge === 'all') return components;

    return components.filter(c => c.badges?.includes(badge));
}

/**
 * Gets badge display info (label and color)
 */
export function getBadgeInfo(badge: FrequencyBadge): { label: string; color: string } {
    switch (badge) {
        case 'P':
            return { label: 'Prime', color: '#f59e0b' }; // Amber
        case 'E':
            return { label: 'Even', color: '#6b7280' }; // Gray
        case 'O':
            return { label: 'Odd', color: '#9ca3af' }; // Light gray
        case 'H2':
        case 'H3':
        case 'H4':
        case 'H5':
        case 'H6':
        case 'H7':
        case 'H8':
            return { label: badge, color: '#3b82f6' }; // Blue
        case 'φ':
            return { label: 'φ', color: '#eab308' }; // Gold
        default:
            return { label: badge, color: '#6b7280' };
    }
}

/**
 * Gets statistics about badges in a component set
 */
export function getBadgeStats(components: FrequencyComponent[]): Record<FrequencyBadge, number> {
    const stats: Partial<Record<FrequencyBadge, number>> = {};

    for (const component of components) {
        if (!component.badges) continue;
        for (const badge of component.badges) {
            stats[badge] = (stats[badge] || 0) + 1;
        }
    }

    return stats as Record<FrequencyBadge, number>;
}
