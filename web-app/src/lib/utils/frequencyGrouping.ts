/**
 * Frequency Grouping - Utilities for grouping frequency components
 * 
 * This module provides functions for:
 * - Detecting harmonic series (fₙ = n × f₀)
 * - Clustering by temporal co-persistence
 * - Combined grouping strategies
 * 
 * Phase 1: Task 1.2
 */

import type { FrequencyComponent, FFTResult } from '../types';

/**
 * A group of related frequency components
 */
export interface FrequencyGroup {
    /** Unique group identifier */
    id: string;
    /** Group label */
    label: string;
    /** Type of grouping */
    type: 'harmonic' | 'correlated' | 'outlier';
    /** Fundamental frequency (for harmonic groups) */
    fundamental?: number;
    /** Component IDs in this group */
    componentIds: string[];
    /** Group color for visualization */
    color: string;
    /** Whether the group is expanded in UI */
    expanded: boolean;
    /** Whether the group is selected for rendering */
    selected: boolean;
}

/**
 * Group colors palette
 */
const GROUP_COLORS = [
    '#3b82f6', // Blue - Primary harmonics
    '#f59e0b', // Amber - Secondary
    '#10b981', // Emerald - Tertiary
    '#8b5cf6', // Violet
    '#ec4899', // Pink
    '#06b6d4', // Cyan
    '#f97316', // Orange
    '#84cc16', // Lime
];

/**
 * Detects harmonic series in frequency components.
 * 
 * A harmonic series is defined as frequencies that are integer multiples
 * of a fundamental frequency: f₁, 2f₁, 3f₁, 4f₁, ...
 * 
 * @param components - Frequency components to analyze
 * @param tolerancePercent - Tolerance for matching (default: 5%)
 * @returns Array of frequency groups
 */
export function detectHarmonics(
    components: FrequencyComponent[],
    tolerancePercent: number = 5
): FrequencyGroup[] {
    if (components.length === 0) return [];

    // Sort by frequency
    const sorted = [...components].sort((a, b) => a.frequencyHz - b.frequencyHz);
    const used = new Set<string>();
    const groups: FrequencyGroup[] = [];
    let colorIndex = 0;

    // Try each component as a potential fundamental
    for (const fundamental of sorted) {
        if (used.has(fundamental.id)) continue;

        const harmonics: FrequencyComponent[] = [fundamental];
        const tolerance = fundamental.frequencyHz * (tolerancePercent / 100);

        // Find harmonics (2f, 3f, 4f, ...)
        for (let n = 2; n <= 8; n++) {
            const expectedFreq = fundamental.frequencyHz * n;

            // Find component closest to expected frequency
            const match = sorted.find(c => {
                if (used.has(c.id) || c.id === fundamental.id) return false;
                return Math.abs(c.frequencyHz - expectedFreq) <= tolerance * n;
            });

            if (match) {
                harmonics.push(match);
            }
        }

        // Only create group if we found at least one harmonic
        if (harmonics.length >= 2) {
            harmonics.forEach(h => used.add(h.id));

            groups.push({
                id: `harmonic_${groups.length + 1}`,
                label: `Harmonics of ${Math.round(fundamental.frequencyHz)} Hz`,
                type: 'harmonic',
                fundamental: fundamental.frequencyHz,
                componentIds: harmonics.map(h => h.id),
                color: GROUP_COLORS[colorIndex % GROUP_COLORS.length],
                expanded: true,
                selected: false
            });

            colorIndex++;
        }
    }

    // Add remaining components as outliers
    const outliers = sorted.filter(c => !used.has(c.id));
    if (outliers.length > 0) {
        groups.push({
            id: 'outliers',
            label: 'Outliers',
            type: 'outlier',
            componentIds: outliers.map(c => c.id),
            color: '#6b7280', // Gray
            expanded: false,
            selected: false
        });
    }

    return groups;
}

/**
 * Computes correlation between two magnitude arrays.
 * 
 * @param a - First magnitude array
 * @param b - Second magnitude array
 * @returns Correlation coefficient (-1 to 1)
 */
function computeCorrelation(a: number[], b: number[]): number {
    const n = Math.min(a.length, b.length);
    if (n === 0) return 0;

    let sumA = 0, sumB = 0, sumA2 = 0, sumB2 = 0, sumAB = 0;

    for (let i = 0; i < n; i++) {
        sumA += a[i];
        sumB += b[i];
        sumA2 += a[i] * a[i];
        sumB2 += b[i] * b[i];
        sumAB += a[i] * b[i];
    }

    const numerator = n * sumAB - sumA * sumB;
    const denominator = Math.sqrt(
        (n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB)
    );

    if (denominator === 0) return 0;
    return numerator / denominator;
}

/**
 * Clusters frequency components by temporal co-persistence.
 * 
 * Frequencies that appear and disappear together across time windows
 * are grouped together.
 * 
 * @param fftResults - Array of FFT results from STFT
 * @param components - Frequency components to cluster
 * @param threshold - Correlation threshold (default: 0.7)
 * @returns Array of frequency groups
 */
export function clusterByCorrelation(
    fftResults: FFTResult[],
    components: FrequencyComponent[],
    threshold: number = 0.7
): FrequencyGroup[] {
    if (fftResults.length < 2 || components.length === 0) {
        return [{
            id: 'all',
            label: 'All Components',
            type: 'correlated',
            componentIds: components.map(c => c.id),
            color: GROUP_COLORS[0],
            expanded: true,
            selected: false
        }];
    }

    // Build time series for each component (approximate by frequency bin)
    const timeSeries: Map<string, number[]> = new Map();

    for (const component of components) {
        const series: number[] = [];

        for (const result of fftResults) {
            // Find the bin closest to this frequency
            const binIndex = result.frequencies.findIndex(
                f => Math.abs(f - component.frequencyHz) < 50 // 50 Hz tolerance
            );

            if (binIndex >= 0 && binIndex < result.magnitudes.length) {
                series.push(result.magnitudes[binIndex]);
            } else {
                series.push(0);
            }
        }

        timeSeries.set(component.id, series);
    }

    // Cluster by correlation
    const used = new Set<string>();
    const groups: FrequencyGroup[] = [];
    let colorIndex = 0;

    for (const component of components) {
        if (used.has(component.id)) continue;

        const cluster: FrequencyComponent[] = [component];
        const baseSeries = timeSeries.get(component.id) || [];
        used.add(component.id);

        // Find correlated components
        for (const other of components) {
            if (used.has(other.id)) continue;

            const otherSeries = timeSeries.get(other.id) || [];
            const correlation = computeCorrelation(baseSeries, otherSeries);

            if (correlation >= threshold) {
                cluster.push(other);
                used.add(other.id);
            }
        }

        groups.push({
            id: `cluster_${groups.length + 1}`,
            label: cluster.length > 1
                ? `Correlated Group ${groups.length + 1}`
                : `${Math.round(component.frequencyHz)} Hz`,
            type: 'correlated',
            componentIds: cluster.map(c => c.id),
            color: GROUP_COLORS[colorIndex % GROUP_COLORS.length],
            expanded: groups.length === 0, // First group expanded
            selected: false
        });

        colorIndex++;
    }

    return groups;
}

/**
 * Configuration for combined grouping
 */
export interface GroupingConfig {
    /** Use harmonic detection */
    harmonics: boolean;
    /** Use correlation clustering */
    correlation: boolean;
    /** Harmonic tolerance percent */
    harmonicTolerance?: number;
    /** Correlation threshold */
    correlationThreshold?: number;
}

/**
 * Splits frequency components into groups using combined strategies.
 * 
 * @param components - Frequency components to group
 * @param fftResults - Optional FFT results for correlation analysis
 * @param config - Grouping configuration
 * @returns Array of frequency groups
 */
export function splitGroups(
    components: FrequencyComponent[],
    fftResults: FFTResult[] = [],
    config: GroupingConfig = { harmonics: true, correlation: false }
): FrequencyGroup[] {
    if (components.length === 0) return [];

    // If only harmonics, use harmonic detection
    if (config.harmonics && !config.correlation) {
        return detectHarmonics(components, config.harmonicTolerance);
    }

    // If only correlation, use correlation clustering
    if (!config.harmonics && config.correlation) {
        return clusterByCorrelation(
            fftResults,
            components,
            config.correlationThreshold
        );
    }

    // Combined: detect harmonics first, then cluster remaining
    const harmonicGroups = detectHarmonics(components, config.harmonicTolerance);

    // Get IDs of components already in harmonic groups
    const inHarmonics = new Set<string>();
    harmonicGroups.forEach(g => {
        if (g.type === 'harmonic') {
            g.componentIds.forEach(id => inHarmonics.add(id));
        }
    });

    // Cluster remaining components
    const remaining = components.filter(c => !inHarmonics.has(c.id));

    if (remaining.length > 0 && fftResults.length > 0) {
        const correlatedGroups = clusterByCorrelation(
            fftResults,
            remaining,
            config.correlationThreshold
        );

        // Merge groups, keeping harmonics first
        const harmonicsOnly = harmonicGroups.filter(g => g.type === 'harmonic');
        return [...harmonicsOnly, ...correlatedGroups];
    }

    return harmonicGroups;
}

/**
 * Gets components belonging to a group
 */
export function getGroupComponents(
    group: FrequencyGroup,
    allComponents: FrequencyComponent[]
): FrequencyComponent[] {
    const idSet = new Set(group.componentIds);
    return allComponents.filter(c => idSet.has(c.id));
}

/**
 * Toggles group selection
 */
export function toggleGroupSelection(
    groups: FrequencyGroup[],
    groupId: string
): FrequencyGroup[] {
    return groups.map(g =>
        g.id === groupId ? { ...g, selected: !g.selected } : g
    );
}

/**
 * Toggles group expansion
 */
export function toggleGroupExpansion(
    groups: FrequencyGroup[],
    groupId: string
): FrequencyGroup[] {
    return groups.map(g =>
        g.id === groupId ? { ...g, expanded: !g.expanded } : g
    );
}
