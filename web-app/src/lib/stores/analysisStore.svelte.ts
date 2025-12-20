/**
 * Analysis Store - Multi-analysis state management
 * 
 * This store manages:
 * - Multiple analysis tiles for a single audio file
 * - Local overrides that merge with global settings
 * - Selection state for focused analysis
 * - CRUD operations for analyses
 * 
 * Phase 0: Task 0.2
 */

import type {
    AnalysisState,
    FrequencyComponent,
    Shape,
    TimeWindow,
    FrequencyRange,
    RotationState,
    GlobalSettings
} from '../types';
import { globalSettingsStore } from './globalSettingsStore.svelte';

/**
 * Generates a unique ID for analyses
 */
function generateId(): string {
    return `analysis_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

/**
 * Creates an empty analysis state
 */
function createEmptyAnalysis(label?: string): AnalysisState {
    return {
        id: generateId(),
        label: label || `Analysis ${Date.now() % 1000}`,
        createdAt: Date.now(),
        // No local overrides - inherit from global
        frequencyComponents: [],
        shapes: []
    };
}

/**
 * Creates the analysis store with Svelte 5 runes
 */
function createAnalysisStore() {
    // Core state using $state rune
    let audioBuffer = $state<AudioBuffer | null>(null);
    let fileName = $state<string>('');
    let analyses = $state<AnalysisState[]>([]);
    let selectedAnalysisId = $state<string | null>(null);
    let isProcessing = $state<boolean>(false);
    let error = $state<string | null>(null);

    return {
        // Getters for reactive state
        get audioBuffer(): AudioBuffer | null {
            return audioBuffer;
        },

        get fileName(): string {
            return fileName;
        },

        get analyses(): AnalysisState[] {
            return analyses;
        },

        get selectedAnalysisId(): string | null {
            return selectedAnalysisId;
        },

        get isProcessing(): boolean {
            return isProcessing;
        },

        get error(): string | null {
            return error;
        },

        /**
         * Returns the currently selected analysis, or null
         */
        get selectedAnalysis(): AnalysisState | null {
            if (!selectedAnalysisId) return null;
            return analyses.find(a => a.id === selectedAnalysisId) || null;
        },

        /**
         * Returns whether audio has been loaded
         */
        get hasAudio(): boolean {
            return audioBuffer !== null;
        },

        /**
         * Returns the count of analyses
         */
        get analysisCount(): number {
            return analyses.length;
        },

        /**
         * Gets the effective settings for an analysis (global + local overrides)
         * 
         * @param analysisId - The analysis ID to get settings for
         * @returns Merged settings object
         */
        getEffectiveSettings(analysisId: string): GlobalSettings {
            const analysis = analyses.find(a => a.id === analysisId);
            const global = globalSettingsStore.settings;

            if (!analysis) {
                return global;
            }

            return {
                timeWindow: analysis.timeWindow ?? global.timeWindow,
                frequencyRange: analysis.frequencyRange ?? global.frequencyRange,
                amplitude: analysis.amplitude ?? global.amplitude,
                rotation: analysis.rotation ?? global.rotation,
                normalize: analysis.normalize ?? global.normalize,
                suppressTransients: analysis.suppressTransients ?? global.suppressTransients,
                transientThreshold: global.transientThreshold,
                geometryMode: global.geometryMode
            };
        },

        /**
         * Sets the audio buffer and file name
         */
        setAudio(buffer: AudioBuffer, name: string): void {
            audioBuffer = buffer;
            fileName = name;
            error = null;
        },

        /**
         * Sets the processing state
         */
        setProcessing(processing: boolean): void {
            isProcessing = processing;
        },

        /**
         * Sets an error message
         */
        setError(errorMessage: string | null): void {
            error = errorMessage;
        },

        /**
         * Adds a new analysis tile
         * 
         * @param label - Optional label for the analysis
         * @returns The newly created analysis
         */
        addAnalysis(label?: string): AnalysisState {
            const newAnalysis = createEmptyAnalysis(label);
            analyses = [...analyses, newAnalysis];
            return newAnalysis;
        },

        /**
         * Removes an analysis by ID
         * 
         * @param id - The analysis ID to remove
         * @returns true if removed, false if not found
         */
        removeAnalysis(id: string): boolean {
            const initialLength = analyses.length;
            analyses = analyses.filter(a => a.id !== id);

            // Clear selection if the selected analysis was removed
            if (selectedAnalysisId === id) {
                selectedAnalysisId = null;
            }

            return analyses.length < initialLength;
        },

        /**
         * Duplicates an existing analysis
         * 
         * @param id - The analysis ID to duplicate
         * @returns The new duplicated analysis, or null if not found
         */
        duplicateAnalysis(id: string): AnalysisState | null {
            const source = analyses.find(a => a.id === id);
            if (!source) return null;

            const duplicate: AnalysisState = {
                ...source,
                id: generateId(),
                label: `${source.label} (copy)`,
                createdAt: Date.now(),
                // Deep copy arrays
                frequencyComponents: source.frequencyComponents.map(c => ({ ...c })),
                shapes: source.shapes.map(s => ({ ...s, id: `shape_${Date.now()}_${Math.random().toString(36).substring(2, 9)}` }))
            };

            analyses = [...analyses, duplicate];
            return duplicate;
        },

        /**
         * Selects an analysis for focused editing
         * 
         * @param id - The analysis ID to select, or null to deselect
         */
        selectAnalysis(id: string | null): void {
            selectedAnalysisId = id;
        },

        /**
         * Updates the label of an analysis
         */
        setAnalysisLabel(id: string, label: string): void {
            analyses = analyses.map(a =>
                a.id === id ? { ...a, label } : a
            );
        },

        /**
         * Sets a local override for a specific analysis
         * 
         * @param id - The analysis ID
         * @param key - The setting key to override
         * @param value - The override value (undefined to clear override)
         */
        setLocalOverride<K extends keyof AnalysisState>(
            id: string,
            key: K,
            value: AnalysisState[K]
        ): void {
            analyses = analyses.map(a =>
                a.id === id ? { ...a, [key]: value } : a
            );
        },

        /**
         * Clears all local overrides for an analysis
         */
        clearLocalOverrides(id: string): void {
            analyses = analyses.map(a => {
                if (a.id !== id) return a;
                return {
                    id: a.id,
                    label: a.label,
                    createdAt: a.createdAt,
                    frequencyComponents: a.frequencyComponents,
                    shapes: a.shapes,
                    stabilityScore: a.stabilityScore,
                    energyInvariant: a.energyInvariant,
                    transientScore: a.transientScore
                };
            });
        },

        /**
         * Updates the frequency components for an analysis
         */
        setFrequencyComponents(id: string, components: FrequencyComponent[]): void {
            analyses = analyses.map(a =>
                a.id === id ? { ...a, frequencyComponents: components } : a
            );
        },

        /**
         * Toggles selection of a frequency component
         */
        toggleComponentSelection(analysisId: string, componentId: string): void {
            analyses = analyses.map(a => {
                if (a.id !== analysisId) return a;
                return {
                    ...a,
                    frequencyComponents: a.frequencyComponents.map(c =>
                        c.id === componentId ? { ...c, selected: !c.selected } : c
                    )
                };
            });
        },

        /**
         * Updates the shapes for an analysis
         */
        setShapes(id: string, shapes: Shape[]): void {
            analyses = analyses.map(a =>
                a.id === id ? { ...a, shapes } : a
            );
        },

        /**
         * Updates computed metrics for an analysis
         */
        setMetrics(id: string, metrics: {
            stabilityScore?: number;
            energyInvariant?: boolean;
            transientScore?: number;
        }): void {
            analyses = analyses.map(a =>
                a.id === id ? { ...a, ...metrics } : a
            );
        },

        /**
         * Reorders analyses (for drag-and-drop)
         */
        reorderAnalyses(fromIndex: number, toIndex: number): void {
            if (fromIndex < 0 || fromIndex >= analyses.length) return;
            if (toIndex < 0 || toIndex >= analyses.length) return;

            const newAnalyses = [...analyses];
            const [moved] = newAnalyses.splice(fromIndex, 1);
            newAnalyses.splice(toIndex, 0, moved);
            analyses = newAnalyses;
        },

        /**
         * Clears all analyses
         */
        clearAnalyses(): void {
            analyses = [];
            selectedAnalysisId = null;
        },

        /**
         * Resets the entire store
         */
        reset(): void {
            audioBuffer = null;
            fileName = '';
            analyses = [];
            selectedAnalysisId = null;
            isProcessing = false;
            error = null;
        }
    };
}

/**
 * Singleton instance of the analysis store
 */
export const analysisStore = createAnalysisStore();

/**
 * Export the factory for testing
 */
export { createAnalysisStore };

/**
 * Export type for external use
 */
export type AnalysisStore = ReturnType<typeof createAnalysisStore>;
