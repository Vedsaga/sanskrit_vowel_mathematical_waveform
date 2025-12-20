/**
 * Global Settings Store - Centralized settings that all analyses inherit from
 * 
 * This store manages global defaults for:
 * - Time window configuration (STFT parameters)
 * - Frequency range filtering
 * - Amplitude and rotation settings
 * - Energy normalization and transient suppression
 * - Geometry rendering mode
 * 
 * Phase 0: Task 0.1
 */

import type {
    GlobalSettings,
    TimeWindow,
    FrequencyRange,
    RotationState,
    GeometryMode
} from '../types';

/**
 * Default time window configuration
 */
const DEFAULT_TIME_WINDOW: TimeWindow = {
    start: 0,
    width: 500,    // 500ms
    step: 100,     // 100ms step
    type: 'hann'
};

/**
 * Default frequency range
 */
const DEFAULT_FREQUENCY_RANGE: FrequencyRange = {
    min: 20,
    max: 20000
};

/**
 * Default rotation state
 */
const DEFAULT_ROTATION: RotationState = {
    isAnimating: false,
    direction: 'clockwise',
    mode: 'loop',
    speed: 1.0
};

/**
 * Default global settings
 */
const DEFAULT_SETTINGS: GlobalSettings = {
    timeWindow: { ...DEFAULT_TIME_WINDOW },
    frequencyRange: { ...DEFAULT_FREQUENCY_RANGE },
    amplitude: 20,
    rotation: { ...DEFAULT_ROTATION },
    normalize: true,
    suppressTransients: false,
    transientThreshold: 0.5,
    geometryMode: 'single'
};

/**
 * Creates the global settings store with Svelte 5 runes
 */
function createGlobalSettingsStore() {
    // Core state using $state rune
    let settings = $state<GlobalSettings>({ ...DEFAULT_SETTINGS });

    return {
        // Getter for the full settings object
        get settings(): GlobalSettings {
            return settings;
        },

        // Individual getters for common access patterns
        get timeWindow(): TimeWindow {
            return settings.timeWindow;
        },

        get frequencyRange(): FrequencyRange {
            return settings.frequencyRange;
        },

        get amplitude(): number {
            return settings.amplitude;
        },

        get rotation(): RotationState {
            return settings.rotation;
        },

        get normalize(): boolean {
            return settings.normalize;
        },

        get suppressTransients(): boolean {
            return settings.suppressTransients;
        },

        get transientThreshold(): number {
            return settings.transientThreshold;
        },

        get geometryMode(): GeometryMode {
            return settings.geometryMode;
        },

        /**
         * Sets all global settings at once
         * 
         * @param newSettings - Complete or partial settings to merge
         */
        setGlobal(newSettings: Partial<GlobalSettings>): void {
            settings = {
                ...settings,
                ...newSettings,
                // Deep merge nested objects
                timeWindow: newSettings.timeWindow
                    ? { ...settings.timeWindow, ...newSettings.timeWindow }
                    : settings.timeWindow,
                frequencyRange: newSettings.frequencyRange
                    ? { ...settings.frequencyRange, ...newSettings.frequencyRange }
                    : settings.frequencyRange,
                rotation: newSettings.rotation
                    ? { ...settings.rotation, ...newSettings.rotation }
                    : settings.rotation
            };
        },

        /**
         * Sets the time window configuration
         */
        setTimeWindow(timeWindow: Partial<TimeWindow>): void {
            settings = {
                ...settings,
                timeWindow: { ...settings.timeWindow, ...timeWindow }
            };
        },

        /**
         * Sets the frequency range filter
         */
        setFrequencyRange(range: Partial<FrequencyRange>): void {
            settings = {
                ...settings,
                frequencyRange: { ...settings.frequencyRange, ...range }
            };
        },

        /**
         * Sets the amplitude value
         */
        setAmplitude(amplitude: number): void {
            settings = {
                ...settings,
                amplitude: Math.max(1, amplitude)
            };
        },

        /**
         * Sets the rotation state
         */
        setRotation(rotation: Partial<RotationState>): void {
            settings = {
                ...settings,
                rotation: { ...settings.rotation, ...rotation }
            };
        },

        /**
         * Toggles the normalize setting
         */
        setNormalize(normalize: boolean): void {
            settings = { ...settings, normalize };
        },

        /**
         * Toggles transient suppression
         */
        setSuppressTransients(suppress: boolean): void {
            settings = { ...settings, suppressTransients: suppress };
        },

        /**
         * Sets the transient detection threshold
         */
        setTransientThreshold(threshold: number): void {
            settings = {
                ...settings,
                transientThreshold: Math.max(0, Math.min(1, threshold))
            };
        },

        /**
         * Sets the geometry rendering mode
         */
        setGeometryMode(mode: GeometryMode): void {
            settings = { ...settings, geometryMode: mode };
        },

        /**
         * Starts rotation animation
         */
        startRotation(): void {
            settings = {
                ...settings,
                rotation: { ...settings.rotation, isAnimating: true }
            };
        },

        /**
         * Stops rotation animation
         */
        stopRotation(): void {
            settings = {
                ...settings,
                rotation: { ...settings.rotation, isAnimating: false }
            };
        },

        /**
         * Resets all settings to defaults
         */
        reset(): void {
            settings = {
                timeWindow: { ...DEFAULT_TIME_WINDOW },
                frequencyRange: { ...DEFAULT_FREQUENCY_RANGE },
                amplitude: 20,
                rotation: { ...DEFAULT_ROTATION },
                normalize: true,
                suppressTransients: false,
                transientThreshold: 0.5,
                geometryMode: 'single'
            };
        }
    };
}

/**
 * Singleton instance of the global settings store
 */
export const globalSettingsStore = createGlobalSettingsStore();

/**
 * Export the factory for testing
 */
export { createGlobalSettingsStore };

/**
 * Export type for external use
 */
export type GlobalSettingsStore = ReturnType<typeof createGlobalSettingsStore>;
