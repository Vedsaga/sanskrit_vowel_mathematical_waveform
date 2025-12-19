/**
 * Audio Store - State management for audio analysis using Svelte 5 runes
 * 
 * This store manages:
 * - Audio buffer and file information
 * - FFT processing state and results
 * - Frequency components extracted from FFT
 * - Normalization strategy and fq range settings
 * 
 * Requirements: 4.2, 4.3
 */

import type { FFTResult, FrequencyComponent } from '../types';
import type { NormalizationStrategy, NormalizationOptions } from '../fftProcessor';
import { computeFFTSync, extractTopFrequencies } from '../fftProcessor';

/**
 * Audio store state interface
 */
export interface AudioStoreState {
	audioBuffer: AudioBuffer | null;
	fileName: string;
	isProcessing: boolean;
	error: string | null;
	fftResult: FFTResult | null;
	frequencyComponents: FrequencyComponent[];
	normalizationStrategy: NormalizationStrategy;
	fqRange: { min: number; max: number };
}

/**
 * Default normalization options
 */
const DEFAULT_FQ_RANGE = { min: 1, max: 50 };
const DEFAULT_STRATEGY: NormalizationStrategy = 'logarithmic';
const DEFAULT_TOP_FREQUENCIES = 20;

/**
 * Creates an audio store with Svelte 5 runes
 */
function createAudioStore() {
	// Core state using $state rune
	let audioBuffer = $state<AudioBuffer | null>(null);
	let fileName = $state<string>('');
	let isProcessing = $state<boolean>(false);
	let error = $state<string | null>(null);
	let fftResult = $state<FFTResult | null>(null);
	let frequencyComponents = $state<FrequencyComponent[]>([]);
	let normalizationStrategy = $state<NormalizationStrategy>(DEFAULT_STRATEGY);
	let fqRange = $state<{ min: number; max: number }>({ ...DEFAULT_FQ_RANGE });

	/**
	 * Gets normalization options based on current settings
	 */
	function getNormalizationOptions(): NormalizationOptions {
		return {
			minFq: fqRange.min,
			maxFq: fqRange.max,
			minFrequency: 20,
			maxFrequency: 20000
		};
	}

	return {
		// Getters for reactive state
		get audioBuffer() {
			return audioBuffer;
		},
		get fileName() {
			return fileName;
		},
		get isProcessing() {
			return isProcessing;
		},
		get error() {
			return error;
		},
		get fftResult() {
			return fftResult;
		},
		get frequencyComponents() {
			return frequencyComponents;
		},
		get normalizationStrategy() {
			return normalizationStrategy;
		},
		get fqRange() {
			return fqRange;
		},

		/**
		 * Returns selected frequency components
		 */
		get selectedComponents(): FrequencyComponent[] {
			return frequencyComponents.filter(c => c.selected);
		},

		/**
		 * Returns whether audio is loaded and ready for analysis
		 */
		get hasAudio(): boolean {
			return audioBuffer !== null;
		},

		/**
		 * Returns whether FFT analysis has been performed
		 */
		get hasFFTResult(): boolean {
			return fftResult !== null;
		},

		/**
		 * Loads an audio buffer and processes it
		 * 
		 * @param buffer - The AudioBuffer to load
		 * @param name - The file name
		 */
		async loadAudio(buffer: AudioBuffer, name: string): Promise<void> {
			isProcessing = true;
			error = null;
			
			try {
				audioBuffer = buffer;
				fileName = name;
				
				// Automatically compute FFT
				await this.computeFFT();
			} catch (err) {
				error = err instanceof Error ? err.message : 'Failed to process audio';
				console.error('Audio processing error:', err);
			} finally {
				isProcessing = false;
			}
		},

		/**
		 * Computes FFT on the loaded audio buffer
		 * 
		 * @param fftSize - Size of the FFT (default: 2048)
		 */
		async computeFFT(fftSize: number = 2048): Promise<void> {
			if (!audioBuffer) {
				error = 'No audio loaded';
				return;
			}

			isProcessing = true;
			error = null;

			try {
				// Use synchronous FFT for reliability
				const result = computeFFTSync(audioBuffer, fftSize);
				fftResult = result;

				// Extract top frequency components
				const components = extractTopFrequencies(
					result,
					DEFAULT_TOP_FREQUENCIES,
					normalizationStrategy,
					getNormalizationOptions()
				);
				frequencyComponents = components;
			} catch (err) {
				error = err instanceof Error ? err.message : 'FFT computation failed';
				console.error('FFT error:', err);
			} finally {
				isProcessing = false;
			}
		},

		/**
		 * Toggles selection of a frequency component
		 * 
		 * @param id - The component ID to toggle
		 */
		toggleComponentSelection(id: string): void {
			frequencyComponents = frequencyComponents.map(c =>
				c.id === id ? { ...c, selected: !c.selected } : c
			);
		},

		/**
		 * Selects all frequency components
		 */
		selectAllComponents(): void {
			frequencyComponents = frequencyComponents.map(c => ({ ...c, selected: true }));
		},

		/**
		 * Deselects all frequency components
		 */
		deselectAllComponents(): void {
			frequencyComponents = frequencyComponents.map(c => ({ ...c, selected: false }));
		},

		/**
		 * Sets the normalization strategy and recomputes components
		 * 
		 * @param strategy - The normalization strategy to use
		 */
		setNormalizationStrategy(strategy: NormalizationStrategy): void {
			normalizationStrategy = strategy;
			
			// Recompute frequency components with new strategy
			if (fftResult) {
				const components = extractTopFrequencies(
					fftResult,
					DEFAULT_TOP_FREQUENCIES,
					strategy,
					getNormalizationOptions()
				);
				// Preserve selection state
				const selectedIds = new Set(
					frequencyComponents.filter(c => c.selected).map(c => c.id)
				);
				frequencyComponents = components.map(c => ({
					...c,
					selected: selectedIds.has(c.id)
				}));
			}
		},

		/**
		 * Sets the fq range and recomputes components
		 * 
		 * @param range - The new fq range
		 */
		setFqRange(range: { min: number; max: number }): void {
			fqRange = { ...range };
			
			// Recompute frequency components with new range
			if (fftResult) {
				const components = extractTopFrequencies(
					fftResult,
					DEFAULT_TOP_FREQUENCIES,
					normalizationStrategy,
					getNormalizationOptions()
				);
				// Preserve selection state
				const selectedIds = new Set(
					frequencyComponents.filter(c => c.selected).map(c => c.id)
				);
				frequencyComponents = components.map(c => ({
					...c,
					selected: selectedIds.has(c.id)
				}));
			}
		},

		/**
		 * Clears the error state
		 */
		clearError(): void {
			error = null;
		},

		/**
		 * Resets the store to initial state
		 */
		reset(): void {
			audioBuffer = null;
			fileName = '';
			isProcessing = false;
			error = null;
			fftResult = null;
			frequencyComponents = [];
			normalizationStrategy = DEFAULT_STRATEGY;
			fqRange = { ...DEFAULT_FQ_RANGE };
		}
	};
}

/**
 * Singleton instance of the audio store
 */
export const audioStore = createAudioStore();

/**
 * Export types for external use
 */
export type AudioStore = ReturnType<typeof createAudioStore>;
