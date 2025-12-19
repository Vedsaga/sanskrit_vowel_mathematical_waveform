/**
 * Comparison Store - State management for side-by-side audio comparison
 * 
 * This store manages:
 * - Left and right panel AudioStoreState independently
 * - Sync mode toggle (independent/synchronized)
 * - Shared frequency scale calculation from both panels
 * 
 * Requirements: 5.1, 5.2, 5.3
 */

import type { FFTResult, FrequencyComponent, Shape, ShapeConfig } from '../types';
import type { NormalizationStrategy, NormalizationOptions } from '../fftProcessor';
import { computeFFTSync, extractTopFrequencies } from '../fftProcessor';

/**
 * Panel-specific audio state interface
 */
export interface PanelAudioState {
	audioBuffer: AudioBuffer | null;
	fileName: string;
	isProcessing: boolean;
	error: string | null;
	fftResult: FFTResult | null;
	frequencyComponents: FrequencyComponent[];
	shapes: Shape[];
	selectedShapeIds: Set<string>;
}

/**
 * Comparison store state interface
 */
export interface ComparisonStoreState {
	leftPanel: PanelAudioState;
	rightPanel: PanelAudioState;
	syncMode: 'independent' | 'synchronized';
	sharedFrequencyScale: { min: number; max: number };
	normalizationStrategy: NormalizationStrategy;
	fqRange: { min: number; max: number };
	config: ShapeConfig;
}

/**
 * Default values
 */
const DEFAULT_FQ_RANGE = { min: 1, max: 50 };
const DEFAULT_STRATEGY: NormalizationStrategy = 'logarithmic';
const DEFAULT_TOP_FREQUENCIES = 20;
const DEFAULT_CONFIG: ShapeConfig = { A: 20, resolution: 360, canvasSize: 400 };

/**
 * Creates an empty panel state
 */
function createEmptyPanelState(): PanelAudioState {
	return {
		audioBuffer: null,
		fileName: '',
		isProcessing: false,
		error: null,
		fftResult: null,
		frequencyComponents: [],
		shapes: [],
		selectedShapeIds: new Set()
	};
}

/**
 * Generates a unique ID for shapes
 */
function generateId(): string {
	return `shape-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Default shape colors palette
 */
const SHAPE_COLORS = [
	'#df728b', // Brand color
	'#6366f1', // Indigo
	'#22c55e', // Green
	'#f59e0b', // Amber
	'#06b6d4', // Cyan
	'#8b5cf6', // Violet
	'#ec4899', // Pink
	'#14b8a6', // Teal
];

/**
 * Creates the comparison store with Svelte 5 runes
 */
function createComparisonStore() {
	// Panel states using $state rune
	let leftPanel = $state<PanelAudioState>(createEmptyPanelState());
	let rightPanel = $state<PanelAudioState>(createEmptyPanelState());
	
	// Sync mode
	let syncMode = $state<'independent' | 'synchronized'>('independent');
	
	// Shared settings
	let normalizationStrategy = $state<NormalizationStrategy>(DEFAULT_STRATEGY);
	let fqRange = $state<{ min: number; max: number }>({ ...DEFAULT_FQ_RANGE });
	let config = $state<ShapeConfig>({ ...DEFAULT_CONFIG });

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

	/**
	 * Calculates shared frequency scale from both panels
	 */
	function calculateSharedFrequencyScale(): { min: number; max: number } {
		const leftFreqs = leftPanel.frequencyComponents.map(c => c.frequencyHz);
		const rightFreqs = rightPanel.frequencyComponents.map(c => c.frequencyHz);
		const allFreqs = [...leftFreqs, ...rightFreqs];
		
		if (allFreqs.length === 0) {
			return { min: 20, max: 20000 };
		}
		
		return {
			min: Math.min(...allFreqs),
			max: Math.max(...allFreqs)
		};
	}

	/**
	 * Gets the next available color for a shape
	 */
	function getNextColor(existingShapes: Shape[]): string {
		const usedColors = new Set(existingShapes.map(s => s.color));
		for (const color of SHAPE_COLORS) {
			if (!usedColors.has(color)) {
				return color;
			}
		}
		// If all colors used, cycle back
		return SHAPE_COLORS[existingShapes.length % SHAPE_COLORS.length];
	}

	return {
		// Getters for reactive state
		get leftPanel() {
			return leftPanel;
		},
		get rightPanel() {
			return rightPanel;
		},
		get syncMode() {
			return syncMode;
		},
		get normalizationStrategy() {
			return normalizationStrategy;
		},
		get fqRange() {
			return fqRange;
		},
		get config() {
			return config;
		},

		/**
		 * Returns calculated shared frequency scale
		 */
		get sharedFrequencyScale(): { min: number; max: number } {
			return calculateSharedFrequencyScale();
		},

		/**
		 * Returns whether left panel has audio loaded
		 */
		get leftHasAudio(): boolean {
			return leftPanel.audioBuffer !== null;
		},

		/**
		 * Returns whether right panel has audio loaded
		 */
		get rightHasAudio(): boolean {
			return rightPanel.audioBuffer !== null;
		},

		/**
		 * Returns whether both panels have audio loaded
		 */
		get bothHaveAudio(): boolean {
			return leftPanel.audioBuffer !== null && rightPanel.audioBuffer !== null;
		},

		/**
		 * Loads audio to a specific panel
		 * 
		 * @param panel - Which panel to load to ('left' or 'right')
		 * @param buffer - The AudioBuffer to load
		 * @param name - The file name
		 */
		async loadAudio(panel: 'left' | 'right', buffer: AudioBuffer, name: string): Promise<void> {
			const targetPanel = panel === 'left' ? leftPanel : rightPanel;
			
			// Update processing state
			if (panel === 'left') {
				leftPanel = { ...leftPanel, isProcessing: true, error: null };
			} else {
				rightPanel = { ...rightPanel, isProcessing: true, error: null };
			}
			
			try {
				// Compute FFT
				const fftResult = computeFFTSync(buffer, 2048);
				
				// Extract frequency components
				const components = extractTopFrequencies(
					fftResult,
					DEFAULT_TOP_FREQUENCIES,
					normalizationStrategy,
					getNormalizationOptions()
				);
				
				// Update panel state
				const newState: PanelAudioState = {
					audioBuffer: buffer,
					fileName: name,
					isProcessing: false,
					error: null,
					fftResult,
					frequencyComponents: components,
					shapes: [],
					selectedShapeIds: new Set()
				};
				
				if (panel === 'left') {
					leftPanel = newState;
				} else {
					rightPanel = newState;
				}
			} catch (err) {
				const errorMsg = err instanceof Error ? err.message : 'Failed to process audio';
				if (panel === 'left') {
					leftPanel = { ...leftPanel, isProcessing: false, error: errorMsg };
				} else {
					rightPanel = { ...rightPanel, isProcessing: false, error: errorMsg };
				}
			}
		},

		/**
		 * Toggles selection of a frequency component in a panel
		 */
		toggleComponentSelection(panel: 'left' | 'right', id: string): void {
			if (panel === 'left') {
				leftPanel = {
					...leftPanel,
					frequencyComponents: leftPanel.frequencyComponents.map(c =>
						c.id === id ? { ...c, selected: !c.selected } : c
					)
				};
			} else {
				rightPanel = {
					...rightPanel,
					frequencyComponents: rightPanel.frequencyComponents.map(c =>
						c.id === id ? { ...c, selected: !c.selected } : c
					)
				};
			}
		},

		/**
		 * Selects all frequency components in a panel
		 */
		selectAllComponents(panel: 'left' | 'right'): void {
			if (panel === 'left') {
				leftPanel = {
					...leftPanel,
					frequencyComponents: leftPanel.frequencyComponents.map(c => ({ ...c, selected: true }))
				};
			} else {
				rightPanel = {
					...rightPanel,
					frequencyComponents: rightPanel.frequencyComponents.map(c => ({ ...c, selected: true }))
				};
			}
		},

		/**
		 * Deselects all frequency components in a panel
		 */
		deselectAllComponents(panel: 'left' | 'right'): void {
			if (panel === 'left') {
				leftPanel = {
					...leftPanel,
					frequencyComponents: leftPanel.frequencyComponents.map(c => ({ ...c, selected: false }))
				};
			} else {
				rightPanel = {
					...rightPanel,
					frequencyComponents: rightPanel.frequencyComponents.map(c => ({ ...c, selected: false }))
				};
			}
		},

		/**
		 * Generates shapes from selected frequency components in a panel
		 */
		generateShapes(panel: 'left' | 'right', components: FrequencyComponent[]): void {
			const targetPanel = panel === 'left' ? leftPanel : rightPanel;
			const newShapes: Shape[] = [];
			
			for (const component of components) {
				const shape: Shape = {
					id: generateId(),
					fq: component.fq,
					R: config.canvasSize / 4, // Default radius
					phi: 0,
					color: getNextColor([...targetPanel.shapes, ...newShapes]),
					opacity: 0.8,
					strokeWidth: 2,
					selected: false
				};
				newShapes.push(shape);
			}
			
			if (panel === 'left') {
				leftPanel = {
					...leftPanel,
					shapes: [...leftPanel.shapes, ...newShapes],
					frequencyComponents: leftPanel.frequencyComponents.map(c => ({ ...c, selected: false }))
				};
			} else {
				rightPanel = {
					...rightPanel,
					shapes: [...rightPanel.shapes, ...newShapes],
					frequencyComponents: rightPanel.frequencyComponents.map(c => ({ ...c, selected: false }))
				};
			}
		},

		/**
		 * Removes a shape from a panel
		 */
		removeShape(panel: 'left' | 'right', shapeId: string): void {
			if (panel === 'left') {
				const newSelectedIds = new Set(leftPanel.selectedShapeIds);
				newSelectedIds.delete(shapeId);
				leftPanel = {
					...leftPanel,
					shapes: leftPanel.shapes.filter(s => s.id !== shapeId),
					selectedShapeIds: newSelectedIds
				};
			} else {
				const newSelectedIds = new Set(rightPanel.selectedShapeIds);
				newSelectedIds.delete(shapeId);
				rightPanel = {
					...rightPanel,
					shapes: rightPanel.shapes.filter(s => s.id !== shapeId),
					selectedShapeIds: newSelectedIds
				};
			}
		},

		/**
		 * Toggles shape selection in a panel
		 */
		toggleShapeSelection(panel: 'left' | 'right', shapeId: string, multi: boolean = false): void {
			if (panel === 'left') {
				const newSelectedIds = new Set(multi ? leftPanel.selectedShapeIds : []);
				if (leftPanel.selectedShapeIds.has(shapeId)) {
					newSelectedIds.delete(shapeId);
				} else {
					newSelectedIds.add(shapeId);
				}
				leftPanel = { ...leftPanel, selectedShapeIds: newSelectedIds };
			} else {
				const newSelectedIds = new Set(multi ? rightPanel.selectedShapeIds : []);
				if (rightPanel.selectedShapeIds.has(shapeId)) {
					newSelectedIds.delete(shapeId);
				} else {
					newSelectedIds.add(shapeId);
				}
				rightPanel = { ...rightPanel, selectedShapeIds: newSelectedIds };
			}
		},

		/**
		 * Updates a shape property in a panel
		 */
		updateShapeProperty(panel: 'left' | 'right', shapeId: string, property: Partial<Shape>): void {
			if (panel === 'left') {
				leftPanel = {
					...leftPanel,
					shapes: leftPanel.shapes.map(s =>
						s.id === shapeId ? { ...s, ...property } : s
					)
				};
			} else {
				rightPanel = {
					...rightPanel,
					shapes: rightPanel.shapes.map(s =>
						s.id === shapeId ? { ...s, ...property } : s
					)
				};
			}
		},

		/**
		 * Sets the sync mode
		 */
		setSyncMode(mode: 'independent' | 'synchronized'): void {
			syncMode = mode;
		},

		/**
		 * Sets the normalization strategy and recomputes components for both panels
		 */
		setNormalizationStrategy(strategy: NormalizationStrategy): void {
			normalizationStrategy = strategy;
			
			// Recompute for left panel
			if (leftPanel.fftResult) {
				const components = extractTopFrequencies(
					leftPanel.fftResult,
					DEFAULT_TOP_FREQUENCIES,
					strategy,
					getNormalizationOptions()
				);
				const selectedIds = new Set(
					leftPanel.frequencyComponents.filter(c => c.selected).map(c => c.id)
				);
				leftPanel = {
					...leftPanel,
					frequencyComponents: components.map(c => ({
						...c,
						selected: selectedIds.has(c.id)
					}))
				};
			}
			
			// Recompute for right panel
			if (rightPanel.fftResult) {
				const components = extractTopFrequencies(
					rightPanel.fftResult,
					DEFAULT_TOP_FREQUENCIES,
					strategy,
					getNormalizationOptions()
				);
				const selectedIds = new Set(
					rightPanel.frequencyComponents.filter(c => c.selected).map(c => c.id)
				);
				rightPanel = {
					...rightPanel,
					frequencyComponents: components.map(c => ({
						...c,
						selected: selectedIds.has(c.id)
					}))
				};
			}
		},

		/**
		 * Sets the fq range
		 */
		setFqRange(range: { min: number; max: number }): void {
			fqRange = { ...range };
		},

		/**
		 * Sets the shape config
		 */
		setConfig(newConfig: Partial<ShapeConfig>): void {
			config = { ...config, ...newConfig };
		},

		/**
		 * Clears error for a panel
		 */
		clearError(panel: 'left' | 'right'): void {
			if (panel === 'left') {
				leftPanel = { ...leftPanel, error: null };
			} else {
				rightPanel = { ...rightPanel, error: null };
			}
		},

		/**
		 * Resets a specific panel
		 */
		resetPanel(panel: 'left' | 'right'): void {
			if (panel === 'left') {
				leftPanel = createEmptyPanelState();
			} else {
				rightPanel = createEmptyPanelState();
			}
		},

		/**
		 * Resets the entire store
		 */
		reset(): void {
			leftPanel = createEmptyPanelState();
			rightPanel = createEmptyPanelState();
			syncMode = 'independent';
			normalizationStrategy = DEFAULT_STRATEGY;
			fqRange = { ...DEFAULT_FQ_RANGE };
			config = { ...DEFAULT_CONFIG };
		}
	};
}

/**
 * Singleton instance of the comparison store
 */
export const comparisonStore = createComparisonStore();

/**
 * Export types for external use
 */
export type ComparisonStore = ReturnType<typeof createComparisonStore>;
