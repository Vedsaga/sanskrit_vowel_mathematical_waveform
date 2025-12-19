<script lang="ts">
	/**
	 * Comparison Page
	 * 
	 * Side-by-side audio comparison with two parallel analysis panels.
	 * Supports synchronized or independent manipulation of shapes.
	 * 
	 * Requirements: 5.1, 5.4, 5.5
	 */
	import { GitCompare } from '@lucide/svelte';
	import ComparisonPanel from '$lib/components/ComparisonPanel.svelte';
	import SyncControls from '$lib/components/SyncControls.svelte';
	import { comparisonStore } from '$lib/stores';
	import type { FrequencyComponent } from '$lib/types';

	// Get reactive state from store
	const leftPanel = $derived(comparisonStore.leftPanel);
	const rightPanel = $derived(comparisonStore.rightPanel);
	const syncMode = $derived(comparisonStore.syncMode);
	const sharedFrequencyScale = $derived(comparisonStore.sharedFrequencyScale);
	const config = $derived(comparisonStore.config);
	const leftHasAudio = $derived(comparisonStore.leftHasAudio);
	const rightHasAudio = $derived(comparisonStore.rightHasAudio);

	/**
	 * Handles audio file loaded
	 */
	function handleFileLoaded(panel: 'left' | 'right', buffer: AudioBuffer, fileName: string) {
		comparisonStore.loadAudio(panel, buffer, fileName);
	}

	/**
	 * Handles frequency component selection toggle
	 */
	function handleToggleSelection(panel: 'left' | 'right', id: string) {
		comparisonStore.toggleComponentSelection(panel, id);
	}

	/**
	 * Handles select all components
	 */
	function handleSelectAll(panel: 'left' | 'right') {
		comparisonStore.selectAllComponents(panel);
	}

	/**
	 * Handles deselect all components
	 */
	function handleDeselectAll(panel: 'left' | 'right') {
		comparisonStore.deselectAllComponents(panel);
	}

	/**
	 * Handles generate shapes from selected components
	 */
	function handleGenerateShapes(panel: 'left' | 'right', components: FrequencyComponent[]) {
		comparisonStore.generateShapes(panel, components);
	}

	/**
	 * Handles shape removal
	 */
	function handleRemoveShape(panel: 'left' | 'right', shapeId: string) {
		comparisonStore.removeShape(panel, shapeId);
	}

	/**
	 * Handles shape selection toggle
	 */
	function handleToggleShapeSelection(panel: 'left' | 'right', shapeId: string) {
		comparisonStore.toggleShapeSelection(panel, shapeId);
	}

	/**
	 * Handles shape property update
	 */
	function handleUpdateShapeProperty(panel: 'left' | 'right', shapeId: string, property: any) {
		comparisonStore.updateShapeProperty(panel, shapeId, property);
	}

	/**
	 * Handles sync mode change
	 */
	function handleSyncModeChange(mode: 'independent' | 'synchronized') {
		comparisonStore.setSyncMode(mode);
	}
</script>

<div class="page-container">
	<header class="page-header">
		<div class="header-icon">
			<GitCompare size={32} />
		</div>
		<div class="header-content">
			<h1>Audio Comparison</h1>
			<p>Compare two audio files side by side to analyze differences in their frequency compositions</p>
		</div>
	</header>

	<div class="comparison-layout">
		<!-- Left Panel -->
		<div class="panel-container">
			<ComparisonPanel
				panel="left"
				panelState={leftPanel}
				{config}
				title="Audio A"
				onFileLoaded={handleFileLoaded}
				onToggleSelection={handleToggleSelection}
				onSelectAll={handleSelectAll}
				onDeselectAll={handleDeselectAll}
				onGenerateShapes={handleGenerateShapes}
				onRemoveShape={handleRemoveShape}
				onToggleShapeSelection={handleToggleShapeSelection}
				onUpdateShapeProperty={handleUpdateShapeProperty}
			/>
		</div>

		<!-- Center Controls -->
		<div class="controls-container">
			<SyncControls
				{syncMode}
				{sharedFrequencyScale}
				{leftHasAudio}
				{rightHasAudio}
				onSyncModeChange={handleSyncModeChange}
			/>
		</div>

		<!-- Right Panel -->
		<div class="panel-container">
			<ComparisonPanel
				panel="right"
				panelState={rightPanel}
				{config}
				title="Audio B"
				onFileLoaded={handleFileLoaded}
				onToggleSelection={handleToggleSelection}
				onSelectAll={handleSelectAll}
				onDeselectAll={handleDeselectAll}
				onGenerateShapes={handleGenerateShapes}
				onRemoveShape={handleRemoveShape}
				onToggleShapeSelection={handleToggleShapeSelection}
				onUpdateShapeProperty={handleUpdateShapeProperty}
			/>
		</div>
	</div>
</div>

<style>
	.page-container {
		padding: 1.5rem;
		max-width: 1800px;
		margin: 0 auto;
		min-height: 100%;
	}

	.page-header {
		display: flex;
		align-items: flex-start;
		gap: 1.5rem;
		margin-bottom: 2rem;
	}

	.header-icon {
		width: 64px;
		height: 64px;
		background: var(--color-brand);
		border-radius: var(--radius-lg);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-brand-foreground);
		box-shadow: 0 4px 20px color-mix(in srgb, var(--color-brand) 30%, transparent);
		flex-shrink: 0;
	}

	.header-content h1 {
		font-size: 1.75rem;
		font-weight: 700;
		margin-bottom: 0.5rem;
		letter-spacing: -0.5px;
	}

	.header-content p {
		color: var(--color-muted-foreground);
		font-size: 0.95rem;
		line-height: 1.6;
	}

	.comparison-layout {
		display: grid;
		grid-template-columns: 1fr 240px 1fr;
		gap: 1.5rem;
		align-items: start;
	}

	.panel-container {
		min-width: 0;
	}

	.controls-container {
		position: sticky;
		top: 1.5rem;
	}

	/* Responsive layout */
	@media (max-width: 1400px) {
		.comparison-layout {
			grid-template-columns: 1fr 200px 1fr;
			gap: 1rem;
		}
	}

	@media (max-width: 1100px) {
		.comparison-layout {
			grid-template-columns: 1fr 1fr;
			gap: 1.5rem;
		}

		.controls-container {
			grid-column: 1 / -1;
			order: -1;
			position: static;
		}
	}

	@media (max-width: 768px) {
		.page-container {
			padding: 1rem;
		}

		.comparison-layout {
			grid-template-columns: 1fr;
		}

		.page-header {
			flex-direction: column;
			align-items: center;
			text-align: center;
			gap: 1rem;
		}

		.header-content h1 {
			font-size: 1.5rem;
		}
	}
</style>
