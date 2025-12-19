<script lang="ts">
	/**
	 * ComparisonPanel Component
	 * 
	 * A reusable panel for the comparison page that combines
	 * AudioUploader, FFTDisplay, and ShapeCanvas components.
	 * Connects to the appropriate panel state (left/right).
	 * 
	 * Requirements: 5.2
	 */
	import AudioUploader from './AudioUploader.svelte';
	import FFTDisplay from './FFTDisplay.svelte';
	import ShapeCanvas from './ShapeCanvas.svelte';
	import * as Card from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Trash2 } from '@lucide/svelte';
	import type { PanelAudioState, ComparisonStore } from '$lib/stores/comparisonStore.svelte';
	import type { FrequencyComponent, ShapeConfig } from '$lib/types';

	// Props
	interface Props {
		panel: 'left' | 'right';
		panelState: PanelAudioState;
		config: ShapeConfig;
		onFileLoaded: (panel: 'left' | 'right', buffer: AudioBuffer, fileName: string) => void;
		onToggleSelection: (panel: 'left' | 'right', id: string) => void;
		onSelectAll: (panel: 'left' | 'right') => void;
		onDeselectAll: (panel: 'left' | 'right') => void;
		onGenerateShapes: (panel: 'left' | 'right', components: FrequencyComponent[]) => void;
		onRemoveShape: (panel: 'left' | 'right', shapeId: string) => void;
		onToggleShapeSelection: (panel: 'left' | 'right', shapeId: string) => void;
		onUpdateShapeProperty: (panel: 'left' | 'right', shapeId: string, property: any) => void;
		title?: string;
	}

	let {
		panel,
		panelState,
		config,
		onFileLoaded,
		onToggleSelection,
		onSelectAll,
		onDeselectAll,
		onGenerateShapes,
		onRemoveShape,
		onToggleShapeSelection,
		onUpdateShapeProperty,
		title = panel === 'left' ? 'Audio A' : 'Audio B'
	}: Props = $props();

	// Derived state
	let selectedComponents = $derived(panelState.frequencyComponents.filter(c => c.selected));
	let hasShapes = $derived(panelState.shapes.length > 0);

	/**
	 * Handles file loaded from uploader
	 */
	function handleFileLoaded(buffer: AudioBuffer, fileName: string) {
		onFileLoaded(panel, buffer, fileName);
	}

	/**
	 * Handles component selection toggle
	 */
	function handleToggleSelection(id: string) {
		onToggleSelection(panel, id);
	}

	/**
	 * Handles select all
	 */
	function handleSelectAll() {
		onSelectAll(panel);
	}

	/**
	 * Handles deselect all
	 */
	function handleDeselectAll() {
		onDeselectAll(panel);
	}

	/**
	 * Handles generate shapes
	 */
	function handleGenerateShapes(components: FrequencyComponent[]) {
		onGenerateShapes(panel, components);
	}

	/**
	 * Handles shape removal
	 */
	function handleRemoveShape(shapeId: string) {
		onRemoveShape(panel, shapeId);
	}

	/**
	 * Handles shape selection toggle
	 */
	function handleToggleShapeSelection(shapeId: string) {
		onToggleShapeSelection(panel, shapeId);
	}

	/**
	 * Handles shape color change
	 */
	function handleColorChange(shapeId: string, color: string) {
		onUpdateShapeProperty(panel, shapeId, { color });
	}
</script>

<div class="comparison-panel">
	<div class="panel-header">
		<h3 class="panel-title">{title}</h3>
		{#if panelState.fileName}
			<span class="panel-filename">{panelState.fileName}</span>
		{/if}
	</div>

	<div class="panel-content">
		<!-- Audio Uploader -->
		<Card.Root class="uploader-card">
			<Card.Content class="uploader-content">
				<AudioUploader
					onFileLoaded={handleFileLoaded}
					isProcessing={panelState.isProcessing}
					error={panelState.error}
					fileName={panelState.fileName}
				/>
			</Card.Content>
		</Card.Root>

		<!-- Shape Canvas -->
		<Card.Root class="canvas-card">
			<Card.Content class="canvas-content">
				<ShapeCanvas
					shapes={panelState.shapes}
					{config}
					selectedIds={panelState.selectedShapeIds}
					width={config.canvasSize}
					height={config.canvasSize}
					showGrid={true}
				/>
				{#if !hasShapes && panelState.frequencyComponents.length === 0}
					<div class="empty-state">
						<p>Upload audio to analyze</p>
					</div>
				{:else if !hasShapes}
					<div class="empty-state">
						<p>Select frequencies to generate shapes</p>
					</div>
				{/if}
			</Card.Content>
		</Card.Root>

		<!-- FFT Display -->
		<Card.Root class="fft-card">
			<Card.Header class="pb-2">
				<Card.Title class="text-sm">Frequencies</Card.Title>
			</Card.Header>
			<Card.Content class="fft-content">
				<FFTDisplay
					components={panelState.frequencyComponents}
					onToggleSelection={handleToggleSelection}
					onSelectAll={handleSelectAll}
					onDeselectAll={handleDeselectAll}
					onGenerateShapes={handleGenerateShapes}
					showAmplitudeMapping={true}
					isProcessing={panelState.isProcessing}
				/>
			</Card.Content>
		</Card.Root>

		<!-- Shape List (simplified) -->
		{#if hasShapes}
			<Card.Root class="shapes-card">
				<Card.Header class="pb-2">
					<Card.Title class="text-sm">Shapes ({panelState.shapes.length})</Card.Title>
				</Card.Header>
				<Card.Content class="shapes-content">
					<div class="shape-list">
						{#each panelState.shapes as shape (shape.id)}
							<div 
								class="shape-item"
								class:selected={panelState.selectedShapeIds.has(shape.id)}
							>
								<button
									type="button"
									class="shape-info"
									onclick={() => handleToggleShapeSelection(shape.id)}
								>
									<div 
										class="shape-color"
										style="background-color: {shape.color}"
									></div>
									<span class="shape-fq">fq = {shape.fq}</span>
								</button>
								<input
									type="color"
									value={shape.color}
									onchange={(e) => handleColorChange(shape.id, e.currentTarget.value)}
									class="color-picker"
									title="Change color"
								/>
								<Button
									variant="ghost"
									size="icon"
									class="delete-btn"
									onclick={() => handleRemoveShape(shape.id)}
								>
									<Trash2 size={14} />
								</Button>
							</div>
						{/each}
					</div>
				</Card.Content>
			</Card.Root>
		{/if}
	</div>
</div>

<style>
	.comparison-panel {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		height: 100%;
	}

	.panel-header {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding-bottom: 0.5rem;
		border-bottom: 1px solid var(--color-border);
	}

	.panel-title {
		font-size: 1rem;
		font-weight: 600;
		color: var(--color-foreground);
	}

	.panel-filename {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		background-color: var(--color-muted);
		padding: 0.25rem 0.5rem;
		border-radius: var(--radius-sm);
		max-width: 150px;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.panel-content {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		flex: 1;
		min-height: 0;
	}

	:global(.uploader-card) {
		flex-shrink: 0;
	}

	:global(.uploader-content) {
		padding: 0.75rem !important;
	}

	:global(.canvas-card) {
		flex-shrink: 0;
	}

	:global(.canvas-content) {
		padding: 0 !important;
		position: relative;
	}

	.empty-state {
		position: absolute;
		top: 50%;
		left: 50%;
		transform: translate(-50%, -50%);
		text-align: center;
		color: var(--color-muted-foreground);
		font-size: 0.8rem;
		pointer-events: none;
	}

	:global(.fft-card) {
		flex: 1;
		min-height: 200px;
		display: flex;
		flex-direction: column;
	}

	:global(.fft-content) {
		flex: 1;
		overflow: hidden;
		padding: 0 !important;
	}

	:global(.shapes-card) {
		flex-shrink: 0;
	}

	:global(.shapes-content) {
		padding: 0.5rem !important;
	}

	.shape-list {
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
		max-height: 150px;
		overflow-y: auto;
	}

	.shape-item {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		padding: 0.375rem 0.5rem;
		border-radius: var(--radius-sm);
		background-color: transparent;
		transition: background-color 0.15s ease-out;
	}

	.shape-item:hover {
		background-color: var(--color-muted);
	}

	.shape-item.selected {
		background-color: color-mix(in srgb, var(--color-brand) 10%, var(--color-card));
	}

	.shape-info {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		flex: 1;
		background: none;
		border: none;
		cursor: pointer;
		padding: 0;
		text-align: left;
	}

	.shape-color {
		width: 16px;
		height: 16px;
		border-radius: var(--radius-sm);
		flex-shrink: 0;
	}

	.shape-fq {
		font-size: 0.75rem;
		color: var(--color-foreground);
	}

	.color-picker {
		width: 24px;
		height: 24px;
		padding: 0;
		border: none;
		border-radius: var(--radius-sm);
		cursor: pointer;
		background: transparent;
	}

	.color-picker::-webkit-color-swatch-wrapper {
		padding: 2px;
	}

	.color-picker::-webkit-color-swatch {
		border-radius: var(--radius-sm);
		border: 1px solid var(--color-border);
	}

	:global(.delete-btn) {
		width: 24px;
		height: 24px;
		padding: 0;
		color: var(--color-muted-foreground);
	}

	:global(.delete-btn:hover) {
		color: var(--color-destructive);
	}
</style>
