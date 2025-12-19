<script lang="ts">
	import { AudioLines } from '@lucide/svelte';
	import AudioUploader from '$lib/components/AudioUploader.svelte';
	import FFTDisplay from '$lib/components/FFTDisplay.svelte';
	import ShapeCanvas from '$lib/components/ShapeCanvas.svelte';
	import ShapeList from '$lib/components/ShapeList.svelte';
	import RotationControls from '$lib/components/RotationControls.svelte';
	import * as Card from '$lib/components/ui/card';
	import { audioStore, shapeStore } from '$lib/stores';
	import type { FrequencyComponent } from '$lib/types';

	/**
	 * Audio Analysis Page
	 * 
	 * Integrates AudioUploader, FFTDisplay, and ShapeCanvas for
	 * audio file FFT analysis and shape generation.
	 * 
	 * Requirements: 4.5
	 */

	// Get reactive state from stores
	const shapes = $derived(shapeStore.shapes);
	const config = $derived(shapeStore.config);
	const selectedIds = $derived(shapeStore.selectedIds);
	const frequencyComponents = $derived(audioStore.frequencyComponents);
	const isProcessing = $derived(audioStore.isProcessing);
	const error = $derived(audioStore.error);
	const fileName = $derived(audioStore.fileName);

	/**
	 * Handles audio file loaded from uploader
	 */
	function handleFileLoaded(buffer: AudioBuffer, name: string) {
		audioStore.loadAudio(buffer, name);
	}

	/**
	 * Handles frequency component selection toggle
	 */
	function handleToggleSelection(id: string) {
		audioStore.toggleComponentSelection(id);
	}

	/**
	 * Handles select all components
	 */
	function handleSelectAll() {
		audioStore.selectAllComponents();
	}

	/**
	 * Handles deselect all components
	 */
	function handleDeselectAll() {
		audioStore.deselectAllComponents();
	}

	/**
	 * Generates shapes from selected frequency components
	 */
	function handleGenerateShapes(components: FrequencyComponent[]) {
		// Add each selected component as a shape
		for (const component of components) {
			shapeStore.addShape(component.fq);
		}
		
		// Optionally deselect components after generating
		audioStore.deselectAllComponents();
	}
</script>

<div class="page-container">
	<header class="page-header">
		<div class="header-icon">
			<AudioLines size={32} />
		</div>
		<div class="header-content">
			<h1>Audio Analysis</h1>
			<p>Upload an audio file and apply Fourier transformation to visualize frequency components as shapes</p>
		</div>
	</header>
	
	<div class="analysis-layout">
		<!-- Left Column: Audio Upload and FFT Display -->
		<div class="analysis-column">
			<!-- Audio Uploader Card -->
			<Card.Root>
				<Card.Header class="pb-3">
					<Card.Title class="text-base">Audio File</Card.Title>
					<Card.Description>
						Upload an audio file to analyze its frequency components
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<AudioUploader
						onFileLoaded={handleFileLoaded}
						{isProcessing}
						{error}
						{fileName}
					/>
				</Card.Content>
			</Card.Root>

			<!-- FFT Display Card -->
			<Card.Root class="fft-card">
				<Card.Header class="pb-3">
					<Card.Title class="text-base">Frequency Components</Card.Title>
					<Card.Description>
						Select components to generate corresponding shapes
					</Card.Description>
				</Card.Header>
				<Card.Content class="fft-content">
					<FFTDisplay
						components={frequencyComponents}
						onToggleSelection={handleToggleSelection}
						onSelectAll={handleSelectAll}
						onDeselectAll={handleDeselectAll}
						onGenerateShapes={handleGenerateShapes}
						showAmplitudeMapping={true}
					/>
				</Card.Content>
			</Card.Root>
		</div>

		<!-- Center Column: Canvas -->
		<div class="canvas-column">
			<Card.Root class="canvas-card">
				<Card.Content class="canvas-content">
					<ShapeCanvas 
						{shapes}
						{config}
						{selectedIds}
						width={config.canvasSize}
						height={config.canvasSize}
						showGrid={true}
					/>
					
					{#if shapes.length === 0}
						<div class="empty-state">
							<p>No shapes yet. Generate shapes from frequency components.</p>
						</div>
					{/if}
				</Card.Content>
			</Card.Root>
		</div>

		<!-- Right Column: Shape Management -->
		<div class="controls-column">
			<!-- Shape List Card -->
			<Card.Root>
				<Card.Header class="pb-3">
					<Card.Title class="text-base">Generated Shapes</Card.Title>
					<Card.Description>
						Manage shapes created from audio analysis
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<ShapeList />
				</Card.Content>
			</Card.Root>

			<!-- Rotation Controls Card -->
			<Card.Root>
				<Card.Header class="pb-3">
					<Card.Title class="text-base">Rotation</Card.Title>
					<Card.Description>
						Animate selected shapes with rotation
					</Card.Description>
				</Card.Header>
				<Card.Content>
					<RotationControls />
				</Card.Content>
			</Card.Root>
		</div>
	</div>
</div>

<style>
	.page-container {
		padding: 1.5rem;
		max-width: 1600px;
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
	
	.analysis-layout {
		display: grid;
		grid-template-columns: 340px auto 320px;
		gap: 1.5rem;
		align-items: start;
	}
	
	.analysis-column {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	:global(.fft-card) {
		flex: 1;
		display: flex;
		flex-direction: column;
	}

	:global(.fft-content) {
		flex: 1;
		display: flex;
		flex-direction: column;
		min-height: 300px;
	}
	
	.canvas-column {
		display: flex;
		justify-content: center;
		position: sticky;
		top: 1.5rem;
	}
	
	:global(.canvas-card) {
		overflow: hidden;
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
		font-size: 0.9rem;
		pointer-events: none;
		z-index: 10;
		max-width: 200px;
	}
	
	.controls-column {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}
	
	@media (max-width: 1200px) {
		.analysis-layout {
			grid-template-columns: 1fr 1fr;
		}
		
		.canvas-column {
			grid-column: 1 / -1;
			order: -1;
			position: static;
		}
	}
	
	@media (max-width: 768px) {
		.page-container {
			padding: 1rem;
		}
		
		.analysis-layout {
			grid-template-columns: 1fr;
		}
		
		.canvas-column {
			order: 0;
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
