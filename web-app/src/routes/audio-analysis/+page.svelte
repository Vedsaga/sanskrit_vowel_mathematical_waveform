<script lang="ts">
	/**
	 * Analysis Observatory Page
	 *
	 * Multi-analysis grid with advanced controls for audio visualization.
	 * Supports focused mode, global/local settings, and Guna analysis.
	 *
	 * Phase 1: Task 1.8
	 */
	import { AudioLines, Plus, Settings, ChevronLeft } from "@lucide/svelte";
	import AudioUploader from "$lib/components/AudioUploader.svelte";
	import ShapeCanvas from "$lib/components/ShapeCanvas.svelte";
	import AnalysisGrid from "$lib/components/layout/AnalysisGrid.svelte";
	import AnalysisTile from "$lib/components/layout/AnalysisTile.svelte";
	import GlobalControlPanel from "$lib/components/controls/GlobalControlPanel.svelte";
	import LocalControlPanel from "$lib/components/controls/LocalControlPanel.svelte";
	import TemporalNavigator from "$lib/components/controls/TemporalNavigator.svelte";
	import AudioPlayer from "$lib/components/audio/AudioPlayer.svelte";
	import AudioMetadata from "$lib/components/audio/AudioMetadata.svelte";
	import GunaStrengthIndicator from "$lib/components/analysis/GunaStrengthIndicator.svelte";
	import FrequencyBadges from "$lib/components/analysis/FrequencyBadges.svelte";
	import ComponentGroups from "$lib/components/analysis/ComponentGroups.svelte";
	import SpectrumGraph from "$lib/components/analysis/SpectrumGraph.svelte";
	import * as Card from "$lib/components/ui/card";
	import { Button } from "$lib/components/ui/button";
	import {
		audioStore,
		shapeStore,
		globalSettingsStore,
		analysisStore,
	} from "$lib/stores";
	import type {
		FrequencyComponent,
		AnalysisState,
		GlobalSettings,
		TimeWindow,
		GeometryMode,
	} from "$lib/types";

	// Store state
	const audioBuffer = $derived(audioStore.audioBuffer);
	const fileName = $derived(audioStore.fileName);
	const isProcessing = $derived(audioStore.isProcessing);
	const error = $derived(audioStore.error);
	const frequencyComponents = $derived(audioStore.frequencyComponents);

	// Analysis store state
	const analyses = $derived(analysisStore.analyses);
	const selectedAnalysisId = $derived(analysisStore.selectedAnalysisId);
	const selectedAnalysis = $derived(
		analyses.find((a) => a.id === selectedAnalysisId) ?? null,
	);

	// Global settings
	const globalSettings = $derived(globalSettingsStore.settings);

	// Shape store
	const shapes = $derived(shapeStore.shapes);
	const config = $derived(shapeStore.config);
	const selectedShapeIds = $derived(shapeStore.selectedIds);

	// UI state
	let showSidebar = $state(true);
	let isSliding = $state(false);

	/**
	 * Handles audio file loaded
	 */
	function handleFileLoaded(buffer: AudioBuffer, name: string) {
		audioStore.loadAudio(buffer, name);
		// Create initial analysis when audio is loaded
		if (analyses.length === 0) {
			const newAnalysis = analysisStore.addAnalysis("Analysis 1");
			analysisStore.setFrequencyComponents(
				newAnalysis.id,
				frequencyComponents,
			);
		}
	}

	/**
	 * Handles analysis selection
	 */
	function handleSelectAnalysis(id: string | null) {
		analysisStore.selectAnalysis(id);
	}

	/**
	 * Handles adding a new analysis
	 */
	function handleAddAnalysis() {
		const newIndex = analyses.length + 1;
		const newAnalysis = analysisStore.addAnalysis(`Analysis ${newIndex}`);
		analysisStore.setFrequencyComponents(
			newAnalysis.id,
			frequencyComponents,
		);
	}

	/**
	 * Handles removing an analysis
	 */
	function handleRemoveAnalysis(id: string) {
		analysisStore.removeAnalysis(id);
	}

	/**
	 * Handles global settings change
	 */
	function handleGlobalSettingsChange(newSettings: Partial<GlobalSettings>) {
		globalSettingsStore.setGlobal(newSettings);
	}

	/**
	 * Handles time window change
	 */
	function handleTimeWindowChange(partialWindow: Partial<TimeWindow>) {
		const currentWindow = globalSettings.timeWindow;
		globalSettingsStore.setTimeWindow({
			...currentWindow,
			...partialWindow,
		});
	}

	/**
	 * Handles geometry mode change
	 */
	function handleGeometryModeChange(mode: GeometryMode) {
		globalSettingsStore.setGeometryMode(mode);
	}

	/**
	 * Handles sliding animation
	 */
	function handleSlide() {
		isSliding = !isSliding;
		// TODO: Implement sliding window animation
	}

	/**
	 * Handles local override change
	 */
	function handleLocalOverrideChange(key: keyof AnalysisState, value: any) {
		if (selectedAnalysisId) {
			analysisStore.setLocalOverride(selectedAnalysisId, key, value);
		}
	}

	/**
	 * Clears all local overrides
	 */
	function handleClearOverrides() {
		if (selectedAnalysisId) {
			analysisStore.clearLocalOverrides(selectedAnalysisId);
		}
	}

	/**
	 * Handles frequency component click in spectrum
	 */
	function handleComponentClick(id: string) {
		audioStore.toggleComponentSelection(id);
	}

	// Determine which shapes to show based on mode and selection
	let displayShapes = $derived(() => {
		if (selectedAnalysis) {
			return selectedAnalysis.shapes;
		}
		return shapes;
	});
</script>

<div class="observatory-container">
	<!-- Header -->
	<header class="observatory-header">
		<div class="header-left">
			<div class="header-icon">
				<AudioLines size={28} />
			</div>
			<div class="header-content">
				<h1>Analysis Observatory</h1>
				<p>Multi-analysis audio visualization with Guna metrics</p>
			</div>
		</div>

		{#if audioBuffer}
			<div class="header-audio">
				<AudioMetadata
					{fileName}
					duration={audioBuffer.duration}
					sampleRate={audioBuffer.sampleRate}
					channels={audioBuffer.numberOfChannels}
					compact
				/>
				<AudioPlayer
					{audioBuffer}
					timeWindow={globalSettings.timeWindow}
				/>
			</div>
		{/if}

		<Button
			variant="ghost"
			size="sm"
			onclick={() => (showSidebar = !showSidebar)}
			class="sidebar-toggle"
		>
			<Settings size={18} />
		</Button>
	</header>

	<div class="observatory-layout" class:sidebar-open={showSidebar}>
		<!-- Main Content Area -->
		<main class="main-content">
			{#if !audioBuffer}
				<!-- Upload State -->
				<Card.Root class="upload-card">
					<Card.Content class="upload-content">
						<AudioUploader
							onFileLoaded={handleFileLoaded}
							{isProcessing}
							{error}
							{fileName}
						/>
					</Card.Content>
				</Card.Root>
			{:else if selectedAnalysis}
				<!-- Focused Analysis View -->
				<div class="focused-view">
					<div class="focused-header">
						<Button
							variant="ghost"
							size="sm"
							onclick={() => handleSelectAnalysis(null)}
						>
							<ChevronLeft size={16} />
							Back to Grid
						</Button>
						<h2>{selectedAnalysis.label}</h2>
					</div>

					<div class="focused-content">
						<!-- Main Canvas -->
						<div class="focused-canvas">
							<ShapeCanvas
								shapes={selectedAnalysis.shapes}
								{config}
								selectedIds={selectedShapeIds}
								width={500}
								height={500}
								showGrid={true}
								mode={globalSettings.geometryMode}
							/>
						</div>

						<!-- Analysis Details -->
						<div class="focused-details">
							<!-- Guna Metrics -->
							<Card.Root>
								<Card.Header class="pb-2">
									<Card.Title class="text-sm"
										>Guna Metrics</Card.Title
									>
								</Card.Header>
								<Card.Content>
									<GunaStrengthIndicator
										metrics={{
											stabilityScore:
												selectedAnalysis.stabilityScore ??
												0.5,
											stabilityLabel: "Variable",
											energyInvariant:
												selectedAnalysis.energyInvariant ??
												true,
											transientScore: 0.3,
											transientLabel: "Mixed",
										}}
									/>
								</Card.Content>
							</Card.Root>

							<!-- Frequency Components -->
							<Card.Root>
								<Card.Header class="pb-2">
									<Card.Title class="text-sm"
										>Spectrum</Card.Title
									>
								</Card.Header>
								<Card.Content>
									<SpectrumGraph
										components={selectedAnalysis.frequencyComponents}
										frequencyRange={globalSettings.frequencyRange}
										height={150}
										onComponentClick={handleComponentClick}
									/>
								</Card.Content>
							</Card.Root>
						</div>
					</div>

					<!-- Thumbnail Strip -->
					<div class="thumbnail-strip">
						{#each analyses as analysis (analysis.id)}
							<AnalysisTile
								{analysis}
								{config}
								{globalSettings}
								isSelected={analysis.id === selectedAnalysisId}
								size={100}
								onSelect={handleSelectAnalysis}
								onRemove={handleRemoveAnalysis}
							/>
						{/each}
						<button class="add-tile" onclick={handleAddAnalysis}>
							<Plus size={24} />
						</button>
					</div>
				</div>
			{:else}
				<!-- Grid View -->
				<div class="grid-view">
					<!-- Temporal Navigator -->
					<div class="temporal-section">
						<TemporalNavigator
							{audioBuffer}
							timeWindow={globalSettings.timeWindow}
							onChange={handleTimeWindowChange}
							onSlide={handleSlide}
							{isSliding}
						/>
					</div>

					<!-- Analysis Grid -->
					<AnalysisGrid
						{analyses}
						selectedId={selectedAnalysisId}
						{config}
						{globalSettings}
						onSelect={handleSelectAnalysis}
						onAdd={handleAddAnalysis}
						onRemove={handleRemoveAnalysis}
					/>
				</div>
			{/if}
		</main>

		<!-- Sidebar -->
		{#if showSidebar}
			<aside class="sidebar">
				{#if selectedAnalysis}
					<!-- Local Control Panel when analysis selected -->
					<LocalControlPanel
						analysis={selectedAnalysis}
						{globalSettings}
						onOverrideChange={handleLocalOverrideChange}
						onClearOverrides={handleClearOverrides}
					/>
				{:else}
					<!-- Global Control Panel -->
					<GlobalControlPanel
						settings={globalSettings}
						{audioBuffer}
						onChange={handleGlobalSettingsChange}
					/>
				{/if}
			</aside>
		{/if}
	</div>
</div>

<style>
	.observatory-container {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	.observatory-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 1rem 1.5rem;
		border-bottom: 1px solid var(--color-border);
		background-color: var(--color-card);
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.header-icon {
		width: 48px;
		height: 48px;
		background: var(--color-brand);
		border-radius: var(--radius-md);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-brand-foreground);
	}

	.header-content h1 {
		font-size: 1.25rem;
		font-weight: 600;
		margin: 0;
	}

	.header-content p {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		margin: 0;
	}

	.header-audio {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.observatory-layout {
		display: grid;
		grid-template-columns: 1fr;
		flex: 1;
		overflow: hidden;
	}

	.observatory-layout.sidebar-open {
		grid-template-columns: 1fr 320px;
	}

	.main-content {
		overflow: auto;
		padding: 1.5rem;
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.sidebar {
		border-left: 1px solid var(--color-border);
		overflow: auto;
		padding: 1rem;
		background-color: var(--color-card);
	}

	:global(.upload-card) {
		max-width: 600px;
		margin: 2rem auto;
	}

	:global(.upload-content) {
		padding: 2rem;
	}

	/* Focused View */
	.focused-view {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
		height: 100%;
	}

	.focused-header {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.focused-header h2 {
		font-size: 1.125rem;
		font-weight: 600;
		margin: 0;
	}

	.focused-content {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 1.5rem;
		flex: 1;
	}

	.focused-canvas {
		background-color: var(--color-card);
		border-radius: var(--radius-lg);
		border: 1px solid var(--color-border);
		overflow: hidden;
	}

	.focused-details {
		display: flex;
		flex-direction: column;
		gap: 1rem;
	}

	.thumbnail-strip {
		display: flex;
		gap: 0.75rem;
		overflow-x: auto;
		padding: 0.5rem 0;
	}

	.add-tile {
		width: 100px;
		height: 130px;
		border: 2px dashed var(--color-border);
		border-radius: var(--radius-lg);
		background: none;
		color: var(--color-muted-foreground);
		display: flex;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		transition: all 0.15s ease-out;
		flex-shrink: 0;
	}

	.add-tile:hover {
		border-color: var(--color-brand);
		color: var(--color-brand);
	}

	/* Grid View */
	.grid-view {
		display: flex;
		flex-direction: column;
		gap: 1.5rem;
	}

	.temporal-section {
		background-color: var(--color-card);
		border-radius: var(--radius-lg);
		border: 1px solid var(--color-border);
		padding: 1rem;
	}

	/* Responsive */
	@media (max-width: 1024px) {
		.observatory-layout.sidebar-open {
			grid-template-columns: 1fr;
		}

		.sidebar {
			position: fixed;
			right: 0;
			top: 0;
			bottom: 0;
			width: 320px;
			z-index: 100;
			box-shadow: var(--shadow-lg);
		}

		.focused-content {
			grid-template-columns: 1fr;
		}

		.focused-canvas {
			justify-self: center;
		}
	}

	@media (max-width: 768px) {
		.observatory-header {
			flex-wrap: wrap;
			gap: 0.75rem;
		}

		.header-audio {
			order: 3;
			width: 100%;
		}

		.main-content {
			padding: 1rem;
		}
	}
</style>
