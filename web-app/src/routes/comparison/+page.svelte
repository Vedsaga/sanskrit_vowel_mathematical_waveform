<script lang="ts">
	/**
	 * Convergence Studio Page
	 *
	 * Dual audio comparison with shared canvas and convergence detection.
	 *
	 * Phase 2: Task 2.6
	 */
	import { GitCompare, Settings } from "@lucide/svelte";
	import DualAnalysisGrid from "$lib/components/layout/DualAnalysisGrid.svelte";
	import ConvergenceIndicator from "$lib/components/analysis/ConvergenceIndicator.svelte";
	import ObservationLog from "$lib/components/analysis/ObservationLog.svelte";
	import { Button } from "$lib/components/ui/button";
	import * as ToggleGroup from "$lib/components/ui/toggle-group";
	import {
		comparisonStore,
		shapeStore,
		globalSettingsStore,
	} from "$lib/stores";
	import { computeQuickConvergence } from "$lib/utils/convergenceAnalysis";
	import type { GlobalSettings } from "$lib/types";

	// Store state
	const leftPanel = $derived(comparisonStore.leftPanel);
	const rightPanel = $derived(comparisonStore.rightPanel);
	const showSharedCanvas = $derived(comparisonStore.showSharedCanvas);
	const linkControls = $derived(comparisonStore.linkControls);
	const comparisonMode = $derived(comparisonStore.comparisonMode);
	const config = $derived(shapeStore.config);
	const globalSettings = $derived(globalSettingsStore.settings);

	// Real-time convergence
	let quickConvergence = $derived(
		computeQuickConvergence(leftPanel.shapes, rightPanel.shapes, config),
	);

	// Current file name for observation log
	let currentFileName = $derived(
		leftPanel.fileName || rightPanel.fileName || "",
	);

	// Combined shapes for observation log
	let currentShapes = $derived([...leftPanel.shapes, ...rightPanel.shapes]);

	/**
	 * Handles comparison mode change
	 */
	function handleModeChange(value: string | undefined): void {
		if (value) {
			comparisonStore.setComparisonMode(
				value as "none" | "overlay" | "intersection" | "difference",
			);
		}
	}

	/**
	 * Handles loading a saved state
	 */
	function handleLoadState(state: any): void {
		// TODO: Implement state loading
		console.log("Loading state:", state);
	}

	/**
	 * Handles overlaying a saved state
	 */
	function handleOverlayState(state: any): void {
		// TODO: Implement state overlay
		console.log("Overlaying state:", state);
	}
</script>

<div class="studio-container">
	<!-- Header -->
	<header class="studio-header">
		<div class="header-left">
			<div class="header-icon">
				<GitCompare size={28} />
			</div>
			<div class="header-content">
				<h1>Convergence Studio</h1>
				<p>Compare two audio files and find geometric convergence</p>
			</div>
		</div>

		<div class="header-center">
			{#if leftPanel.audioBuffer && rightPanel.audioBuffer}
				<ConvergenceIndicator
					result={{
						score: quickConvergence.score,
						label:
							quickConvergence.score >= 0.7
								? "High"
								: quickConvergence.score >= 0.4
									? "Moderate"
									: "Low",
						matchingPairs: [],
						commonFrequencies: [],
					}}
					compact
				/>
			{/if}
		</div>

		<div class="header-right">
			{#if showSharedCanvas}
				<ToggleGroup.Root
					type="single"
					value={comparisonMode}
					onValueChange={handleModeChange}
					class="mode-toggle"
				>
					<ToggleGroup.Item value="overlay" class="mode-item"
						>Overlay</ToggleGroup.Item
					>
					<ToggleGroup.Item value="intersection" class="mode-item"
						>Intersect</ToggleGroup.Item
					>
					<ToggleGroup.Item value="difference" class="mode-item"
						>Diff</ToggleGroup.Item
					>
				</ToggleGroup.Root>
			{/if}

			<ObservationLog
				{config}
				{currentShapes}
				currentSettings={globalSettings}
				{currentFileName}
				onLoadState={handleLoadState}
				onOverlayState={handleOverlayState}
			/>
		</div>
	</header>

	<!-- Main Content -->
	<main class="studio-main">
		<DualAnalysisGrid {globalSettings} />
	</main>

	<!-- Footer with convergence details -->
	{#if leftPanel.audioBuffer && rightPanel.audioBuffer && leftPanel.shapes.length > 0 && rightPanel.shapes.length > 0}
		<footer class="studio-footer">
			<ConvergenceIndicator
				result={{
					score: quickConvergence.score,
					label:
						quickConvergence.score >= 0.7
							? "Very High"
							: quickConvergence.score >= 0.4
								? "Moderate"
								: "Low",
					matchingPairs: [],
					commonFrequencies: [],
				}}
			/>
		</footer>
	{/if}
</div>

<style>
	.studio-container {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	.studio-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		padding: 1rem 1.5rem;
		border-bottom: 1px solid var(--color-border);
		background-color: var(--color-card);
		flex-wrap: wrap;
	}

	.header-left {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	.header-icon {
		width: 48px;
		height: 48px;
		background: linear-gradient(135deg, #f97316, #3b82f6);
		border-radius: var(--radius-md);
		display: flex;
		align-items: center;
		justify-content: center;
		color: white;
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

	.header-center {
		flex: 1;
		display: flex;
		justify-content: center;
	}

	.header-right {
		display: flex;
		align-items: center;
		gap: 0.75rem;
	}

	:global(.mode-toggle) {
		background-color: var(--color-muted);
		border-radius: var(--radius-md);
		padding: 0.125rem;
	}

	:global(.mode-item) {
		font-size: 0.75rem;
		padding: 0.375rem 0.75rem;
	}

	.studio-main {
		flex: 1;
		overflow: auto;
		padding: 1.5rem;
	}

	.studio-footer {
		padding: 1rem 1.5rem;
		border-top: 1px solid var(--color-border);
		background-color: var(--color-card);
	}

	@media (max-width: 768px) {
		.studio-header {
			flex-direction: column;
			align-items: stretch;
			gap: 0.75rem;
		}

		.header-center {
			order: 3;
		}

		.header-right {
			justify-content: flex-end;
		}

		.studio-main {
			padding: 1rem;
		}
	}
</style>
