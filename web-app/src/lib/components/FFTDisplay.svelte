<script lang="ts">
	/**
	 * FFTDisplay Component
	 * 
	 * Displays frequency components extracted from FFT analysis.
	 * Allows selection of components for shape generation.
	 * Optionally maps amplitude to visual properties.
	 * 
	 * Requirements: 4.3, 4.4, 4.8, 6.1
	 */
	import { Checkbox } from '$lib/components/ui/checkbox';
	import { Button } from '$lib/components/ui/button';
	import { Spinner } from '$lib/components/ui/spinner';
	import ErrorState from './ErrorState.svelte';
	import { Waves, Sparkles, CheckSquare, Square, Loader2 } from '@lucide/svelte';
	import type { FrequencyComponent } from '$lib/types';

	// Props
	interface Props {
		components?: FrequencyComponent[];
		onToggleSelection?: (id: string) => void;
		onSelectAll?: () => void;
		onDeselectAll?: () => void;
		onGenerateShapes?: (components: FrequencyComponent[]) => void;
		showAmplitudeMapping?: boolean;
		isProcessing?: boolean;
		error?: string | null;
		onRetry?: () => void;
	}

	let {
		components = [],
		onToggleSelection,
		onSelectAll,
		onDeselectAll,
		onGenerateShapes,
		showAmplitudeMapping = true,
		isProcessing = false,
		error = null,
		onRetry
	}: Props = $props();

	// Derived state
	let selectedComponents = $derived(components.filter(c => c.selected));
	let hasSelection = $derived(selectedComponents.length > 0);
	let allSelected = $derived(components.length > 0 && selectedComponents.length === components.length);

	/**
	 * Formats frequency in Hz to a readable string
	 */
	function formatFrequency(hz: number): string {
		if (hz >= 1000) {
			return `${(hz / 1000).toFixed(2)} kHz`;
		}
		return `${hz.toFixed(1)} Hz`;
	}

	/**
	 * Formats magnitude as percentage
	 */
	function formatMagnitude(magnitude: number): string {
		return `${(magnitude * 100).toFixed(1)}%`;
	}

	/**
	 * Gets opacity based on magnitude (for amplitude mapping)
	 */
	function getOpacityFromMagnitude(magnitude: number): number {
		// Map magnitude (0-1) to opacity (0.3-1.0)
		return 0.3 + magnitude * 0.7;
	}

	/**
	 * Gets color intensity based on magnitude
	 */
	function getColorFromMagnitude(magnitude: number): string {
		// Interpolate from muted to brand color based on magnitude
		const intensity = Math.round(magnitude * 100);
		return `color-mix(in srgb, var(--color-brand) ${intensity}%, var(--color-muted-foreground))`;
	}

	/**
	 * Handles checkbox change
	 */
	function handleCheckboxChange(id: string) {
		onToggleSelection?.(id);
	}

	/**
	 * Handles generate shapes button click
	 */
	function handleGenerateShapes() {
		onGenerateShapes?.(selectedComponents);
	}

	/**
	 * Handles select/deselect all toggle
	 */
	function handleToggleAll() {
		if (allSelected) {
			onDeselectAll?.();
		} else {
			onSelectAll?.();
		}
	}
</script>

<div class="fft-display">
	{#if error}
		<!-- Error state -->
		<ErrorState
			title="Analysis Failed"
			message={error}
			onRetry={onRetry}
			showRetry={!!onRetry}
		/>
	{:else if isProcessing}
		<!-- Processing state -->
		<div class="processing-state">
			<div class="processing-icon">
				<Loader2 size={32} class="animate-spin" />
			</div>
			<p class="processing-title">Analyzing audio...</p>
			<p class="processing-subtitle">Computing frequency components</p>
		</div>
	{:else if components.length === 0}
		<!-- Empty state -->
		<div class="empty-state">
			<div class="empty-icon">
				<Waves size={32} />
			</div>
			<p class="empty-title">No frequency data</p>
			<p class="empty-subtitle">Upload an audio file to analyze its frequency components</p>
		</div>
	{:else}
		<!-- Header with actions -->
		<div class="fft-header">
			<div class="header-info">
				<h3 class="header-title">Frequency Components</h3>
				<p class="header-subtitle">
					{selectedComponents.length} of {components.length} selected
				</p>
			</div>
			<div class="header-actions">
				<Button
					variant="ghost"
					size="sm"
					onclick={handleToggleAll}
					class="select-all-btn"
				>
					{#if allSelected}
						<Square size={16} />
						<span>Deselect All</span>
					{:else}
						<CheckSquare size={16} />
						<span>Select All</span>
					{/if}
				</Button>
			</div>
		</div>

		<!-- Component list -->
		<div class="component-list">
			{#each components as component (component.id)}
				<button
					type="button"
					class="component-item"
					class:selected={component.selected}
					onclick={() => handleCheckboxChange(component.id)}
					style={showAmplitudeMapping ? `--item-opacity: ${getOpacityFromMagnitude(component.magnitude)}` : ''}
				>
					<div class="component-checkbox">
						<Checkbox
							checked={component.selected}
							onCheckedChange={() => handleCheckboxChange(component.id)}
							aria-label={`Select ${formatFrequency(component.frequencyHz)}`}
						/>
					</div>
					
					<div class="component-info">
						<div class="component-frequency">
							{formatFrequency(component.frequencyHz)}
						</div>
						<div class="component-fq">
							fq = {component.fq}
						</div>
					</div>

					<div class="component-magnitude">
						<div 
							class="magnitude-bar"
							style="width: {component.magnitude * 100}%; background-color: {showAmplitudeMapping ? getColorFromMagnitude(component.magnitude) : 'var(--color-brand)'}"
						></div>
						<span class="magnitude-value">{formatMagnitude(component.magnitude)}</span>
					</div>

					{#if showAmplitudeMapping}
						<div class="component-preview">
							<div 
								class="preview-dot"
								style="opacity: {getOpacityFromMagnitude(component.magnitude)}; background-color: {getColorFromMagnitude(component.magnitude)}"
							></div>
						</div>
					{/if}
				</button>
			{/each}
		</div>

		<!-- Generate shapes button -->
		<div class="fft-footer">
			<Button
				onclick={handleGenerateShapes}
				disabled={!hasSelection}
				class="generate-btn"
			>
				<Sparkles size={16} />
				<span>Generate {selectedComponents.length} Shape{selectedComponents.length !== 1 ? 's' : ''}</span>
			</Button>
		</div>
	{/if}
</div>

<style>
	.fft-display {
		display: flex;
		flex-direction: column;
		height: 100%;
		min-height: 300px;
	}

	/* Processing state */
	.processing-state {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 2rem;
		text-align: center;
		color: var(--color-muted-foreground);
	}

	.processing-icon {
		width: 64px;
		height: 64px;
		border-radius: var(--radius-full);
		background-color: color-mix(in srgb, var(--color-brand) 15%, var(--color-muted));
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
		color: var(--color-brand);
	}

	.processing-title {
		font-size: 1rem;
		font-weight: 500;
		color: var(--color-foreground);
		margin-bottom: 0.25rem;
	}

	.processing-subtitle {
		font-size: 0.875rem;
	}

	/* Empty state */
	.empty-state {
		flex: 1;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 2rem;
		text-align: center;
		color: var(--color-muted-foreground);
	}

	.empty-icon {
		width: 64px;
		height: 64px;
		border-radius: var(--radius-full);
		background-color: var(--color-muted);
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
	}

	.empty-title {
		font-size: 1rem;
		font-weight: 500;
		color: var(--color-foreground);
		margin-bottom: 0.25rem;
	}

	.empty-subtitle {
		font-size: 0.875rem;
	}

	/* Header */
	.fft-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 0.75rem 1rem;
		border-bottom: 1px solid var(--color-border);
		background-color: var(--color-muted);
		border-radius: var(--radius-lg) var(--radius-lg) 0 0;
	}

	.header-title {
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--color-foreground);
	}

	.header-subtitle {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
	}

	.header-actions {
		display: flex;
		gap: 0.5rem;
	}

	:global(.select-all-btn) {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		font-size: 0.75rem;
	}

	/* Component list */
	.component-list {
		flex: 1;
		overflow-y: auto;
		padding: 0.5rem;
		display: flex;
		flex-direction: column;
		gap: 0.25rem;
	}

	.component-item {
		display: flex;
		align-items: center;
		gap: 0.75rem;
		padding: 0.625rem 0.75rem;
		border-radius: var(--radius-md);
		background-color: transparent;
		border: 1px solid transparent;
		cursor: pointer;
		transition: all 0.15s ease-out;
		width: 100%;
		text-align: left;
	}

	.component-item:hover {
		background-color: var(--color-muted);
	}

	.component-item.selected {
		background-color: color-mix(in srgb, var(--color-brand) 10%, var(--color-card));
		border-color: color-mix(in srgb, var(--color-brand) 30%, transparent);
	}

	.component-checkbox {
		flex-shrink: 0;
	}

	.component-info {
		flex: 1;
		min-width: 0;
	}

	.component-frequency {
		font-size: 0.875rem;
		font-weight: 500;
		color: var(--color-foreground);
		font-variant-numeric: tabular-nums;
	}

	.component-fq {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
	}

	.component-magnitude {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		width: 120px;
		flex-shrink: 0;
	}

	.magnitude-bar {
		height: 4px;
		border-radius: 2px;
		transition: width 0.2s ease-out;
	}

	.magnitude-value {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		font-variant-numeric: tabular-nums;
		min-width: 40px;
		text-align: right;
	}

	.component-preview {
		flex-shrink: 0;
		width: 24px;
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.preview-dot {
		width: 12px;
		height: 12px;
		border-radius: var(--radius-full);
		transition: all 0.2s ease-out;
	}

	/* Footer */
	.fft-footer {
		padding: 0.75rem 1rem;
		border-top: 1px solid var(--color-border);
		background-color: var(--color-muted);
		border-radius: 0 0 var(--radius-lg) var(--radius-lg);
	}

	:global(.generate-btn) {
		width: 100%;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 0.5rem;
	}
</style>
