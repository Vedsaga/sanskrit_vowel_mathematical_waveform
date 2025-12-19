<script lang="ts">
	/**
	 * SyncControls Component
	 * 
	 * Provides controls for synchronization mode and displays
	 * the shared frequency scale range for comparison panels.
	 * 
	 * Requirements: 5.3, 5.4
	 */
	import { Button } from '$lib/components/ui/button';
	import * as ToggleGroup from '$lib/components/ui/toggle-group';
	import { Link, Unlink, Activity } from '@lucide/svelte';

	// Props
	interface Props {
		syncMode: 'independent' | 'synchronized';
		sharedFrequencyScale: { min: number; max: number };
		leftHasAudio: boolean;
		rightHasAudio: boolean;
		onSyncModeChange: (mode: 'independent' | 'synchronized') => void;
	}

	let {
		syncMode,
		sharedFrequencyScale,
		leftHasAudio,
		rightHasAudio,
		onSyncModeChange
	}: Props = $props();

	// Derived state
	let bothHaveAudio = $derived(leftHasAudio && rightHasAudio);

	/**
	 * Formats frequency for display
	 */
	function formatFrequency(hz: number): string {
		if (hz >= 1000) {
			return `${(hz / 1000).toFixed(1)}k`;
		}
		return `${Math.round(hz)}`;
	}

	/**
	 * Handles sync mode toggle
	 */
	function handleSyncModeChange(value: string | undefined) {
		if (value === 'independent' || value === 'synchronized') {
			onSyncModeChange(value);
		}
	}
</script>

<div class="sync-controls">
	<div class="sync-header">
		<div class="sync-icon">
			{#if syncMode === 'synchronized'}
				<Link size={16} />
			{:else}
				<Unlink size={16} />
			{/if}
		</div>
		<span class="sync-label">Comparison Mode</span>
	</div>

	<div class="sync-toggle">
		<ToggleGroup.Root
			type="single"
			value={syncMode}
			onValueChange={handleSyncModeChange}
			class="toggle-group"
		>
			<ToggleGroup.Item value="independent" class="toggle-item">
				<Unlink size={14} />
				<span>Independent</span>
			</ToggleGroup.Item>
			<ToggleGroup.Item value="synchronized" class="toggle-item">
				<Link size={14} />
				<span>Synchronized</span>
			</ToggleGroup.Item>
		</ToggleGroup.Root>
	</div>

	<div class="sync-description">
		{#if syncMode === 'synchronized'}
			<p>Both panels share the same frequency scale for accurate visual comparison.</p>
		{:else}
			<p>Each panel operates independently with its own frequency scale.</p>
		{/if}
	</div>

	{#if bothHaveAudio}
		<div class="frequency-scale">
			<div class="scale-header">
				<Activity size={14} />
				<span>Shared Frequency Range</span>
			</div>
			<div class="scale-values">
				<span class="scale-min">{formatFrequency(sharedFrequencyScale.min)} Hz</span>
				<div class="scale-bar"></div>
				<span class="scale-max">{formatFrequency(sharedFrequencyScale.max)} Hz</span>
			</div>
		</div>
	{:else}
		<div class="frequency-scale empty">
			<div class="scale-header">
				<Activity size={14} />
				<span>Frequency Range</span>
			</div>
			<p class="scale-empty-text">
				Upload audio to both panels to see the shared frequency range
			</p>
		</div>
	{/if}
</div>

<style>
	.sync-controls {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		padding: 1rem;
		background-color: var(--color-card);
		border-radius: var(--radius-lg);
		border: 1px solid var(--color-border);
	}

	.sync-header {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.sync-icon {
		width: 28px;
		height: 28px;
		border-radius: var(--radius-md);
		background-color: var(--color-muted);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-muted-foreground);
	}

	.sync-label {
		font-size: 0.875rem;
		font-weight: 600;
		color: var(--color-foreground);
	}

	.sync-toggle {
		display: flex;
		justify-content: center;
	}

	:global(.toggle-group) {
		display: flex;
		gap: 0;
		background-color: var(--color-muted);
		border-radius: var(--radius-md);
		padding: 0.25rem;
	}

	:global(.toggle-item) {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		padding: 0.5rem 0.75rem;
		font-size: 0.75rem;
		border-radius: var(--radius-sm);
		transition: all 0.15s ease-out;
	}

	:global(.toggle-item[data-state="on"]) {
		background-color: var(--color-background);
		color: var(--color-foreground);
		box-shadow: var(--shadow-sm);
	}

	:global(.toggle-item[data-state="off"]) {
		color: var(--color-muted-foreground);
	}

	.sync-description {
		text-align: center;
	}

	.sync-description p {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		line-height: 1.4;
	}

	.frequency-scale {
		padding: 0.75rem;
		background-color: var(--color-muted);
		border-radius: var(--radius-md);
	}

	.frequency-scale.empty {
		opacity: 0.7;
	}

	.scale-header {
		display: flex;
		align-items: center;
		gap: 0.375rem;
		margin-bottom: 0.5rem;
		color: var(--color-muted-foreground);
		font-size: 0.75rem;
	}

	.scale-values {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}

	.scale-min,
	.scale-max {
		font-size: 0.75rem;
		font-weight: 500;
		color: var(--color-foreground);
		font-variant-numeric: tabular-nums;
	}

	.scale-bar {
		flex: 1;
		height: 4px;
		background: linear-gradient(
			to right,
			var(--color-brand),
			color-mix(in srgb, var(--color-brand) 50%, var(--color-muted-foreground))
		);
		border-radius: 2px;
	}

	.scale-empty-text {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		text-align: center;
		margin: 0;
	}
</style>
