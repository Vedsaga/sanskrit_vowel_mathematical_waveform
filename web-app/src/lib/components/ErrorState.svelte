<script lang="ts">
	/**
	 * ErrorState Component
	 * 
	 * A reusable error state component that displays an error message
	 * with an optional retry button.
	 * 
	 * Requirements: 6.2
	 */
	import { Button } from '$lib/components/ui/button';
	import { AlertCircle, RefreshCw } from '@lucide/svelte';

	interface Props {
		title?: string;
		message?: string;
		onRetry?: () => void;
		showRetry?: boolean;
		class?: string;
	}

	let {
		title = 'Something went wrong',
		message = 'An unexpected error occurred. Please try again.',
		onRetry,
		showRetry = true,
		class: className = ''
	}: Props = $props();
</script>

<div class="error-state {className}">
	<div class="error-icon">
		<AlertCircle size={32} />
	</div>
	<h3 class="error-title">{title}</h3>
	<p class="error-message">{message}</p>
	{#if showRetry && onRetry}
		<Button
			variant="outline"
			size="sm"
			onclick={onRetry}
			class="retry-btn"
		>
			<RefreshCw size={16} />
			<span>Try Again</span>
		</Button>
	{/if}
</div>

<style>
	.error-state {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		padding: 2rem;
		text-align: center;
		min-height: 200px;
	}

	.error-icon {
		width: 64px;
		height: 64px;
		border-radius: var(--radius-full);
		background-color: color-mix(in srgb, var(--color-destructive) 15%, var(--color-muted));
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
		color: var(--color-destructive);
	}

	.error-title {
		font-size: 1rem;
		font-weight: 600;
		color: var(--color-foreground);
		margin-bottom: 0.5rem;
	}

	.error-message {
		font-size: 0.875rem;
		color: var(--color-muted-foreground);
		max-width: 300px;
		margin-bottom: 1rem;
	}

	:global(.retry-btn) {
		display: flex;
		align-items: center;
		gap: 0.5rem;
	}
</style>
