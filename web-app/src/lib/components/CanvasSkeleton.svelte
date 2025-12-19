<script lang="ts">
	/**
	 * CanvasSkeleton Component
	 * 
	 * A skeleton placeholder for the ShapeCanvas component while loading.
	 * Shows a pulsing animation with a centered icon.
	 * 
	 * Requirements: 6.1
	 */
	import { Skeleton } from '$lib/components/ui/skeleton';
	import { Loader2 } from '@lucide/svelte';

	interface Props {
		width?: number;
		height?: number;
	}

	let { width = 400, height = 400 }: Props = $props();
</script>

<div 
	class="canvas-skeleton"
	style="width: {width}px; height: {height}px;"
>
	<Skeleton class="absolute inset-0 rounded-xl" />
	<div class="skeleton-content">
		<div class="skeleton-icon">
			<Loader2 size={32} class="animate-spin" />
		</div>
		<p class="skeleton-text">Loading canvas...</p>
	</div>
</div>

<style>
	.canvas-skeleton {
		position: relative;
		background-color: var(--color-card);
		border-radius: var(--radius-xl);
		border: 1px solid var(--color-border);
		overflow: hidden;
	}

	.skeleton-content {
		position: absolute;
		inset: 0;
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
		gap: 0.75rem;
		z-index: 10;
	}

	.skeleton-icon {
		width: 64px;
		height: 64px;
		border-radius: var(--radius-full);
		background-color: var(--color-muted);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-muted-foreground);
	}

	.skeleton-text {
		font-size: 0.875rem;
		color: var(--color-muted-foreground);
	}
</style>
