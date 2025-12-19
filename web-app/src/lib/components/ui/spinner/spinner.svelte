<script lang="ts">
	import { cn } from '$lib/utils';
	import type { HTMLAttributes } from 'svelte/elements';

	/**
	 * Spinner Component
	 * 
	 * A loading spinner that indicates processing is in progress.
	 * 
	 * Requirements: 6.1
	 */
	interface Props extends HTMLAttributes<HTMLDivElement> {
		class?: string;
		size?: 'sm' | 'md' | 'lg';
	}

	let { class: className, size = 'md', ...restProps }: Props = $props();

	const sizeClasses = {
		sm: 'h-4 w-4 border-2',
		md: 'h-6 w-6 border-2',
		lg: 'h-8 w-8 border-3'
	};
</script>

<div
	class={cn(
		'animate-spin rounded-full border-solid border-muted-foreground border-t-transparent',
		sizeClasses[size],
		className
	)}
	role="status"
	aria-label="Loading"
	{...restProps}
>
	<span class="sr-only">Loading...</span>
</div>

<style>
	.sr-only {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border-width: 0;
	}
</style>
