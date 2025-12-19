<script lang="ts">
	import { cn } from '$lib/utils';
	import type { HTMLAttributes } from 'svelte/elements';

	/**
	 * Skeleton Component
	 * 
	 * A placeholder loading component that shows a pulsing animation
	 * to indicate content is loading.
	 * 
	 * Requirements: 6.1
	 */
	interface Props extends HTMLAttributes<HTMLDivElement> {
		class?: string;
	}

	let { class: className, ...restProps }: Props = $props();
</script>

<div
	class={cn(
		'animate-pulse rounded-md bg-muted',
		className
	)}
	{...restProps}
></div>
