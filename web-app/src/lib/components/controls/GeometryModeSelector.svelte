<script lang="ts">
    /**
     * GeometryModeSelector Component
     *
     * Toggle group for selecting geometry rendering mode:
     * - Single: Render only selected group
     * - Overlay: Render all selected groups overlaid
     * - Accumulation: Render accumulated geometry
     *
     * Phase 1: Task 1.4
     */
    import * as ToggleGroup from "$lib/components/ui/toggle-group";
    import { Layers, Square, BarChart3 } from "@lucide/svelte";
    import type { GeometryMode } from "$lib/types";

    interface Props {
        value: GeometryMode;
        onChange?: (mode: GeometryMode) => void;
        disabled?: boolean;
    }

    let { value, onChange, disabled = false }: Props = $props();

    function handleChange(newValue: string | undefined) {
        if (
            newValue &&
            (newValue === "single" ||
                newValue === "overlay" ||
                newValue === "accumulation")
        ) {
            onChange?.(newValue);
        }
    }
</script>

<div class="geometry-mode-selector">
    <div class="selector-header">
        <span class="selector-label">Geometry Mode</span>
    </div>

    <ToggleGroup.Root
        type="single"
        {value}
        onValueChange={handleChange}
        class="mode-toggle-group"
        {disabled}
    >
        <ToggleGroup.Item
            value="single"
            class="mode-toggle-item"
            title="Single Group"
        >
            <Square size={16} />
            <span>Single</span>
        </ToggleGroup.Item>

        <ToggleGroup.Item
            value="overlay"
            class="mode-toggle-item"
            title="Overlay Groups"
        >
            <Layers size={16} />
            <span>Overlay</span>
        </ToggleGroup.Item>

        <ToggleGroup.Item
            value="accumulation"
            class="mode-toggle-item"
            title="Time Accumulation"
        >
            <BarChart3 size={16} />
            <span>Accumulation</span>
        </ToggleGroup.Item>
    </ToggleGroup.Root>

    <p class="mode-description">
        {#if value === "single"}
            Render only the selected frequency group in isolation.
        {:else if value === "overlay"}
            Overlay all selected groups on the same canvas.
        {:else}
            Accumulate geometry from sliding time window.
        {/if}
    </p>
</div>

<style>
    .geometry-mode-selector {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .selector-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .selector-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-foreground);
    }

    :global(.mode-toggle-group) {
        display: flex;
        gap: 0;
        background-color: var(--color-muted);
        border-radius: var(--radius-md);
        padding: 0.25rem;
    }

    :global(.mode-toggle-item) {
        display: flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.5rem 0.75rem;
        font-size: 0.75rem;
        border-radius: var(--radius-sm);
        transition: all 0.15s ease-out;
        flex: 1;
        justify-content: center;
    }

    :global(.mode-toggle-item[data-state="on"]) {
        background-color: var(--color-background);
        color: var(--color-foreground);
        box-shadow: var(--shadow-sm);
    }

    :global(.mode-toggle-item[data-state="off"]) {
        color: var(--color-muted-foreground);
    }

    :global(.mode-toggle-item[data-disabled]) {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .mode-description {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
        line-height: 1.4;
    }
</style>
