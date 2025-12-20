<script lang="ts">
    /**
     * StateCard Component
     *
     * Displays a saved analysis state with thumbnail, metadata, and actions.
     *
     * Phase 2: Task 2.4
     */
    import { Button } from "$lib/components/ui/button";
    import * as Card from "$lib/components/ui/card";
    import { Upload, Layers, Trash2, Clock, Music } from "@lucide/svelte";
    import type { Shape, ShapeConfig, TimeWindow } from "$lib/types";

    interface SavedState {
        id: string;
        label: string;
        audioFileName: string;
        timeWindow: TimeWindow;
        frequencyRange: { min: number; max: number };
        shapes: Shape[];
        createdAt: number;
    }

    interface Props {
        state: SavedState;
        config: ShapeConfig;
        onLoad?: (state: SavedState) => void;
        onOverlay?: (state: SavedState) => void;
        onDelete?: (id: string) => void;
    }

    let { state, config, onLoad, onOverlay, onDelete }: Props = $props();

    // Format time window
    let timeWindowStr = $derived(
        `${state.timeWindow.start.toFixed(2)}s - ${(state.timeWindow.start + state.timeWindow.width / 1000).toFixed(2)}s`,
    );

    // Format frequency range
    let freqRangeStr = $derived(
        `${state.frequencyRange.min}Hz - ${state.frequencyRange.max}Hz`,
    );

    // Format date
    let dateStr = $derived(
        new Date(state.createdAt).toLocaleDateString("en-US", {
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit",
        }),
    );
</script>

<Card.Root class="state-card">
    <Card.Header class="card-header">
        <div class="header-row">
            <Card.Title class="card-title">{state.label}</Card.Title>
            <Button
                variant="ghost"
                size="icon"
                class="delete-btn"
                onclick={() => onDelete?.(state.id)}
            >
                <Trash2 size={14} />
            </Button>
        </div>
        <div class="card-meta">
            <span class="meta-item">
                <Music size={12} />
                {state.audioFileName}
            </span>
            <span class="meta-item">
                <Clock size={12} />
                {dateStr}
            </span>
        </div>
    </Card.Header>

    <Card.Content class="card-content">
        <!-- Thumbnail placeholder -->
        <div class="thumbnail">
            <div class="shape-count">{state.shapes.length} shapes</div>
        </div>

        <div class="state-details">
            <div class="detail-row">
                <span class="detail-label">Time:</span>
                <span class="detail-value">{timeWindowStr}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Freq:</span>
                <span class="detail-value">{freqRangeStr}</span>
            </div>
        </div>
    </Card.Content>

    <Card.Footer class="card-footer">
        <Button variant="outline" size="sm" onclick={() => onLoad?.(state)}>
            <Upload size={14} />
            Load
        </Button>
        <Button variant="outline" size="sm" onclick={() => onOverlay?.(state)}>
            <Layers size={14} />
            Overlay
        </Button>
    </Card.Footer>
</Card.Root>

<style>
    :global(.state-card) {
        width: 200px;
        flex-shrink: 0;
    }

    :global(.card-header) {
        padding: 0.75rem;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
    }

    :global(.card-title) {
        font-size: 0.875rem;
        font-weight: 600;
        line-height: 1.2;
    }

    :global(.delete-btn) {
        width: 24px;
        height: 24px;
        opacity: 0.5;
    }

    :global(.delete-btn:hover) {
        opacity: 1;
        color: var(--color-destructive);
    }

    .card-meta {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }

    .meta-item {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.65rem;
        color: var(--color-muted-foreground);
    }

    .meta-item :global(svg) {
        flex-shrink: 0;
    }

    :global(.card-content) {
        padding: 0.75rem;
        padding-top: 0;
    }

    .thumbnail {
        width: 100%;
        height: 80px;
        background-color: var(--color-muted);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 0.5rem;
    }

    .shape-count {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
    }

    .state-details {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.65rem;
    }

    .detail-label {
        color: var(--color-muted-foreground);
    }

    .detail-value {
        color: var(--color-foreground);
        font-variant-numeric: tabular-nums;
    }

    :global(.card-footer) {
        padding: 0.75rem;
        padding-top: 0;
        display: flex;
        gap: 0.5rem;
    }

    :global(.card-footer button) {
        flex: 1;
        font-size: 0.75rem;
    }
</style>
