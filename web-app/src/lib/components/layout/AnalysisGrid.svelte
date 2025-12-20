<script lang="ts">
    /**
     * AnalysisGrid Component
     *
     * Responsive grid layout for multiple analysis tiles.
     * Includes "Add Analysis" button and focused mode support.
     *
     * Phase 1: Task 1.6
     */
    import type {
        AnalysisState,
        ShapeConfig,
        GlobalSettings,
    } from "$lib/types";
    import AnalysisTile from "./AnalysisTile.svelte";
    import { Plus } from "@lucide/svelte";

    interface Props {
        analyses: AnalysisState[];
        selectedId: string | null;
        config: ShapeConfig;
        globalSettings: GlobalSettings;
        tileSize?: number;
        onSelect?: (id: string | null) => void;
        onAdd?: () => void;
        onRemove?: (id: string) => void;
    }

    let {
        analyses,
        selectedId,
        config,
        globalSettings,
        tileSize = 150,
        onSelect,
        onAdd,
        onRemove,
    }: Props = $props();

    // Derived state
    let hasAnalyses = $derived(analyses.length > 0);
    let selectedAnalysis = $derived(
        selectedId ? analyses.find((a) => a.id === selectedId) : null,
    );
    let isInFocusedMode = $derived(selectedId !== null);

    function handleTileSelect(id: string) {
        // Toggle selection
        if (selectedId === id) {
            onSelect?.(null);
        } else {
            onSelect?.(id);
        }
    }

    function handleAdd() {
        onAdd?.();
    }

    function handleRemove(id: string) {
        onRemove?.(id);
    }

    function handleBackdropClick() {
        onSelect?.(null);
    }
</script>

<div class="analysis-grid-container" class:focused={isInFocusedMode}>
    {#if isInFocusedMode && selectedAnalysis}
        <!-- Focused Mode: Large selected analysis -->
        <div class="focused-view">
            <div class="focused-main">
                <AnalysisTile
                    analysis={selectedAnalysis}
                    {config}
                    {globalSettings}
                    isSelected={true}
                    size={400}
                />
            </div>

            <!-- Thumbnail strip at bottom -->
            <div class="thumbnail-strip">
                {#each analyses as analysis (analysis.id)}
                    <AnalysisTile
                        {analysis}
                        {config}
                        {globalSettings}
                        isSelected={analysis.id === selectedId}
                        size={100}
                        onSelect={handleTileSelect}
                        onRemove={handleRemove}
                    />
                {/each}

                {#if onAdd}
                    <button class="add-tile mini" onclick={handleAdd}>
                        <Plus size={20} />
                    </button>
                {/if}
            </div>
        </div>
    {:else}
        <!-- Grid Mode -->
        <div class="grid-view" style="--tile-size: {tileSize}px">
            {#each analyses as analysis (analysis.id)}
                <AnalysisTile
                    {analysis}
                    {config}
                    {globalSettings}
                    isSelected={analysis.id === selectedId}
                    size={tileSize}
                    onSelect={handleTileSelect}
                    onRemove={handleRemove}
                />
            {/each}

            {#if onAdd}
                <button
                    class="add-tile"
                    onclick={handleAdd}
                    style="--tile-size: {tileSize}px"
                >
                    <Plus size={32} />
                    <span>Add Analysis</span>
                </button>
            {/if}
        </div>

        {#if !hasAnalyses && !onAdd}
            <div class="empty-state">
                <p>No analyses yet. Upload audio to begin.</p>
            </div>
        {/if}
    {/if}
</div>

<style>
    .analysis-grid-container {
        width: 100%;
        min-height: 200px;
    }

    .grid-view {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(var(--tile-size), 1fr));
        gap: 1rem;
        justify-items: center;
    }

    .focused-view {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }

    .focused-main {
        display: flex;
        justify-content: center;
        padding: 1rem;
        background-color: var(--color-muted);
        border-radius: var(--radius-xl);
    }

    .thumbnail-strip {
        display: flex;
        gap: 0.75rem;
        overflow-x: auto;
        padding: 0.75rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .thumbnail-strip::-webkit-scrollbar {
        height: 6px;
    }

    .thumbnail-strip::-webkit-scrollbar-track {
        background: var(--color-muted);
        border-radius: 3px;
    }

    .thumbnail-strip::-webkit-scrollbar-thumb {
        background: var(--color-muted-foreground);
        border-radius: 3px;
    }

    .add-tile {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        width: var(--tile-size);
        height: calc(var(--tile-size) + 32px);
        background-color: var(--color-muted);
        border: 2px dashed var(--color-border);
        border-radius: var(--radius-lg);
        color: var(--color-muted-foreground);
        cursor: pointer;
        transition: all 0.2s ease-out;
    }

    .add-tile:hover {
        border-color: var(--color-brand);
        color: var(--color-brand);
        background-color: color-mix(
            in srgb,
            var(--color-brand) 10%,
            var(--color-muted)
        );
    }

    .add-tile span {
        font-size: 0.75rem;
        font-weight: 500;
    }

    .add-tile.mini {
        width: 100px;
        height: 100px;
        flex-shrink: 0;
    }

    .add-tile.mini span {
        display: none;
    }

    .empty-state {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 200px;
        color: var(--color-muted-foreground);
        font-size: 0.875rem;
    }

    @media (max-width: 768px) {
        .grid-view {
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 0.75rem;
        }

        .focused-main {
            padding: 0.5rem;
        }

        .thumbnail-strip {
            gap: 0.5rem;
            padding: 0.5rem;
        }
    }
</style>
