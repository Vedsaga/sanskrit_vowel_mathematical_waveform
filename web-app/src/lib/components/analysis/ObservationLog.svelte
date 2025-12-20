<script lang="ts">
    /**
     * ObservationLog Component
     *
     * Expandable panel containing saved states for comparison.
     * Persists to localStorage.
     *
     * Phase 2: Task 2.4
     */
    import { onMount } from "svelte";
    import { Button } from "$lib/components/ui/button";
    import { Input } from "$lib/components/ui/input";
    import * as Card from "$lib/components/ui/card";
    import StateCard from "./StateCard.svelte";
    import {
        BookOpen,
        Plus,
        Save,
        X,
        ChevronDown,
        ChevronUp,
    } from "@lucide/svelte";
    import type {
        Shape,
        ShapeConfig,
        TimeWindow,
        GlobalSettings,
    } from "$lib/types";

    const STORAGE_KEY = "vak-observation-log";

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
        config: ShapeConfig;
        currentShapes?: Shape[];
        currentSettings?: GlobalSettings;
        currentFileName?: string;
        onLoadState?: (state: SavedState) => void;
        onOverlayState?: (state: SavedState) => void;
    }

    let {
        config,
        currentShapes = [],
        currentSettings,
        currentFileName = "",
        onLoadState,
        onOverlayState,
    }: Props = $props();

    let isExpanded = $state(false);
    let savedStates = $state<SavedState[]>([]);
    let newLabel = $state("");
    let showSaveForm = $state(false);

    /**
     * Loads saved states from localStorage
     */
    function loadFromStorage(): void {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                savedStates = JSON.parse(stored);
            }
        } catch (e) {
            console.error("Failed to load observation log:", e);
            savedStates = [];
        }
    }

    /**
     * Saves states to localStorage
     */
    function saveToStorage(): void {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(savedStates));
        } catch (e) {
            console.error("Failed to save observation log:", e);
        }
    }

    /**
     * Saves the current state
     */
    function handleSaveState(): void {
        if (!newLabel.trim() || !currentSettings) return;

        const newState: SavedState = {
            id: `state_${Date.now()}`,
            label: newLabel.trim(),
            audioFileName: currentFileName,
            timeWindow: currentSettings.timeWindow,
            frequencyRange: currentSettings.frequencyRange,
            shapes: currentShapes.map((s) => ({ ...s })),
            createdAt: Date.now(),
        };

        savedStates = [newState, ...savedStates];
        saveToStorage();

        newLabel = "";
        showSaveForm = false;
    }

    /**
     * Deletes a saved state
     */
    function handleDelete(id: string): void {
        savedStates = savedStates.filter((s) => s.id !== id);
        saveToStorage();
    }

    /**
     * Handles loading a state
     */
    function handleLoad(state: SavedState): void {
        onLoadState?.(state);
    }

    /**
     * Handles overlaying a state
     */
    function handleOverlay(state: SavedState): void {
        onOverlayState?.(state);
    }

    onMount(() => {
        loadFromStorage();
    });
</script>

<div class="observation-log">
    <Button
        variant="outline"
        size="sm"
        class="log-trigger"
        onclick={() => (isExpanded = !isExpanded)}
    >
        <BookOpen size={16} />
        Observations
        {#if savedStates.length > 0}
            <span class="badge">{savedStates.length}</span>
        {/if}
        {#if isExpanded}
            <ChevronUp size={14} />
        {:else}
            <ChevronDown size={14} />
        {/if}
    </Button>

    {#if isExpanded}
        <Card.Root class="log-panel">
            <Card.Header class="panel-header">
                <Card.Title class="panel-title">Observation Log</Card.Title>
                <Button
                    variant="ghost"
                    size="icon"
                    onclick={() => (isExpanded = false)}
                >
                    <X size={16} />
                </Button>
            </Card.Header>

            <Card.Content class="panel-content">
                <!-- Save Current State -->
                <div class="save-section">
                    {#if showSaveForm}
                        <div class="save-form">
                            <Input
                                bind:value={newLabel}
                                placeholder="State label..."
                            />
                            <div class="save-buttons">
                                <Button
                                    size="sm"
                                    onclick={handleSaveState}
                                    disabled={!newLabel.trim()}
                                >
                                    <Save size={14} />
                                    Save
                                </Button>
                                <Button
                                    size="sm"
                                    variant="ghost"
                                    onclick={() => {
                                        showSaveForm = false;
                                        newLabel = "";
                                    }}
                                >
                                    Cancel
                                </Button>
                            </div>
                        </div>
                    {:else}
                        <Button
                            variant="outline"
                            size="sm"
                            class="add-button"
                            onclick={() => (showSaveForm = true)}
                            disabled={currentShapes.length === 0}
                        >
                            <Plus size={16} />
                            Save Current State
                        </Button>
                    {/if}
                </div>

                <!-- Saved States List -->
                <div class="states-list">
                    {#if savedStates.length === 0}
                        <div class="empty-state">
                            <BookOpen size={24} />
                            <p>No saved observations</p>
                            <p class="empty-hint">
                                Save analysis states to compare later
                            </p>
                        </div>
                    {:else}
                        {#each savedStates as state (state.id)}
                            <StateCard
                                {state}
                                {config}
                                onLoad={handleLoad}
                                onOverlay={handleOverlay}
                                onDelete={handleDelete}
                            />
                        {/each}
                    {/if}
                </div>
            </Card.Content>
        </Card.Root>
    {/if}
</div>

<style>
    .observation-log {
        position: relative;
    }

    :global(.log-trigger) {
        gap: 0.5rem;
    }

    .badge {
        background-color: var(--color-brand);
        color: var(--color-brand-foreground);
        font-size: 0.65rem;
        padding: 0.125rem 0.375rem;
        border-radius: var(--radius-full);
        font-weight: 600;
    }

    :global(.log-panel) {
        position: absolute;
        top: 100%;
        right: 0;
        width: 280px;
        max-height: 400px;
        margin-top: 0.5rem;
        z-index: 50;
        overflow: hidden;
    }

    :global(.panel-header) {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1rem !important;
    }

    :global(.panel-title) {
        font-size: 0.875rem !important;
    }

    :global(.panel-content) {
        padding: 0.75rem 1rem !important;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        max-height: 300px;
        overflow-y: auto;
    }

    .save-section {
        flex-shrink: 0;
    }

    .save-form {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .save-buttons {
        display: flex;
        gap: 0.5rem;
    }

    :global(.add-button) {
        width: 100%;
        justify-content: center;
        gap: 0.5rem;
    }

    .states-list {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        overflow-y: auto;
    }

    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
        padding: 1rem;
        text-align: center;
        color: var(--color-muted-foreground);
    }

    .empty-state p {
        font-size: 0.75rem;
        margin: 0;
    }

    .empty-hint {
        font-size: 0.65rem !important;
        opacity: 0.7;
    }
</style>
