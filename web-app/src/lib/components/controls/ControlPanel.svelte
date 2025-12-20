<script lang="ts">
    /**
     * ControlPanel Component
     *
     * Unified floating control panel with tabs for Compose Lab.
     * Tabs: Compose (add shapes), Layers (manage shapes), Animate (rotation)
     *
     * Phase 4: Task 4.2
     */
    import * as Tabs from "$lib/components/ui/toggle-group";
    import * as Card from "$lib/components/ui/card";
    import { Button } from "$lib/components/ui/button";
    import { Input } from "$lib/components/ui/input";
    import { Slider } from "$lib/components/ui/slider";
    import { Checkbox } from "$lib/components/ui/checkbox";
    import { shapeStore } from "$lib/stores";
    import {
        Plus,
        Layers,
        PlayCircle,
        Trash2,
        Eye,
        EyeOff,
        RotateCcw,
        RotateCw,
        Pause,
    } from "@lucide/svelte";
    import type { Shape } from "$lib/types";

    // Props
    interface Props {
        class?: string;
    }

    let { class: className = "" }: Props = $props();

    // Store state
    const shapes = $derived(shapeStore.shapes);
    const rotation = $derived(shapeStore.rotation);
    const selectedIds = $derived(shapeStore.selectedIds);

    // Local state
    let activeTab = $state<"compose" | "layers" | "animate">("compose");
    let frequencyInput = $state("");
    let radiusInput = $state(80);

    // Speed presets
    const SPEED_PRESETS = [
        { label: "Slow", value: 0.5 },
        { label: "Medium", value: 1.0 },
        { label: "Fast", value: 2.0 },
    ];

    /**
     * Adds a new shape with the current inputs
     */
    function handleAddShape(): void {
        const fq = parseInt(frequencyInput, 10);
        if (isNaN(fq) || fq < 1) return;

        shapeStore.addShape(fq);
        frequencyInput = "";
    }

    /**
     * Gets a random color from the palette
     */
    function getRandomColor(): string {
        const colors = [
            "#df728b",
            "#6366f1",
            "#22c55e",
            "#f59e0b",
            "#06b6d4",
            "#8b5cf6",
            "#ec4899",
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    /**
     * Deletes a shape by ID
     */
    function handleDeleteShape(id: string): void {
        shapeStore.removeShape(id);
    }

    /**
     * Toggles shape visibility
     */
    function handleToggleVisibility(id: string): void {
        shapeStore.updateShape(id, {
            opacity:
                shapeStore.shapes.find((s) => s.id === id)?.opacity === 0
                    ? 1
                    : 0,
        });
    }

    /**
     * Starts rotation animation
     */
    function handleStartRotation(
        direction: "clockwise" | "counterclockwise",
    ): void {
        shapeStore.startRotation(direction, "loop");
    }

    /**
     * Stops rotation animation
     */
    function handleStopRotation(): void {
        shapeStore.stopRotation();
    }

    /**
     * Sets rotation speed
     */
    function handleSpeedChange(speed: number): void {
        shapeStore.setRotationSpeed(speed);
    }
</script>

<div class="control-panel {className}">
    <Card.Root class="panel-card">
        <!-- Tabs -->
        <div class="tab-bar">
            <button
                class="tab"
                class:active={activeTab === "compose"}
                onclick={() => (activeTab = "compose")}
            >
                <Plus size={16} />
                Compose
            </button>
            <button
                class="tab"
                class:active={activeTab === "layers"}
                onclick={() => (activeTab = "layers")}
            >
                <Layers size={16} />
                Layers
            </button>
            <button
                class="tab"
                class:active={activeTab === "animate"}
                onclick={() => (activeTab = "animate")}
            >
                <PlayCircle size={16} />
                Animate
            </button>
        </div>

        <Card.Content class="panel-content">
            <!-- Compose Tab -->
            {#if activeTab === "compose"}
                <div class="tab-content">
                    <div class="input-group">
                        <label>Frequency (fq)</label>
                        <Input
                            type="number"
                            bind:value={frequencyInput}
                            placeholder="e.g., 3"
                            min="1"
                        />
                    </div>

                    <div class="input-group">
                        <label>Radius: {radiusInput}px</label>
                        <Slider
                            type="single"
                            value={radiusInput}
                            onValueChange={(v) => (radiusInput = v)}
                            min={20}
                            max={150}
                            step={5}
                        />
                    </div>

                    <Button
                        onclick={handleAddShape}
                        disabled={!frequencyInput}
                        class="add-btn"
                    >
                        <Plus size={16} />
                        Add Shape
                    </Button>
                </div>
            {/if}

            <!-- Layers Tab -->
            {#if activeTab === "layers"}
                <div class="tab-content layers-tab">
                    {#if shapes.length === 0}
                        <div class="empty-layers">
                            <p>No shapes yet</p>
                        </div>
                    {:else}
                        <div class="layer-list">
                            {#each shapes as shape (shape.id)}
                                <div class="layer-item">
                                    <button
                                        class="color-dot"
                                        style="background-color: {shape.color};"
                                        aria-label="Shape color"
                                    ></button>
                                    <span class="layer-label"
                                        >fq={shape.fq}</span
                                    >
                                    <div class="layer-actions">
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onclick={() =>
                                                handleToggleVisibility(
                                                    shape.id,
                                                )}
                                        >
                                            {#if shape.opacity > 0}
                                                <Eye size={14} />
                                            {:else}
                                                <EyeOff size={14} />
                                            {/if}
                                        </Button>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onclick={() =>
                                                handleDeleteShape(shape.id)}
                                        >
                                            <Trash2 size={14} />
                                        </Button>
                                    </div>
                                </div>
                            {/each}
                        </div>
                    {/if}
                </div>
            {/if}

            <!-- Animate Tab -->
            {#if activeTab === "animate"}
                <div class="tab-content">
                    <div class="direction-controls">
                        <Button
                            variant={rotation.direction ===
                                "counterclockwise" && rotation.isAnimating
                                ? "default"
                                : "outline"}
                            onclick={() =>
                                handleStartRotation("counterclockwise")}
                        >
                            <RotateCcw size={16} />
                            CCW
                        </Button>
                        <Button
                            variant={!rotation.isAnimating
                                ? "default"
                                : "outline"}
                            onclick={handleStopRotation}
                        >
                            <Pause size={16} />
                            Stop
                        </Button>
                        <Button
                            variant={rotation.direction === "clockwise" &&
                            rotation.isAnimating
                                ? "default"
                                : "outline"}
                            onclick={() => handleStartRotation("clockwise")}
                        >
                            <RotateCw size={16} />
                            CW
                        </Button>
                    </div>

                    <div class="speed-presets">
                        <label>Speed</label>
                        <div class="preset-buttons">
                            {#each SPEED_PRESETS as preset}
                                <Button
                                    size="sm"
                                    variant={rotation.speed === preset.value
                                        ? "default"
                                        : "outline"}
                                    onclick={() =>
                                        handleSpeedChange(preset.value)}
                                >
                                    {preset.label}
                                </Button>
                            {/each}
                        </div>
                    </div>
                </div>
            {/if}
        </Card.Content>
    </Card.Root>
</div>

<style>
    .control-panel {
        position: fixed;
        right: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        z-index: 50;
    }

    :global(.panel-card) {
        width: 280px;
        background: color-mix(
            in srgb,
            var(--color-card) 85%,
            transparent
        ) !important;
        backdrop-filter: blur(16px);
        border: 1px solid
            color-mix(in srgb, var(--color-border) 50%, transparent) !important;
    }

    .tab-bar {
        display: flex;
        border-bottom: 1px solid var(--color-border);
    }

    .tab {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.375rem;
        padding: 0.75rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
        background: transparent;
        border: none;
        cursor: pointer;
        transition: all 0.15s ease;
    }

    .tab:hover {
        color: var(--color-foreground);
        background: color-mix(in srgb, var(--color-muted) 30%, transparent);
    }

    .tab.active {
        color: var(--color-brand);
        border-bottom: 2px solid var(--color-brand);
    }

    :global(.panel-content) {
        padding: 1rem !important;
    }

    .tab-content {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    .input-group {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
    }

    .input-group label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
    }

    :global(.add-btn) {
        width: 100%;
        gap: 0.5rem;
    }

    /* Layers Tab */
    .layers-tab {
        max-height: 300px;
        overflow-y: auto;
    }

    .empty-layers {
        text-align: center;
        padding: 2rem 1rem;
        color: var(--color-muted-foreground);
        font-size: 0.875rem;
    }

    .layer-list {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .layer-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: var(--radius-sm);
        transition: background 0.15s ease;
    }

    .layer-item:hover {
        background: color-mix(in srgb, var(--color-muted) 30%, transparent);
    }

    .color-dot {
        width: 16px;
        height: 16px;
        border-radius: 50%;
        border: none;
        cursor: pointer;
    }

    .layer-label {
        flex: 1;
        font-size: 0.8rem;
        font-variant-numeric: tabular-nums;
    }

    .layer-actions {
        display: flex;
        gap: 0.25rem;
    }

    /* Animate Tab */
    .direction-controls {
        display: flex;
        gap: 0.5rem;
    }

    .direction-controls :global(button) {
        flex: 1;
        gap: 0.25rem;
    }

    .speed-presets {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
    }

    .speed-presets label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
    }

    .preset-buttons {
        display: flex;
        gap: 0.375rem;
    }

    .preset-buttons :global(button) {
        flex: 1;
    }
</style>
