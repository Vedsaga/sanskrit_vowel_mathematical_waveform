<script lang="ts">
    /**
     * ShapePopover Component
     *
     * Contextual popover for editing selected shape properties.
     * Positioned near the selected shape on canvas.
     *
     * Phase 3: Task 3.2
     */
    import { Button } from "$lib/components/ui/button";
    import { Slider } from "$lib/components/ui/slider";
    import * as Card from "$lib/components/ui/card";
    import * as ToggleGroup from "$lib/components/ui/toggle-group";
    import { Trash2, RotateCcw, RotateCw, Pause, X } from "@lucide/svelte";
    import type { Shape, AnimationDirection, AnimationMode } from "$lib/types";

    // Color presets
    const COLOR_PRESETS = [
        "#df728b", // Brand pink
        "#6366f1", // Indigo
        "#22c55e", // Green
        "#f59e0b", // Amber
        "#06b6d4", // Cyan
        "#8b5cf6", // Violet
        "#ec4899", // Pink
        "#14b8a6", // Teal
        "#ffffff", // White
    ];

    interface Props {
        selectedShapes: Shape[];
        position: { x: number; y: number };
        onColorChange?: (color: string) => void;
        onOpacityChange?: (opacity: number) => void;
        onSpeedChange?: (speed: number) => void;
        onDirectionChange?: (direction: AnimationDirection) => void;
        onModeChange?: (mode: AnimationMode) => void;
        onDelete?: () => void;
        onClose?: () => void;
    }

    let {
        selectedShapes,
        position,
        onColorChange,
        onOpacityChange,
        onSpeedChange,
        onDirectionChange,
        onModeChange,
        onDelete,
        onClose,
    }: Props = $props();

    // Derived values from selection
    let selectionCount = $derived(selectedShapes.length);
    let isSingleShape = $derived(selectionCount === 1);
    let firstShape = $derived(selectedShapes[0]);

    // Current values (from first selected shape or defaults)
    let currentColor = $derived(firstShape?.color || "#df728b");
    let currentOpacity = $derived(firstShape?.opacity ?? 1);
    let currentSpeed = $derived(firstShape?.animationOverride?.speed ?? 1);
    let currentDirection = $derived(
        firstShape?.animationOverride?.direction ?? "none",
    );
    let currentMode = $derived(
        firstShape?.animationOverride?.mode ?? "continuous",
    );

    // Local state for controlled inputs
    let opacityValue = $state(100);
    let speedValue = $state(1);

    // Sync local state with derived values
    $effect(() => {
        opacityValue = Math.round(currentOpacity * 100);
        speedValue = currentSpeed;
    });

    function handleColorClick(color: string): void {
        onColorChange?.(color);
    }

    function handleOpacityChange(value: number): void {
        opacityValue = value;
        onOpacityChange?.(value / 100);
    }

    function handleSpeedChange(value: number): void {
        speedValue = value;
        onSpeedChange?.(value);
    }

    function handleDirectionChange(value: string | undefined): void {
        if (value) {
            onDirectionChange?.(value as AnimationDirection);
        }
    }

    function handleModeChange(value: string | undefined): void {
        if (value) {
            onModeChange?.(value as AnimationMode);
        }
    }

    function handleKeydown(event: KeyboardEvent): void {
        if (event.key === "Escape") {
            onClose?.();
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />

<div class="shape-popover" style="left: {position.x}px; top: {position.y}px;">
    <Card.Root class="popover-card">
        <Card.Header class="popover-header">
            <div class="header-row">
                <Card.Title class="popover-title">
                    {#if isSingleShape}
                        Shape
                    {:else}
                        {selectionCount} Shapes
                    {/if}
                </Card.Title>
                <Button variant="ghost" size="icon" onclick={onClose}>
                    <X size={14} />
                </Button>
            </div>
        </Card.Header>

        <Card.Content class="popover-content">
            <!-- Color Picker -->
            <div class="control-group">
                <span class="control-label">Color</span>
                <div class="color-grid">
                    {#each COLOR_PRESETS as color}
                        <button
                            class="color-swatch"
                            class:active={color === currentColor}
                            style="background-color: {color};"
                            onclick={() => handleColorClick(color)}
                            aria-label="Select color {color}"
                        ></button>
                    {/each}
                </div>
            </div>

            <!-- Opacity Slider -->
            <div class="control-group">
                <span class="control-label">
                    Opacity
                    <span class="value-label">{opacityValue}%</span>
                </span>
                <Slider
                    type="single"
                    value={opacityValue}
                    onValueChange={handleOpacityChange}
                    min={0}
                    max={100}
                    step={5}
                />
            </div>

            <!-- Rotation Speed -->
            <div class="control-group">
                <span class="control-label">
                    Speed
                    <span class="value-label"
                        >{speedValue.toFixed(1)} rad/s</span
                    >
                </span>
                <Slider
                    type="single"
                    value={speedValue}
                    onValueChange={handleSpeedChange}
                    min={0}
                    max={5}
                    step={0.1}
                />
            </div>

            <!-- Direction Toggle -->
            <div class="control-group">
                <span class="control-label">Direction</span>
                <ToggleGroup.Root
                    type="single"
                    value={currentDirection}
                    onValueChange={handleDirectionChange}
                    class="direction-toggle"
                >
                    <ToggleGroup.Item value="ccw" class="toggle-item">
                        <RotateCcw size={14} />
                    </ToggleGroup.Item>
                    <ToggleGroup.Item value="none" class="toggle-item">
                        <Pause size={14} />
                    </ToggleGroup.Item>
                    <ToggleGroup.Item value="cw" class="toggle-item">
                        <RotateCw size={14} />
                    </ToggleGroup.Item>
                </ToggleGroup.Root>
            </div>

            <!-- Loop Mode -->
            <div class="control-group">
                <span class="control-label">Loop</span>
                <ToggleGroup.Root
                    type="single"
                    value={currentMode}
                    onValueChange={handleModeChange}
                    class="mode-toggle"
                >
                    <ToggleGroup.Item value="continuous" class="toggle-item"
                        >Loop</ToggleGroup.Item
                    >
                    <ToggleGroup.Item value="once" class="toggle-item"
                        >Once</ToggleGroup.Item
                    >
                    <ToggleGroup.Item value="off" class="toggle-item"
                        >Off</ToggleGroup.Item
                    >
                </ToggleGroup.Root>
            </div>
        </Card.Content>

        <Card.Footer class="popover-footer">
            <Button
                variant="destructive"
                size="sm"
                onclick={onDelete}
                class="delete-btn"
            >
                <Trash2 size={14} />
                Delete
            </Button>
        </Card.Footer>
    </Card.Root>
</div>

<style>
    .shape-popover {
        position: absolute;
        z-index: 100;
        filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.3));
    }

    :global(.popover-card) {
        width: 220px;
        background-color: var(--color-card) !important;
        border-color: var(--color-border) !important;
    }

    :global(.popover-header) {
        padding: 0.75rem !important;
        padding-bottom: 0.5rem !important;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    :global(.popover-title) {
        font-size: 0.875rem !important;
    }

    :global(.popover-content) {
        padding: 0.75rem !important;
        padding-top: 0 !important;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .control-group {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
    }

    .control-label {
        font-size: 0.7rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .value-label {
        font-variant-numeric: tabular-nums;
        color: var(--color-foreground);
    }

    .color-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.25rem;
    }

    .color-swatch {
        width: 28px;
        height: 28px;
        border-radius: var(--radius-sm);
        border: 2px solid transparent;
        cursor: pointer;
        transition: transform 0.1s ease;
    }

    .color-swatch:hover {
        transform: scale(1.1);
    }

    .color-swatch.active {
        border-color: var(--color-foreground);
    }

    :global(.direction-toggle),
    :global(.mode-toggle) {
        width: 100%;
    }

    :global(.toggle-item) {
        flex: 1;
        font-size: 0.7rem;
    }

    :global(.popover-footer) {
        padding: 0.75rem !important;
        padding-top: 0 !important;
    }

    :global(.delete-btn) {
        width: 100%;
        gap: 0.5rem;
    }
</style>
