<script lang="ts">
    import { Sparkles } from "@lucide/svelte";
    import ShapeCanvas from "$lib/components/ShapeCanvas.svelte";
    import ControlPanel from "$lib/components/controls/ControlPanel.svelte";
    import ShapePopover from "$lib/components/controls/ShapePopover.svelte";
    import { shapeStore } from "$lib/stores";
    import { getShapeCenter, getSelectionBounds } from "$lib/utils/hitTesting";
    import type { Shape } from "$lib/types";

    /**
     * Compose Lab Page
     *
     * Full-screen canvas for manual frequency shape visualization
     * with floating glass control panel.
     *
     * Phase 4: Compose Lab
     */

    // Store state
    const shapes = $derived(shapeStore.shapes);
    const config = $derived(shapeStore.config);
    const selectedIds = $derived(shapeStore.selectedIds);

    // Popover state
    let showPopover = $state(false);
    let popoverPosition = $state({ x: 0, y: 0 });

    // Get selected shapes
    let selectedShapes = $derived(shapes.filter((s) => selectedIds.has(s.id)));

    /**
     * Handles shape click from canvas
     */
    function handleShapeClick(shapeId: string | null, event: MouseEvent): void {
        if (shapeId && selectedIds.has(shapeId)) {
            // Clicked on selected shape - show popover
            showPopover = true;
            popoverPosition = { x: event.clientX + 10, y: event.clientY - 50 };
        } else if (!shapeId) {
            // Clicked empty space - close popover
            showPopover = false;
        }
    }

    /**
     * Handles selection change from canvas
     */
    function handleSelectionChange(newSelection: Set<string>): void {
        shapeStore.setSelectedIds(newSelection);
        if (newSelection.size === 0) {
            showPopover = false;
        }
    }

    /**
     * Popover handlers
     */
    function handleColorChange(color: string): void {
        selectedIds.forEach((id) => shapeStore.updateShape(id, { color }));
    }

    function handleOpacityChange(opacity: number): void {
        selectedIds.forEach((id) => shapeStore.updateShape(id, { opacity }));
    }

    function handleSpeedChange(speed: number): void {
        selectedIds.forEach((id) => {
            const shape = shapes.find((s) => s.id === id);
            if (shape) {
                shapeStore.updateShape(id, {
                    animationOverride: { ...shape.animationOverride, speed },
                });
            }
        });
    }

    function handleDirectionChange(
        direction: import("$lib/types").AnimationDirection,
    ): void {
        selectedIds.forEach((id) => {
            const shape = shapes.find((s) => s.id === id);
            if (shape) {
                shapeStore.updateShape(id, {
                    animationOverride: {
                        ...shape.animationOverride,
                        direction,
                    },
                });
            }
        });
    }

    function handleModeChange(mode: import("$lib/types").AnimationMode): void {
        selectedIds.forEach((id) => {
            const shape = shapes.find((s) => s.id === id);
            if (shape) {
                shapeStore.updateShape(id, {
                    animationOverride: { ...shape.animationOverride, mode },
                });
            }
        });
    }

    function handleDeleteSelected(): void {
        selectedIds.forEach((id) => shapeStore.removeShape(id));
        showPopover = false;
    }

    function handleClosePopover(): void {
        showPopover = false;
    }
</script>

<svelte:head>
    <title>Compose Lab | Vak</title>
</svelte:head>

<div class="compose-lab">
    <!-- Page Title Overlay -->
    <div class="title-overlay">
        <Sparkles size={20} />
        <span>Compose Lab</span>
    </div>

    <!-- Full-Screen Canvas -->
    <div class="canvas-wrapper">
        <ShapeCanvas
            {shapes}
            {config}
            {selectedIds}
            width={800}
            height={600}
            showGrid={true}
            onShapeClick={handleShapeClick}
            onSelectionChange={handleSelectionChange}
        />
    </div>

    <!-- Floating Control Panel -->
    <ControlPanel />

    <!-- Shape Popover (when shapes selected) -->
    {#if showPopover && selectedShapes.length > 0}
        <ShapePopover
            {selectedShapes}
            position={popoverPosition}
            onColorChange={handleColorChange}
            onOpacityChange={handleOpacityChange}
            onSpeedChange={handleSpeedChange}
            onDirectionChange={handleDirectionChange}
            onModeChange={handleModeChange}
            onDelete={handleDeleteSelected}
            onClose={handleClosePopover}
        />
    {/if}
</div>

<style>
    .compose-lab {
        position: relative;
        width: 100%;
        height: 100vh;
        overflow: hidden;
        background: radial-gradient(
            ellipse at center,
            color-mix(in srgb, var(--color-card) 100%, transparent) 0%,
            var(--color-background) 70%
        );
    }

    .title-overlay {
        position: fixed;
        top: 1rem;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: color-mix(in srgb, var(--color-card) 80%, transparent);
        backdrop-filter: blur(12px);
        border-radius: var(--radius-full);
        border: 1px solid var(--color-border);
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-brand);
        z-index: 40;
    }

    .canvas-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
    }

    /* Responsive adjustments */
    @media (max-width: 900px) {
        .compose-lab :global(.control-panel) {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            top: auto;
            transform: none;
        }
    }
</style>
