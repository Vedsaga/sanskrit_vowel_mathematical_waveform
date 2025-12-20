<script lang="ts">
    /**
     * AnalysisTile Component
     *
     * A mini-canvas tile for displaying a single analysis in the grid.
     * Shows geometry preview, label, and selection state.
     *
     * Phase 1: Task 1.6
     */
    import { onMount } from "svelte";
    import type {
        AnalysisState,
        ShapeConfig,
        GlobalSettings,
    } from "$lib/types";
    import { generateShapePoints } from "$lib/shapeEngine";
    import { X } from "@lucide/svelte";

    interface Props {
        analysis: AnalysisState;
        config: ShapeConfig;
        globalSettings: GlobalSettings;
        isSelected?: boolean;
        size?: number;
        onSelect?: (id: string) => void;
        onRemove?: (id: string) => void;
    }

    let {
        analysis,
        config,
        globalSettings,
        isSelected = false,
        size = 150,
        onSelect,
        onRemove,
    }: Props = $props();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = null;

    // Derived metrics
    let shapeCount = $derived(analysis.shapes.length);
    let stabilityDisplay = $derived(
        analysis.stabilityScore !== undefined
            ? `${Math.round(analysis.stabilityScore * 100)}%`
            : "--",
    );

    function initCanvas(): void {
        if (!canvas) return;
        ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        ctx.scale(dpr, dpr);
    }

    function renderShapes(): void {
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, size, size);

        // Draw subtle grid
        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.lineWidth = 1;
        const center = size / 2;
        ctx.beginPath();
        ctx.moveTo(center, 0);
        ctx.lineTo(center, size);
        ctx.moveTo(0, center);
        ctx.lineTo(size, center);
        ctx.stroke();
        ctx.restore();

        // Draw shapes
        const scale = (size - 20) / config.canvasSize;

        for (const shape of analysis.shapes) {
            const points = generateShapePoints(
                shape.fq,
                shape.R * scale,
                config.A * scale,
                shape.phi,
                Math.min(config.resolution, 180), // Lower resolution for thumbnails
            );

            if (points.length === 0) continue;

            ctx.save();
            ctx.strokeStyle = shape.color;
            ctx.globalAlpha = shape.opacity * 0.8;
            ctx.lineWidth = 1.5;
            ctx.lineCap = "round";
            ctx.lineJoin = "round";

            ctx.beginPath();
            ctx.moveTo(center + points[0].x, center + points[0].y);
            for (let i = 1; i < points.length; i++) {
                ctx.lineTo(center + points[i].x, center + points[i].y);
            }
            ctx.closePath();
            ctx.stroke();
            ctx.restore();
        }

        // If no shapes, show placeholder
        if (analysis.shapes.length === 0) {
            ctx.save();
            ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
            ctx.font = "12px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText("No shapes", center, center);
            ctx.restore();
        }
    }

    onMount(() => {
        initCanvas();
        renderShapes();
    });

    $effect(() => {
        analysis.shapes;
        config;
        if (ctx) {
            renderShapes();
        }
    });

    function handleClick() {
        onSelect?.(analysis.id);
    }

    function handleRemove(e: MouseEvent) {
        e.stopPropagation();
        onRemove?.(analysis.id);
    }
</script>

<div
    class="analysis-tile"
    class:selected={isSelected}
    style="--tile-size: {size}px"
    title="{analysis.label} | Shapes: {shapeCount} | Stability: {stabilityDisplay}"
>
    <button
        class="tile-button"
        onclick={handleClick}
        aria-label="Select {analysis.label}"
    >
        <canvas
            bind:this={canvas}
            class="tile-canvas"
            style="width: {size}px; height: {size}px;"
        ></canvas>

        <div class="tile-label">
            <span class="label-text">{analysis.label}</span>
            {#if shapeCount > 0}
                <span class="shape-count">{shapeCount}</span>
            {/if}
        </div>
    </button>

    {#if onRemove}
        <button
            class="remove-button"
            onclick={handleRemove}
            aria-label="Remove analysis"
        >
            <X size={14} />
        </button>
    {/if}
</div>

<style>
    .analysis-tile {
        position: relative;
        display: flex;
        flex-direction: column;
        background-color: var(--color-card);
        border: 2px solid var(--color-border);
        border-radius: var(--radius-lg);
        overflow: hidden;
        transition: all 0.2s ease-out;
        width: var(--tile-size);
    }

    .analysis-tile:hover {
        border-color: var(--color-muted-foreground);
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }

    .analysis-tile.selected {
        border-color: var(--color-brand);
        box-shadow: 0 0 0 3px
            color-mix(in srgb, var(--color-brand) 30%, transparent);
    }

    .tile-button {
        display: flex;
        flex-direction: column;
        background: none;
        border: none;
        padding: 0;
        cursor: pointer;
        width: 100%;
    }

    .tile-canvas {
        display: block;
        background-color: var(--color-background);
    }

    .tile-label {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.75rem;
        background-color: var(--color-muted);
        border-top: 1px solid var(--color-border);
    }

    .label-text {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-foreground);
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .shape-count {
        font-size: 0.65rem;
        padding: 0.125rem 0.375rem;
        background-color: var(--color-background);
        border-radius: var(--radius-sm);
        color: var(--color-muted-foreground);
    }

    .remove-button {
        position: absolute;
        top: 0.25rem;
        right: 0.25rem;
        width: 22px;
        height: 22px;
        border-radius: var(--radius-sm);
        background-color: rgba(0, 0, 0, 0.6);
        color: var(--color-foreground);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.15s ease-out;
        border: none;
        cursor: pointer;
    }

    .analysis-tile:hover .remove-button {
        opacity: 1;
    }

    .remove-button:hover {
        background-color: var(--color-destructive);
    }
</style>
