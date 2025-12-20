<script lang="ts">
    /**
     * SharedCanvas Component
     *
     * Renders shapes from both audio panels on a single canvas.
     * Supports comparison modes: overlay, intersection, difference.
     *
     * Phase 2: Task 2.2
     */
    import { onMount } from "svelte";
    import { generateShapePoints } from "$lib/shapeEngine";
    import type { Shape, ShapeConfig, Point } from "$lib/types";

    // Color palettes for each source
    const LEFT_COLORS = ["#f97316", "#ef4444", "#dc2626", "#fb923c"]; // Warm
    const RIGHT_COLORS = ["#3b82f6", "#06b6d4", "#8b5cf6", "#22d3ee"]; // Cool

    interface Props {
        leftShapes: Shape[];
        rightShapes: Shape[];
        config: ShapeConfig;
        comparisonMode?: "none" | "overlay" | "intersection" | "difference";
        width?: number;
        height?: number;
    }

    let {
        leftShapes,
        rightShapes,
        config,
        comparisonMode = "overlay",
        width = 400,
        height = 400,
    }: Props = $props();

    let canvas = $state<HTMLCanvasElement | null>(null);
    let ctx: CanvasRenderingContext2D | null = null;
    let dpr = $state(1);

    function initCanvas(): void {
        if (!canvas) return;

        ctx = canvas.getContext("2d");
        if (!ctx) return;

        dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);
    }

    function clearCanvas(): void {
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);
    }

    function drawGrid(): void {
        if (!ctx) return;

        const centerX = width / 2;
        const centerY = height / 2;

        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;

        // Center crosshair
        ctx.beginPath();
        ctx.moveTo(centerX, 0);
        ctx.lineTo(centerX, height);
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.stroke();

        ctx.restore();
    }

    function getColorForSource(
        source: "left" | "right",
        index: number,
    ): string {
        const palette = source === "left" ? LEFT_COLORS : RIGHT_COLORS;
        return palette[index % palette.length];
    }

    function renderShape(
        shape: Shape,
        source: "left" | "right",
        index: number,
    ): void {
        if (!ctx) return;

        const centerX = width / 2;
        const centerY = height / 2;

        const points = generateShapePoints(
            shape.fq,
            shape.R,
            config.A,
            shape.phi,
            config.resolution,
        );

        if (points.length === 0) return;

        ctx.save();

        // Use source-specific color in overlay mode
        const color =
            comparisonMode === "overlay"
                ? getColorForSource(source, index)
                : shape.color;

        ctx.strokeStyle = color;
        ctx.globalAlpha = comparisonMode === "overlay" ? 0.7 : shape.opacity;
        ctx.lineWidth = shape.strokeWidth;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        ctx.beginPath();
        ctx.moveTo(centerX + points[0].x, centerY + points[0].y);

        for (let i = 1; i < points.length; i++) {
            ctx.lineTo(centerX + points[i].x, centerY + points[i].y);
        }

        ctx.closePath();
        ctx.stroke();

        ctx.restore();
    }

    function renderOverlayMode(): void {
        // Render left shapes first (warm colors)
        leftShapes.forEach((shape, i) => renderShape(shape, "left", i));

        // Render right shapes on top (cool colors)
        rightShapes.forEach((shape, i) => renderShape(shape, "right", i));
    }

    function renderIntersectionMode(): void {
        if (!ctx) return;

        // Create temporary canvases for each source
        const leftCanvas = document.createElement("canvas");
        const rightCanvas = document.createElement("canvas");
        leftCanvas.width = width;
        leftCanvas.height = height;
        rightCanvas.width = width;
        rightCanvas.height = height;

        const leftCtx = leftCanvas.getContext("2d");
        const rightCtx = rightCanvas.getContext("2d");

        if (!leftCtx || !rightCtx) return;

        const centerX = width / 2;
        const centerY = height / 2;

        // Draw left shapes
        leftCtx.fillStyle = "white";
        leftShapes.forEach((shape) => {
            const points = generateShapePoints(
                shape.fq,
                shape.R,
                config.A,
                shape.phi,
                config.resolution,
            );
            if (points.length === 0) return;

            leftCtx.beginPath();
            leftCtx.moveTo(centerX + points[0].x, centerY + points[0].y);
            points.forEach((p) => leftCtx.lineTo(centerX + p.x, centerY + p.y));
            leftCtx.closePath();
            leftCtx.fill();
        });

        // Draw right shapes
        rightCtx.fillStyle = "white";
        rightShapes.forEach((shape) => {
            const points = generateShapePoints(
                shape.fq,
                shape.R,
                config.A,
                shape.phi,
                config.resolution,
            );
            if (points.length === 0) return;

            rightCtx.beginPath();
            rightCtx.moveTo(centerX + points[0].x, centerY + points[0].y);
            points.forEach((p) =>
                rightCtx.lineTo(centerX + p.x, centerY + p.y),
            );
            rightCtx.closePath();
            rightCtx.fill();
        });

        // Compute intersection using composite operation
        leftCtx.globalCompositeOperation = "destination-in";
        leftCtx.drawImage(rightCanvas, 0, 0);

        // Draw intersection to main canvas
        ctx.globalAlpha = 0.8;
        ctx.drawImage(leftCanvas, 0, 0);

        // Also draw outlines for context
        ctx.globalAlpha = 0.3;
        leftShapes.forEach((shape, i) => renderShape(shape, "left", i));
        rightShapes.forEach((shape, i) => renderShape(shape, "right", i));
    }

    function renderDifferenceMode(): void {
        // Render both with different opacities to show differences
        if (!ctx) return;

        ctx.save();
        ctx.globalCompositeOperation = "screen";

        // Left shapes in warm color
        ctx.globalAlpha = 0.6;
        leftShapes.forEach((shape, i) => {
            renderShape({ ...shape, color: LEFT_COLORS[0] }, "left", i);
        });

        // Right shapes in cool color
        rightShapes.forEach((shape, i) => {
            renderShape({ ...shape, color: RIGHT_COLORS[0] }, "right", i);
        });

        ctx.restore();

        // Draw labels
        ctx.save();
        ctx.font = "12px Inter, sans-serif";
        ctx.fillStyle = LEFT_COLORS[0];
        ctx.fillText("Audio A", 10, 20);
        ctx.fillStyle = RIGHT_COLORS[0];
        ctx.fillText("Audio B", 10, 35);
        ctx.restore();
    }

    function render(): void {
        if (!ctx) return;

        clearCanvas();
        drawGrid();

        if (leftShapes.length === 0 && rightShapes.length === 0) {
            // Empty state
            ctx.save();
            ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
            ctx.font = "14px Inter, sans-serif";
            ctx.textAlign = "center";
            ctx.fillText(
                "Load audio in both panels to compare",
                width / 2,
                height / 2,
            );
            ctx.restore();
            return;
        }

        switch (comparisonMode) {
            case "intersection":
                renderIntersectionMode();
                break;
            case "difference":
                renderDifferenceMode();
                break;
            case "overlay":
            default:
                renderOverlayMode();
                break;
        }
    }

    onMount(() => {
        initCanvas();
        render();

        const handleResize = () => {
            initCanvas();
            render();
        };

        window.addEventListener("resize", handleResize);
        return () => window.removeEventListener("resize", handleResize);
    });

    $effect(() => {
        leftShapes;
        rightShapes;
        config;
        comparisonMode;
        if (ctx) {
            render();
        }
    });
</script>

<div class="shared-canvas">
    <div class="canvas-header">
        <span class="mode-label"
            >{comparisonMode.charAt(0).toUpperCase() + comparisonMode.slice(1)} Mode</span
        >
        <div class="legend">
            <span class="legend-item left">Audio A</span>
            <span class="legend-item right">Audio B</span>
        </div>
    </div>
    <canvas
        bind:this={canvas}
        class="canvas-element"
        style="width: {width}px; height: {height}px;"
    ></canvas>
</div>

<style>
    .shared-canvas {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        height: 100%;
    }

    .canvas-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.75rem;
    }

    .mode-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-foreground);
    }

    .legend {
        display: flex;
        gap: 1rem;
    }

    .legend-item {
        font-size: 0.65rem;
        padding: 0.125rem 0.5rem;
        border-radius: var(--radius-sm);
    }

    .legend-item.left {
        background-color: rgba(249, 115, 22, 0.2);
        color: #f97316;
    }

    .legend-item.right {
        background-color: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }

    .canvas-element {
        flex: 1;
        width: 100%;
        background-color: var(--color-background);
        border-radius: var(--radius-md);
    }
</style>
