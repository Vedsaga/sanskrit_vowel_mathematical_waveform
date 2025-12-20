<script lang="ts">
    /**
     * SpectrumGraph Component
     *
     * Visual frequency plot showing magnitude vs frequency.
     * Clickable peaks to select/deselect frequency components.
     *
     * Phase 1: Task 1.2
     */
    import { onMount } from "svelte";
    import type { FrequencyComponent } from "$lib/types";

    interface Props {
        components: FrequencyComponent[];
        frequencyRange?: { min: number; max: number };
        height?: number;
        onComponentClick?: (id: string) => void;
    }

    let {
        components,
        frequencyRange = { min: 20, max: 20000 },
        height = 200,
        onComponentClick,
    }: Props = $props();

    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D | null = null;
    let width = $state(600);

    // Sorted components by frequency
    let sortedComponents = $derived(
        [...components].sort((a, b) => a.frequencyHz - b.frequencyHz),
    );

    // Max magnitude for scaling
    let maxMagnitude = $derived(
        components.length > 0
            ? Math.max(...components.map((c) => c.magnitude))
            : 1,
    );

    function initCanvas(): void {
        if (!canvas) return;
        ctx = canvas.getContext("2d");
        if (!ctx) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();
        width = rect.width;

        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);
    }

    function freqToX(freq: number): number {
        // Logarithmic frequency scale
        const minLog = Math.log10(frequencyRange.min);
        const maxLog = Math.log10(frequencyRange.max);
        const freqLog = Math.log10(Math.max(freq, frequencyRange.min));
        return ((freqLog - minLog) / (maxLog - minLog)) * width;
    }

    function magToY(magnitude: number): number {
        const normalized = magnitude / maxMagnitude;
        return height - normalized * (height - 30); // Leave space for labels
    }

    function draw(): void {
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, width, height);

        // Draw grid
        drawGrid();

        // Draw spectrum line
        drawSpectrum();

        // Draw peaks
        drawPeaks();

        // Draw frequency labels
        drawLabels();
    }

    function drawGrid(): void {
        if (!ctx) return;

        ctx.save();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
        ctx.lineWidth = 1;

        // Horizontal grid lines (magnitude)
        for (let i = 0; i <= 4; i++) {
            const y = (height - 30) * (i / 4) + 5;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        // Vertical grid lines (frequency - log scale)
        const freqMarkers = [100, 500, 1000, 5000, 10000];
        for (const freq of freqMarkers) {
            if (freq >= frequencyRange.min && freq <= frequencyRange.max) {
                const x = freqToX(freq);
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, height - 25);
                ctx.stroke();
            }
        }

        ctx.restore();
    }

    function drawSpectrum(): void {
        if (!ctx || sortedComponents.length === 0) return;

        ctx.save();
        ctx.strokeStyle = "rgba(99, 102, 241, 0.5)";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.moveTo(0, height - 30);

        for (const comp of sortedComponents) {
            const x = freqToX(comp.frequencyHz);
            const y = magToY(comp.magnitude);
            ctx.lineTo(x, y);
        }

        ctx.lineTo(width, height - 30);
        ctx.stroke();

        // Fill area
        ctx.fillStyle = "rgba(99, 102, 241, 0.1)";
        ctx.fill();

        ctx.restore();
    }

    function drawPeaks(): void {
        if (!ctx) return;

        for (const comp of components) {
            const x = freqToX(comp.frequencyHz);
            const y = magToY(comp.magnitude);

            ctx.save();

            // Peak marker
            const isSelected = comp.selected;
            const radius = isSelected ? 6 : 4;

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);

            if (isSelected) {
                ctx.fillStyle = "var(--color-brand, #df728b)";
                ctx.strokeStyle = "white";
                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();
            } else {
                ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
                ctx.fill();
            }

            ctx.restore();
        }
    }

    function drawLabels(): void {
        if (!ctx) return;

        ctx.save();
        ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
        ctx.font = "10px Inter, sans-serif";
        ctx.textAlign = "center";

        const freqLabels = [100, 1000, 10000];
        for (const freq of freqLabels) {
            if (freq >= frequencyRange.min && freq <= frequencyRange.max) {
                const x = freqToX(freq);
                const label = freq >= 1000 ? `${freq / 1000}kHz` : `${freq}Hz`;
                ctx.fillText(label, x, height - 5);
            }
        }

        ctx.restore();
    }

    function handleClick(e: MouseEvent): void {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Find clicked component
        for (const comp of components) {
            const cx = freqToX(comp.frequencyHz);
            const cy = magToY(comp.magnitude);
            const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

            if (dist < 10) {
                onComponentClick?.(comp.id);
                break;
            }
        }
    }

    onMount(() => {
        initCanvas();
        draw();

        const resizeObserver = new ResizeObserver(() => {
            initCanvas();
            draw();
        });
        resizeObserver.observe(canvas);

        return () => resizeObserver.disconnect();
    });

    $effect(() => {
        components;
        frequencyRange;
        if (ctx) {
            draw();
        }
    });
</script>

<div class="spectrum-graph">
    <div class="graph-header">
        <span class="graph-title">Frequency Spectrum</span>
        <span class="graph-subtitle">Click peaks to select</span>
    </div>
    <canvas
        bind:this={canvas}
        class="spectrum-canvas"
        style="height: {height}px"
        onclick={handleClick}
    ></canvas>
</div>

<style>
    .spectrum-graph {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .graph-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .graph-title {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-foreground);
    }

    .graph-subtitle {
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
    }

    .spectrum-canvas {
        width: 100%;
        cursor: crosshair;
        border-radius: var(--radius-md);
        background-color: var(--color-background);
    }
</style>
