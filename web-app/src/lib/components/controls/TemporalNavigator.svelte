<script lang="ts">
    /**
     * TemporalNavigator Component
     *
     * Time window control with waveform display and draggable handles.
     *
     * Phase 1: Task 1.1
     */
    import { onMount } from "svelte";
    import { Button } from "$lib/components/ui/button";
    import { Input } from "$lib/components/ui/input";
    import * as Select from "$lib/components/ui/select";
    import { Play, Pause, SkipForward } from "@lucide/svelte";
    import type { TimeWindow, WindowType } from "$lib/types";

    interface Props {
        audioBuffer: AudioBuffer | null;
        timeWindow: TimeWindow;
        onChange?: (window: Partial<TimeWindow>) => void;
        onSlide?: () => void;
        isSliding?: boolean;
    }

    let {
        audioBuffer,
        timeWindow,
        onChange,
        onSlide,
        isSliding = false,
    }: Props = $props();

    let waveformCanvas: HTMLCanvasElement;
    let containerWidth = $state(400);
    let isDragging = $state<"start" | "end" | "window" | null>(null);
    let dragStartX = $state(0);

    // Derived values
    let duration = $derived(audioBuffer?.duration ?? 0);
    let windowStart = $derived(timeWindow.start);
    let windowEnd = $derived(timeWindow.start + timeWindow.width / 1000);

    // Pixel calculations
    let pixelsPerSecond = $derived(
        duration > 0 ? containerWidth / duration : 0,
    );
    let windowStartPx = $derived(windowStart * pixelsPerSecond);
    let windowWidthPx = $derived((timeWindow.width / 1000) * pixelsPerSecond);

    function drawWaveform(): void {
        if (!waveformCanvas || !audioBuffer) return;

        const ctx = waveformCanvas.getContext("2d");
        if (!ctx) return;

        const { width, height } = waveformCanvas;
        const channelData = audioBuffer.getChannelData(0);
        const samples = channelData.length;
        const samplesPerPixel = samples / width;

        ctx.clearRect(0, 0, width, height);

        // Draw waveform
        ctx.fillStyle = "rgba(223, 114, 139, 0.3)";
        ctx.strokeStyle = "rgba(223, 114, 139, 0.6)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);

        for (let x = 0; x < width; x++) {
            const startSample = Math.floor(x * samplesPerPixel);
            const endSample = Math.floor((x + 1) * samplesPerPixel);

            let min = 0,
                max = 0;
            for (let i = startSample; i < endSample && i < samples; i++) {
                const sample = channelData[i];
                if (sample < min) min = sample;
                if (sample > max) max = sample;
            }

            const yMin = ((1 - max) * height) / 2;
            const yMax = ((1 - min) * height) / 2;

            ctx.lineTo(x, yMin);
        }

        ctx.lineTo(width, height / 2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    }

    function handleMouseDown(e: MouseEvent, type: "start" | "end" | "window") {
        isDragging = type;
        dragStartX = e.clientX;
    }

    function handleMouseMove(e: MouseEvent) {
        if (!isDragging || !duration) return;

        const deltaX = e.clientX - dragStartX;
        const deltaSeconds = deltaX / pixelsPerSecond;
        dragStartX = e.clientX;

        if (isDragging === "start") {
            const newStart = Math.max(
                0,
                Math.min(windowEnd - 0.1, windowStart + deltaSeconds),
            );
            const newWidth = (windowEnd - newStart) * 1000;
            onChange?.({ start: newStart, width: newWidth });
        } else if (isDragging === "end") {
            const newEnd = Math.max(
                windowStart + 0.1,
                Math.min(duration, windowEnd + deltaSeconds),
            );
            const newWidth = (newEnd - windowStart) * 1000;
            onChange?.({ width: newWidth });
        } else if (isDragging === "window") {
            const windowDuration = timeWindow.width / 1000;
            const newStart = Math.max(
                0,
                Math.min(duration - windowDuration, windowStart + deltaSeconds),
            );
            onChange?.({ start: newStart });
        }
    }

    function handleMouseUp() {
        isDragging = null;
    }

    function handleStartChange(e: Event) {
        const value = parseFloat((e.target as HTMLInputElement).value);
        if (!isNaN(value) && value >= 0) {
            onChange?.({
                start: Math.min(value, duration - timeWindow.width / 1000),
            });
        }
    }

    function handleWidthChange(e: Event) {
        const value = parseFloat((e.target as HTMLInputElement).value);
        if (!isNaN(value) && value > 0) {
            onChange?.({ width: value });
        }
    }

    function handleStepChange(e: Event) {
        const value = parseFloat((e.target as HTMLInputElement).value);
        if (!isNaN(value) && value > 0) {
            onChange?.({ step: value });
        }
    }

    function handleWindowTypeChange(value: string | undefined) {
        if (
            value &&
            ["hann", "rectangular", "hamming", "blackman"].includes(value)
        ) {
            onChange?.({ type: value as WindowType });
        }
    }

    onMount(() => {
        const resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                containerWidth = entry.contentRect.width;
                if (waveformCanvas) {
                    waveformCanvas.width = containerWidth;
                    drawWaveform();
                }
            }
        });

        if (waveformCanvas?.parentElement) {
            resizeObserver.observe(waveformCanvas.parentElement);
        }

        return () => resizeObserver.disconnect();
    });

    $effect(() => {
        audioBuffer;
        if (waveformCanvas) {
            drawWaveform();
        }
    });
</script>

<svelte:window onmousemove={handleMouseMove} onmouseup={handleMouseUp} />

<div class="temporal-navigator">
    <div class="navigator-header">
        <span class="navigator-label">Time Window</span>
        <span class="duration-label">
            {#if duration > 0}
                {duration.toFixed(2)}s total
            {:else}
                No audio
            {/if}
        </span>
    </div>

    <div class="waveform-container">
        <canvas bind:this={waveformCanvas} class="waveform-canvas" height="60"
        ></canvas>

        {#if audioBuffer && duration > 0}
            <!-- Window overlay -->
            <div
                class="window-overlay"
                style="left: {windowStartPx}px; width: {windowWidthPx}px;"
                role="slider"
                tabindex="0"
            >
                <!-- Start handle -->
                <div
                    class="window-handle start"
                    onmousedown={(e) => handleMouseDown(e, "start")}
                    role="slider"
                    tabindex="0"
                    aria-label="Window start"
                ></div>

                <!-- Draggable center -->
                <div
                    class="window-center"
                    onmousedown={(e) => handleMouseDown(e, "window")}
                ></div>

                <!-- End handle -->
                <div
                    class="window-handle end"
                    onmousedown={(e) => handleMouseDown(e, "end")}
                    role="slider"
                    tabindex="0"
                    aria-label="Window end"
                ></div>
            </div>
        {/if}
    </div>

    <div class="controls-row">
        <div class="input-group">
            <label for="start-time">Start (s)</label>
            <Input
                id="start-time"
                type="number"
                step="0.1"
                min="0"
                max={duration}
                value={timeWindow.start.toFixed(2)}
                onchange={handleStartChange}
                class="input-sm"
            />
        </div>

        <div class="input-group">
            <label for="window-width">Width (ms)</label>
            <Input
                id="window-width"
                type="number"
                step="50"
                min="50"
                value={timeWindow.width}
                onchange={handleWidthChange}
                class="input-sm"
            />
        </div>

        <div class="input-group">
            <label for="step-size">Step (ms)</label>
            <Input
                id="step-size"
                type="number"
                step="10"
                min="10"
                value={timeWindow.step}
                onchange={handleStepChange}
                class="input-sm"
            />
        </div>

        <div class="input-group">
            <label for="window-type">Window</label>
            <Select.Root
                type="single"
                value={timeWindow.type}
                onValueChange={handleWindowTypeChange}
            >
                <Select.Trigger id="window-type" class="select-sm">
                    {timeWindow.type}
                </Select.Trigger>
                <Select.Content>
                    <Select.Item value="hann">Hann</Select.Item>
                    <Select.Item value="hamming">Hamming</Select.Item>
                    <Select.Item value="blackman">Blackman</Select.Item>
                    <Select.Item value="rectangular">Rectangular</Select.Item>
                </Select.Content>
            </Select.Root>
        </div>
    </div>

    <div class="slide-controls">
        <Button
            variant="outline"
            size="sm"
            onclick={onSlide}
            disabled={!audioBuffer || isSliding}
        >
            {#if isSliding}
                <Pause size={16} />
                <span>Stop</span>
            {:else}
                <SkipForward size={16} />
                <span>Slide</span>
            {/if}
        </Button>
    </div>
</div>

<style>
    .temporal-navigator {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .navigator-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .navigator-label {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-foreground);
    }

    .duration-label {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
    }

    .waveform-container {
        position: relative;
        height: 60px;
        background-color: var(--color-muted);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    .waveform-canvas {
        width: 100%;
        height: 100%;
    }

    .window-overlay {
        position: absolute;
        top: 0;
        height: 100%;
        background-color: rgba(223, 114, 139, 0.2);
        border: 2px solid var(--color-brand);
        border-radius: var(--radius-sm);
        cursor: move;
    }

    .window-handle {
        position: absolute;
        top: 0;
        width: 8px;
        height: 100%;
        background-color: var(--color-brand);
        cursor: ew-resize;
    }

    .window-handle.start {
        left: -4px;
        border-radius: var(--radius-sm) 0 0 var(--radius-sm);
    }

    .window-handle.end {
        right: -4px;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    }

    .window-center {
        position: absolute;
        left: 8px;
        right: 8px;
        top: 0;
        height: 100%;
    }

    .controls-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
    }

    .input-group {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        flex: 1;
        min-width: 80px;
    }

    .input-group label {
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
    }

    :global(.input-sm) {
        height: 32px;
        font-size: 0.75rem;
    }

    :global(.select-sm) {
        height: 32px;
        font-size: 0.75rem;
    }

    .slide-controls {
        display: flex;
        justify-content: flex-end;
    }

    .slide-controls :global(button) {
        display: flex;
        align-items: center;
        gap: 0.375rem;
    }
</style>
