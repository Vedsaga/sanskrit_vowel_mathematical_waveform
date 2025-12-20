<script lang="ts">
    /**
     * GlobalControlPanel Component
     *
     * Sidebar panel containing all global controls:
     * - Time Window (TemporalNavigator)
     * - Frequency Range
     * - Amplitude
     * - Rotation
     * - Toggles (Normalize, Suppress Transients)
     * - Geometry Mode
     *
     * Phase 1: Task 1.6
     */
    import TemporalNavigator from "./TemporalNavigator.svelte";
    import GeometryModeSelector from "./GeometryModeSelector.svelte";
    import { Slider } from "$lib/components/ui/slider";
    import { Checkbox } from "$lib/components/ui/checkbox";
    import { Button } from "$lib/components/ui/button";
    import { RotateCw, RotateCcw, Play, Pause } from "@lucide/svelte";
    import type { GlobalSettings, TimeWindow, GeometryMode } from "$lib/types";

    interface Props {
        settings: GlobalSettings;
        audioBuffer: AudioBuffer | null;
        onChange?: (settings: Partial<GlobalSettings>) => void;
        onRotationToggle?: () => void;
    }

    let { settings, audioBuffer, onChange, onRotationToggle }: Props = $props();

    function handleTimeWindowChange(window: Partial<TimeWindow>) {
        onChange?.({ timeWindow: { ...settings.timeWindow, ...window } });
    }

    function handleFrequencyMinChange(value: number) {
        onChange?.({
            frequencyRange: { ...settings.frequencyRange, min: value },
        });
    }

    function handleFrequencyMaxChange(value: number) {
        onChange?.({
            frequencyRange: { ...settings.frequencyRange, max: value },
        });
    }

    function handleAmplitudeChange(value: number) {
        onChange?.({ amplitude: value });
    }

    function handleSpeedChange(value: number) {
        onChange?.({
            rotation: { ...settings.rotation, speed: value },
        });
    }

    function toggleDirection() {
        const newDir =
            settings.rotation.direction === "clockwise"
                ? "counterclockwise"
                : "clockwise";
        onChange?.({ rotation: { ...settings.rotation, direction: newDir } });
    }

    function handleNormalizeChange(checked: boolean | "indeterminate") {
        if (typeof checked === "boolean") {
            onChange?.({ normalize: checked });
        }
    }

    function handleTransientChange(checked: boolean | "indeterminate") {
        if (typeof checked === "boolean") {
            onChange?.({ suppressTransients: checked });
        }
    }

    function handleGeometryModeChange(mode: GeometryMode) {
        onChange?.({ geometryMode: mode });
    }
</script>

<div class="global-control-panel">
    <h3 class="panel-title">Global Controls</h3>

    <!-- Time Window -->
    <section class="control-section">
        <TemporalNavigator
            {audioBuffer}
            timeWindow={settings.timeWindow}
            onChange={handleTimeWindowChange}
        />
    </section>

    <!-- Frequency Range -->
    <section class="control-section">
        <h4 class="section-title">Frequency Range</h4>
        <div class="range-control">
            <label class="control-label"
                >Min: {settings.frequencyRange.min} Hz</label
            >
            <Slider
                type="single"
                value={settings.frequencyRange.min}
                onValueChange={handleFrequencyMinChange}
                min={20}
                max={settings.frequencyRange.max - 100}
                step={10}
            />
        </div>
        <div class="range-control">
            <label class="control-label"
                >Max: {settings.frequencyRange.max} Hz</label
            >
            <Slider
                type="single"
                value={settings.frequencyRange.max}
                onValueChange={handleFrequencyMaxChange}
                min={settings.frequencyRange.min + 100}
                max={20000}
                step={100}
            />
        </div>
    </section>

    <!-- Amplitude -->
    <section class="control-section">
        <h4 class="section-title">Wiggle Amplitude (A)</h4>
        <div class="range-control">
            <label class="control-label">{settings.amplitude}</label>
            <Slider
                type="single"
                value={settings.amplitude}
                onValueChange={handleAmplitudeChange}
                min={1}
                max={50}
                step={1}
            />
        </div>
    </section>

    <!-- Rotation -->
    <section class="control-section">
        <h4 class="section-title">Rotation</h4>
        <div class="rotation-controls">
            <Button variant="outline" size="icon" onclick={onRotationToggle}>
                {#if settings.rotation.isAnimating}
                    <Pause size={16} />
                {:else}
                    <Play size={16} />
                {/if}
            </Button>

            <Button
                variant="ghost"
                size="icon"
                onclick={toggleDirection}
                title={settings.rotation.direction}
            >
                {#if settings.rotation.direction === "clockwise"}
                    <RotateCw size={16} />
                {:else}
                    <RotateCcw size={16} />
                {/if}
            </Button>

            <div class="speed-slider">
                <label class="control-label"
                    >Speed: {settings.rotation.speed.toFixed(1)}x</label
                >
                <Slider
                    type="single"
                    value={settings.rotation.speed}
                    onValueChange={handleSpeedChange}
                    min={0.1}
                    max={3}
                    step={0.1}
                />
            </div>
        </div>
    </section>

    <!-- Toggles -->
    <section class="control-section">
        <h4 class="section-title">Options</h4>
        <div class="toggle-row">
            <label for="normalize" class="toggle-label">Normalize Energy</label>
            <Checkbox
                id="normalize"
                checked={settings.normalize}
                onCheckedChange={handleNormalizeChange}
            />
        </div>
        <div class="toggle-row">
            <label for="transients" class="toggle-label"
                >Suppress Transients</label
            >
            <Checkbox
                id="transients"
                checked={settings.suppressTransients}
                onCheckedChange={handleTransientChange}
            />
        </div>
    </section>

    <!-- Geometry Mode -->
    <section class="control-section">
        <GeometryModeSelector
            value={settings.geometryMode}
            onChange={handleGeometryModeChange}
        />
    </section>
</div>

<style>
    .global-control-panel {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
        overflow-y: auto;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--color-foreground);
        margin: 0;
    }

    .control-section {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border);
    }

    .control-section:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.025em;
        margin: 0;
    }

    .range-control {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
    }

    .control-label {
        font-size: 0.75rem;
        color: var(--color-foreground);
    }

    .rotation-controls {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .speed-slider {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .toggle-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.25rem 0;
    }

    .toggle-label {
        font-size: 0.8rem;
        color: var(--color-foreground);
    }
</style>
