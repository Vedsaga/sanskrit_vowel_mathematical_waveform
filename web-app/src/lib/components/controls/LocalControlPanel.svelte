<script lang="ts">
    /**
     * LocalControlPanel Component
     *
     * Per-analysis override controls. Shows which settings are
     * inherited vs overridden.
     *
     * Phase 1: Task 1.6
     */
    import { Slider } from "$lib/components/ui/slider";
    import { Checkbox } from "$lib/components/ui/checkbox";
    import { Button } from "$lib/components/ui/button";
    import { RotateCcw, Unlink, Link } from "@lucide/svelte";
    import type {
        AnalysisState,
        GlobalSettings,
        FrequencyRange,
    } from "$lib/types";

    interface Props {
        analysis: AnalysisState;
        globalSettings: GlobalSettings;
        onOverrideChange?: (key: keyof AnalysisState, value: any) => void;
        onClearOverrides?: () => void;
    }

    let {
        analysis,
        globalSettings,
        onOverrideChange,
        onClearOverrides,
    }: Props = $props();

    // Check which settings are overridden
    let hasFrequencyOverride = $derived(analysis.frequencyRange !== undefined);
    let hasAmplitudeOverride = $derived(analysis.amplitude !== undefined);
    let hasNormalizeOverride = $derived(analysis.normalize !== undefined);

    // Get effective values
    let effectiveFrequencyRange = $derived(
        analysis.frequencyRange ?? globalSettings.frequencyRange,
    );
    let effectiveAmplitude = $derived(
        analysis.amplitude ?? globalSettings.amplitude,
    );
    let effectiveNormalize = $derived(
        analysis.normalize ?? globalSettings.normalize,
    );

    function toggleOverride(key: keyof AnalysisState, currentValue: any) {
        if ((analysis as any)[key] !== undefined) {
            // Clear override
            onOverrideChange?.(key, undefined);
        } else {
            // Set override to current global value
            onOverrideChange?.(key, currentValue);
        }
    }

    function handleFrequencyMinChange(value: number) {
        onOverrideChange?.("frequencyRange", {
            ...effectiveFrequencyRange,
            min: value,
        });
    }

    function handleFrequencyMaxChange(value: number) {
        onOverrideChange?.("frequencyRange", {
            ...effectiveFrequencyRange,
            max: value,
        });
    }

    function handleAmplitudeChange(value: number) {
        onOverrideChange?.("amplitude", value);
    }

    function handleNormalizeChange(checked: boolean | "indeterminate") {
        if (typeof checked === "boolean") {
            onOverrideChange?.("normalize", checked);
        }
    }
</script>

<div class="local-control-panel">
    <div class="panel-header">
        <h3 class="panel-title">{analysis.label}</h3>
        <Button
            variant="ghost"
            size="sm"
            onclick={onClearOverrides}
            title="Reset all to global"
        >
            <RotateCcw size={14} />
            <span>Reset</span>
        </Button>
    </div>

    <p class="panel-description">
        Override global settings for this analysis only.
    </p>

    <!-- Frequency Range -->
    <section class="control-section">
        <div class="section-header">
            <h4 class="section-title">Frequency Range</h4>
            <button
                class="override-toggle"
                class:active={hasFrequencyOverride}
                onclick={() =>
                    toggleOverride("frequencyRange", effectiveFrequencyRange)}
                title={hasFrequencyOverride
                    ? "Using local override"
                    : "Using global setting"}
            >
                {#if hasFrequencyOverride}
                    <Unlink size={12} />
                {:else}
                    <Link size={12} />
                {/if}
            </button>
        </div>
        <div class="range-control" class:inherited={!hasFrequencyOverride}>
            <label class="control-label"
                >Min: {effectiveFrequencyRange.min} Hz</label
            >
            <Slider
                type="single"
                value={effectiveFrequencyRange.min}
                onValueChange={handleFrequencyMinChange}
                min={20}
                max={effectiveFrequencyRange.max - 100}
                step={10}
                disabled={!hasFrequencyOverride}
            />
        </div>
        <div class="range-control" class:inherited={!hasFrequencyOverride}>
            <label class="control-label"
                >Max: {effectiveFrequencyRange.max} Hz</label
            >
            <Slider
                type="single"
                value={effectiveFrequencyRange.max}
                onValueChange={handleFrequencyMaxChange}
                min={effectiveFrequencyRange.min + 100}
                max={20000}
                step={100}
                disabled={!hasFrequencyOverride}
            />
        </div>
    </section>

    <!-- Amplitude -->
    <section class="control-section">
        <div class="section-header">
            <h4 class="section-title">Amplitude (A)</h4>
            <button
                class="override-toggle"
                class:active={hasAmplitudeOverride}
                onclick={() => toggleOverride("amplitude", effectiveAmplitude)}
            >
                {#if hasAmplitudeOverride}
                    <Unlink size={12} />
                {:else}
                    <Link size={12} />
                {/if}
            </button>
        </div>
        <div class="range-control" class:inherited={!hasAmplitudeOverride}>
            <label class="control-label">{effectiveAmplitude}</label>
            <Slider
                type="single"
                value={effectiveAmplitude}
                onValueChange={handleAmplitudeChange}
                min={1}
                max={50}
                step={1}
                disabled={!hasAmplitudeOverride}
            />
        </div>
    </section>

    <!-- Normalize Toggle -->
    <section class="control-section">
        <div class="toggle-row">
            <div class="toggle-left">
                <label for="local-normalize" class="toggle-label"
                    >Normalize Energy</label
                >
                {#if !hasNormalizeOverride}
                    <span class="inherited-badge">Inherited</span>
                {/if}
            </div>
            <div class="toggle-right">
                <button
                    class="override-toggle"
                    class:active={hasNormalizeOverride}
                    onclick={() =>
                        toggleOverride("normalize", effectiveNormalize)}
                >
                    {#if hasNormalizeOverride}
                        <Unlink size={12} />
                    {:else}
                        <Link size={12} />
                    {/if}
                </button>
                <Checkbox
                    id="local-normalize"
                    checked={effectiveNormalize}
                    onCheckedChange={handleNormalizeChange}
                    disabled={!hasNormalizeOverride}
                />
            </div>
        </div>
    </section>

    <!-- Metrics Summary -->
    <section class="metrics-section">
        <h4 class="section-title">Analysis Metrics</h4>
        <div class="metric-row">
            <span class="metric-label">Shapes</span>
            <span class="metric-value">{analysis.shapes.length}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Frequencies</span>
            <span class="metric-value"
                >{analysis.frequencyComponents.length}</span
            >
        </div>
        {#if analysis.stabilityScore !== undefined}
            <div class="metric-row">
                <span class="metric-label">Stability</span>
                <span class="metric-value"
                    >{Math.round(analysis.stabilityScore * 100)}%</span
                >
            </div>
        {/if}
    </section>
</div>

<style>
    .local-control-panel {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--color-foreground);
        margin: 0;
    }

    .panel-description {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
        margin: 0;
    }

    .control-section {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border);
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-title {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--color-muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.025em;
        margin: 0;
    }

    .override-toggle {
        width: 24px;
        height: 24px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--color-border);
        background: var(--color-muted);
        color: var(--color-muted-foreground);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.15s ease-out;
    }

    .override-toggle:hover {
        background: var(--color-background);
    }

    .override-toggle.active {
        background: var(--color-brand);
        border-color: var(--color-brand);
        color: var(--color-brand-foreground);
    }

    .range-control {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
    }

    .range-control.inherited {
        opacity: 0.6;
    }

    .control-label {
        font-size: 0.75rem;
        color: var(--color-foreground);
    }

    .toggle-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .toggle-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .toggle-right {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .toggle-label {
        font-size: 0.8rem;
        color: var(--color-foreground);
    }

    .inherited-badge {
        font-size: 0.6rem;
        padding: 0.125rem 0.25rem;
        background-color: var(--color-muted);
        color: var(--color-muted-foreground);
        border-radius: var(--radius-sm);
    }

    .metrics-section {
        display: flex;
        flex-direction: column;
        gap: 0.375rem;
        padding-top: 0.5rem;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
    }

    .metric-label {
        color: var(--color-muted-foreground);
    }

    .metric-value {
        color: var(--color-foreground);
        font-weight: 500;
        font-variant-numeric: tabular-nums;
    }
</style>
