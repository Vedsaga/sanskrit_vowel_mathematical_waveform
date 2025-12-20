<script lang="ts">
    /**
     * GunaStrengthIndicator Component
     *
     * Displays Guna analysis metrics:
     * - Stability Score (percentage bar)
     * - Energy Invariant (checkmark)
     * - Transient Score (percentage bar)
     *
     * Phase 1: Task 1.5
     */
    import type { GunaMetrics } from "$lib/utils/gunaAnalysis";
    import {
        formatStabilityScore,
        getStabilityColor,
        getTransientColor,
    } from "$lib/utils/gunaAnalysis";
    import { Check, X } from "@lucide/svelte";

    interface Props {
        metrics: GunaMetrics | null;
        compact?: boolean;
    }

    let { metrics, compact = false }: Props = $props();

    // Derived display values
    let stabilityPercent = $derived(
        metrics ? Math.round(metrics.stabilityScore * 100) : 0,
    );
    let transientPercent = $derived(
        metrics ? Math.round(metrics.transientScore * 100) : 0,
    );
    let stabilityColor = $derived(
        metrics ? getStabilityColor(metrics.stabilityScore) : "#6b7280",
    );
    let transientColor = $derived(
        metrics ? getTransientColor(metrics.transientScore) : "#6b7280",
    );
</script>

<div class="guna-indicator" class:compact>
    {#if metrics}
        <!-- Stability Score -->
        <div class="metric-row">
            <div class="metric-header">
                <span class="metric-label">Stability</span>
                <span class="metric-value" style="color: {stabilityColor}">
                    {stabilityPercent}%
                </span>
            </div>
            <div class="progress-bar">
                <div
                    class="progress-fill"
                    style="width: {stabilityPercent}%; background-color: {stabilityColor}"
                ></div>
            </div>
            {#if !compact}
                <span class="metric-description">{metrics.stabilityLabel}</span>
            {/if}
        </div>

        <!-- Energy Invariant -->
        <div class="metric-row inline">
            <span class="metric-label">Energy Invariant</span>
            <span
                class="invariant-badge"
                class:positive={metrics.energyInvariant}
            >
                {#if metrics.energyInvariant}
                    <Check size={12} />
                    <span>Yes</span>
                {:else}
                    <X size={12} />
                    <span>No</span>
                {/if}
            </span>
        </div>

        <!-- Transient Score -->
        <div class="metric-row">
            <div class="metric-header">
                <span class="metric-label">Transients</span>
                <span class="metric-value" style="color: {transientColor}">
                    {transientPercent}%
                </span>
            </div>
            <div class="progress-bar">
                <div
                    class="progress-fill"
                    style="width: {transientPercent}%; background-color: {transientColor}"
                ></div>
            </div>
            {#if !compact}
                <span class="metric-description">{metrics.transientLabel}</span>
            {/if}
        </div>
    {:else}
        <div class="empty-state">
            <span>No analysis data</span>
        </div>
    {/if}
</div>

<style>
    .guna-indicator {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .guna-indicator.compact {
        gap: 0.5rem;
        padding: 0.75rem;
    }

    .metric-row {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .metric-row.inline {
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .metric-label {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
    }

    .metric-value {
        font-size: 0.875rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    .progress-bar {
        height: 6px;
        background-color: var(--color-muted);
        border-radius: 3px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s ease-out;
    }

    .metric-description {
        font-size: 0.65rem;
        color: var(--color-muted-foreground);
    }

    .invariant-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.125rem 0.375rem;
        font-size: 0.7rem;
        font-weight: 500;
        border-radius: var(--radius-sm);
        background-color: var(--color-muted);
        color: var(--color-muted-foreground);
    }

    .invariant-badge.positive {
        background-color: color-mix(in srgb, #22c55e 20%, transparent);
        color: #22c55e;
    }

    .empty-state {
        padding: 1rem;
        text-align: center;
        color: var(--color-muted-foreground);
        font-size: 0.75rem;
    }
</style>
