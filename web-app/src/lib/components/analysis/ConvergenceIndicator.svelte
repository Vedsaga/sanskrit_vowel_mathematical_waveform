<script lang="ts">
    /**
     * ConvergenceIndicator Component
     *
     * Displays convergence score between two audio sources.
     * Highlights matching analysis tiles when convergence is detected.
     *
     * Phase 2: Task 2.5
     */
    import type { ConvergenceResult } from "$lib/utils/convergenceAnalysis";
    import { Zap, CheckCircle, AlertCircle, MinusCircle } from "@lucide/svelte";

    interface Props {
        result: ConvergenceResult | null;
        compact?: boolean;
    }

    let { result, compact = false }: Props = $props();

    // Score percentage
    let scorePercent = $derived(result ? Math.round(result.score * 100) : 0);

    // Icon type based on score
    let iconType = $derived(() => {
        if (!result) return "none";
        if (result.score >= 0.7) return "high";
        if (result.score >= 0.4) return "medium";
        return "low";
    });

    // Color based on score
    let color = $derived(() => {
        if (!result) return "var(--color-muted-foreground)";
        if (result.score >= 0.7) return "#22c55e"; // Green
        if (result.score >= 0.4) return "#f59e0b"; // Amber
        return "#6b7280"; // Gray
    });
</script>

{#if compact}
    <div class="convergence-compact" style="--indicator-color: {color()}">
        {#if iconType() === "high"}
            <CheckCircle size={16} />
        {:else if iconType() === "medium"}
            <Zap size={16} />
        {:else if iconType() === "low"}
            <AlertCircle size={16} />
        {:else}
            <MinusCircle size={16} />
        {/if}
        <span class="score">{scorePercent}%</span>
    </div>
{:else}
    <div class="convergence-indicator" style="--indicator-color: {color()}">
        <div class="indicator-header">
            {#if iconType() === "high"}
                <CheckCircle size={20} />
            {:else if iconType() === "medium"}
                <Zap size={20} />
            {:else if iconType() === "low"}
                <AlertCircle size={20} />
            {:else}
                <MinusCircle size={20} />
            {/if}
            <span class="title">Convergence</span>
        </div>

        <div class="score-display">
            <span class="score-value">{scorePercent}%</span>
            {#if result}
                <span class="score-label">{result.label}</span>
            {/if}
        </div>

        <!-- Progress bar -->
        <div class="progress-bar">
            <div class="progress-fill" style="width: {scorePercent}%"></div>
        </div>

        {#if result && result.matchingPairs.length > 0}
            <div class="matching-info">
                <span class="matching-label">Best matches:</span>
                {#each result.matchingPairs.slice(0, 3) as pair}
                    <div class="match-pair">
                        <span class="match-a">{pair.stateA.label}</span>
                        <span class="match-arrow">↔</span>
                        <span class="match-b">{pair.stateB.label}</span>
                        <span class="match-score"
                            >{Math.round(pair.similarity * 100)}%</span
                        >
                    </div>
                {/each}
            </div>
        {/if}

        {#if result && result.commonFrequencies.length > 0}
            <div class="common-freqs">
                <span class="freq-label">Common frequencies:</span>
                <div class="freq-list">
                    {#each result.commonFrequencies.slice(0, 5) as freq}
                        <span class="freq-badge">{freq}Hz</span>
                    {/each}
                    {#if result.commonFrequencies.length > 5}
                        <span class="freq-more"
                            >+{result.commonFrequencies.length - 5}</span
                        >
                    {/if}
                </div>
            </div>
        {/if}
    </div>
{/if}

<style>
    .convergence-compact {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        background-color: color-mix(
            in srgb,
            var(--indicator-color) 15%,
            transparent
        );
        border-radius: var(--radius-sm);
        color: var(--indicator-color);
    }

    .convergence-compact .score {
        font-size: 0.75rem;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
    }

    .convergence-indicator {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .indicator-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--indicator-color);
    }

    .title {
        font-size: 0.875rem;
        font-weight: 600;
        color: var(--color-foreground);
    }

    .score-display {
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
    }

    .score-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--indicator-color);
        font-variant-numeric: tabular-nums;
        line-height: 1;
    }

    .score-label {
        font-size: 0.875rem;
        color: var(--color-muted-foreground);
    }

    .progress-bar {
        height: 6px;
        background-color: var(--color-muted);
        border-radius: 3px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background-color: var(--indicator-color);
        border-radius: 3px;
        transition: width 0.3s ease-out;
    }

    .matching-info,
    .common-freqs {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .matching-label,
    .freq-label {
        font-size: 0.65rem;
        color: var(--color-muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .match-pair {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        font-size: 0.75rem;
    }

    .match-a {
        color: #f97316;
    }

    .match-arrow {
        color: var(--color-muted-foreground);
    }

    .match-b {
        color: #3b82f6;
    }

    .match-score {
        margin-left: auto;
        font-weight: 600;
        color: var(--color-foreground);
        font-variant-numeric: tabular-nums;
    }

    .freq-list {
        display: flex;
        flex-wrap: wrap;
        gap: 0.25rem;
    }

    .freq-badge {
        font-size: 0.65rem;
        padding: 0.125rem 0.375rem;
        background-color: var(--color-muted);
        border-radius: var(--radius-sm);
        color: var(--color-foreground);
        font-variant-numeric: tabular-nums;
    }

    .freq-more {
        font-size: 0.65rem;
        color: var(--color-muted-foreground);
    }
</style>
