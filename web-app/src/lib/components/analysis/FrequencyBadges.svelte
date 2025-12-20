<script lang="ts">
    /**
     * FrequencyBadges Component
     *
     * Displays badges for frequency component properties:
     * - P: Prime fq
     * - E/O: Even/Odd fq
     * - H2-H8: Harmonic order
     * - φ: Golden ratio related
     *
     * Phase 1: Task 1.3
     */
    import { getBadgeInfo } from "$lib/utils/frequencyAnalysis";
    import type { FrequencyBadge } from "$lib/utils/frequencyAnalysis";

    interface Props {
        badges: FrequencyBadge[];
        size?: "sm" | "md";
        showLabels?: boolean;
    }

    let { badges, size = "sm", showLabels = false }: Props = $props();

    // Get info for each badge
    let badgeInfos = $derived(
        badges.map((b) => ({ badge: b, ...getBadgeInfo(b) })),
    );
</script>

<div class="frequency-badges" class:size-md={size === "md"}>
    {#each badgeInfos as info (info.badge)}
        <span
            class="badge"
            style="--badge-color: {info.color}"
            title={info.label}
        >
            {info.badge}
            {#if showLabels}
                <span class="badge-label">{info.label}</span>
            {/if}
        </span>
    {/each}
</div>

<style>
    .frequency-badges {
        display: flex;
        gap: 0.25rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.125rem 0.375rem;
        font-size: 0.625rem;
        font-weight: 600;
        font-family: "SF Mono", Monaco, "Fira Code", monospace;
        border-radius: var(--radius-sm);
        background-color: color-mix(
            in srgb,
            var(--badge-color) 20%,
            transparent
        );
        color: var(--badge-color);
        border: 1px solid
            color-mix(in srgb, var(--badge-color) 40%, transparent);
    }

    .size-md .badge {
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
    }

    .badge-label {
        font-weight: 400;
        opacity: 0.8;
    }
</style>
