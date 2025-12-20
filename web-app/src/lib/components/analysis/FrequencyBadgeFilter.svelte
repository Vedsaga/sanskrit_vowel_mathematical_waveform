<script lang="ts">
    /**
     * FrequencyBadgeFilter Component
     *
     * Filter toggles for frequency component badges.
     * Allows filtering by: All, Harmonics, Primes, Golden Ratio
     *
     * Phase 1: Task 1.3
     */
    import type { FrequencyBadge } from "$lib/utils/frequencyAnalysis";
    import { Button } from "$lib/components/ui/button";

    type FilterType = "all" | "harmonics" | "primes" | "golden";

    interface Props {
        activeFilter: FilterType;
        onFilterChange?: (filter: FilterType) => void;
    }

    let { activeFilter = "all", onFilterChange }: Props = $props();

    const FILTERS: { id: FilterType; label: string; icon: string }[] = [
        { id: "all", label: "All", icon: "⊕" },
        { id: "harmonics", label: "Harmonics", icon: "H" },
        { id: "primes", label: "Primes", icon: "P" },
        { id: "golden", label: "Golden", icon: "φ" },
    ];

    /**
     * Filters badges based on active filter
     */
    export function filterBadges(
        badges: FrequencyBadge[],
        filter: FilterType,
    ): FrequencyBadge[] {
        switch (filter) {
            case "harmonics":
                return badges.filter((b) => b.startsWith("H"));
            case "primes":
                return badges.filter((b) => b === "P");
            case "golden":
                return badges.filter((b) => b === "φ");
            case "all":
            default:
                return badges;
        }
    }

    /**
     * Checks if a component passes the filter based on its badges
     */
    export function componentPassesFilter(
        badges: FrequencyBadge[],
        filter: FilterType,
    ): boolean {
        if (filter === "all") return true;
        const filtered = filterBadges(badges, filter);
        return filtered.length > 0;
    }
</script>

<div class="badge-filter">
    {#each FILTERS as filter (filter.id)}
        <button
            class="filter-button"
            class:active={activeFilter === filter.id}
            onclick={() => onFilterChange?.(filter.id)}
        >
            <span class="filter-icon">{filter.icon}</span>
            <span class="filter-label">{filter.label}</span>
        </button>
    {/each}
</div>

<style>
    .badge-filter {
        display: flex;
        gap: 0.25rem;
        padding: 0.25rem;
        background-color: var(--color-muted);
        border-radius: var(--radius-md);
    }

    .filter-button {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.375rem 0.75rem;
        border: none;
        border-radius: var(--radius-sm);
        background: none;
        color: var(--color-muted-foreground);
        font-size: 0.75rem;
        cursor: pointer;
        transition: all 0.15s ease-out;
    }

    .filter-button:hover {
        background-color: var(--color-background);
        color: var(--color-foreground);
    }

    .filter-button.active {
        background-color: var(--color-background);
        color: var(--color-foreground);
        font-weight: 500;
    }

    .filter-icon {
        font-family: "SF Mono", Monaco, monospace;
        font-weight: 600;
    }

    .filter-label {
        font-size: 0.7rem;
    }

    @media (max-width: 400px) {
        .filter-label {
            display: none;
        }
    }
</style>
