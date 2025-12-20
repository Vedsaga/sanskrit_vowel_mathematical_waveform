<script lang="ts">
    /**
     * ComponentGroups Component
     *
     * Collapsible list of frequency component groups.
     * Each group shows its components and allows group-level selection.
     *
     * Phase 1: Task 1.2
     */
    import type { FrequencyGroup } from "$lib/utils/frequencyGrouping";
    import type { FrequencyComponent } from "$lib/types";
    import { getGroupComponents } from "$lib/utils/frequencyGrouping";
    import FrequencyBadges from "./FrequencyBadges.svelte";
    import { ChevronDown, ChevronRight, Check } from "@lucide/svelte";
    import { Button } from "$lib/components/ui/button";

    interface Props {
        groups: FrequencyGroup[];
        components: FrequencyComponent[];
        onToggleGroup?: (groupId: string) => void;
        onToggleExpand?: (groupId: string) => void;
        onSelectComponent?: (componentId: string) => void;
    }

    let {
        groups,
        components,
        onToggleGroup,
        onToggleExpand,
        onSelectComponent,
    }: Props = $props();

    // Get components for each group
    function getComponentsForGroup(
        group: FrequencyGroup,
    ): FrequencyComponent[] {
        return getGroupComponents(group, components);
    }

    function formatFrequency(hz: number): string {
        if (hz >= 1000) {
            return `${(hz / 1000).toFixed(1)}k`;
        }
        return `${Math.round(hz)}`;
    }
</script>

<div class="component-groups">
    {#if groups.length === 0}
        <div class="empty-state">
            <p>No frequency groups detected. Upload audio and run analysis.</p>
        </div>
    {:else}
        {#each groups as group (group.id)}
            {@const groupComponents = getComponentsForGroup(group)}
            <div class="group" class:selected={group.selected}>
                <div
                    class="group-header"
                    onclick={() => onToggleExpand?.(group.id)}
                    role="button"
                    tabindex="0"
                    onkeydown={(e) =>
                        e.key === "Enter" && onToggleExpand?.(group.id)}
                >
                    <div class="header-left">
                        <span class="expand-icon">
                            {#if group.expanded}
                                <ChevronDown size={16} />
                            {:else}
                                <ChevronRight size={16} />
                            {/if}
                        </span>
                        <span
                            class="group-color"
                            style="background-color: {group.color}"
                        ></span>
                        <span class="group-label">{group.label}</span>
                        <span class="component-count"
                            >{groupComponents.length}</span
                        >
                    </div>

                    <span
                        class="select-button"
                        class:active={group.selected}
                        onclick={(e) => {
                            e.stopPropagation();
                            onToggleGroup?.(group.id);
                        }}
                        role="checkbox"
                        tabindex="0"
                        aria-checked={group.selected}
                        aria-label={group.selected
                            ? "Deselect group"
                            : "Select group"}
                        onkeydown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                                e.stopPropagation();
                                onToggleGroup?.(group.id);
                            }
                        }}
                    >
                        {#if group.selected}
                            <Check size={14} />
                        {/if}
                    </span>
                </div>

                {#if group.expanded}
                    <div class="group-content">
                        {#each groupComponents as comp (comp.id)}
                            <button
                                class="component-row"
                                class:selected={comp.selected}
                                onclick={() => onSelectComponent?.(comp.id)}
                            >
                                <span class="comp-freq"
                                    >{formatFrequency(comp.frequencyHz)} Hz</span
                                >
                                <span class="comp-fq">fq={comp.fq}</span>
                                <div class="comp-badges">
                                    {#if comp.badges}
                                        <FrequencyBadges
                                            badges={comp.badges}
                                            size="sm"
                                        />
                                    {/if}
                                </div>
                                <span class="comp-mag"
                                    >{(comp.magnitude * 100).toFixed(0)}%</span
                                >
                            </button>
                        {/each}
                    </div>
                {/if}
            </div>
        {/each}
    {/if}
</div>

<style>
    .component-groups {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .empty-state {
        padding: 1.5rem;
        text-align: center;
        color: var(--color-muted-foreground);
        font-size: 0.875rem;
    }

    .group {
        background-color: var(--color-card);
        border: 1px solid var(--color-border);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    .group.selected {
        border-color: var(--color-brand);
    }

    .group-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
        padding: 0.75rem;
        background: none;
        border: none;
        cursor: pointer;
        transition: background-color 0.15s ease-out;
    }

    .group-header:hover {
        background-color: var(--color-muted);
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .expand-icon {
        color: var(--color-muted-foreground);
    }

    .group-color {
        width: 12px;
        height: 12px;
        border-radius: 3px;
    }

    .group-label {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-foreground);
    }

    .component-count {
        font-size: 0.7rem;
        padding: 0.125rem 0.375rem;
        background-color: var(--color-muted);
        border-radius: var(--radius-sm);
        color: var(--color-muted-foreground);
    }

    .select-button {
        width: 24px;
        height: 24px;
        border-radius: var(--radius-sm);
        border: 2px solid var(--color-border);
        background: none;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease-out;
    }

    .select-button.active {
        background-color: var(--color-brand);
        border-color: var(--color-brand);
        color: var(--color-brand-foreground);
    }

    .group-content {
        border-top: 1px solid var(--color-border);
        padding: 0.25rem;
    }

    .component-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.5rem 0.75rem;
        background: none;
        border: none;
        border-radius: var(--radius-sm);
        cursor: pointer;
        transition: background-color 0.15s ease-out;
        text-align: left;
    }

    .component-row:hover {
        background-color: var(--color-muted);
    }

    .component-row.selected {
        background-color: color-mix(
            in srgb,
            var(--color-brand) 15%,
            transparent
        );
    }

    .comp-freq {
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--color-foreground);
        min-width: 70px;
    }

    .comp-fq {
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
        font-family: "SF Mono", Monaco, monospace;
        min-width: 50px;
    }

    .comp-badges {
        flex: 1;
    }

    .comp-mag {
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
        font-variant-numeric: tabular-nums;
    }
</style>
