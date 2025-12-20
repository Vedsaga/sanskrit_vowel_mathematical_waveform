<script lang="ts">
    /**
     * AudioMetadata Component
     *
     * Displays audio file metadata in a compact format.
     *
     * Phase 1: Task 1.7
     */
    import { FileAudio } from "@lucide/svelte";

    interface Props {
        fileName: string;
        duration: number;
        sampleRate: number;
        channels: number;
        compact?: boolean;
    }

    let {
        fileName,
        duration,
        sampleRate,
        channels,
        compact = false,
    }: Props = $props();

    function formatDuration(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return mins > 0
            ? `${mins}:${secs.toFixed(1).padStart(4, "0")}`
            : `${secs.toFixed(2)}s`;
    }

    function formatSampleRate(rate: number): string {
        return `${(rate / 1000).toFixed(1)}kHz`;
    }
</script>

{#if compact}
    <div class="metadata-compact">
        <FileAudio size={14} />
        <span class="filename">{fileName}</span>
        <span class="separator">|</span>
        <span>{formatDuration(duration)}</span>
        <span class="separator">|</span>
        <span>{formatSampleRate(sampleRate)}</span>
    </div>
{:else}
    <div class="metadata-full">
        <div class="metadata-row">
            <FileAudio size={18} />
            <span class="filename">{fileName}</span>
        </div>
        <div class="metadata-details">
            <span class="detail">
                <span class="label">Duration:</span>
                <span class="value">{formatDuration(duration)}</span>
            </span>
            <span class="detail">
                <span class="label">Sample Rate:</span>
                <span class="value">{formatSampleRate(sampleRate)}</span>
            </span>
            <span class="detail">
                <span class="label">Channels:</span>
                <span class="value">{channels === 1 ? "Mono" : "Stereo"}</span>
            </span>
        </div>
    </div>
{/if}

<style>
    .metadata-compact {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
    }

    .metadata-compact .filename {
        color: var(--color-foreground);
        font-weight: 500;
        max-width: 150px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .separator {
        color: var(--color-border);
    }

    .metadata-full {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .metadata-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--color-muted-foreground);
    }

    .metadata-full .filename {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--color-foreground);
    }

    .metadata-details {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .detail {
        display: flex;
        gap: 0.25rem;
        font-size: 0.75rem;
    }

    .label {
        color: var(--color-muted-foreground);
    }

    .value {
        color: var(--color-foreground);
        font-variant-numeric: tabular-nums;
    }
</style>
