<script lang="ts">
    /**
     * AudioPlayer Component
     *
     * Play/pause audio with playhead position synced to time window.
     *
     * Phase 1: Task 1.7
     */
    import { onMount } from "svelte";
    import { Button } from "$lib/components/ui/button";
    import { Play, Pause, RotateCcw } from "@lucide/svelte";
    import type { TimeWindow } from "$lib/types";

    interface Props {
        audioBuffer: AudioBuffer | null;
        timeWindow?: TimeWindow;
        onTimeUpdate?: (time: number) => void;
    }

    let { audioBuffer, timeWindow, onTimeUpdate }: Props = $props();

    let audioContext: AudioContext | null = null;
    let sourceNode: AudioBufferSourceNode | null = null;
    let isPlaying = $state(false);
    let currentTime = $state(0);
    let startTimeRef = $state(0);

    // Derived playback range
    let playbackStart = $derived(timeWindow?.start ?? 0);
    let playbackEnd = $derived(
        timeWindow
            ? timeWindow.start + timeWindow.width / 1000
            : (audioBuffer?.duration ?? 0),
    );
    let playbackDuration = $derived(playbackEnd - playbackStart);

    function formatTime(seconds: number): string {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        if (mins > 0) {
            return `${mins}:${secs.toFixed(1).padStart(4, "0")}`;
        }
        return `${secs.toFixed(2)}s`;
    }

    async function play() {
        if (!audioBuffer) return;

        // Create audio context if needed
        if (!audioContext) {
            audioContext = new AudioContext();
        }

        // Stop any existing playback
        stop();

        // Create source node
        sourceNode = audioContext.createBufferSource();
        sourceNode.buffer = audioBuffer;
        sourceNode.connect(audioContext.destination);

        // Start playback from window start
        const offset = playbackStart;
        const duration = playbackDuration;

        startTimeRef = audioContext.currentTime - (currentTime - playbackStart);
        sourceNode.start(
            0,
            offset + (currentTime - playbackStart),
            duration - (currentTime - playbackStart),
        );

        isPlaying = true;

        // Track playback position
        sourceNode.onended = () => {
            isPlaying = false;
            currentTime = playbackStart;
        };
    }

    function stop() {
        if (sourceNode) {
            try {
                sourceNode.stop();
            } catch {}
            sourceNode.disconnect();
            sourceNode = null;
        }
        isPlaying = false;
    }

    function reset() {
        stop();
        currentTime = playbackStart;
        onTimeUpdate?.(currentTime);
    }

    function togglePlayback() {
        if (isPlaying) {
            stop();
        } else {
            play();
        }
    }

    // Update current time during playback
    $effect(() => {
        if (!isPlaying || !audioContext) return;

        let animationId: number;

        function updateTime() {
            if (!audioContext || !isPlaying) return;

            const elapsed = audioContext.currentTime - startTimeRef;
            currentTime = Math.min(playbackStart + elapsed, playbackEnd);

            onTimeUpdate?.(currentTime);

            if (currentTime >= playbackEnd) {
                isPlaying = false;
                currentTime = playbackStart;
            } else {
                animationId = requestAnimationFrame(updateTime);
            }
        }

        animationId = requestAnimationFrame(updateTime);

        return () => {
            if (animationId) {
                cancelAnimationFrame(animationId);
            }
        };
    });

    // Reset current time when window changes
    $effect(() => {
        playbackStart;
        if (!isPlaying) {
            currentTime = playbackStart;
        }
    });

    onMount(() => {
        return () => {
            stop();
            if (audioContext) {
                audioContext.close();
            }
        };
    });
</script>

<div class="audio-player">
    <div class="player-controls">
        <Button
            variant="outline"
            size="icon"
            onclick={togglePlayback}
            disabled={!audioBuffer}
        >
            {#if isPlaying}
                <Pause size={18} />
            {:else}
                <Play size={18} />
            {/if}
        </Button>

        <Button
            variant="ghost"
            size="icon"
            onclick={reset}
            disabled={!audioBuffer}
        >
            <RotateCcw size={16} />
        </Button>
    </div>

    <div class="player-progress">
        <div class="progress-bar">
            <div
                class="progress-fill"
                style="width: {playbackDuration > 0
                    ? ((currentTime - playbackStart) / playbackDuration) * 100
                    : 0}%"
            ></div>
        </div>
        <div class="time-display">
            <span>{formatTime(currentTime)}</span>
            <span class="separator">/</span>
            <span>{formatTime(playbackEnd)}</span>
        </div>
    </div>

    {#if timeWindow}
        <div class="window-indicator">
            <span class="window-label">Window:</span>
            <span>{formatTime(playbackStart)} - {formatTime(playbackEnd)}</span>
        </div>
    {/if}
</div>

<style>
    .audio-player {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.75rem 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
    }

    .player-controls {
        display: flex;
        align-items: center;
        gap: 0.25rem;
    }

    .player-progress {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .progress-bar {
        height: 4px;
        background-color: var(--color-muted);
        border-radius: 2px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background-color: var(--color-brand);
        border-radius: 2px;
        transition: width 0.1s linear;
    }

    .time-display {
        display: flex;
        gap: 0.25rem;
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
        font-variant-numeric: tabular-nums;
    }

    .separator {
        opacity: 0.5;
    }

    .window-indicator {
        font-size: 0.7rem;
        color: var(--color-muted-foreground);
        display: flex;
        gap: 0.25rem;
    }

    .window-label {
        opacity: 0.7;
    }

    @media (max-width: 500px) {
        .audio-player {
            flex-wrap: wrap;
        }

        .window-indicator {
            width: 100%;
            justify-content: center;
        }
    }
</style>
