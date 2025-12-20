<script lang="ts">
    /**
     * DualAnalysisGrid Component
     *
     * Side-by-side layout for comparing two audio analyses.
     * Features: Audio A (left) | Central Spine (controls) | Audio B (right)
     *
     * Phase 2: Task 2.1
     */
    import AnalysisGrid from "./AnalysisGrid.svelte";
    import SharedCanvas from "$lib/components/canvas/SharedCanvas.svelte";
    import SyncControls from "$lib/components/SyncControls.svelte";
    import AudioUploader from "$lib/components/AudioUploader.svelte";
    import * as Card from "$lib/components/ui/card";
    import { Button } from "$lib/components/ui/button";
    import { comparisonStore, shapeStore } from "$lib/stores";
    import type {
        AnalysisState,
        Shape,
        ShapeConfig,
        GlobalSettings,
    } from "$lib/types";
    import { ArrowLeftRight, Maximize2 } from "@lucide/svelte";

    interface Props {
        globalSettings: GlobalSettings;
    }

    let { globalSettings }: Props = $props();

    // Store state
    const leftPanel = $derived(comparisonStore.leftPanel);
    const rightPanel = $derived(comparisonStore.rightPanel);
    const syncMode = $derived(comparisonStore.syncMode);
    const showSharedCanvas = $derived(comparisonStore.showSharedCanvas);
    const linkControls = $derived(comparisonStore.linkControls);
    const comparisonMode = $derived(comparisonStore.comparisonMode);
    const sharedFrequencyScale = $derived(comparisonStore.sharedFrequencyScale);
    const config = $derived(shapeStore.config);

    // Left panel analyses (simulated from shapes)
    let leftAnalyses = $derived<AnalysisState[]>(
        !leftPanel.audioBuffer
            ? []
            : leftPanel.analyses.length > 0
              ? leftPanel.analyses
              : [
                    {
                        id: "left-default",
                        label: leftPanel.fileName || "Audio A",
                        createdAt: Date.now(),
                        frequencyComponents: leftPanel.frequencyComponents,
                        shapes: leftPanel.shapes,
                    },
                ],
    );

    // Right panel analyses
    let rightAnalyses = $derived<AnalysisState[]>(
        !rightPanel.audioBuffer
            ? []
            : rightPanel.analyses.length > 0
              ? rightPanel.analyses
              : [
                    {
                        id: "right-default",
                        label: rightPanel.fileName || "Audio B",
                        createdAt: Date.now(),
                        frequencyComponents: rightPanel.frequencyComponents,
                        shapes: rightPanel.shapes,
                    },
                ],
    );

    // Selected analysis IDs
    let selectedLeftId = $state<string | null>(null);
    let selectedRightId = $state<string | null>(null);

    /**
     * Handles left panel audio load
     */
    function handleLeftAudioLoaded(buffer: AudioBuffer, name: string) {
        comparisonStore.loadAudio("left", buffer, name);
    }

    /**
     * Handles right panel audio load
     */
    function handleRightAudioLoaded(buffer: AudioBuffer, name: string) {
        comparisonStore.loadAudio("right", buffer, name);
    }

    /**
     * Handles left analysis selection
     */
    function handleLeftSelect(id: string | null) {
        selectedLeftId = id;
    }

    /**
     * Handles right analysis selection
     */
    function handleRightSelect(id: string | null) {
        selectedRightId = id;
    }

    /**
     * Handles sync mode change
     */
    function handleSyncModeChange(mode: "independent" | "synchronized") {
        comparisonStore.setSyncMode(mode);
    }

    /**
     * Toggles link controls
     */
    function handleLinkToggle() {
        comparisonStore.setLinkControls(!linkControls);
    }

    /**
     * Toggles shared canvas
     */
    function handleSharedCanvasToggle() {
        comparisonStore.setShowSharedCanvas(!showSharedCanvas);
    }
</script>

<div class="dual-grid-container">
    <!-- Left Panel: Audio A -->
    <div class="panel panel-left">
        <div class="panel-header">
            <h2 class="panel-title">Audio A</h2>
            {#if leftPanel.fileName}
                <span class="panel-filename">{leftPanel.fileName}</span>
            {/if}
        </div>

        {#if !leftPanel.audioBuffer}
            <Card.Root class="upload-card">
                <Card.Content>
                    <AudioUploader
                        onFileLoaded={handleLeftAudioLoaded}
                        isProcessing={leftPanel.isProcessing}
                        error={leftPanel.error}
                        fileName={leftPanel.fileName}
                        showPlayerAfterUpload={false}
                    />
                </Card.Content>
            </Card.Root>
        {:else}
            <AnalysisGrid
                analyses={leftAnalyses}
                selectedId={selectedLeftId}
                {config}
                {globalSettings}
                onSelect={handleLeftSelect}
                tileSize={150}
            />
        {/if}
    </div>

    <!-- Central Spine -->
    <div class="central-spine">
        <div class="spine-controls">
            <Button
                variant={linkControls ? "default" : "outline"}
                size="sm"
                onclick={handleLinkToggle}
                class="link-button"
            >
                <ArrowLeftRight size={16} />
                {linkControls ? "Linked" : "Link"}
            </Button>

            <Button
                variant={showSharedCanvas ? "default" : "outline"}
                size="sm"
                onclick={handleSharedCanvasToggle}
            >
                <Maximize2 size={16} />
                {showSharedCanvas ? "Shared" : "Split"}
            </Button>

            <SyncControls
                {syncMode}
                {sharedFrequencyScale}
                leftHasAudio={leftPanel.audioBuffer !== null}
                rightHasAudio={rightPanel.audioBuffer !== null}
                onSyncModeChange={handleSyncModeChange}
            />
        </div>

        {#if showSharedCanvas}
            <div class="shared-canvas-container">
                <SharedCanvas
                    leftShapes={leftPanel.shapes}
                    rightShapes={rightPanel.shapes}
                    {config}
                    {comparisonMode}
                />
            </div>
        {/if}
    </div>

    <!-- Right Panel: Audio B -->
    <div class="panel panel-right">
        <div class="panel-header">
            <h2 class="panel-title">Audio B</h2>
            {#if rightPanel.fileName}
                <span class="panel-filename">{rightPanel.fileName}</span>
            {/if}
        </div>

        {#if !rightPanel.audioBuffer}
            <Card.Root class="upload-card">
                <Card.Content>
                    <AudioUploader
                        onFileLoaded={handleRightAudioLoaded}
                        isProcessing={rightPanel.isProcessing}
                        error={rightPanel.error}
                        fileName={rightPanel.fileName}
                        showPlayerAfterUpload={false}
                    />
                </Card.Content>
            </Card.Root>
        {:else}
            <AnalysisGrid
                analyses={rightAnalyses}
                selectedId={selectedRightId}
                {config}
                {globalSettings}
                onSelect={handleRightSelect}
                tileSize={150}
            />
        {/if}
    </div>
</div>

<style>
    .dual-grid-container {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 1rem;
        height: 100%;
        min-height: 500px;
    }

    .panel {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
        overflow: auto;
    }

    .panel-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--color-border);
    }

    .panel-title {
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
    }

    .panel-filename {
        font-size: 0.75rem;
        color: var(--color-muted-foreground);
        max-width: 150px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .central-spine {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        padding: 1rem;
        min-width: 180px;
    }

    .spine-controls {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    :global(.link-button) {
        justify-content: flex-start;
        gap: 0.5rem;
    }

    .shared-canvas-container {
        flex: 1;
        min-height: 300px;
        background-color: var(--color-card);
        border-radius: var(--radius-lg);
        border: 1px solid var(--color-border);
        overflow: hidden;
    }

    :global(.upload-card) {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 200px;
    }

    @media (max-width: 1024px) {
        .dual-grid-container {
            grid-template-columns: 1fr;
            grid-template-rows: auto auto auto;
        }

        .central-spine {
            flex-direction: row;
            flex-wrap: wrap;
            justify-content: center;
        }

        .spine-controls {
            flex-direction: row;
        }

        .shared-canvas-container {
            width: 100%;
            min-height: 250px;
        }
    }
</style>
