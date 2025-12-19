<script lang="ts">
    import { Waves } from '@lucide/svelte';
    import ShapeCanvas from '$lib/components/ShapeCanvas.svelte';
    import ShapeControls from '$lib/components/ShapeControls.svelte';
    import ShapeList from '$lib/components/ShapeList.svelte';
    import RotationControls from '$lib/components/RotationControls.svelte';
    import * as Card from '$lib/components/ui/card';
    import { Separator } from '$lib/components/ui/separator';
    import { shapeStore } from '$lib/stores';

    /**
     * Visualizer Page
     * 
     * Main page for manual frequency shape visualization.
     * Integrates ShapeCanvas, ShapeControls, ShapeList, and RotationControls.
     * 
     * Requirements: 6.3
     */

    // Get reactive state from store
    const shapes = $derived(shapeStore.shapes);
    const config = $derived(shapeStore.config);
    const selectedIds = $derived(shapeStore.selectedIds);
</script>

<div class="page-container">
    <header class="page-header">
        <div class="header-icon">
            <Waves size={32} />
        </div>
        <div class="header-content">
            <h1>Frequency Shape Visualizer</h1>
            <p>Enter a frequency number to visualize it as a 2D geometric shape with wiggles</p>
        </div>
    </header>
    
    <div class="visualizer-layout">
        <!-- Canvas Section -->
        <div class="canvas-section">
            <Card.Root class="canvas-card">
                <Card.Content class="canvas-content">
                    <ShapeCanvas 
                        {shapes}
                        {config}
                        {selectedIds}
                        width={config.canvasSize}
                        height={config.canvasSize}
                        showGrid={true}
                    />
                    
                    {#if shapes.length === 0}
                        <div class="empty-state">
                            <p>No shapes yet. Add a frequency to get started.</p>
                        </div>
                    {/if}
                </Card.Content>
            </Card.Root>
        </div>
        
        <!-- Controls Section -->
        <div class="controls-section">
            <!-- Shape Controls Card -->
            <Card.Root>
                <Card.Header class="pb-3">
                    <Card.Title class="text-base">Add Shape</Card.Title>
                    <Card.Description>
                        Create shapes by entering frequency values
                    </Card.Description>
                </Card.Header>
                <Card.Content>
                    <ShapeControls />
                </Card.Content>
            </Card.Root>

            <!-- Shape List Card -->
            <Card.Root>
                <Card.Header class="pb-3">
                    <Card.Title class="text-base">Shape List</Card.Title>
                    <Card.Description>
                        Manage and select shapes for manipulation
                    </Card.Description>
                </Card.Header>
                <Card.Content>
                    <ShapeList />
                </Card.Content>
            </Card.Root>

            <!-- Rotation Controls Card -->
            <Card.Root>
                <Card.Header class="pb-3">
                    <Card.Title class="text-base">Rotation</Card.Title>
                    <Card.Description>
                        Animate selected shapes with rotation
                    </Card.Description>
                </Card.Header>
                <Card.Content>
                    <RotationControls />
                </Card.Content>
            </Card.Root>
        </div>
    </div>
</div>

<style>
    .page-container {
        padding: 1.5rem;
        max-width: 1400px;
        margin: 0 auto;
        min-height: 100%;
    }
    
    .page-header {
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .header-icon {
        width: 64px;
        height: 64px;
        background: var(--color-brand);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--color-brand-foreground);
        box-shadow: 0 4px 20px color-mix(in srgb, var(--color-brand) 30%, transparent);
        flex-shrink: 0;
    }
    
    .header-content h1 {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-content p {
        color: var(--color-muted-foreground);
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .visualizer-layout {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 1.5rem;
        align-items: start;
    }
    
    .canvas-section {
        position: sticky;
        top: 1.5rem;
    }
    
    :global(.canvas-card) {
        overflow: hidden;
    }
    
    :global(.canvas-content) {
        padding: 0 !important;
        position: relative;
    }
    
    .empty-state {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: var(--color-muted-foreground);
        font-size: 0.9rem;
        pointer-events: none;
        z-index: 10;
    }
    
    .controls-section {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        min-width: 320px;
        max-width: 400px;
    }
    
    @media (max-width: 900px) {
        .page-container {
            padding: 1rem;
        }
        
        .visualizer-layout {
            grid-template-columns: 1fr;
        }
        
        .canvas-section {
            position: static;
            display: flex;
            justify-content: center;
        }
        
        .controls-section {
            max-width: none;
        }
        
        .page-header {
            flex-direction: column;
            align-items: center;
            text-align: center;
            gap: 1rem;
        }
        
        .header-content h1 {
            font-size: 1.5rem;
        }
    }
</style>
