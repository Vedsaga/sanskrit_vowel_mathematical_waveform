<script lang="ts">
    import { Waves } from '@lucide/svelte';
    import ShapeCanvas from '$lib/components/ShapeCanvas.svelte';
    import { shapeStore } from '$lib/stores';

    /**
     * Visualizer Page
     * 
     * Main page for manual frequency shape visualization.
     * Uses ShapeCanvas component to render shapes from the store.
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
        </div>
        
        <!-- Controls Section (placeholder for future tasks) -->
        <div class="controls-section glass-card">
            <h2>Controls</h2>
            <p class="placeholder-text">Shape controls will be implemented in upcoming tasks.</p>
            
            <!-- Temporary demo buttons for testing -->
            <div class="demo-controls">
                <button 
                    class="demo-btn"
                    onclick={() => shapeStore.addShape(1)}
                >
                    Add Circle (fq=1)
                </button>
                <button 
                    class="demo-btn"
                    onclick={() => shapeStore.addShape(3)}
                >
                    Add 2-Wiggle (fq=3)
                </button>
                <button 
                    class="demo-btn"
                    onclick={() => shapeStore.addShape(5)}
                >
                    Add 4-Wiggle (fq=5)
                </button>
                <button 
                    class="demo-btn"
                    onclick={() => shapeStore.addShape(8)}
                >
                    Add 7-Wiggle (fq=8)
                </button>
                <button 
                    class="demo-btn secondary"
                    onclick={() => shapeStore.reset()}
                >
                    Clear All
                </button>
            </div>
            
            <!-- Shape List (placeholder) -->
            {#if shapes.length > 0}
                <div class="shape-list-preview">
                    <h3>Shapes ({shapes.length})</h3>
                    <ul>
                        {#each shapes as shape (shape.id)}
                            <li class="shape-item">
                                <span 
                                    class="color-dot" 
                                    style="background-color: {shape.color}"
                                ></span>
                                <span>fq={shape.fq} ({shape.fq - 1} wiggles)</span>
                                <button 
                                    class="select-btn"
                                    onclick={() => shapeStore.selectShape(shape.id)}
                                >
                                    {selectedIds.has(shape.id) ? '✓' : 'Select'}
                                </button>
                                <button 
                                    class="delete-btn"
                                    onclick={() => shapeStore.removeShape(shape.id)}
                                >
                                    ×
                                </button>
                            </li>
                        {/each}
                    </ul>
                </div>
            {/if}
        </div>
    </div>
</div>

<style>
    .page-container {
        padding: 2rem;
        max-width: 1400px;
        margin: 0 auto;
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
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    
    .header-content p {
        color: var(--color-muted-foreground);
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .visualizer-layout {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 2rem;
        align-items: start;
    }
    
    .canvas-section {
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
    }
    
    .controls-section {
        padding: 1.5rem;
        min-width: 300px;
    }
    
    .controls-section h2 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .placeholder-text {
        color: var(--color-muted-foreground);
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
    
    .demo-controls {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .demo-btn {
        padding: 0.75rem 1rem;
        background: var(--color-brand);
        color: var(--color-brand-foreground);
        border: none;
        border-radius: var(--radius);
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: opacity var(--transition-fast);
    }
    
    .demo-btn:hover {
        opacity: 0.9;
    }
    
    .demo-btn.secondary {
        background: var(--color-secondary);
        color: var(--color-secondary-foreground);
    }
    
    .shape-list-preview {
        border-top: 1px solid var(--color-border);
        padding-top: 1rem;
    }
    
    .shape-list-preview h3 {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .shape-list-preview ul {
        list-style: none;
        padding: 0;
        margin: 0;
    }
    
    .shape-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--color-border);
    }
    
    .shape-item:last-child {
        border-bottom: none;
    }
    
    .color-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .shape-item span:nth-child(2) {
        flex: 1;
        font-size: 0.9rem;
    }
    
    .select-btn {
        padding: 0.25rem 0.5rem;
        background: var(--color-secondary);
        color: var(--color-secondary-foreground);
        border: none;
        border-radius: var(--radius-sm);
        font-size: 0.75rem;
        cursor: pointer;
    }
    
    .delete-btn {
        padding: 0.25rem 0.5rem;
        background: var(--color-destructive);
        color: var(--color-destructive-foreground);
        border: none;
        border-radius: var(--radius-sm);
        font-size: 0.9rem;
        cursor: pointer;
        line-height: 1;
    }
    
    @media (max-width: 900px) {
        .visualizer-layout {
            grid-template-columns: 1fr;
        }
        
        .canvas-section {
            display: flex;
            justify-content: center;
        }
    }
</style>
