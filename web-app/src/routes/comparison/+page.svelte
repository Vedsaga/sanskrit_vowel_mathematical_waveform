<script lang="ts">
    import { GitCompare } from '@lucide/svelte';
</script>

<div class="page-container">
    <header class="page-header">
        <div class="header-icon">
            <GitCompare size={32} />
        </div>
        <div class="header-content">
            <h1>Audio Comparison</h1>
            <p>Compare two audio files side by side to analyze differences in their frequency compositions</p>
        </div>
    </header>
    
    <div class="placeholder-content glass-card">
        <div class="placeholder-icon">
            <GitCompare size={64} class="animate-breathing" />
        </div>
        <h2>Coming Soon</h2>
        <p>The side-by-side audio comparison feature will be implemented here.</p>
        <ul class="feature-list">
            <li>Upload two audio files for comparison</li>
            <li>View parallel frequency analysis panels</li>
            <li>Synchronized or independent shape manipulation</li>
            <li>Aligned frequency scales for accurate comparison</li>
        </ul>
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
    
    .placeholder-content {
        padding: 4rem 2rem;
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 1rem;
    }
    
    .placeholder-icon {
        color: var(--color-brand);
        opacity: 0.6;
        margin-bottom: 1rem;
    }
    
    .placeholder-content h2 {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .placeholder-content > p {
        color: var(--color-muted-foreground);
        max-width: 400px;
    }
    
    .feature-list {
        list-style: none;
        padding: 0;
        margin-top: 1.5rem;
        text-align: left;
    }
    
    .feature-list li {
        padding: 0.5rem 0;
        padding-left: 1.5rem;
        position: relative;
        color: var(--color-muted-foreground);
        font-size: 0.9rem;
    }
    
    .feature-list li::before {
        content: '→';
        position: absolute;
        left: 0;
        color: var(--color-brand);
    }
</style>
