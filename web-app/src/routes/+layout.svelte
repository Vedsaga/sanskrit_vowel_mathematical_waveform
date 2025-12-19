<script lang="ts">
    import "../app.css";
    import Sidebar from "$lib/components/Sidebar.svelte";
    
    let { children } = $props();
</script>

<div class="app-layout">
    <Sidebar />
    <main class="main-content bg-noise">
        {@render children()}
    </main>
</div>

<style>
    .app-layout {
        display: flex;
        min-height: 100vh;
    }
    
    .main-content {
        flex: 1;
        margin-left: 260px;
        min-height: 100vh;
        background-color: var(--color-background);
    }
</style>
