<script lang="ts">
    import { page } from "$app/stores";
    import { Home, Sparkles, Eye, GitMerge, Sun, Moon } from "@lucide/svelte";

    let isDark = $state(true);

    const navItems = [
        { href: "/", label: "Home", icon: Home },
        { href: "/visualizer", label: "Compose Lab", icon: Sparkles },
        { href: "/audio-analysis", label: "Analysis Observatory", icon: Eye },
        { href: "/comparison", label: "Convergence Studio", icon: GitMerge },
    ];

    function toggleTheme() {
        isDark = !isDark;
        if (typeof document !== "undefined") {
            document.documentElement.classList.toggle("dark", isDark);
        }
    }

    // Initialize dark mode on mount
    $effect(() => {
        if (typeof document !== "undefined") {
            document.documentElement.classList.toggle("dark", isDark);
        }
    });
</script>

<aside class="sidebar">
    <div class="sidebar-header">
        <div class="logo">
            <span class="logo-icon">V</span>
            <span class="logo-text">Vak</span>
        </div>
    </div>

    <nav class="sidebar-nav">
        {#each navItems as item}
            <a
                href={item.href}
                class="nav-link"
                class:active={$page.url.pathname === item.href}
            >
                <item.icon size={20} />
                <span>{item.label}</span>
            </a>
        {/each}
    </nav>

    <div class="sidebar-footer">
        <button
            class="theme-toggle"
            onclick={toggleTheme}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
            {#if isDark}
                <Sun size={20} />
                <span>Light Mode</span>
            {:else}
                <Moon size={20} />
                <span>Dark Mode</span>
            {/if}
        </button>
    </div>
</aside>

<style>
    .sidebar {
        width: 260px;
        height: 100vh;
        position: fixed;
        left: 0;
        top: 0;
        display: flex;
        flex-direction: column;
        background-color: var(--color-card);
        border-right: 1px solid var(--color-border);
        padding: 1.5rem 1rem;
        z-index: 50;
    }

    .sidebar-header {
        padding: 0 0.5rem 1.5rem;
        border-bottom: 1px solid var(--color-border);
        margin-bottom: 1.5rem;
    }

    .logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .logo-icon {
        width: 40px;
        height: 40px;
        background: var(--color-brand);
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.25rem;
        color: var(--color-brand-foreground);
        box-shadow: 0 0 20px
            color-mix(in srgb, var(--color-brand) 40%, transparent);
    }

    .logo-text {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    .sidebar-nav {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }

    .nav-link {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        color: var(--color-muted-foreground);
        text-decoration: none;
        font-weight: 500;
        font-size: 0.9rem;
        transition: all var(--transition-fast);
    }

    .nav-link:hover {
        background-color: var(--color-muted);
        color: var(--color-foreground);
    }

    .nav-link.active {
        background-color: var(--color-brand);
        color: var(--color-brand-foreground);
        box-shadow: 0 2px 8px
            color-mix(in srgb, var(--color-brand) 30%, transparent);
    }

    .sidebar-footer {
        padding-top: 1rem;
        border-top: 1px solid var(--color-border);
    }

    .theme-toggle {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        background: transparent;
        border: 1px solid var(--color-border);
        color: var(--color-muted-foreground);
        font-weight: 500;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all var(--transition-fast);
    }

    .theme-toggle:hover {
        background-color: var(--color-muted);
        color: var(--color-foreground);
        border-color: var(--color-muted-foreground);
    }
</style>
