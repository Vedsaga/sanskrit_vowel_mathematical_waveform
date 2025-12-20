<script lang="ts">
    import { Sparkles, Eye, GitMerge, ArrowRight } from "@lucide/svelte";

    const features = [
        {
            href: "/visualizer",
            icon: Sparkles,
            title: "Compose Lab",
            description:
                "Create geometric shapes from frequencies. Overlay multiple forms and rotate them to explore visual patterns emerging from mathematics.",
            highlights: [
                "Manual frequency entry",
                "Multi-shape overlay",
                "Per-shape animation",
            ],
        },
        {
            href: "/audio-analysis",
            icon: Eye,
            title: "Analysis Observatory",
            description:
                "Upload audio and visualize its frequency spectrum as geometric shapes. Analyze multiple time windows side by side.",
            highlights: [
                "FFT analysis",
                "Multi-tile grid",
                "Frequency grouping",
            ],
        },
        {
            href: "/comparison",
            icon: GitMerge,
            title: "Convergence Studio",
            description:
                "Compare two audio sources to find where their frequency signatures converge. Detect matching patterns across recordings.",
            highlights: [
                "Dual-grid layout",
                "Convergence detection",
                "State snapshots",
            ],
        },
    ];
</script>

<div class="page-container">
    <header class="hero">
        <div class="hero-content">
            <div class="logo-large">
                <span class="logo-icon">V</span>
            </div>
            <h1>Vak</h1>
            <p class="tagline">Frequency Shape Visualizer</p>
            <p class="description">
                Explore the mathematical relationship between frequency and
                geometric form. Transform frequencies into beautiful 2D shapes
                using in-phase radial modulation.
            </p>
        </div>
    </header>

    <section class="features">
        <h2 class="section-title">Features</h2>
        <div class="feature-grid">
            {#each features as feature}
                <a href={feature.href} class="feature-card glass-card">
                    <div class="feature-icon">
                        <feature.icon size={28} />
                    </div>
                    <h3>{feature.title}</h3>
                    <p>{feature.description}</p>
                    <ul class="highlights">
                        {#each feature.highlights as highlight}
                            <li>{highlight}</li>
                        {/each}
                    </ul>
                    <div class="feature-link">
                        <span>Explore</span>
                        <ArrowRight size={16} />
                    </div>
                </a>
            {/each}
        </div>
    </section>

    <section class="formula-section glass-card">
        <h2>The Shape Formula</h2>
        <div class="formula">
            <code>r(θ) = R + A·sin((fq-1)·θ + φ)</code>
        </div>
        <div class="formula-explanation">
            <div class="param">
                <span class="param-name">R</span>
                <span class="param-desc">Base radius of the circle</span>
            </div>
            <div class="param">
                <span class="param-name">A</span>
                <span class="param-desc"
                    >Wiggle amplitude (how far wiggles extend)</span
                >
            </div>
            <div class="param">
                <span class="param-name">fq</span>
                <span class="param-desc"
                    >Frequency value (determines wiggle count = fq - 1)</span
                >
            </div>
            <div class="param">
                <span class="param-name">φ</span>
                <span class="param-desc">Phase offset (rotation angle)</span>
            </div>
            <div class="param">
                <span class="param-name">θ</span>
                <span class="param-desc">Angle parameter (0 to 2π)</span>
            </div>
        </div>
    </section>
</div>

<style>
    .page-container {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }

    .hero {
        text-align: center;
        padding: 4rem 2rem;
        margin-bottom: 3rem;
    }

    .hero-content {
        max-width: 600px;
        margin: 0 auto;
    }

    .logo-large {
        margin-bottom: 1.5rem;
    }

    .logo-icon {
        display: inline-flex;
        width: 80px;
        height: 80px;
        background: var(--color-brand);
        border-radius: var(--radius-xl);
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 2.5rem;
        color: var(--color-brand-foreground);
        box-shadow: 0 8px 32px
            color-mix(in srgb, var(--color-brand) 40%, transparent);
    }

    .hero h1 {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        margin-bottom: 0.5rem;
    }

    .tagline {
        font-size: 1.25rem;
        color: var(--color-brand);
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    .description {
        color: var(--color-muted-foreground);
        font-size: 1.1rem;
        line-height: 1.7;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 1.5rem;
        margin-bottom: 3rem;
    }

    .feature-card {
        padding: 2rem;
        text-decoration: none;
        color: inherit;
        transition: all var(--transition-base);
        display: flex;
        flex-direction: column;
    }

    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px
            color-mix(in srgb, var(--color-brand) 15%, transparent);
    }

    .feature-icon {
        width: 56px;
        height: 56px;
        background: var(--color-brand);
        border-radius: var(--radius-lg);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--color-brand-foreground);
        margin-bottom: 1.25rem;
    }

    .feature-card h3 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }

    .feature-card > p {
        color: var(--color-muted-foreground);
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 1rem;
        flex: 1;
    }

    .highlights {
        list-style: none;
        padding: 0;
        margin-bottom: 1.5rem;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .highlights li {
        font-size: 0.75rem;
        padding: 0.25rem 0.75rem;
        background: var(--color-muted);
        border-radius: var(--radius-full);
        color: var(--color-muted-foreground);
    }

    .feature-link {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: var(--color-brand);
        font-weight: 500;
        font-size: 0.9rem;
    }

    .feature-card:hover .feature-link {
        gap: 0.75rem;
    }

    .formula-section {
        padding: 2.5rem;
        text-align: center;
    }

    .formula-section h2 {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }

    .formula {
        margin-bottom: 2rem;
    }

    .formula code {
        font-family: "JetBrains Mono", "Fira Code", monospace;
        font-size: 1.5rem;
        color: var(--color-brand);
        background: var(--color-muted);
        padding: 1rem 2rem;
        border-radius: var(--radius-lg);
        display: inline-block;
    }

    .formula-explanation {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem;
    }

    .param {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.25rem;
    }

    .param-name {
        font-family: "JetBrains Mono", "Fira Code", monospace;
        font-weight: 600;
        color: var(--color-brand);
        font-size: 1.1rem;
    }

    .param-desc {
        font-size: 0.8rem;
        color: var(--color-muted-foreground);
        max-width: 150px;
        text-align: center;
    }

    @media (max-width: 768px) {
        .hero h1 {
            font-size: 3rem;
        }

        .formula code {
            font-size: 1rem;
            padding: 0.75rem 1rem;
        }

        .formula-explanation {
            gap: 1rem;
        }
    }
</style>
