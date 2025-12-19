# Project Vak: Theme & Design System Mapping

This document provides a comprehensive mapping of the visual identity and design tokens used in the **Project Vak** (Dhiutsa) Svelte application. This mapping serves as a standard for maintaining consistency across future developments and related websites.

## 1. Core Philosophy
The design follows a **Premium Minimalist** aesthetic, characterized by:
- High contrast (Black & White base)
- Subtle textures (Noise)
- Smooth micro-interactions
- Modern typography (Inter)
- Purposeful use of brand colors

---

## 2. Color Palette

### 2.1 Base Colors (Light Mode)
| Token | Value | Usage |
| :--- | :--- | :--- |
| `--color-background` | `#ffffff` | Main page background |
| `--color-foreground` | `#111111` | Primary text |
| `--color-card` | `#fafafa` | Card and container backgrounds |
| `--color-border` | `#e5e5e5` | Subtle dividers and borders |
| `--color-muted` | `#f5f5f5` | Secondary backgrounds |
| `--color-muted-foreground`| `#737373` | De-emphasized text |

### 2.2 Base Colors (Dark Mode)
| Token | Value | Usage |
| :--- | :--- | :--- |
| `--color-background` | `#111111` | Main page background |
| `--color-foreground` | `#eaeaea` | Primary text |
| `--color-card` | `#181818` | Card and container backgrounds |
| `--color-border` | `#333333` | Subtle dividers and borders |
| `--color-muted` | `#222222` | Secondary backgrounds |
| `--color-muted-foreground`| `#888888` | De-emphasized text |

### 2.3 Brand & Accent Colors
| Token | Value | Tailwind Equivalent | Usage |
| :--- | :--- | :--- | :--- |
| `--color-brand` | `#df728b` | `rose-400` (custom) | Primary actions, brand identity, progress |
| `accent-amber` | `#F59E0B` | `amber-500` | Functional highlights (e.g., Anatomy, Sthanas) |
| `destructive` | `#ef4444` | `red-500` | Errors and destructive actions |

---

## 3. Typography

- **Primary Font**: `Inter` (Google Fonts)
- **Fallback Stack**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Weights**: 
  - `400`: Regular (Body text)
  - `500`: Medium (Buttons, UI elements)
  - `600`: Semi-Bold (Subheadings)
  - `700`: Bold (Headings)
- **Letter Spacing**: Subtle tracking for a premium feel (`tracking-wide`, `tracking-widest` for labels).

---

## 4. UI Components & Geometry

### 4.1 Borders & Radius
- **Base Radius**: `0.5rem` (8px)
- **Large Radius**: `1.5rem` (24px) for cards and containers.
- **Full Radius**: `9999px` for buttons and badges.
- **Border Width**: `1px` for standard UI, `2px` for active states.

### 4.2 Shadows & Glassmorphism
- **Shadows**: Subtle `shadow-sm` for cards, `shadow-lg` for primary brand buttons.
- **Glassmorphism**: `backdrop-blur-sm` with `bg-background/50` for floating elements (e.g., navigation bars, progress indicators).

---

## 5. Visual Effects & Animations

### 5.1 Noise Texture
A subtle noise overlay is used to add depth and a "premium paper" feel.
```css
.bg-noise::before {
    content: '';
    opacity: 0.03;
    background-image: url("data:image/svg+xml,..."); /* Fractal Noise SVG */
}
```

### 5.2 Animations
- **Transitions**: `0.2s ease-out` for hover states.
- **Keyframes**:
  - `slide-in-from-top`: Used for toasts and notifications.
  - `fade-in`: Standard entry animation for content.
  - `breathing`: Used for the Tripundra logo and active visualizers.

---

## 6. Brand Elements

### 6.1 Tripundra (Logo/Visualizer)
- **Symbol**: Three horizontal curved lines with a central dot (Bindu).
- **Behavior**: Inherits `currentColor`, often pulses or "breathes" during active states (e.g., audio recording).

### 6.2 Mandala Visualizer
- **Usage**: Encircles audio players or characters during playback.
- **Style**: Thin lines, rotating or pulsing, using `--color-brand`.

---

## 7. Implementation Guidelines for Future Projects

1.  **High Contrast**: Always prioritize readability. Use pure black/white backgrounds with slightly off-white/off-black text.
2.  **Brand Consistency**: Use `#df728b` for the primary call-to-action.
3.  **Micro-Interactions**: Every button should have a subtle scale or color transition on hover.
4.  **Whitespace**: Be generous with padding and margins to maintain a "breathable" and premium layout.
5.  **Dark Mode First**: Ensure all designs look equally stunning (or better) in dark mode.
