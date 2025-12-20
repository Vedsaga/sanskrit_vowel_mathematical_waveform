<script lang="ts">
    /**
     * KeyboardShortcuts Component
     *
     * Global keyboard shortcuts for the application.
     *
     * Phase 5: Task 5.3
     */
    import { shapeStore } from "$lib/stores";
    import { goto } from "$app/navigation";
    import { page } from "$app/stores";

    interface Props {
        onAddTile?: () => void;
        onSaveState?: () => void;
    }

    let { onAddTile, onSaveState }: Props = $props();

    // Store state
    const rotation = $derived(shapeStore.rotation);
    const selectedIds = $derived(shapeStore.selectedIds);

    /**
     * Handles global keydown events
     */
    function handleKeydown(event: KeyboardEvent): void {
        // Ignore if user is typing in an input
        if (isTypingInInput(event)) return;

        switch (event.key) {
            case " ":
                // Space - Toggle rotation play/pause
                event.preventDefault();
                if (rotation.isAnimating) {
                    shapeStore.stopRotation();
                } else {
                    shapeStore.startRotation(rotation.direction, rotation.mode);
                }
                break;

            case "Delete":
            case "Backspace":
                // Delete selected shapes
                if (selectedIds.size > 0) {
                    event.preventDefault();
                    selectedIds.forEach((id) => shapeStore.removeShape(id));
                }
                break;

            case "Escape":
                // Deselect all
                shapeStore.deselectAll();
                break;

            case "a":
            case "A":
                // Add new analysis tile (Analysis Observatory)
                if (isOnPage("/audio-analysis") && onAddTile) {
                    event.preventDefault();
                    onAddTile();
                }
                break;

            case "s":
            case "S":
                // Save state (Convergence Studio)
                if (isOnPage("/comparison") && onSaveState) {
                    event.preventDefault();
                    onSaveState();
                }
                break;

            case "1":
            case "2":
            case "3":
            case "4":
            case "5":
            case "6":
            case "7":
            case "8":
            case "9":
                // Select shape/tile by number
                selectByNumber(parseInt(event.key, 10));
                break;
        }
    }

    /**
     * Checks if user is typing in an input field
     */
    function isTypingInInput(event: KeyboardEvent): boolean {
        const target = event.target as HTMLElement;
        const tagName = target.tagName.toLowerCase();
        return (
            tagName === "input" ||
            tagName === "textarea" ||
            target.isContentEditable
        );
    }

    /**
     * Checks if on a specific page
     */
    function isOnPage(path: string): boolean {
        return $page.url.pathname === path;
    }

    /**
     * Selects a shape or tile by its 1-based index
     */
    function selectByNumber(num: number): void {
        const shapes = shapeStore.shapes;
        if (num <= shapes.length) {
            shapeStore.selectShape(shapes[num - 1].id, false);
        }
    }
</script>

<svelte:window onkeydown={handleKeydown} />
