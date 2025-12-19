<script lang="ts">
  /**
   * ShapeList Component
   * 
   * Displays a list of shapes with controls for:
   * - Multi-selection via checkboxes
   * - Color picker for each shape
   * - Delete button for each shape
   * 
   * Requirements: 3.2, 3.7, 3.8
   */
  import { Button } from '$lib/components/ui/button';
  import { Checkbox } from '$lib/components/ui/checkbox';
  import { shapeStore } from '$lib/stores/shapeStore';
  import Trash2 from '@lucide/svelte/icons/trash-2';

  /**
   * Handles checkbox change for shape selection
   */
  function handleSelectionChange(id: string, checked: boolean) {
    if (checked) {
      // Add to selection (multi-select mode)
      shapeStore.selectShape(id, true);
    } else {
      // Remove from selection
      shapeStore.selectShape(id, true);
    }
  }

  /**
   * Handles color change for a shape
   */
  function handleColorChange(id: string, event: Event) {
    const target = event.target as HTMLInputElement;
    shapeStore.updateShapeProperty(id, { color: target.value });
  }

  /**
   * Handles shape deletion
   */
  function handleDelete(id: string) {
    shapeStore.removeShape(id);
  }

  /**
   * Handles opacity change for a shape
   */
  function handleOpacityChange(id: string, event: Event) {
    const target = event.target as HTMLInputElement;
    const opacity = parseFloat(target.value);
    if (!isNaN(opacity)) {
      shapeStore.updateShapeProperty(id, { opacity });
    }
  }
</script>

<div class="space-y-3">
  <div class="flex items-center justify-between">
    <h3 class="text-sm font-medium text-foreground">Shapes</h3>
    {#if shapeStore.shapes.length > 0}
      <span class="text-xs text-muted-foreground">
        {shapeStore.selectedIds.size} of {shapeStore.shapes.length} selected
      </span>
    {/if}
  </div>

  {#if shapeStore.shapes.length === 0}
    <div class="rounded-lg border border-dashed border-border p-4 text-center">
      <p class="text-sm text-muted-foreground">
        No shapes yet. Add a frequency above to create a shape.
      </p>
    </div>
  {:else}
    <div class="space-y-2 max-h-64 overflow-y-auto pr-1">
      {#each shapeStore.shapes as shape (shape.id)}
        <div 
          class="flex items-center gap-3 rounded-lg border border-border bg-card p-3 transition-colors hover:bg-muted/50"
          class:ring-2={shape.selected}
          class:ring-brand={shape.selected}
        >
          <!-- Selection Checkbox -->
          <Checkbox
            checked={shape.selected}
            onCheckedChange={(checked: boolean | 'indeterminate') => handleSelectionChange(shape.id, checked === true)}
            aria-label={`Select shape with frequency ${shape.fq}`}
          />

          <!-- Shape Info -->
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span class="font-medium text-sm">fq = {shape.fq}</span>
              <span class="text-xs text-muted-foreground">
                ({shape.fq - 1} wiggle{shape.fq - 1 !== 1 ? 's' : ''})
              </span>
            </div>
          </div>

          <!-- Color Picker -->
          <div class="flex items-center gap-2">
            <label class="sr-only" for="color-{shape.id}">Shape color</label>
            <input
              id="color-{shape.id}"
              type="color"
              value={shape.color}
              onchange={(e) => handleColorChange(shape.id, e)}
              class="h-7 w-7 cursor-pointer rounded border border-border bg-transparent p-0.5"
              title="Change shape color"
            />
          </div>

          <!-- Opacity Control -->
          <div class="flex items-center gap-1">
            <label class="sr-only" for="opacity-{shape.id}">Shape opacity</label>
            <input
              id="opacity-{shape.id}"
              type="range"
              min="0.1"
              max="1"
              step="0.1"
              value={shape.opacity}
              onchange={(e) => handleOpacityChange(shape.id, e)}
              class="h-1 w-12 cursor-pointer accent-brand"
              title="Adjust opacity: {Math.round(shape.opacity * 100)}%"
            />
          </div>

          <!-- Delete Button -->
          <Button
            variant="ghost"
            size="icon"
            onclick={() => handleDelete(shape.id)}
            class="h-8 w-8 text-muted-foreground hover:text-destructive"
            aria-label={`Delete shape with frequency ${shape.fq}`}
          >
            <Trash2 class="h-4 w-4" />
          </Button>
        </div>
      {/each}
    </div>
  {/if}
</div>
