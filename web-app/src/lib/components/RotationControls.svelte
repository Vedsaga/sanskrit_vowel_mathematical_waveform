<script lang="ts">
  /**
   * RotationControls Component
   * 
   * Provides controls for rotating selected shapes:
   * - Direction toggle (clockwise/counter-clockwise)
   * - Mode select (loop/fixed)
   * - Speed slider
   * - Angle input for fixed mode
   * - Start/stop button
   * 
   * Requirements: 3.3, 3.4, 3.5, 3.6, 3.9
   */
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Slider } from '$lib/components/ui/slider';
  import * as Select from '$lib/components/ui/select';
  import * as ToggleGroup from '$lib/components/ui/toggle-group';
  import { shapeStore } from '$lib/stores/shapeStore';
  import { animationLoop } from '$lib/animationLoop';
  import { onMount, onDestroy } from 'svelte';
  import RotateCw from '@lucide/svelte/icons/rotate-cw';
  import RotateCcw from '@lucide/svelte/icons/rotate-ccw';
  import Play from '@lucide/svelte/icons/play';
  import Square from '@lucide/svelte/icons/square';

  // Local state
  let direction = $state<'clockwise' | 'counterclockwise'>('clockwise');
  let mode = $state<'loop' | 'fixed'>('loop');
  let speed = $state([1.0]); // radians per second
  let targetAngle = $state('90'); // degrees
  let isAnimating = $state(false);
  let angleError = $state('');

  // Derived state
  let hasSelectedShapes = $derived(shapeStore.selectedIds.size > 0);
  let isStartDisabled = $derived(!hasSelectedShapes);
  let showAngleInput = $derived(mode === 'fixed');

  /**
   * Validates the target angle input
   */
  function validateAngle(): boolean {
    if (mode !== 'fixed') return true;
    
    const angle = parseFloat(targetAngle);
    if (isNaN(angle) || angle <= 0) {
      angleError = 'Angle must be a positive number';
      return false;
    }
    angleError = '';
    return true;
  }

  /**
   * Handles direction change
   */
  function handleDirectionChange(value: string | undefined) {
    if (value === 'clockwise' || value === 'counterclockwise') {
      direction = value;
      if (isAnimating) {
        animationLoop.setDirection(direction);
      }
    }
  }

  /**
   * Handles mode change
   */
  function handleModeChange(value: string | undefined) {
    if (value === 'loop' || value === 'fixed') {
      mode = value;
      angleError = '';
      // If currently animating and switching modes, restart with new mode
      if (isAnimating) {
        stopAnimation();
        startAnimation();
      }
    }
  }

  /**
   * Handles speed slider change
   */
  function handleSpeedChange(value: number[]) {
    if (value.length > 0) {
      speed = value;
      if (isAnimating) {
        animationLoop.setSpeed(value[0]);
      }
    }
  }

  /**
   * Handles angle input change
   */
  function handleAngleChange(event: Event) {
    const target = event.target as HTMLInputElement;
    targetAngle = target.value;
    validateAngle();
  }

  /**
   * Starts the rotation animation
   */
  function startAnimation() {
    if (!hasSelectedShapes) return;
    
    if (mode === 'fixed' && !validateAngle()) {
      return;
    }

    const angle = mode === 'fixed' ? parseFloat(targetAngle) : undefined;
    
    animationLoop.startRotation(direction, mode, angle, speed[0]);
    isAnimating = true;
    
    // Update store rotation state
    shapeStore.startRotation(direction, mode, angle);
  }

  /**
   * Stops the rotation animation
   */
  function stopAnimation() {
    animationLoop.stopRotation();
    isAnimating = false;
    shapeStore.stopRotation();
  }

  /**
   * Toggles animation start/stop
   */
  function toggleAnimation() {
    if (isAnimating) {
      stopAnimation();
    } else {
      startAnimation();
    }
  }

  // Set up animation loop callbacks
  onMount(() => {
    animationLoop.setPhiUpdateCallback((deltaPhi) => {
      shapeStore.updateSelectedShapesPhi(deltaPhi);
    });

    animationLoop.setStopCallback(() => {
      isAnimating = false;
      shapeStore.stopRotation();
    });
  });

  // Clean up on destroy
  onDestroy(() => {
    if (isAnimating) {
      animationLoop.stopRotation();
    }
  });

  // Format speed for display
  function formatSpeed(value: number): string {
    return `${value.toFixed(1)} rad/s`;
  }
</script>

<div class="space-y-4">
  <h3 class="text-sm font-medium text-foreground">Rotation Controls</h3>

  <!-- Direction Toggle -->
  <div class="space-y-2">
    <span id="direction-label" class="text-xs text-muted-foreground">Direction</span>
    <ToggleGroup.Root
      aria-labelledby="direction-label"
      type="single"
      value={direction}
      onValueChange={handleDirectionChange}
      class="w-full"
    >
      <ToggleGroup.Item 
        value="counterclockwise" 
        class="flex-1 gap-1.5"
        aria-label="Counter-clockwise rotation"
      >
        <RotateCcw class="h-4 w-4" />
        <span class="text-xs">CCW</span>
      </ToggleGroup.Item>
      <ToggleGroup.Item 
        value="clockwise" 
        class="flex-1 gap-1.5"
        aria-label="Clockwise rotation"
      >
        <RotateCw class="h-4 w-4" />
        <span class="text-xs">CW</span>
      </ToggleGroup.Item>
    </ToggleGroup.Root>
  </div>

  <!-- Mode Select -->
  <div class="space-y-2">
    <span id="mode-label" class="text-xs text-muted-foreground">Mode</span>
    <Select.Root type="single" value={mode} onValueChange={handleModeChange}>
      <Select.Trigger class="w-full">
        {mode === 'loop' ? 'Loop (continuous)' : 'Fixed angle'}
      </Select.Trigger>
      <Select.Content>
        <Select.Item value="loop">Loop (continuous)</Select.Item>
        <Select.Item value="fixed">Fixed angle</Select.Item>
      </Select.Content>
    </Select.Root>
  </div>

  <!-- Angle Input (for fixed mode) -->
  {#if showAngleInput}
    <div class="space-y-2">
      <label for="angle-input" class="text-xs text-muted-foreground">
        Target Angle (degrees)
      </label>
      <Input
        id="angle-input"
        type="number"
        min="1"
        step="1"
        placeholder="Enter angle"
        value={targetAngle}
        oninput={handleAngleChange}
        class={angleError ? 'border-destructive' : ''}
        aria-invalid={angleError ? 'true' : 'false'}
        aria-describedby={angleError ? 'angle-error' : undefined}
        disabled={isAnimating}
      />
      {#if angleError}
        <p id="angle-error" class="text-xs text-destructive" role="alert">
          {angleError}
        </p>
      {/if}
    </div>
  {/if}

  <!-- Speed Slider -->
  <div class="space-y-2">
    <div class="flex items-center justify-between">
      <span id="speed-label" class="text-xs text-muted-foreground">Speed</span>
      <span class="text-xs text-muted-foreground tabular-nums">
        {formatSpeed(speed[0])}
      </span>
    </div>
    <Slider
      type="multiple"
      value={speed}
      onValueChange={handleSpeedChange}
      min={0.1}
      max={5}
      step={0.1}
      class="w-full"
    />
  </div>

  <!-- Start/Stop Button -->
  <Button
    onclick={toggleAnimation}
    disabled={isStartDisabled}
    variant={isAnimating ? 'destructive' : 'default'}
    class="w-full gap-2"
  >
    {#if isAnimating}
      <Square class="h-4 w-4" />
      Stop
    {:else}
      <Play class="h-4 w-4" />
      Start Rotation
    {/if}
  </Button>

  {#if !hasSelectedShapes}
    <p class="text-xs text-muted-foreground text-center">
      Select one or more shapes to enable rotation
    </p>
  {/if}
</div>
