<script lang="ts">
  /**
   * ShapeControls Component
   * 
   * Provides controls for adding shapes and adjusting global configuration:
   * - Frequency input with validation
   * - Add Shape button
   * - Global amplitude slider
   * 
   * Requirements: 2.4, 2.5, 6.4
   */
  import { Button } from '$lib/components/ui/button';
  import { Input } from '$lib/components/ui/input';
  import { Slider } from '$lib/components/ui/slider';
  import { shapeStore } from '$lib/stores/shapeStore.svelte';
  import { validateFrequencyInput } from '$lib/shapeEngine';

  // Local state for frequency input
  let frequencyInput = $state('');
  let validationError = $state('');
  let isInputTouched = $state(false);

  // Derived state for amplitude slider
  let amplitudeValue = $derived([shapeStore.config.A]);

  /**
   * Validates the current frequency input and updates error state
   */
  function validateInput(): boolean {
    if (!isInputTouched && frequencyInput === '') {
      validationError = '';
      return false;
    }

    const result = validateFrequencyInput(frequencyInput);
    validationError = result.errors.join(', ');
    return result.valid;
  }

  /**
   * Handles adding a new shape
   */
  function handleAddShape() {
    isInputTouched = true;
    
    if (!validateInput()) {
      return;
    }

    const fq = parseInt(frequencyInput, 10);
    const shape = shapeStore.addShape(fq);
    
    if (shape) {
      // Clear input on success
      frequencyInput = '';
      validationError = '';
      isInputTouched = false;
    }
  }

  /**
   * Handles input change with validation
   */
  function handleInputChange(event: Event) {
    const target = event.target as HTMLInputElement;
    frequencyInput = target.value;
    
    if (isInputTouched) {
      validateInput();
    }
  }

  /**
   * Handles input blur to trigger validation
   */
  function handleInputBlur() {
    isInputTouched = true;
    validateInput();
  }

  /**
   * Handles Enter key press to add shape
   */
  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      handleAddShape();
    }
  }

  /**
   * Handles amplitude slider change
   */
  function handleAmplitudeChange(value: number[]) {
    if (value.length > 0) {
      shapeStore.setConfig({ A: value[0] });
    }
  }

  // Check if add button should be disabled
  let isAddDisabled = $derived(
    frequencyInput === '' || 
    (isInputTouched && validationError !== '')
  );
</script>

<div class="space-y-6">
  <!-- Frequency Input Section -->
  <div class="space-y-2">
    <label for="frequency-input" class="text-sm font-medium text-foreground">
      Frequency (fq)
    </label>
    <div class="flex gap-2">
      <div class="flex-1">
        <Input
          id="frequency-input"
          type="number"
          min="1"
          step="1"
          placeholder="Enter frequency (≥ 1)"
          value={frequencyInput}
          oninput={handleInputChange}
          onblur={handleInputBlur}
          onkeydown={handleKeyDown}
          class={validationError ? 'border-destructive focus-visible:ring-destructive' : ''}
          aria-invalid={validationError ? 'true' : 'false'}
          aria-describedby={validationError ? 'frequency-error' : undefined}
        />
      </div>
      <Button 
        onclick={handleAddShape}
        disabled={isAddDisabled}
        class="shrink-0"
      >
        Add Shape
      </Button>
    </div>
    {#if validationError}
      <p id="frequency-error" class="text-sm text-destructive" role="alert">
        {validationError}
      </p>
    {/if}
    <p class="text-xs text-muted-foreground">
      Frequency determines the number of wiggles: fq=1 is a circle, fq=2 has 1 wiggle, etc.
    </p>
  </div>

  <!-- Amplitude Slider Section -->
  <div class="space-y-3">
    <div class="flex items-center justify-between">
      <label id="amplitude-label" class="text-sm font-medium text-foreground">
        Wiggle Amplitude (A)
      </label>
      <span class="text-sm text-muted-foreground tabular-nums">
        {shapeStore.config.A.toFixed(0)}
      </span>
    </div>
    <Slider
      type="multiple"
      value={amplitudeValue}
      onValueChange={handleAmplitudeChange}
      min={1}
      max={80}
      step={1}
      class="w-full"
    />
    <p class="text-xs text-muted-foreground">
      Controls how far wiggles extend from the base circle. Must be less than base radius.
    </p>
  </div>
</div>
