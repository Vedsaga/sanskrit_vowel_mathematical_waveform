/**
 * Shape Store - State management for frequency shapes using Svelte 5 runes
 * 
 * This store manages:
 * - Collection of shapes with their properties
 * - Global configuration (wiggle amplitude, resolution, canvas size)
 * - Selection state for multi-shape operations
 * - Rotation animation state
 * 
 * Requirements: 3.1, 3.2, 3.7
 */

import type { Shape, ShapeConfig, RotationState } from '../types';

/**
 * Default configuration values
 */
const DEFAULT_CONFIG: ShapeConfig = {
  A: 20,           // Default wiggle amplitude
  resolution: 360, // Default sampling points
  canvasSize: 400  // Default canvas size in pixels
};

const DEFAULT_ROTATION: RotationState = {
  isAnimating: false,
  direction: 'clockwise',
  mode: 'loop',
  speed: 1.0  // radians per second
};

/**
 * Default shape properties for new shapes
 */
const DEFAULT_SHAPE_PROPS = {
  R: 100,           // Base radius
  phi: 0,           // Initial phase offset
  opacity: 1,       // Full opacity
  strokeWidth: 2,   // 2px stroke
  selected: false
};

/**
 * Color palette for auto-assigning colors to new shapes
 */
const SHAPE_COLORS = [
  '#df728b', // Brand color (first shape)
  '#6366f1', // Indigo
  '#22c55e', // Green
  '#f59e0b', // Amber
  '#06b6d4', // Cyan
  '#8b5cf6', // Violet
  '#ec4899', // Pink
  '#14b8a6', // Teal
];

/**
 * Generates a unique ID for shapes
 */
function generateId(): string {
  return `shape_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}


/**
 * Creates a shape store with Svelte 5 runes
 * 
 * This function creates a reactive store that manages all shape-related state.
 * It uses Svelte 5's $state rune for reactivity.
 */
function createShapeStore() {
  // Core state using $state rune
  let shapes = $state<Shape[]>([]);
  let config = $state<ShapeConfig>({ ...DEFAULT_CONFIG });
  let selectedIds = $state<Set<string>>(new Set());
  let rotation = $state<RotationState>({ ...DEFAULT_ROTATION });

  // Track color index for auto-assignment
  let colorIndex = 0;

  /**
   * Gets the next color from the palette
   */
  function getNextColor(): string {
    const color = SHAPE_COLORS[colorIndex % SHAPE_COLORS.length];
    colorIndex++;
    return color;
  }

  return {
    // Getters for reactive state
    get shapes() {
      return shapes;
    },
    get config() {
      return config;
    },
    get selectedIds() {
      return selectedIds;
    },
    get rotation() {
      return rotation;
    },

    /**
     * Returns shapes that are currently selected
     */
    get selectedShapes(): Shape[] {
      return shapes.filter(s => selectedIds.has(s.id));
    },

    /**
     * Adds a new shape with the specified frequency
     * 
     * @param fq - Frequency value (integer ≥ 1)
     * @returns The newly created shape, or null if fq is invalid
     * 
     * Requirements: 3.1
     */
    addShape(fq: number): Shape | null {
      // Validate frequency
      if (!Number.isInteger(fq) || fq < 1) {
        return null;
      }

      const newShape: Shape = {
        id: generateId(),
        fq,
        R: DEFAULT_SHAPE_PROPS.R,
        phi: DEFAULT_SHAPE_PROPS.phi,
        color: getNextColor(),
        opacity: DEFAULT_SHAPE_PROPS.opacity,
        strokeWidth: DEFAULT_SHAPE_PROPS.strokeWidth,
        selected: false
      };

      shapes = [...shapes, newShape];
      return newShape;
    },

    /**
     * Removes a shape by its ID
     * 
     * @param id - The ID of the shape to remove
     * @returns true if shape was removed, false if not found
     * 
     * Requirements: 3.7
     */
    removeShape(id: string): boolean {
      const initialLength = shapes.length;
      shapes = shapes.filter(s => s.id !== id);
      
      // Also remove from selection if selected
      if (selectedIds.has(id)) {
        const newSelectedIds = new Set(selectedIds);
        newSelectedIds.delete(id);
        selectedIds = newSelectedIds;
      }

      return shapes.length < initialLength;
    },

    /**
     * Selects a shape, optionally adding to existing selection (multi-select)
     * 
     * @param id - The ID of the shape to select
     * @param multi - If true, adds to selection; if false, replaces selection
     * 
     * Requirements: 3.2
     */
    selectShape(id: string, multi: boolean = false): void {
      const shape = shapes.find(s => s.id === id);
      if (!shape) return;

      if (multi) {
        // Toggle selection in multi-select mode
        const newSelectedIds = new Set(selectedIds);
        if (newSelectedIds.has(id)) {
          newSelectedIds.delete(id);
        } else {
          newSelectedIds.add(id);
        }
        selectedIds = newSelectedIds;
      } else {
        // Single selection mode - replace selection
        selectedIds = new Set([id]);
      }

      // Update selected property on shapes
      shapes = shapes.map(s => ({
        ...s,
        selected: selectedIds.has(s.id)
      }));
    },

    /**
     * Deselects all shapes
     */
    deselectAll(): void {
      selectedIds = new Set();
      shapes = shapes.map(s => ({
        ...s,
        selected: false
      }));
    },

    /**
     * Updates properties of a specific shape
     * 
     * @param id - The ID of the shape to update
     * @param properties - Partial shape properties to update
     * @returns true if shape was updated, false if not found
     */
    updateShapeProperty(id: string, properties: Partial<Omit<Shape, 'id'>>): boolean {
      const index = shapes.findIndex(s => s.id === id);
      if (index === -1) return false;

      shapes = shapes.map(s => 
        s.id === id ? { ...s, ...properties } : s
      );
      return true;
    },

    /**
     * Updates global configuration
     * 
     * @param newConfig - Partial configuration to merge
     */
    setConfig(newConfig: Partial<ShapeConfig>): void {
      config = { ...config, ...newConfig };
    },

    /**
     * Starts rotation animation for selected shapes
     * 
     * @param direction - Rotation direction
     * @param mode - 'loop' for continuous, 'fixed' for specific angle
     * @param targetAngle - Target angle in degrees (for fixed mode)
     */
    startRotation(
      direction: 'clockwise' | 'counterclockwise',
      mode: 'loop' | 'fixed',
      targetAngle?: number
    ): void {
      rotation = {
        ...rotation,
        isAnimating: true,
        direction,
        mode,
        targetAngle: mode === 'fixed' ? targetAngle : undefined
      };
    },

    /**
     * Stops rotation animation
     */
    stopRotation(): void {
      rotation = {
        ...rotation,
        isAnimating: false
      };
    },

    /**
     * Sets the rotation speed
     * 
     * @param speed - Angular velocity in radians per second
     */
    setRotationSpeed(speed: number): void {
      rotation = {
        ...rotation,
        speed: Math.max(0, speed)
      };
    },

    /**
     * Updates the phase offset of selected shapes
     * Used by animation loop to apply rotation
     * 
     * @param deltaPhi - Change in phase offset (radians)
     */
    updateSelectedShapesPhi(deltaPhi: number): void {
      shapes = shapes.map(s => {
        if (selectedIds.has(s.id)) {
          return {
            ...s,
            phi: s.phi + deltaPhi
          };
        }
        return s;
      });
    },

    /**
     * Resets the store to initial state
     */
    reset(): void {
      shapes = [];
      config = { ...DEFAULT_CONFIG };
      selectedIds = new Set();
      rotation = { ...DEFAULT_ROTATION };
      colorIndex = 0;
    }
  };
}

/**
 * Singleton instance of the shape store
 */
export const shapeStore = createShapeStore();

/**
 * Export types for external use
 */
export type ShapeStore = ReturnType<typeof createShapeStore>;
