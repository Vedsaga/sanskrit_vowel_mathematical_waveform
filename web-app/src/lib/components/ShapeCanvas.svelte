<script lang="ts">
	import { onMount } from "svelte";
	import type { Shape, ShapeConfig, Point, GeometryMode } from "$lib/types";
	import { generateShapePoints } from "$lib/shapeEngine";
	import { getClickedShape, canvasEventToPoint } from "$lib/utils/hitTesting";
	import CanvasSkeleton from "./CanvasSkeleton.svelte";
	import ErrorState from "./ErrorState.svelte";

	/**
	 * ShapeCanvas Component
	 *
	 * Renders frequency shapes on an HTML Canvas with proper DPI handling,
	 * Project Vak theme styling, and multi-shape overlay support.
	 *
	 * Requirements: 2.6, 3.1, 3.2, 3.8, 6.1, 6.2, 6.3
	 */

	// Props
	interface Props {
		shapes?: Shape[];
		config?: ShapeConfig;
		selectedIds?: Set<string>;
		width?: number;
		height?: number;
		showGrid?: boolean;
		isLoading?: boolean;
		error?: string | null;
		onRetry?: () => void;
		// Phase 1.4: Geometry mode support
		mode?: GeometryMode;
		overlayShapes?: Shape[];
		accumulationWeights?: number[];
		// Phase 3.1: Click-to-select
		onShapeClick?: (shapeId: string | null, event: MouseEvent) => void;
		onSelectionChange?: (selectedIds: Set<string>) => void;
	}

	let {
		shapes = [],
		config = { A: 20, resolution: 360, canvasSize: 400 },
		selectedIds = new Set<string>(),
		width = 400,
		height = 400,
		showGrid = true,
		isLoading = false,
		error = null,
		onRetry,
		mode = "single",
		overlayShapes = [],
		accumulationWeights = [],
		onShapeClick,
		onSelectionChange,
	}: Props = $props();

	// Local selection state (for internal management if no external handler)
	let localSelection = $state<Set<string>>(new Set());
	let effectiveSelection = $derived(
		onSelectionChange ? selectedIds : localSelection,
	);

	// Canvas element reference
	let canvas = $state<HTMLCanvasElement | null>(null);
	let ctx: CanvasRenderingContext2D | null = null;

	// DPI scaling for crisp rendering on high-DPI displays
	let dpr = $state(1);

	// Brand color for selected shapes
	const BRAND_COLOR = "#df728b";
	const GRID_COLOR = "var(--color-border)";
	const GRID_OPACITY = 0.1;
	const HIT_TOLERANCE = 10; // pixels

	/**
	 * Handles canvas click for shape selection
	 */
	function handleCanvasClick(event: MouseEvent): void {
		if (!canvas) return;

		// Convert click to shape coordinates
		const point = canvasEventToPoint(event, canvas);

		// Find clicked shape
		const clickedId = getClickedShape(point, shapes, config, HIT_TOLERANCE);

		// Emit click event if handler provided
		onShapeClick?.(clickedId, event);

		// Handle selection
		if (event.shiftKey && clickedId) {
			// Shift+Click: Add to selection
			const newSelection = new Set(effectiveSelection);
			newSelection.add(clickedId);
			updateSelection(newSelection);
		} else if ((event.ctrlKey || event.metaKey) && clickedId) {
			// Ctrl/Cmd+Click: Toggle selection
			const newSelection = new Set(effectiveSelection);
			if (newSelection.has(clickedId)) {
				newSelection.delete(clickedId);
			} else {
				newSelection.add(clickedId);
			}
			updateSelection(newSelection);
		} else if (clickedId) {
			// Regular click: Select only this shape
			updateSelection(new Set([clickedId]));
		} else {
			// Click on empty area: Deselect all
			updateSelection(new Set());
		}
	}

	/**
	 * Updates selection state
	 */
	function updateSelection(newSelection: Set<string>): void {
		if (onSelectionChange) {
			onSelectionChange(newSelection);
		} else {
			localSelection = newSelection;
		}
	}

	/**
	 * Initialize canvas context with proper DPI scaling
	 */
	function initCanvas(): void {
		if (!canvas) return;

		ctx = canvas.getContext("2d");
		if (!ctx) return;

		// Get device pixel ratio for crisp rendering
		dpr = window.devicePixelRatio || 1;

		// Set canvas size accounting for DPI
		canvas.width = width * dpr;
		canvas.height = height * dpr;

		// Scale context to match DPI
		ctx.scale(dpr, dpr);
	}

	/**
	 * Clears the canvas
	 */
	function clearCanvas(): void {
		if (!ctx) return;
		ctx.clearRect(0, 0, width, height);
	}

	/**
	 * Draws a subtle grid for visual reference
	 */
	function drawGrid(): void {
		if (!ctx || !showGrid) return;

		const centerX = width / 2;
		const centerY = height / 2;
		const gridSpacing = 50;

		ctx.save();
		ctx.strokeStyle = GRID_COLOR;
		ctx.globalAlpha = GRID_OPACITY;
		ctx.lineWidth = 1;

		// Draw vertical lines
		for (let x = centerX % gridSpacing; x < width; x += gridSpacing) {
			ctx.beginPath();
			ctx.moveTo(x, 0);
			ctx.lineTo(x, height);
			ctx.stroke();
		}

		// Draw horizontal lines
		for (let y = centerY % gridSpacing; y < height; y += gridSpacing) {
			ctx.beginPath();
			ctx.moveTo(0, y);
			ctx.lineTo(width, y);
			ctx.stroke();
		}

		// Draw center crosshair with slightly higher opacity
		ctx.globalAlpha = GRID_OPACITY * 2;
		ctx.beginPath();
		ctx.moveTo(centerX, 0);
		ctx.lineTo(centerX, height);
		ctx.moveTo(0, centerY);
		ctx.lineTo(width, centerY);
		ctx.stroke();

		ctx.restore();
	}

	/**
	 * Renders a single shape on the canvas
	 *
	 * @param shape - The shape to render
	 * @param isSelected - Whether the shape is currently selected
	 */
	function renderShape(shape: Shape, isSelected: boolean): void {
		if (!ctx) return;

		const centerX = width / 2;
		const centerY = height / 2;

		// Generate shape points using the shape engine
		const points = generateShapePoints(
			shape.fq,
			shape.R,
			config.A,
			shape.phi,
			config.resolution,
		);

		if (points.length === 0) return;

		ctx.save();

		// Set stroke style
		ctx.strokeStyle = isSelected ? BRAND_COLOR : shape.color;
		ctx.globalAlpha = shape.opacity;
		ctx.lineWidth = isSelected ? shape.strokeWidth + 1 : shape.strokeWidth;
		ctx.lineCap = "round";
		ctx.lineJoin = "round";

		// Enable anti-aliasing (default in canvas)
		ctx.imageSmoothingEnabled = true;
		ctx.imageSmoothingQuality = "high";

		// Draw the shape path
		ctx.beginPath();

		// Move to first point (centered on canvas)
		ctx.moveTo(centerX + points[0].x, centerY + points[0].y);

		// Draw lines to all subsequent points
		for (let i = 1; i < points.length; i++) {
			ctx.lineTo(centerX + points[i].x, centerY + points[i].y);
		}

		// Close the path
		ctx.closePath();
		ctx.stroke();

		// Draw selection highlight glow for selected shapes
		if (isSelected) {
			ctx.save();
			ctx.strokeStyle = BRAND_COLOR;
			ctx.globalAlpha = 0.3;
			ctx.lineWidth = shape.strokeWidth + 4;
			ctx.stroke();
			ctx.restore();
		}

		ctx.restore();
	}

	/**
	 * Renders all shapes with proper z-ordering
	 * Selected shapes are rendered last (on top)
	 */
	function renderAllShapes(): void {
		if (!ctx) return;

		// Clear canvas
		clearCanvas();

		// Draw grid first (background)
		drawGrid();

		// Render based on geometry mode
		switch (mode) {
			case "overlay":
				renderOverlayMode();
				break;
			case "accumulation":
				renderAccumulationMode();
				break;
			case "single":
			default:
				renderSingleMode();
				break;
		}
	}

	/**
	 * Single mode: Render shapes normally with selection highlighting
	 */
	function renderSingleMode(): void {
		// Sort shapes: non-selected first, selected last (for z-ordering)
		const sortedShapes = [...shapes].sort((a, b) => {
			const aSelected = selectedIds.has(a.id);
			const bSelected = selectedIds.has(b.id);
			if (aSelected === bSelected) return 0;
			return aSelected ? 1 : -1;
		});

		// Render each shape
		for (const shape of sortedShapes) {
			const isSelected = selectedIds.has(shape.id);
			renderShape(shape, isSelected);
		}
	}

	/**
	 * Overlay mode: Render primary shapes + overlay shapes with visual distinction
	 */
	function renderOverlayMode(): void {
		// Render primary shapes first
		for (const shape of shapes) {
			const isSelected = selectedIds.has(shape.id);
			renderShape(shape, isSelected);
		}

		// Render overlay shapes with reduced opacity
		for (const shape of overlayShapes) {
			const overlayShape = {
				...shape,
				opacity: shape.opacity * 0.6,
			};
			renderShape(overlayShape, false);
		}
	}

	/**
	 * Accumulation mode: Blend geometries with increasing opacity for persistent structures
	 */
	function renderAccumulationMode(): void {
		if (shapes.length === 0) return;

		// If we have accumulation weights, use them
		// Otherwise, use equal weights with increasing opacity
		const weights =
			accumulationWeights.length === shapes.length
				? accumulationWeights
				: shapes.map((_, i) => (i + 1) / shapes.length);

		// Render shapes with weighted opacity
		for (let i = 0; i < shapes.length; i++) {
			const shape = shapes[i];
			const weight = weights[i];

			// More recent shapes (higher weight) are more opaque
			const accumulatedShape = {
				...shape,
				opacity: shape.opacity * (0.2 + 0.8 * weight),
			};

			renderShape(accumulatedShape, false);
		}
	}

	// Initialize canvas on mount
	onMount(() => {
		initCanvas();
		renderAllShapes();

		// Handle window resize for DPI changes
		const handleResize = () => {
			initCanvas();
			renderAllShapes();
		};

		window.addEventListener("resize", handleResize);

		return () => {
			window.removeEventListener("resize", handleResize);
		};
	});

	// Re-render when shapes, config, or selection changes
	$effect(() => {
		// Track dependencies
		shapes;
		config;
		selectedIds;
		width;
		height;
		showGrid;
		mode;
		overlayShapes;
		accumulationWeights;

		// Re-render
		if (ctx) {
			renderAllShapes();
		}
	});
</script>

{#if error}
	<div
		class="shape-canvas-container error-container"
		style="width: {width}px; height: {height}px;"
	>
		<ErrorState
			title="Canvas Error"
			message={error}
			{onRetry}
			showRetry={!!onRetry}
		/>
	</div>
{:else if isLoading}
	<CanvasSkeleton {width} {height} />
{:else}
	<div class="shape-canvas-container bg-noise">
		<canvas
			bind:this={canvas}
			style="width: {width}px; height: {height}px;"
			class="shape-canvas"
			onclick={handleCanvasClick}
		></canvas>
	</div>
{/if}

<style>
	.shape-canvas-container {
		position: relative;
		display: inline-block;
		background-color: var(--color-card);
		border-radius: var(--radius-xl);
		border: 1px solid var(--color-border);
		overflow: hidden;
		box-shadow: var(--shadow-md);
	}

	.shape-canvas-container.error-container {
		display: flex;
		align-items: center;
		justify-content: center;
		border-color: var(--color-destructive);
	}

	.shape-canvas {
		display: block;
		/* Canvas uses CSS dimensions, actual pixels are scaled by DPI */
	}
</style>
