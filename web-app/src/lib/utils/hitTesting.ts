/**
 * Hit Testing Utilities
 * 
 * Functions for detecting which shape was clicked on a canvas.
 * Uses point-to-path distance calculations for stroke-based shapes.
 * 
 * Phase 3: Task 3.1
 */

import { generateShapePoints } from '$lib/shapeEngine';
import type { Shape, ShapeConfig, Point } from '$lib/types';

/**
 * Calculates the distance from a point to a line segment
 */
function pointToSegmentDistance(
    px: number,
    py: number,
    x1: number,
    y1: number,
    x2: number,
    y2: number
): number {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lengthSq = dx * dx + dy * dy;

    if (lengthSq === 0) {
        // Segment is a point
        return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }

    // Project point onto line, clamped to segment
    let t = ((px - x1) * dx + (py - y1) * dy) / lengthSq;
    t = Math.max(0, Math.min(1, t));

    const projX = x1 + t * dx;
    const projY = y1 + t * dy;

    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

/**
 * Calculates the minimum distance from a point to a shape's path
 * 
 * @param point - The point to check (in canvas coordinates, relative to center)
 * @param pathPoints - The path points of the shape
 * @returns Minimum distance to the path
 */
export function pointToPathDistance(
    point: Point,
    pathPoints: Point[]
): number {
    if (pathPoints.length === 0) return Infinity;
    if (pathPoints.length === 1) {
        return Math.sqrt((point.x - pathPoints[0].x) ** 2 + (point.y - pathPoints[0].y) ** 2);
    }

    let minDistance = Infinity;

    for (let i = 0; i < pathPoints.length; i++) {
        const p1 = pathPoints[i];
        const p2 = pathPoints[(i + 1) % pathPoints.length];

        const distance = pointToSegmentDistance(
            point.x,
            point.y,
            p1.x,
            p1.y,
            p2.x,
            p2.y
        );

        minDistance = Math.min(minDistance, distance);
    }

    return minDistance;
}

/**
 * Determines which shape was clicked based on canvas coordinates
 * 
 * @param clickPoint - Canvas coordinates of the click (relative to center)
 * @param shapes - Array of shapes to test
 * @param config - Shape configuration
 * @param tolerance - Maximum distance from path to count as a hit (pixels)
 * @returns The ID of the clicked shape, or null if none
 */
export function getClickedShape(
    clickPoint: Point,
    shapes: Shape[],
    config: ShapeConfig,
    tolerance: number = 10
): string | null {
    // Test shapes in reverse order (top shapes first based on z-index)
    // Shapes drawn last are on top
    const sortedShapes = [...shapes].reverse();

    for (const shape of sortedShapes) {
        const pathPoints = generateShapePoints(
            shape.fq,
            shape.R,
            config.A,
            shape.phi,
            config.resolution
        );

        const distance = pointToPathDistance(clickPoint, pathPoints);
        const effectiveTolerance = tolerance + shape.strokeWidth / 2;

        if (distance <= effectiveTolerance) {
            return shape.id;
        }
    }

    return null;
}

/**
 * Converts canvas click event coordinates to shape-relative coordinates
 * 
 * @param event - The mouse event
 * @param canvas - The canvas element
 * @returns Point relative to canvas center
 */
export function canvasEventToPoint(
    event: MouseEvent,
    canvas: HTMLCanvasElement
): Point {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    // Calculate position relative to canvas element
    const canvasX = (event.clientX - rect.left) * scaleX;
    const canvasY = (event.clientY - rect.top) * scaleY;

    // Convert to center-relative coordinates
    // Note: Need to account for DPI scaling
    const dpr = window.devicePixelRatio || 1;
    const centerX = canvas.width / (2 * dpr);
    const centerY = canvas.height / (2 * dpr);

    return {
        x: canvasX / dpr - centerX,
        y: canvasY / dpr - centerY
    };
}

/**
 * Gets all shapes within a rectangular selection box
 * 
 * @param startPoint - Start corner of selection box (center-relative)
 * @param endPoint - End corner of selection box (center-relative)
 * @param shapes - Array of shapes to test
 * @param config - Shape configuration
 * @returns Array of shape IDs within the selection
 */
export function getShapesInSelection(
    startPoint: Point,
    endPoint: Point,
    shapes: Shape[],
    config: ShapeConfig
): string[] {
    const minX = Math.min(startPoint.x, endPoint.x);
    const maxX = Math.max(startPoint.x, endPoint.x);
    const minY = Math.min(startPoint.y, endPoint.y);
    const maxY = Math.max(startPoint.y, endPoint.y);

    const selected: string[] = [];

    for (const shape of shapes) {
        const pathPoints = generateShapePoints(
            shape.fq,
            shape.R,
            config.A,
            shape.phi,
            config.resolution
        );

        // Check if any path point is within the selection box
        const isInSelection = pathPoints.some(
            p => p.x >= minX && p.x <= maxX && p.y >= minY && p.y <= maxY
        );

        if (isInSelection) {
            selected.push(shape.id);
        }
    }

    return selected;
}

/**
 * Calculates the center point of a shape
 * 
 * @param shape - The shape
 * @param config - Shape configuration
 * @returns Center point of the shape's bounding box
 */
export function getShapeCenter(shape: Shape, config: ShapeConfig): Point {
    const pathPoints = generateShapePoints(
        shape.fq,
        shape.R,
        config.A,
        shape.phi,
        config.resolution
    );

    if (pathPoints.length === 0) {
        return { x: 0, y: 0 };
    }

    let sumX = 0;
    let sumY = 0;

    for (const p of pathPoints) {
        sumX += p.x;
        sumY += p.y;
    }

    return {
        x: sumX / pathPoints.length,
        y: sumY / pathPoints.length
    };
}

/**
 * Gets the bounding box of selected shapes
 * 
 * @param shapes - Array of selected shapes
 * @param config - Shape configuration
 * @returns Bounding box { minX, minY, maxX, maxY } or null if no shapes
 */
export function getSelectionBounds(
    shapes: Shape[],
    config: ShapeConfig
): { minX: number; minY: number; maxX: number; maxY: number } | null {
    if (shapes.length === 0) return null;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const shape of shapes) {
        const pathPoints = generateShapePoints(
            shape.fq,
            shape.R,
            config.A,
            shape.phi,
            config.resolution
        );

        for (const p of pathPoints) {
            minX = Math.min(minX, p.x);
            minY = Math.min(minY, p.y);
            maxX = Math.max(maxX, p.x);
            maxY = Math.max(maxY, p.y);
        }
    }

    return { minX, minY, maxX, maxY };
}
