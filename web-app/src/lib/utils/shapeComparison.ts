/**
 * Shape Comparison Utilities
 * 
 * Functions for comparing and analyzing geometric shapes from different sources.
 * Supports intersection detection, difference computation, and similarity scoring.
 * 
 * Phase 2: Task 2.3
 */

import { generateShapePoints } from '$lib/shapeEngine';
import type { Shape, ShapeConfig, Point } from '$lib/types';

/**
 * Represents a 2D grid for shape rasterization
 */
interface RasterGrid {
    width: number;
    height: number;
    data: boolean[];
}

/**
 * Creates an empty raster grid
 */
function createGrid(width: number, height: number): RasterGrid {
    return {
        width,
        height,
        data: new Array(width * height).fill(false)
    };
}

/**
 * Gets grid index for a coordinate
 */
function getGridIndex(grid: RasterGrid, x: number, y: number): number {
    const gridX = Math.floor(x + grid.width / 2);
    const gridY = Math.floor(y + grid.height / 2);

    if (gridX < 0 || gridX >= grid.width || gridY < 0 || gridY >= grid.height) {
        return -1;
    }

    return gridY * grid.width + gridX;
}

/**
 * Rasterizes a shape's outline onto a grid
 */
function rasterizeShape(
    shape: Shape,
    config: ShapeConfig,
    grid: RasterGrid
): void {
    const points = generateShapePoints(
        shape.fq,
        shape.R,
        config.A,
        shape.phi,
        config.resolution
    );

    // Draw lines between consecutive points
    for (let i = 0; i < points.length; i++) {
        const p1 = points[i];
        const p2 = points[(i + 1) % points.length];

        // Bresenham's line algorithm
        rasterizeLine(grid, p1.x, p1.y, p2.x, p2.y);
    }
}

/**
 * Rasterizes a line segment using Bresenham's algorithm
 */
function rasterizeLine(
    grid: RasterGrid,
    x0: number,
    y0: number,
    x1: number,
    y1: number
): void {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = x0 < x1 ? 1 : -1;
    const sy = y0 < y1 ? 1 : -1;
    let err = dx - dy;

    let x = x0;
    let y = y0;

    while (true) {
        const idx = getGridIndex(grid, x, y);
        if (idx >= 0) {
            grid.data[idx] = true;
        }

        if (Math.abs(x - x1) < 1 && Math.abs(y - y1) < 1) break;

        const e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x += sx;
        }
        if (e2 < dx) {
            err += dx;
            y += sy;
        }
    }
}

/**
 * Computes the intersection of two shape sets
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param config - Shape configuration
 * @param resolution - Grid resolution (higher = more precise, slower)
 * @returns Intersection data and statistics
 */
export function computeIntersection(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig,
    resolution: number = 200
): {
    intersectionMask: boolean[];
    intersectionCount: number;
    totalA: number;
    totalB: number;
    overlapRatio: number;
} {
    const gridSize = resolution;
    const gridA = createGrid(gridSize, gridSize);
    const gridB = createGrid(gridSize, gridSize);

    // Rasterize each shape set
    shapesA.forEach(shape => rasterizeShape(shape, config, gridA));
    shapesB.forEach(shape => rasterizeShape(shape, config, gridB));

    // Compute intersection
    const intersectionMask = new Array(gridSize * gridSize).fill(false);
    let intersectionCount = 0;
    let totalA = 0;
    let totalB = 0;

    for (let i = 0; i < gridSize * gridSize; i++) {
        if (gridA.data[i]) totalA++;
        if (gridB.data[i]) totalB++;
        if (gridA.data[i] && gridB.data[i]) {
            intersectionMask[i] = true;
            intersectionCount++;
        }
    }

    const union = totalA + totalB - intersectionCount;
    const overlapRatio = union > 0 ? intersectionCount / union : 0;

    return {
        intersectionMask,
        intersectionCount,
        totalA,
        totalB,
        overlapRatio
    };
}

/**
 * Computes the difference between two shape sets
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param config - Shape configuration
 * @param resolution - Grid resolution
 * @returns Difference data for each source
 */
export function computeDifference(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig,
    resolution: number = 200
): {
    onlyA: boolean[];
    onlyB: boolean[];
    countOnlyA: number;
    countOnlyB: number;
} {
    const gridSize = resolution;
    const gridA = createGrid(gridSize, gridSize);
    const gridB = createGrid(gridSize, gridSize);

    shapesA.forEach(shape => rasterizeShape(shape, config, gridA));
    shapesB.forEach(shape => rasterizeShape(shape, config, gridB));

    const onlyA = new Array(gridSize * gridSize).fill(false);
    const onlyB = new Array(gridSize * gridSize).fill(false);
    let countOnlyA = 0;
    let countOnlyB = 0;

    for (let i = 0; i < gridSize * gridSize; i++) {
        if (gridA.data[i] && !gridB.data[i]) {
            onlyA[i] = true;
            countOnlyA++;
        }
        if (gridB.data[i] && !gridA.data[i]) {
            onlyB[i] = true;
            countOnlyB++;
        }
    }

    return { onlyA, onlyB, countOnlyA, countOnlyB };
}

/**
 * Computes a similarity score between two shape sets
 * Uses Jaccard similarity (intersection over union)
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param config - Shape configuration
 * @param resolution - Grid resolution
 * @returns Similarity score (0-1)
 */
export function computeSimilarityScore(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig,
    resolution: number = 200
): number {
    const { overlapRatio } = computeIntersection(shapesA, shapesB, config, resolution);
    return overlapRatio;
}

/**
 * Finds the most similar shapes between two sets
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param config - Shape configuration
 * @returns Array of shape pairs sorted by similarity
 */
export function findSimilarShapes(
    shapesA: Shape[],
    shapesB: Shape[],
    config: ShapeConfig
): Array<{
    shapeA: Shape;
    shapeB: Shape;
    similarity: number;
}> {
    const pairs: Array<{
        shapeA: Shape;
        shapeB: Shape;
        similarity: number;
    }> = [];

    for (const shapeA of shapesA) {
        for (const shapeB of shapesB) {
            const similarity = computeSimilarityScore(
                [shapeA],
                [shapeB],
                config,
                100 // Lower resolution for performance
            );
            pairs.push({ shapeA, shapeB, similarity });
        }
    }

    // Sort by similarity descending
    return pairs.sort((a, b) => b.similarity - a.similarity);
}

/**
 * Computes frequency overlap between two shape sets
 * 
 * @param shapesA - Shapes from source A
 * @param shapesB - Shapes from source B
 * @param tolerance - Frequency tolerance in Hz
 * @returns Frequency overlap statistics
 */
export function computeFrequencyOverlap(
    shapesA: Shape[],
    shapesB: Shape[],
    tolerance: number = 10
): {
    matchingFrequencies: Array<{ freqA: number; freqB: number }>;
    uniqueToA: number[];
    uniqueToB: number[];
    overlapScore: number;
} {
    const freqsA = shapesA.map(s => s.fq);
    const freqsB = shapesB.map(s => s.fq);

    const matchingFrequencies: Array<{ freqA: number; freqB: number }> = [];
    const matchedA = new Set<number>();
    const matchedB = new Set<number>();

    for (const freqA of freqsA) {
        for (const freqB of freqsB) {
            if (Math.abs(freqA - freqB) <= tolerance) {
                matchingFrequencies.push({ freqA, freqB });
                matchedA.add(freqA);
                matchedB.add(freqB);
            }
        }
    }

    const uniqueToA = freqsA.filter(f => !matchedA.has(f));
    const uniqueToB = freqsB.filter(f => !matchedB.has(f));

    const total = freqsA.length + freqsB.length;
    const matched = matchingFrequencies.length * 2;
    const overlapScore = total > 0 ? matched / total : 0;

    return {
        matchingFrequencies,
        uniqueToA,
        uniqueToB,
        overlapScore
    };
}
