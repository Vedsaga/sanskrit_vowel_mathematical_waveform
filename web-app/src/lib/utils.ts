import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
	return twMerge(clsx(inputs));
}

/**
 * Type utilities for shadcn-svelte components
 */

/**
 * Removes the 'children' property from a type
 */
export type WithoutChildren<T> = T extends { children?: unknown }
	? Omit<T, "children">
	: T;

/**
 * Removes the 'child' property from a type
 */
export type WithoutChild<T> = T extends { child?: unknown }
	? Omit<T, "child">
	: T;

/**
 * Removes both 'children' and 'child' properties from a type
 */
export type WithoutChildrenOrChild<T> = WithoutChildren<WithoutChild<T>>;

/**
 * Adds an optional 'ref' property that can be bound to an element
 */
export type WithElementRef<
	T,
	E extends HTMLElement = HTMLElement
> = T & {
	ref?: E | null;
};
