<script lang="ts">
	/**
	 * AudioUploader Component
	 * 
	 * Provides drag-and-drop and click-to-upload functionality for audio files.
	 * Validates file formats (WAV, MP3, OGG) and displays loading/error states.
	 * 
	 * Requirements: 4.1, 4.6, 4.7, 6.1, 6.2
	 */
	import { Upload, FileAudio, AlertCircle, Loader2, X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';

	// Supported audio formats
	const SUPPORTED_FORMATS = ['audio/wav', 'audio/mpeg', 'audio/ogg', 'audio/mp3', 'audio/x-wav'];
	const SUPPORTED_EXTENSIONS = ['.wav', '.mp3', '.ogg'];

	// Props
	interface Props {
		onFileLoaded?: (buffer: AudioBuffer, fileName: string) => void;
		isProcessing?: boolean;
		error?: string | null;
		fileName?: string | null;
	}

	let {
		onFileLoaded,
		isProcessing = false,
		error = null,
		fileName = null
	}: Props = $props();

	// Local state
	let isDragOver = $state(false);
	let localError = $state<string | null>(null);
	let fileInput: HTMLInputElement;

	// Combined error state
	let displayError = $derived(error || localError);

	/**
	 * Validates if the file is a supported audio format
	 */
	function isValidAudioFile(file: File): boolean {
		// Check MIME type
		if (SUPPORTED_FORMATS.includes(file.type)) {
			return true;
		}
		// Fallback: check file extension
		const extension = '.' + file.name.split('.').pop()?.toLowerCase();
		return SUPPORTED_EXTENSIONS.includes(extension);
	}

	/**
	 * Decodes an audio file to AudioBuffer
	 */
	async function decodeAudioFile(file: File): Promise<AudioBuffer> {
		const arrayBuffer = await file.arrayBuffer();
		const audioContext = new AudioContext();
		try {
			const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
			return audioBuffer;
		} finally {
			await audioContext.close();
		}
	}

	/**
	 * Handles file selection (from drop or input)
	 */
	async function handleFile(file: File) {
		localError = null;

		// Validate file format
		if (!isValidAudioFile(file)) {
			localError = `Unsupported format. Please use ${SUPPORTED_EXTENSIONS.join(', ')} files.`;
			return;
		}

		try {
			const audioBuffer = await decodeAudioFile(file);
			onFileLoaded?.(audioBuffer, file.name);
		} catch (err) {
			localError = 'Failed to decode audio file. The file may be corrupted or invalid.';
			console.error('Audio decode error:', err);
		}
	}

	/**
	 * Handles drag over event
	 */
	function handleDragOver(event: DragEvent) {
		event.preventDefault();
		isDragOver = true;
	}

	/**
	 * Handles drag leave event
	 */
	function handleDragLeave(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;
	}

	/**
	 * Handles drop event
	 */
	function handleDrop(event: DragEvent) {
		event.preventDefault();
		isDragOver = false;

		const files = event.dataTransfer?.files;
		if (files && files.length > 0) {
			handleFile(files[0]);
		}
	}

	/**
	 * Handles file input change
	 */
	function handleInputChange(event: Event) {
		const target = event.target as HTMLInputElement;
		const files = target.files;
		if (files && files.length > 0) {
			handleFile(files[0]);
		}
		// Reset input so same file can be selected again
		target.value = '';
	}

	/**
	 * Opens file picker
	 */
	function openFilePicker() {
		fileInput?.click();
	}

	/**
	 * Clears the error state
	 */
	function clearError() {
		localError = null;
	}

	/**
	 * Retries by opening file picker
	 */
	function handleRetry() {
		clearError();
		openFilePicker();
	}
</script>

<div class="audio-uploader">
	<!-- Hidden file input -->
	<input
		bind:this={fileInput}
		type="file"
		accept={SUPPORTED_EXTENSIONS.join(',')}
		onchange={handleInputChange}
		class="hidden"
		aria-hidden="true"
	/>

	<!-- Drop zone -->
	<button
		type="button"
		class="drop-zone"
		class:drag-over={isDragOver}
		class:has-file={fileName && !displayError}
		class:has-error={displayError}
		class:is-processing={isProcessing}
		ondragover={handleDragOver}
		ondragleave={handleDragLeave}
		ondrop={handleDrop}
		onclick={openFilePicker}
		disabled={isProcessing}
		aria-label={fileName ? `Current file: ${fileName}. Click to change.` : 'Upload audio file'}
	>
		{#if isProcessing}
			<!-- Loading state -->
			<div class="drop-zone-content">
				<div class="icon-container loading">
					<Loader2 size={32} class="animate-spin" />
				</div>
				<p class="drop-zone-title">Processing audio...</p>
				<p class="drop-zone-subtitle">Analyzing frequency components</p>
			</div>
		{:else if displayError}
			<!-- Error state -->
			<div class="drop-zone-content">
				<div class="icon-container error">
					<AlertCircle size={32} />
				</div>
				<p class="drop-zone-title error-text">{displayError}</p>
				<div class="error-actions">
					<Button variant="outline" size="sm" onclick={handleRetry}>
						Try Again
					</Button>
				</div>
			</div>
		{:else if fileName}
			<!-- File loaded state -->
			<div class="drop-zone-content">
				<div class="icon-container success">
					<FileAudio size={32} />
				</div>
				<p class="drop-zone-title">{fileName}</p>
				<p class="drop-zone-subtitle">Click or drop to replace</p>
			</div>
		{:else}
			<!-- Default state -->
			<div class="drop-zone-content">
				<div class="icon-container">
					<Upload size={32} />
				</div>
				<p class="drop-zone-title">Drop audio file here</p>
				<p class="drop-zone-subtitle">or click to browse</p>
				<p class="supported-formats">Supports {SUPPORTED_EXTENSIONS.join(', ')}</p>
			</div>
		{/if}
	</button>
</div>

<style>
	.audio-uploader {
		width: 100%;
	}

	.hidden {
		display: none;
	}

	.drop-zone {
		width: 100%;
		min-height: 180px;
		padding: 2rem;
		border: 2px dashed var(--color-border);
		border-radius: var(--radius-lg);
		background-color: var(--color-card);
		cursor: pointer;
		transition: all 0.2s ease-out;
		display: flex;
		align-items: center;
		justify-content: center;
		text-align: center;
	}

	.drop-zone:hover:not(:disabled) {
		border-color: var(--color-brand);
		background-color: color-mix(in srgb, var(--color-brand) 5%, var(--color-card));
	}

	.drop-zone:focus-visible {
		outline: 2px solid var(--color-brand);
		outline-offset: 2px;
	}

	.drop-zone.drag-over {
		border-color: var(--color-brand);
		background-color: color-mix(in srgb, var(--color-brand) 10%, var(--color-card));
		border-style: solid;
	}

	.drop-zone.has-file {
		border-style: solid;
		border-color: var(--color-border);
	}

	.drop-zone.has-error {
		border-color: var(--color-destructive);
		background-color: color-mix(in srgb, var(--color-destructive) 5%, var(--color-card));
	}

	.drop-zone.is-processing {
		cursor: wait;
		opacity: 0.8;
	}

	.drop-zone:disabled {
		cursor: not-allowed;
	}

	.drop-zone-content {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.75rem;
	}

	.icon-container {
		width: 64px;
		height: 64px;
		border-radius: var(--radius-full);
		background-color: var(--color-muted);
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--color-muted-foreground);
		transition: all 0.2s ease-out;
	}

	.drop-zone:hover:not(:disabled) .icon-container {
		background-color: color-mix(in srgb, var(--color-brand) 15%, var(--color-muted));
		color: var(--color-brand);
	}

	.icon-container.loading {
		background-color: color-mix(in srgb, var(--color-brand) 15%, var(--color-muted));
		color: var(--color-brand);
	}

	.icon-container.error {
		background-color: color-mix(in srgb, var(--color-destructive) 15%, var(--color-muted));
		color: var(--color-destructive);
	}

	.icon-container.success {
		background-color: color-mix(in srgb, var(--color-brand) 15%, var(--color-muted));
		color: var(--color-brand);
	}

	.drop-zone-title {
		font-size: 1rem;
		font-weight: 500;
		color: var(--color-foreground);
	}

	.drop-zone-title.error-text {
		color: var(--color-destructive);
		max-width: 300px;
	}

	.drop-zone-subtitle {
		font-size: 0.875rem;
		color: var(--color-muted-foreground);
	}

	.supported-formats {
		font-size: 0.75rem;
		color: var(--color-muted-foreground);
		margin-top: 0.5rem;
	}

	.error-actions {
		margin-top: 0.5rem;
	}

	/* Animation for loading spinner */
	:global(.animate-spin) {
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		from {
			transform: rotate(0deg);
		}
		to {
			transform: rotate(360deg);
		}
	}
</style>
