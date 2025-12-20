/**
 * FFT Processor - Audio frequency analysis using Web Audio API
 * 
 * This module provides functions for computing FFT on audio buffers,
 * mapping frequencies to shape parameters (fq), and extracting
 * top frequency components for visualization.
 */

import type { FFTResult, FrequencyComponent, WindowType } from './types';

/**
 * Normalization strategy for mapping FFT frequencies to shape fq values
 */
export type NormalizationStrategy = 'linear' | 'logarithmic';

/**
 * Options for frequency-to-fq mapping
 */
export interface NormalizationOptions {
  /** Minimum fq value (default: 1) */
  minFq?: number;
  /** Maximum fq value (default: 100) */
  maxFq?: number;
  /** Minimum frequency in Hz to consider (default: 20) */
  minFrequency?: number;
  /** Maximum frequency in Hz to consider (default: 20000) */
  maxFrequency?: number;
  /** Base frequency for logarithmic scaling (default: 20) */
  baseFrequency?: number;
}

const DEFAULT_OPTIONS: Required<NormalizationOptions> = {
  minFq: 1,
  maxFq: 100,
  minFrequency: 20,
  maxFrequency: 20000,
  baseFrequency: 20
};

/**
 * Computes FFT on an audio buffer using Web Audio API.
 * 
 * Uses OfflineAudioContext and AnalyserNode to perform FFT analysis.
 * Returns frequency bins and their corresponding magnitudes.
 * 
 * @param audioBuffer - The audio buffer to analyze
 * @param fftSize - Size of the FFT (must be power of 2, default: 2048)
 * @returns FFTResult with frequencies and magnitudes arrays
 */
export async function computeFFT(
  audioBuffer: AudioBuffer,
  fftSize: number = 2048
): Promise<FFTResult> {
  // Validate fftSize is a power of 2
  if (fftSize < 32 || fftSize > 32768 || (fftSize & (fftSize - 1)) !== 0) {
    throw new Error('FFT size must be a power of 2 between 32 and 32768');
  }

  const sampleRate = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);

  // Create offline audio context for processing
  const offlineContext = new OfflineAudioContext(
    1,
    channelData.length,
    sampleRate
  );

  // Create analyser node
  const analyser = offlineContext.createAnalyser();
  analyser.fftSize = fftSize;
  analyser.smoothingTimeConstant = 0;

  // Create buffer source
  const source = offlineContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(analyser);
  analyser.connect(offlineContext.destination);

  // Start playback
  source.start(0);

  // Render the audio
  await offlineContext.startRendering();

  // Get frequency data
  const frequencyBinCount = analyser.frequencyBinCount;
  const frequencyData = new Float32Array(frequencyBinCount);
  analyser.getFloatFrequencyData(frequencyData);

  // Calculate frequency for each bin
  const frequencies: number[] = [];
  const magnitudes: number[] = [];
  const binWidth = sampleRate / fftSize;

  for (let i = 0; i < frequencyBinCount; i++) {
    const frequency = i * binWidth;
    // Convert from dB to linear magnitude (0-1 range)
    // getFloatFrequencyData returns values in dB (typically -100 to 0)
    const dbValue = frequencyData[i];
    // Normalize: -100dB -> 0, 0dB -> 1
    const magnitude = Math.max(0, (dbValue + 100) / 100);

    frequencies.push(frequency);
    magnitudes.push(magnitude);
  }

  return {
    frequencies,
    magnitudes,
    sampleRate
  };
}

/**
 * Computes FFT synchronously using a simple DFT approach.
 * 
 * This is a fallback for environments where OfflineAudioContext
 * may not work as expected, or for simpler use cases.
 * 
 * @param audioBuffer - The audio buffer to analyze
 * @param fftSize - Size of the FFT (must be power of 2, default: 2048)
 * @returns FFTResult with frequencies and magnitudes arrays
 */
export function computeFFTSync(
  audioBuffer: AudioBuffer,
  fftSize: number = 2048
): FFTResult {
  // Validate fftSize is a power of 2
  if (fftSize < 32 || fftSize > 32768 || (fftSize & (fftSize - 1)) !== 0) {
    throw new Error('FFT size must be a power of 2 between 32 and 32768');
  }

  const sampleRate = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);

  // Take a sample from the middle of the audio for analysis
  const startIndex = Math.max(0, Math.floor(channelData.length / 2) - fftSize / 2);
  const samples = new Float32Array(fftSize);

  for (let i = 0; i < fftSize; i++) {
    const idx = startIndex + i;
    samples[i] = idx < channelData.length ? channelData[idx] : 0;
  }

  // Apply Hanning window to reduce spectral leakage
  for (let i = 0; i < fftSize; i++) {
    const window = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
    samples[i] *= window;
  }

  // Compute DFT (simplified FFT for demonstration)
  const frequencyBinCount = fftSize / 2;
  const frequencies: number[] = [];
  const magnitudes: number[] = [];
  const binWidth = sampleRate / fftSize;

  for (let k = 0; k < frequencyBinCount; k++) {
    let real = 0;
    let imag = 0;

    for (let n = 0; n < fftSize; n++) {
      const angle = (2 * Math.PI * k * n) / fftSize;
      real += samples[n] * Math.cos(angle);
      imag -= samples[n] * Math.sin(angle);
    }

    const magnitude = Math.sqrt(real * real + imag * imag) / fftSize;
    frequencies.push(k * binWidth);
    magnitudes.push(magnitude);
  }

  // Normalize magnitudes to 0-1 range
  const maxMagnitude = Math.max(...magnitudes, 1e-10);
  const normalizedMagnitudes = magnitudes.map(m => m / maxMagnitude);

  return {
    frequencies,
    magnitudes: normalizedMagnitudes,
    sampleRate
  };
}

/**
 * Maps an FFT frequency (Hz) to an integer fq value for shape generation.
 * 
 * Strategies:
 * - linear: fq = minFq + (freq - minFreq) / (maxFreq - minFreq) * (maxFq - minFq)
 * - logarithmic: fq = minFq + log2(freq / baseFreq) / log2(maxFreq / baseFreq) * (maxFq - minFq)
 * 
 * The mapping is deterministic: same input always produces same output.
 * 
 * @param frequencyHz - Frequency in Hz to map
 * @param strategy - Normalization strategy ('linear' or 'logarithmic')
 * @param options - Configuration options for the mapping
 * @returns Integer fq value (≥ 1)
 */
export function mapFrequencyToFq(
  frequencyHz: number,
  strategy: NormalizationStrategy,
  options: NormalizationOptions = {}
): number {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const { minFq, maxFq, minFrequency, maxFrequency, baseFrequency } = opts;

  // Clamp frequency to valid range
  const clampedFreq = Math.max(minFrequency, Math.min(maxFrequency, frequencyHz));

  let normalizedValue: number;

  if (strategy === 'linear') {
    // Linear mapping
    normalizedValue = (clampedFreq - minFrequency) / (maxFrequency - minFrequency);
  } else {
    // Logarithmic mapping (better for audio perception)
    const logMin = Math.log2(minFrequency / baseFrequency);
    const logMax = Math.log2(maxFrequency / baseFrequency);
    const logFreq = Math.log2(clampedFreq / baseFrequency);
    normalizedValue = (logFreq - logMin) / (logMax - logMin);
  }

  // Map to fq range and round to integer
  const fq = Math.round(minFq + normalizedValue * (maxFq - minFq));

  // Ensure fq is at least 1
  return Math.max(1, fq);
}


/**
 * Extracts the top N frequency components by magnitude from FFT results.
 * 
 * Sorts frequency components by magnitude (descending) and returns
 * the top N components with their fq values mapped using the specified strategy.
 * 
 * @param fftResult - The FFT result to extract from
 * @param count - Number of top components to extract
 * @param strategy - Normalization strategy for fq mapping (default: 'logarithmic')
 * @param options - Options for frequency-to-fq mapping
 * @returns Array of FrequencyComponent objects sorted by magnitude (descending)
 */
export function extractTopFrequencies(
  fftResult: FFTResult,
  count: number,
  strategy: NormalizationStrategy = 'logarithmic',
  options: NormalizationOptions = {}
): FrequencyComponent[] {
  const { frequencies, magnitudes } = fftResult;
  const opts = { ...DEFAULT_OPTIONS, ...options };

  // Create array of frequency-magnitude pairs with indices
  const components: Array<{ index: number; frequency: number; magnitude: number }> = [];

  for (let i = 0; i < frequencies.length; i++) {
    const freq = frequencies[i];
    // Filter out frequencies outside the audible/useful range
    if (freq >= opts.minFrequency && freq <= opts.maxFrequency) {
      components.push({
        index: i,
        frequency: freq,
        magnitude: magnitudes[i]
      });
    }
  }

  // Sort by magnitude (descending)
  components.sort((a, b) => b.magnitude - a.magnitude);

  // Take top N components
  const topComponents = components.slice(0, Math.min(count, components.length));

  // Map to FrequencyComponent interface
  return topComponents.map((comp, idx) => ({
    id: `freq-${idx}-${comp.index}`,
    frequencyHz: comp.frequency,
    magnitude: comp.magnitude,
    fq: mapFrequencyToFq(comp.frequency, strategy, options),
    selected: false
  }));
}

/**
 * Validates FFT parameters.
 * 
 * @param fftSize - The FFT size to validate
 * @returns Object with valid flag and error message if invalid
 */
export function validateFFTParams(fftSize: number): { valid: boolean; error?: string } {
  if (!Number.isInteger(fftSize)) {
    return { valid: false, error: 'FFT size must be an integer' };
  }
  if (fftSize < 32 || fftSize > 32768) {
    return { valid: false, error: 'FFT size must be between 32 and 32768' };
  }
  if ((fftSize & (fftSize - 1)) !== 0) {
    return { valid: false, error: 'FFT size must be a power of 2' };
  }
  return { valid: true };
}

/**
 * Options for Short-Time Fourier Transform (STFT)
 */
export interface STFTOptions {
  /** Start time in seconds */
  startTime: number;
  /** Window width in milliseconds */
  windowWidth: number;
  /** Step size in milliseconds for sliding window */
  stepSize: number;
  /** Window function type */
  windowType: WindowType;
  /** FFT size (power of 2, default: 2048) */
  fftSize?: number;
}

/**
 * Applies a window function to samples
 * 
 * @param samples - The sample array to window (modified in place)
 * @param windowType - Type of window function
 */
function applyWindow(samples: Float32Array, windowType: WindowType): void {
  const N = samples.length;

  for (let i = 0; i < N; i++) {
    let w: number;

    switch (windowType) {
      case 'hann':
        w = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
        break;
      case 'hamming':
        w = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (N - 1));
        break;
      case 'blackman':
        w = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1))
          + 0.08 * Math.cos((4 * Math.PI * i) / (N - 1));
        break;
      case 'rectangular':
      default:
        w = 1;
        break;
    }

    samples[i] *= w;
  }
}

/**
 * Computes DFT on a windowed sample array
 * 
 * @param samples - Windowed samples
 * @param sampleRate - Sample rate of the audio
 * @returns FFTResult for this window
 */
function computeDFT(samples: Float32Array, sampleRate: number): FFTResult {
  const fftSize = samples.length;
  const frequencyBinCount = fftSize / 2;
  const frequencies: number[] = [];
  const magnitudes: number[] = [];
  const binWidth = sampleRate / fftSize;

  for (let k = 0; k < frequencyBinCount; k++) {
    let real = 0;
    let imag = 0;

    for (let n = 0; n < fftSize; n++) {
      const angle = (2 * Math.PI * k * n) / fftSize;
      real += samples[n] * Math.cos(angle);
      imag -= samples[n] * Math.sin(angle);
    }

    const magnitude = Math.sqrt(real * real + imag * imag) / fftSize;
    frequencies.push(k * binWidth);
    magnitudes.push(magnitude);
  }

  // Normalize magnitudes to 0-1 range
  const maxMagnitude = Math.max(...magnitudes, 1e-10);
  const normalizedMagnitudes = magnitudes.map(m => m / maxMagnitude);

  return {
    frequencies,
    magnitudes: normalizedMagnitudes,
    sampleRate
  };
}

/**
 * Computes Short-Time Fourier Transform (STFT) on an audio buffer.
 * 
 * Slides a window across the audio and computes FFT for each position,
 * enabling time-frequency analysis.
 * 
 * @param audioBuffer - The audio buffer to analyze
 * @param options - STFT configuration options
 * @returns Array of FFTResult for each window position
 */
export function computeSTFT(
  audioBuffer: AudioBuffer,
  options: STFTOptions
): FFTResult[] {
  const { startTime, windowWidth, stepSize, windowType, fftSize = 2048 } = options;
  const sampleRate = audioBuffer.sampleRate;
  const channelData = audioBuffer.getChannelData(0);

  // Convert times to sample indices
  const startSample = Math.floor(startTime * sampleRate);
  const windowSamples = Math.floor((windowWidth / 1000) * sampleRate);
  const stepSamples = Math.floor((stepSize / 1000) * sampleRate);

  // Determine how many samples to use (limited by fftSize)
  const actualWindowSize = Math.min(windowSamples, fftSize);

  // Calculate number of windows
  const totalSamples = channelData.length;
  const results: FFTResult[] = [];

  let currentStart = startSample;

  while (currentStart + actualWindowSize <= totalSamples) {
    // Extract samples for this window
    const samples = new Float32Array(fftSize);

    for (let i = 0; i < actualWindowSize && currentStart + i < totalSamples; i++) {
      samples[i] = channelData[currentStart + i];
    }
    // Remaining samples are zero-padded

    // Apply window function
    applyWindow(samples, windowType);

    // Compute DFT for this window
    const result = computeDFT(samples, sampleRate);
    results.push(result);

    // Move to next window
    currentStart += stepSamples;

    // Safety limit to prevent infinite loops
    if (results.length > 10000) {
      console.warn('STFT: Maximum window count reached');
      break;
    }
  }

  return results;
}

/**
 * Computes a single FFT window at a specific time position.
 * 
 * @param audioBuffer - The audio buffer to analyze
 * @param startTime - Start time in seconds
 * @param windowWidth - Window width in milliseconds
 * @param windowType - Window function type
 * @param fftSize - FFT size (default: 2048)
 * @returns FFTResult for this time window
 */
export function computeFFTAtTime(
  audioBuffer: AudioBuffer,
  startTime: number,
  windowWidth: number,
  windowType: WindowType = 'hann',
  fftSize: number = 2048
): FFTResult {
  const results = computeSTFT(audioBuffer, {
    startTime,
    windowWidth,
    stepSize: windowWidth, // Single step
    windowType,
    fftSize
  });

  return results[0] || {
    frequencies: [],
    magnitudes: [],
    sampleRate: audioBuffer.sampleRate
  };
}

/**
 * Computes spectral flux for transient detection.
 * 
 * Spectral flux measures the rate of change in the spectrum over time.
 * High values indicate transients (sudden changes in frequency content).
 * 
 * Flux(t) = Σf (|X(t,f)| - |X(t-1,f)|)²
 * 
 * @param fftResults - Array of FFT results from STFT
 * @returns Array of spectral flux values (one per frame, starting from index 1)
 */
export function computeSpectralFlux(fftResults: FFTResult[]): number[] {
  if (fftResults.length < 2) {
    return [];
  }

  const fluxValues: number[] = [];

  for (let t = 1; t < fftResults.length; t++) {
    const current = fftResults[t].magnitudes;
    const previous = fftResults[t - 1].magnitudes;

    let flux = 0;
    const binCount = Math.min(current.length, previous.length);

    for (let f = 0; f < binCount; f++) {
      // Only consider positive changes (onset detection)
      const diff = current[f] - previous[f];
      if (diff > 0) {
        flux += diff * diff;
      }
    }

    fluxValues.push(Math.sqrt(flux));
  }

  // Normalize flux values
  const maxFlux = Math.max(...fluxValues, 1e-10);
  return fluxValues.map(f => f / maxFlux);
}

/**
 * Detects transients based on spectral flux threshold.
 * 
 * @param fluxValues - Normalized spectral flux values
 * @param threshold - Detection threshold (0-1, default: 0.5)
 * @returns Array of indices where transients are detected
 */
export function detectTransients(
  fluxValues: number[],
  threshold: number = 0.5
): number[] {
  const transients: number[] = [];

  for (let i = 0; i < fluxValues.length; i++) {
    if (fluxValues[i] > threshold) {
      transients.push(i + 1); // Offset by 1 because flux starts at frame 1
    }
  }

  return transients;
}
