/**
 * FFT Processor - Audio frequency analysis using Web Audio API
 * 
 * This module provides functions for computing FFT on audio buffers,
 * mapping frequencies to shape parameters (fq), and extracting
 * top frequency components for visualization.
 */

import type { FFTResult, FrequencyComponent } from './types';

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
