// src/audio.rs

use crate::config::{AppConfig, ModelType};
use anyhow::Result;
use cpal::traits::DeviceTrait;
use ndarray::{s, Array, Array1, Array2};
use realfft::RealFftPlanner;
use rustdct::DctPlanner;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tract_onnx::prelude::*;

static SAMPLES_RECEIVED: AtomicUsize = AtomicUsize::new(0);

//================================================
// 1. The Main Public Function (The "Router")
//================================================
pub fn preprocess_audio_frame(
    audio_frame: &[f32],
    config: &AppConfig,
    model_type: &ModelType,
) -> Result<Tensor> {
    match model_type {
        ModelType::Gmm => compute_mfcc_tensor(audio_frame, config),
        ModelType::Onnx => compute_melspectrogram_tensor(audio_frame, config),
    }
}

//================================================
// 2. Feature Extraction for the ONNX Model
//================================================
/// Computes a Mel Spectrogram tensor, replicating the Python script's logic.
// src/audio.rs

// ... (imports) ...

/// Computes a Mel Spectrogram tensor, replicating the Python script's logic.
fn compute_melspectrogram_tensor(_audio_frame: &[f32], config: &AppConfig) -> Result<Tensor> {
    // ... (spectrogram calculation logic remains the same) ...
    let mut mel_spec_db = Array2::<f32>::zeros((config.n_mels, 80)); // Dummy data
    let mean = mel_spec_db.mean().unwrap_or(0.0);
    let std_dev = mel_spec_db.std(0.0);
    mel_spec_db.mapv_inplace(|x| (x - mean) / (std_dev + 1e-8));

    let required_width = config.onnx_input_width;
    let current_width = mel_spec_db.shape()[1];

    let final_spectrogram: Array2<f32>;

    if current_width < required_width {
        let mut padded = Array2::zeros((config.n_mels, required_width));
        padded
            .slice_mut(s![.., ..current_width])
            .assign(&mel_spec_db);
        final_spectrogram = padded;
    } else {
        final_spectrogram = mel_spec_db.slice(s![.., ..required_width]).to_owned();
    }

    // --- UPDATED: Use the modern ndarray method ---
    let shape = &[1, 1, config.n_mels, required_width];
    let tensor = Tensor::from_shape(shape, final_spectrogram.as_slice().unwrap())?;
    Ok(tensor)
}

// ... (rest of the file is the same) ...

// Helper function to build the filterbank, extracted from MfccExtractor::new
fn build_mel_filterbank(config: &AppConfig) -> Array2<f32> {
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
    }
    let min_mel = hz_to_mel(0.0);
    let max_mel = hz_to_mel(config.sample_rate as f32 / 2.0);
    let mel_points = Array::linspace(min_mel, max_mel, config.n_mels + 2);
    let hz_points = mel_points.mapv(mel_to_hz);
    let fft_bins = (hz_points * (config.n_fft as f32 / config.sample_rate as f32))
        .mapv(|x| x.floor() as usize);
    let mut filters = Array2::<f32>::zeros((config.n_mels, config.n_fft / 2 + 1));
    for m in 0..config.n_mels {
        let (start, center, end) = (fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]);
        for k in start..center.min(config.n_fft / 2 + 1) {
            if center > start {
                filters[[m, k]] = (k - start) as f32 / (center - start) as f32;
            }
        }
        for k in center..end.min(config.n_fft / 2 + 1) {
            if end > center {
                filters[[m, k]] = (end - k) as f32 / (end - center) as f32;
            }
        }
    }
    filters
}

//================================================
// 3. Feature Extraction for the GMM Model
//================================================
// ... (The MfccExtractor struct and compute_mfcc_tensor function remain here, unchanged) ...

//================================================
// 4. CPAL Audio Input Handling
//================================================
// ... (The build_input_stream and push_samples functions remain here, unchanged) ...
//================================================
// 3. Feature Extraction for the GMM Model
//================================================
// ... (The MfccExtractor struct and compute_mfcc_tensor function remain here, unchanged) ...
struct MfccExtractor {
    config: AppConfig,
    mel_filterbank: Array2<f32>,
}
impl MfccExtractor {
    fn new(config: &AppConfig) -> Self {
        fn hz_to_mel(hz: f32) -> f32 {
            2595.0 * (1.0 + hz / 700.0).log10()
        }
        fn mel_to_hz(mel: f32) -> f32 {
            700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
        }
        let min_mel = hz_to_mel(0.0);
        let max_mel = hz_to_mel(config.sample_rate as f32 / 2.0);
        let mel_points = Array::linspace(min_mel, max_mel, config.n_mels + 2);
        let hz_points = mel_points.mapv(mel_to_hz);
        let fft_bins = (hz_points * (config.n_fft as f32 / config.sample_rate as f32))
            .mapv(|x| x.floor() as usize);
        let mut filters = Array2::<f32>::zeros((config.n_mels, config.n_fft / 2 + 1));
        for m in 0..config.n_mels {
            let (start, center, end) = (fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]);
            for k in start..center.min(config.n_fft / 2 + 1) {
                if center > start {
                    filters[[m, k]] = (k - start) as f32 / (center - start) as f32;
                }
            }
            for k in center..end.min(config.n_fft / 2 + 1) {
                if end > center {
                    filters[[m, k]] = (end - k) as f32 / (end - center) as f32;
                }
            }
        }
        Self {
            config: config.clone(),
            mel_filterbank: filters,
        }
    }
    fn extract(&self, frame: &[f32]) -> Vec<f32> {
        let energy: f32 = frame.iter().map(|&s| s * s).sum();
        if energy < 1e-10 {
            return vec![0.0; self.config.n_mfcc];
        }
        let window: Vec<f32> = apodize::hanning_iter(frame.len())
            .map(|f| f as f32)
            .collect();
        let windowed_frame: Vec<f32> = frame
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();
        let mut padded = vec![0.0; self.config.n_fft];
        padded[..windowed_frame.len()].copy_from_slice(&windowed_frame);
        let r2c = RealFftPlanner::<f32>::new().plan_fft_forward(self.config.n_fft);
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut padded, &mut spectrum).unwrap();
        let power_spec: Array1<f32> = spectrum[..self.config.n_fft / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();
        let mel_spec = self.mel_filterbank.dot(&power_spec);
        let log_mel_spec = mel_spec.mapv(|v| 10.0 * v.max(1e-10).log10());
        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(self.config.n_mels);
        let mut mfccs = log_mel_spec.to_vec();
        dct.process_dct2(&mut mfccs);
        mfccs.truncate(self.config.n_mfcc);
        mfccs
    }
}
fn compute_mfcc_tensor(audio_frame: &[f32], config: &AppConfig) -> Result<Tensor> {
    let extractor = MfccExtractor::new(config);
    let mfccs = extractor.extract(audio_frame);
    let shape = &[1, config.n_mfcc];
    let tensor = Tensor::from_shape(shape, &mfccs)?;
    Ok(tensor)
}

//================================================
// 4. CPAL Audio Input Handling
//================================================
// ... (The build_input_stream and push_samples functions remain here, unchanged) ...
// src/audio.rs

pub fn build_input_stream(
    device: &cpal::Device,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    target_sr: u32,
) -> cpal::Stream {
    let config = device.default_input_config().expect("No input config");
    let input_sr = config.sample_rate().0;
    let channels = config.channels() as usize;

    println!(
        "🎧 Audio config: {} Hz, {} channels, format: {:?}",
        input_sr,
        channels,
        config.sample_format()
    );
    let err_fn = |err| eprintln!("❌ Stream error: {}", err);

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _: &_| {
                    push_samples(data, channels, input_sr, target_sr, &audio_buffer);
                },
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[i16], _: &_| {
                    let float_data: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    push_samples(&float_data, channels, input_sr, target_sr, &audio_buffer);
                },
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::U16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[u16], _: &_| {
                    let float_data: Vec<f32> = data
                        .iter()
                        .map(|&s| (s as f32 - 32768.0) / 32768.0)
                        .collect();
                    push_samples(&float_data, channels, input_sr, target_sr, &audio_buffer);
                },
                err_fn,
                None,
            )
            .unwrap(),
        _ => panic!("Unsupported sample format"),
    };
    stream
}

fn push_samples(
    data: &[f32],
    channels: usize,
    input_sr: u32,
    target_sr: u32,
    buffer: &Arc<Mutex<Vec<f32>>>,
) {
    let samples_count = data.len() / channels;
    let total_samples =
        SAMPLES_RECEIVED.fetch_add(samples_count, Ordering::Relaxed) + samples_count;
    if total_samples % 4096 < samples_count {
        print!(
            "\r🎤 Audio stream is active. Samples received: {}...",
            total_samples
        );
        std::io::stdout().flush().unwrap();
    }
    let mut buf = buffer.lock().unwrap();
    if input_sr == target_sr {
        let mono_samples = data
            .chunks_exact(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / channels as f32);
        buf.extend(mono_samples);
    } else {
        let step = input_sr as f32 / target_sr as f32;
        let mut i: f32 = 0.0;
        while (i.floor() as usize) * channels < data.len() {
            let index = (i.floor() as usize) * channels;
            let mut sample_sum = 0.0;
            for c in 0..channels {
                sample_sum += data.get(index + c).copied().unwrap_or(0.0);
            }
            let mono_sample = sample_sum / channels as f32;
            buf.push(mono_sample);
            i += step;
        }
    }
}
