// src/main.rs

use std::fs;
use std::io::{Cursor, Read};
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};
use rustfft::{FftPlanner, num_complex::Complex, DctPlanner};
use ndarray::{Array, Array1, Array2};

// --- Constants ---
const TARGET_SR: u32 = 16000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_STRIDE_MS: u32 = 10;
const FRAME_LENGTH_SAMPLES: usize = (TARGET_SR as f32 * (FRAME_LENGTH_MS as f32 / 1000.0)) as usize; // 400
const FRAME_STRIDE_SAMPLES: usize = (TARGET_SR as f32 * (FRAME_STRIDE_MS as f32 / 1000.0)) as usize; // 160
const N_MFCC: usize = 13;
const N_FFT: usize = FRAME_LENGTH_SAMPLES;
const N_MELS: usize = 40;

#[derive(Debug)]
struct MasterPattern { /* ... same as before ... */
    phoneme: String,
    energy_template: Vec<f32>,
    centroid_template: Vec<f32>,
    zcr_template: Vec<u8>,
    gmm_weights: Vec<f32>,
    gmm_means: Vec<f32>,
    gmm_covariances: Vec<f32>,
    transition_matrix: Vec<f32>,
    min_duration: f32,
    max_duration: f32,
    mfcc_mean: Vec<f32>,
    mfcc_std: Vec<f32>,
}
#[derive(Debug)]
struct LiveFeatures { mfccs: Vec<f32> }

fn parse_dbp_file(bytes: &[u8], phoneme: String) -> Result<MasterPattern, std::io::Error> {
    // ... (this function is correct and remains the same)
    let mut cursor = Cursor::new(bytes);

    fn read_f32_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> std::io::Result<Vec<f32>> {
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len { vec.push(cursor.read_f32::<LittleEndian>()?); } Ok(vec)
    }
    fn read_f16_as_f32_vec(cursor: &mut Cursor<&[u8]>, len: usize) -> std::io::Result<Vec<f32>> {
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len { let bits = cursor.read_u16::<LittleEndian>()?; vec.push(f16::from_bits(bits).to_f32()); } Ok(vec)
    }
    let energy_template = read_f16_as_f32_vec(&mut cursor, 200)?;
    let centroid_template = read_f16_as_f32_vec(&mut cursor, 200)?;
    let mut zcr_template = vec![0u8; 200];
    cursor.read_exact(&mut zcr_template)?;
    cursor.set_position(10 * 1024);
    let gmm_weights = read_f32_vec(&mut cursor, 5)?;
    let gmm_means = read_f32_vec(&mut cursor, 5 * 13)?;
    let gmm_covariances = read_f32_vec(&mut cursor, 5 * 13 * 13)?;
    let transition_matrix = read_f16_as_f32_vec(&mut cursor, 50 * 50)?;
    cursor.set_position(80 * 1024);
    let min_duration = cursor.read_f32::<LittleEndian>()?;
    let max_duration = cursor.read_f32::<LittleEndian>()?;
    let mfcc_mean = read_f32_vec(&mut cursor, 13)?;
    let mfcc_std = read_f32_vec(&mut cursor, 13)?;
    Ok(MasterPattern {
        phoneme, energy_template, centroid_template, zcr_template, gmm_weights, gmm_means,
        gmm_covariances, transition_matrix, min_duration, max_duration, mfcc_mean, mfcc_std,
    })
}

// --- Self-contained MFCC implementation ---
struct MfccExtractor {
    mel_filterbank: Array2<f32>,
}

impl MfccExtractor {
    fn new() -> Self {
        fn hz_to_mel(hz: f32) -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() }
        fn mel_to_hz(mel: f32) -> f32 { 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0) }

        let min_mel = hz_to_mel(0.0);
        let max_mel = hz_to_mel(TARGET_SR as f32 / 2.0);
        let mel_points = Array::linspace(min_mel, max_mel, N_MELS + 2);
        let hz_points = mel_points.mapv(mel_to_hz);
        let fft_bins = (hz_points / (TARGET_SR as f32 / 2.0) * (N_FFT as f32 / 2.0)).mapv(|x| x.floor() as usize);

        let mut filters = Array2::<f32>::zeros((N_MELS, N_FFT / 2 + 1));
        for m in 0..N_MELS {
            let start = fft_bins[m];
            let center = fft_bins[m + 1];
            let end = fft_bins[m + 2];
            for k in start..center {
                if center != start { filters[[m, k]] = (k - start) as f32 / (center - start) as f32; }
            }
            for k in center..end {
                if end != center { filters[[m, k]] = (end - k) as f32 / (end - center) as f32; }
            }
        }
        Self { mel_filterbank: filters }
    }

    fn extract(&self, frame: &[f32]) -> Vec<f32> {
        let window = apodize::hanning_array(frame.len());
        let mut windowed_frame: Vec<f32> = frame.iter().zip(window.iter()).map(|(s, w)| s * w).collect();

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut buffer: Vec<Complex<f32>> = windowed_frame.iter().map(|&x| Complex::new(x, 0.0)).collect();
        buffer.resize(N_FFT, Complex::default());
        fft.process(&mut buffer);

        let power_spec: Array1<f32> = buffer[..N_FFT / 2 + 1].iter().map(|c| c.norm_sqr()).collect();
        let mel_spec = self.mel_filterbank.dot(&power_spec);
        let log_mel_spec = mel_spec.mapv(|v| if v > 1e-6 { v.ln() } else { 0.0 });

        let mut dct_planner = DctPlanner::new();
        let dct = dct_planner.plan_dct2(N_MELS);
        let mut mfccs = log_mel_spec.to_vec();
        dct.process_dct2(&mut mfccs);

        mfccs.truncate(N_MFCC);
        mfccs
    }
}

fn main() {
    println!("Starting Dhvani Real-Time Engine...");

    let patterns = Arc::new(Mutex::new(Vec::<MasterPattern>::new()));
    // ... (loading patterns is the same)
    let pattern_files = fs::read_dir("data/05_patterns/").expect("Could not read patterns directory");
    let mut loaded_patterns = patterns.lock().unwrap();
    for entry in pattern_files {
        let path = entry.unwrap().path();
        if path.extension().map_or(false, |s| s == "dbp") {
            let phoneme = path.file_stem().unwrap().to_str().unwrap().to_string();
            let file_bytes = fs::read(&path).unwrap();
            if let Ok(pattern) = parse_dbp_file(&file_bytes, phoneme) { loaded_patterns.push(pattern); }
        }
    }
    println!("✅ {} patterns loaded successfully.", loaded_patterns.len());
    drop(loaded_patterns);

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let host = cpal::default_host();
    let device = host.default_input_device().expect("No input device available");
    let config = device.default_input_config().expect("Failed to get default input config");
    let input_sr = config.sample_rate().0;
    let err_fn = |err| eprintln!("an error occurred on the audio stream: {}", err);
    let buffer_clone = audio_buffer.clone();

    let stream = device.build_input_stream(&config.into(), move |data: &[f32], _: &cpal::InputCallbackInfo| {
        let mut buffer = buffer_clone.lock().unwrap();
        let step = input_sr as f32 / TARGET_SR as f32;
        let mut i: f32 = 0.0; // Explicitly define type as f32
        while (i.floor() as usize) < data.len() {
            let index = i.floor() as usize;
            let mono_sample = (data[index] + data.get(index + 1).unwrap_or(&data[index])) / 2.0;
            buffer.push(mono_sample);
            i += step;
        }
    }, err_fn, None).expect("Failed to build input stream");

    stream.play().expect("Failed to play stream");
    println!("\nListening... (Press Ctrl+C to stop)");

    let extractor = MfccExtractor::new();

    loop {
        let mut buffer = audio_buffer.lock().unwrap();
        while buffer.len() >= FRAME_LENGTH_SAMPLES {
            let frame: Vec<f32> = buffer[..FRAME_LENGTH_SAMPLES].to_vec();

            let mfccs = extractor.extract(&frame);

            println!("Live fingerprint (first 3 MFCCs): [{:.2}, {:.2}, {:.2}]", mfccs[0], mfccs[1], mfccs[2]);

            // TODO: Final Step - Calculate probabilities and update state machines

            buffer.drain(..FRAME_STRIDE_SAMPLES);
        }
        drop(buffer);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
