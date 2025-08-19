// src/main.rs

use byteorder::{LittleEndian, ReadBytesExt};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Array2};
use realfft::RealFftPlanner;
use rustdct::DctPlanner;
use std::f64::consts::PI;
use std::fs;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::LazyLock;
use std::sync::{Arc, Mutex};

// --- Constants ---
const TARGET_SR: u32 = 16000;
const FRAME_LENGTH_MS: u32 = 25;
const FRAME_STRIDE_MS: u32 = 10;
const FRAME_LENGTH_SAMPLES: usize = (TARGET_SR as f32 * (FRAME_LENGTH_MS as f32 / 1000.0)) as usize; // 400
const FRAME_STRIDE_SAMPLES: usize = (TARGET_SR as f32 * (FRAME_STRIDE_MS as f32 / 1000.0)) as usize; // 160
const N_MFCC: usize = 13;
const N_MFCC_USED: usize = 12;
const N_FFT: usize = 512; // Changed to next power of 2 for better FFT performance
const N_MELS: usize = 40;

#[derive(Debug, Clone)]
struct MasterPattern {
    phoneme: String,
    gmm_weights: DVector<f64>,
    gmm_means: Vec<DVector<f64>>,
    gmm_cov_invs: Vec<DMatrix<f64>>,
    gmm_cov_dets: Vec<f64>,
    mfcc_mean: DVector<f64>,
    mfcc_std: DVector<f64>,
    min_duration: f32,
    max_duration: f32,
}

fn parse_dbp_file(bytes: &[u8], phoneme: String) -> Result<MasterPattern, std::io::Error> {
    let mut cursor = Cursor::new(bytes);
    fn read_f32_vec(c: &mut Cursor<&[u8]>, l: usize) -> std::io::Result<Vec<f32>> {
        let mut v = Vec::with_capacity(l);
        for _ in 0..l {
            v.push(c.read_f32::<LittleEndian>()?);
        }
        Ok(v)
    }

    cursor.set_position(10240);

    let gmm_weights_vec = read_f32_vec(&mut cursor, 5)?;
    let gmm_means_vec = read_f32_vec(&mut cursor, 5 * N_MFCC)?;
    let gmm_covariances_vec = read_f32_vec(&mut cursor, 5 * N_MFCC * N_MFCC)?;

    cursor.set_position(81920);
    let min_duration = cursor.read_f32::<LittleEndian>()?;
    let max_duration = cursor.read_f32::<LittleEndian>()?;
    let mfcc_mean_vec = read_f32_vec(&mut cursor, N_MFCC)?;
    let mfcc_std_vec = read_f32_vec(&mut cursor, N_MFCC)?;

    let gmm_weights = DVector::from_vec(gmm_weights_vec.iter().map(|&x| x as f64).collect());
    let mfcc_mean = DVector::from_vec(mfcc_mean_vec.iter().map(|&x| x as f64).collect());
    let mut mfcc_std = DVector::from_vec(mfcc_std_vec.iter().map(|&x| x as f64).collect());
    mfcc_std.add_scalar_mut(1e-8);

    let mut gmm_means = Vec::new();
    for i in 0..5 {
        gmm_means.push(
            DVector::from_row_slice(&gmm_means_vec[i * N_MFCC..(i + 1) * N_MFCC]).map(|x| x as f64),
        );
    }

    let mut gmm_cov_invs = Vec::new();
    let mut gmm_cov_dets = Vec::new();
    for i in 0..5 {
        let cov_full = DMatrix::from_row_slice(
            N_MFCC,
            N_MFCC,
            &gmm_covariances_vec[i * N_MFCC * N_MFCC..(i + 1) * N_MFCC * N_MFCC],
        )
        .map(|x| x as f64);
        let cov_sliced = cov_full
            .view((1, 1), (N_MFCC_USED, N_MFCC_USED))
            .clone_owned();

        // Add regularization to avoid singular matrices
        let mut cov_regularized = cov_sliced.clone();
        for i in 0..N_MFCC_USED {
            cov_regularized[(i, i)] += 1e-6;
        }

        gmm_cov_dets.push(cov_regularized.determinant());
        gmm_cov_invs.push(cov_regularized.try_inverse().unwrap_or_else(|| {
            eprintln!(
                "Warning: Could not invert covariance matrix for component {}",
                i
            );
            DMatrix::identity(N_MFCC_USED, N_MFCC_USED)
        }));
    }

    Ok(MasterPattern {
        phoneme,
        gmm_weights,
        gmm_means,
        gmm_cov_invs,
        gmm_cov_dets,
        mfcc_mean,
        mfcc_std,
        min_duration,
        max_duration,
    })
}

struct MfccExtractor {
    mel_filterbank: Array2<f32>,
}

impl MfccExtractor {
    fn new() -> Self {
        fn hz_to_mel(hz: f32) -> f32 {
            2595.0 * (1.0 + hz / 700.0).log10()
        }
        fn mel_to_hz(mel: f32) -> f32 {
            700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
        }
        let min_mel = hz_to_mel(0.0);
        let max_mel = hz_to_mel(TARGET_SR as f32 / 2.0);
        let mel_points = Array::linspace(min_mel, max_mel, N_MELS + 2);
        let hz_points = mel_points.mapv(mel_to_hz);
        let fft_bins = (hz_points * (N_FFT as f32 / TARGET_SR as f32)).mapv(|x| x.floor() as usize);
        let mut filters = Array2::<f32>::zeros((N_MELS, N_FFT / 2 + 1));
        for m in 0..N_MELS {
            let (start, center, end) = (fft_bins[m], fft_bins[m + 1], fft_bins[m + 2]);
            for k in start..center.min(N_FFT / 2 + 1) {
                if center > start {
                    filters[[m, k]] = (k - start) as f32 / (center - start) as f32;
                }
            }
            for k in center..end.min(N_FFT / 2 + 1) {
                if end > center {
                    filters[[m, k]] = (end - k) as f32 / (end - center) as f32;
                }
            }
        }
        Self {
            mel_filterbank: filters,
        }
    }

    fn extract(&self, frame: &[f32]) -> Vec<f32> {
        // Check if we have silence
        let energy: f32 = frame.iter().map(|&s| s * s).sum();
        if energy < 1e-10 {
            return vec![0.0; N_MFCC];
        }

        let window: Vec<f32> = apodize::hanning_iter(frame.len())
            .map(|f| f as f32)
            .collect();
        let windowed_frame: Vec<f32> = frame
            .iter()
            .zip(window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Pad to N_FFT size
        let mut padded = vec![0.0; N_FFT];
        padded[..windowed_frame.len()].copy_from_slice(&windowed_frame);

        let r2c = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let mut spectrum = r2c.make_output_vec();
        let mut real_input = padded;
        r2c.process(&mut real_input, &mut spectrum).unwrap();

        let power_spec: Array1<f32> = spectrum[..N_FFT / 2 + 1]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();
        let mel_spec = self.mel_filterbank.dot(&power_spec);
        let log_mel_spec = mel_spec.mapv(|v| 10.0 * v.max(1e-10).log10());

        let mut planner = DctPlanner::new();
        let dct = planner.plan_dct2(N_MELS);
        let mut mfccs = log_mel_spec.to_vec();
        dct.process_dct2(&mut mfccs);
        mfccs.truncate(N_MFCC);
        mfccs
    }
}

fn score_gmm(mfccs_12d: &DVector<f64>, pattern: &MasterPattern) -> f64 {
    let mut weighted_probs = 0.0;
    for i in 0..5 {
        let weight = pattern.gmm_weights[i];
        let mean_12d = pattern.gmm_means[i].rows(1, N_MFCC_USED);
        let cov_inv = &pattern.gmm_cov_invs[i];
        let det = pattern.gmm_cov_dets[i];

        if det > 1e-10 && weight > 1e-10 {
            let diff = mfccs_12d - mean_12d;
            let exponent = -0.5 * (diff.transpose() * cov_inv * &diff).x;

            // Avoid numerical underflow
            if exponent > -100.0 {
                let prob_density =
                    (1.0 / ((2.0 * PI).powi(N_MFCC_USED as i32 / 2) * det).sqrt()) * exponent.exp();
                weighted_probs += weight * prob_density;
            }
        }
    }

    if weighted_probs > 1e-30 {
        weighted_probs.ln()
    } else {
        -45.0
    }
}

#[derive(PartialEq, Debug)]
enum State {
    Idle,
    Matching,
    Mismatch,
}

struct MatcherStateMachine {
    state: State,
    phoneme: String,
    start_frame: u64,
    mismatch_counter: u32,
    max_mismatch_frames: u32,
    min_duration_frames: u32,
    max_duration_frames: u32,
}

impl MatcherStateMachine {
    fn new(pattern: &MasterPattern) -> Self {
        Self {
            state: State::Idle,
            phoneme: pattern.phoneme.clone(),
            start_frame: 0,
            mismatch_counter: 0,
            max_mismatch_frames: 5,
            min_duration_frames: (pattern.min_duration / (FRAME_STRIDE_MS as f32 / 1000.0)) as u32,
            max_duration_frames: (pattern.max_duration / (FRAME_STRIDE_MS as f32 / 1000.0)) as u32,
        }
    }

    fn update(&mut self, log_prob: f64, threshold: f64, current_frame: u64) {
        match self.state {
            State::Idle => {
                if log_prob > threshold {
                    self.state = State::Matching;
                    self.start_frame = current_frame;
                    self.mismatch_counter = 0;
                }
            }
            State::Matching => {
                if log_prob < threshold {
                    self.state = State::Mismatch;
                    self.mismatch_counter = 1;
                }
            }
            State::Mismatch => {
                if log_prob > threshold {
                    self.state = State::Matching;
                    self.mismatch_counter = 0;
                } else {
                    self.mismatch_counter += 1;
                    if self.mismatch_counter > self.max_mismatch_frames {
                        self.fire_event(current_frame);
                        self.state = State::Idle;
                    }
                }
            }
        }
    }

    fn fire_event(&self, current_frame: u64) {
        let end_frame = current_frame - self.mismatch_counter as u64;
        let duration_frames = end_frame - self.start_frame;
        if duration_frames >= self.min_duration_frames as u64
            && duration_frames <= self.max_duration_frames as u64
        {
            let start_time = self.start_frame as f32 * (FRAME_STRIDE_MS as f32 / 1000.0);
            let end_time = end_frame as f32 * (FRAME_STRIDE_MS as f32 / 1000.0);
            println!(
                "\n✅ DETECTED: '{}' (start: {:.2}s, end: {:.2}s)",
                self.phoneme, start_time, end_time
            );
        }
    }
}

// Global counter for debugging
static SAMPLES_RECEIVED: AtomicUsize = AtomicUsize::new(0);
static LAST_PRINT_TIME: LazyLock<Mutex<std::time::Instant>> =
    LazyLock::new(|| Mutex::new(std::time::Instant::now()));

fn push_samples(
    data: &[f32],
    channels: usize,
    input_sr: u32,
    target_sr: u32,
    buffer: &Arc<Mutex<Vec<f32>>>,
) {
    // Debug: Track samples received
    let samples_count = data.len() / channels;
    SAMPLES_RECEIVED.fetch_add(samples_count, Ordering::Relaxed);

    // Print debug info every second
    {
        let mut last_print = LAST_PRINT_TIME.lock().unwrap();
        if last_print.elapsed().as_secs() >= 1 {
            let total_samples = SAMPLES_RECEIVED.load(Ordering::Relaxed);
            println!(
                "\n[DEBUG] Total samples received: {}, Buffer size: {}",
                total_samples,
                buffer.lock().unwrap().len()
            );
            *last_print = std::time::Instant::now();
        }
    }

    let step = input_sr as f32 / target_sr as f32;
    let mut i: f32 = 0.0;
    let mut buf = buffer.lock().unwrap();

    while (i.floor() as usize) * channels < data.len() {
        let index = (i.floor() as usize) * channels;
        let mut sample = data[index];

        // Mix to mono if multi-channel
        if channels > 1 && index + channels <= data.len() {
            let mut sum = 0.0;
            for c in 0..channels {
                sum += data[index + c];
            }
            sample = sum / channels as f32;
        }

        buf.push(sample);
        i += step;
    }
}

fn build_input_stream(device: &cpal::Device, audio_buffer: Arc<Mutex<Vec<f32>>>) -> cpal::Stream {
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

    match config.sample_format() {
        cpal::SampleFormat::F32 => {
            println!("Using F32 format");
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[f32], _| {
                        let non_zero = data.iter().any(|&s| s.abs() > 1e-10);
                        push_samples(data, channels, input_sr, TARGET_SR, &audio_buffer);
                        if non_zero {
                            push_samples(data, channels, input_sr, TARGET_SR, &audio_buffer);
                        }
                    },
                    err_fn,
                    None,
                )
                .unwrap()
        }
        cpal::SampleFormat::I16 => {
            println!("Using I16 format");
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[i16], _| {
                        let float_data: Vec<f32> =
                            data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                        push_samples(&float_data, channels, input_sr, TARGET_SR, &audio_buffer);
                    },
                    err_fn,
                    None,
                )
                .unwrap()
        }
        cpal::SampleFormat::U16 => {
            println!("Using U16 format");
            device
                .build_input_stream(
                    &config.into(),
                    move |data: &[u16], _| {
                        let float_data: Vec<f32> = data
                            .iter()
                            .map(|&s| (s as f32 - 32768.0) / 32768.0)
                            .collect();
                        push_samples(&float_data, channels, input_sr, TARGET_SR, &audio_buffer);
                    },
                    err_fn,
                    None,
                )
                .unwrap()
        }
        _ => panic!("Unsupported sample format"),
    }
}

fn main() {
    println!("Starting Dhvani Real-Time Engine...");

    // Initialize last print time
    *LAST_PRINT_TIME.lock().unwrap() = std::time::Instant::now();

    let patterns = Arc::new(Mutex::new(Vec::<MasterPattern>::new()));
    {
        let pattern_files =
            fs::read_dir("data/05_patterns/").expect("Could not read patterns directory");
        let mut loaded_patterns = patterns.lock().unwrap();
        for entry in pattern_files {
            let path = entry.unwrap().path();
            if path.extension().map_or(false, |s| s == "dbp") {
                let phoneme = path.file_stem().unwrap().to_str().unwrap().to_string();
                println!("Loading pattern: {}", phoneme);
                let file_bytes = fs::read(&path).unwrap();
                match parse_dbp_file(&file_bytes, phoneme.clone()) {
                    Ok(pattern) => {
                        loaded_patterns.push(pattern);
                    }
                    Err(e) => {
                        eprintln!("Failed to load pattern {}: {}", phoneme, e);
                    }
                }
            }
        }
        println!("✅ {} patterns loaded successfully.", loaded_patterns.len());
    }

    let mut state_machines: Vec<MatcherStateMachine> = patterns
        .lock()
        .unwrap()
        .iter()
        .map(MatcherStateMachine::new)
        .collect();

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));

    // Setup audio input
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");

    println!(
        "🎤 Using input device: {}",
        device.name().unwrap_or_else(|_| "Unknown".to_string())
    );

    // List all available devices for debugging
    println!("\n📋 Available input devices:");
    for dev in host.input_devices().unwrap() {
        println!(
            "  - {}",
            dev.name().unwrap_or_else(|_| "Unknown".to_string())
        );
    }

    let stream = build_input_stream(&device, audio_buffer.clone());
    stream.play().expect("Failed to play stream");

    println!("\n🔊 Listening... (Press Ctrl+C to stop)");
    println!("Make sure your microphone is enabled and not muted!");
    println!("Try speaking loudly and clearly.\n");

    let extractor = MfccExtractor::new();
    let mut frame_counter: u64 = 0;
    let mut last_debug_time = std::time::Instant::now();

    loop {
        let mut buffer = audio_buffer.lock().unwrap();

        // Debug buffer status every 2 seconds
        if last_debug_time.elapsed().as_secs() >= 2 {
            println!(
                "\n[MAIN] Buffer size: {} samples, Frames processed: {}",
                buffer.len(),
                frame_counter
            );
            last_debug_time = std::time::Instant::now();
        }

        while buffer.len() >= FRAME_LENGTH_SAMPLES {
            let frame: Vec<f32> = buffer[..FRAME_LENGTH_SAMPLES].to_vec();

            // Check frame energy
            let frame_energy: f32 = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;

            let mfccs = extractor.extract(&frame);
            let mfccs_dvec = DVector::from_vec(mfccs.iter().map(|&x| x as f64).collect());

            let patterns_guard = patterns.lock().unwrap();
            for (i, pattern) in patterns_guard.iter().enumerate() {
                let detection_threshold = -28.0;
                let normalized_mfccs =
                    (&mfccs_dvec - &pattern.mfcc_mean).component_div(&pattern.mfcc_std);
                let mfccs_12d = normalized_mfccs.rows(1, N_MFCC_USED).clone_owned();
                let log_prob = score_gmm(&mfccs_12d, pattern);

                if pattern.phoneme == "अ" {
                    print!(
                        "\rLive score for 'अ': {:.2} | Energy: {:.6}      ",
                        log_prob, frame_energy
                    );
                    use std::io::{self, Write};
                    io::stdout().flush().unwrap();
                }

                state_machines[i].update(log_prob, detection_threshold, frame_counter);
            }

            buffer.drain(..FRAME_STRIDE_SAMPLES);
            frame_counter += 1;
        }
        drop(buffer);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
