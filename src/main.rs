use byteorder::{LittleEndian, ReadBytesExt};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array, Array1, Array2};
use realfft::RealFftPlanner;
use rustdct::DctPlanner;
use serde::Deserialize;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::{self, Cursor, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ===================================================================
// == Configuration Structs (for loading dhvani_config.toml) ==
// ===================================================================

#[derive(Deserialize, Debug)]
struct Config {
    active_profile: String,
    profiles: Profiles,
    features: Features,
}

#[derive(Deserialize, Debug)]
struct Profiles {
    standard: Profile,
    high_detail: Profile,
    ultra_detail: Profile,
}

#[derive(Deserialize, Debug, Clone)]
struct Profile {
    pattern_size_kb: u64,
    n_mfcc: usize,
    gmm_components: usize,
}

#[derive(Deserialize, Debug)]
struct Features {
    sample_rate: u32,
    frame_length_samples: usize,
    hop_length_samples: usize,
    n_fft: usize,
}

// This struct holds the FINAL, ACTIVE configuration for easy passing to functions.
#[derive(Debug, Clone)]
struct AppConfig {
    pattern_size_bytes: u64,
    sample_rate: u32,
    frame_length_samples: usize,
    hop_length_samples: usize,
    n_fft: usize,
    n_mfcc: usize,
    n_mfcc_used: usize, // Determines which MFCCs to use for scoring
    n_mels: usize,      // Based on librosa's default, should match training
    gmm_components: usize,
}

// ===================================================================
// == Application Logic ==
// ===================================================================

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

fn parse_dbp_file(
    bytes: &[u8],
    phoneme: String,
    config: &AppConfig,
) -> Result<MasterPattern, std::io::Error> {
    let mut cursor = Cursor::new(bytes);
    fn read_f32_vec(c: &mut Cursor<&[u8]>, l: usize) -> std::io::Result<Vec<f32>> {
        let mut v = Vec::with_capacity(l);
        for _ in 0..l {
            v.push(c.read_f32::<LittleEndian>()?);
        }
        Ok(v)
    }

    let gmm_data_start = (config.pattern_size_bytes as f64 * 0.1) as u64;
    cursor.set_position(gmm_data_start);

    let gmm_weights_vec = read_f32_vec(&mut cursor, config.gmm_components)?;
    let gmm_means_vec = read_f32_vec(&mut cursor, config.gmm_components * config.n_mfcc)?;
    let gmm_covariances_vec = read_f32_vec(
        &mut cursor,
        config.gmm_components * config.n_mfcc * config.n_mfcc,
    )?;

    let metadata_start = (config.pattern_size_bytes as f64 * 0.8) as u64;
    cursor.set_position(metadata_start);

    let min_duration = cursor.read_f32::<LittleEndian>()?;
    let max_duration = cursor.read_f32::<LittleEndian>()?;
    let mfcc_mean_vec = read_f32_vec(&mut cursor, config.n_mfcc)?;
    let mfcc_std_vec = read_f32_vec(&mut cursor, config.n_mfcc)?;

    let gmm_weights = DVector::from_vec(gmm_weights_vec.iter().map(|&x| x as f64).collect());
    let mfcc_mean = DVector::from_vec(mfcc_mean_vec.iter().map(|&x| x as f64).collect());
    let mut mfcc_std = DVector::from_vec(mfcc_std_vec.iter().map(|&x| x as f64).collect());
    mfcc_std.add_scalar_mut(1e-8); // Regularization

    let mut gmm_means = Vec::new();
    for i in 0..config.gmm_components {
        gmm_means.push(
            DVector::from_row_slice(&gmm_means_vec[i * config.n_mfcc..(i + 1) * config.n_mfcc])
                .map(|x| x as f64),
        );
    }

    let mut gmm_cov_invs = Vec::new();
    let mut gmm_cov_dets = Vec::new();
    for i in 0..config.gmm_components {
        let cov_full = DMatrix::from_row_slice(
            config.n_mfcc,
            config.n_mfcc,
            &gmm_covariances_vec
                [i * config.n_mfcc * config.n_mfcc..(i + 1) * config.n_mfcc * config.n_mfcc],
        )
        .map(|x| x as f64);

        let start_coeff = config.n_mfcc - config.n_mfcc_used;
        let cov_sliced = cov_full
            .view(
                (start_coeff, start_coeff),
                (config.n_mfcc_used, config.n_mfcc_used),
            )
            .clone_owned();

        let mut cov_regularized = cov_sliced.clone();
        for j in 0..config.n_mfcc_used {
            cov_regularized[(j, j)] += 1e-6;
        }

        gmm_cov_dets.push(cov_regularized.determinant());
        gmm_cov_invs.push(cov_regularized.try_inverse().unwrap_or_else(|| {
            eprintln!(
                "Warning: Could not invert covariance matrix for component {}",
                i
            );
            DMatrix::identity(config.n_mfcc_used, config.n_mfcc_used)
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

fn score_gmm(mfccs_used: &DVector<f64>, pattern: &MasterPattern, config: &AppConfig) -> f64 {
    let mut weighted_probs = 0.0;
    for i in 0..config.gmm_components {
        let weight = pattern.gmm_weights[i];

        let start_coeff = config.n_mfcc - config.n_mfcc_used;
        let mean_used = pattern.gmm_means[i].rows(start_coeff, config.n_mfcc_used);

        let cov_inv = &pattern.gmm_cov_invs[i];
        let det = pattern.gmm_cov_dets[i];

        if det > 1e-10 && weight > 1e-10 {
            let diff = mfccs_used - mean_used;
            let exponent = -0.5 * (diff.transpose() * cov_inv * &diff).x;

            if exponent > -100.0 {
                let prob_density = (1.0
                    / ((2.0 * PI).powi(config.n_mfcc_used as i32 / 2) * det).sqrt())
                    * exponent.exp();
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
    config: AppConfig,
}

impl MatcherStateMachine {
    fn new(pattern: &MasterPattern, config: &AppConfig) -> Self {
        // Calculate frames per second to convert duration from seconds to frames
        let frames_per_second = config.sample_rate as f32 / config.hop_length_samples as f32;
        Self {
            state: State::Idle,
            phoneme: pattern.phoneme.clone(),
            start_frame: 0,
            mismatch_counter: 0,
            max_mismatch_frames: 5, // 5 frames of silence/mismatch allowed
            min_duration_frames: (pattern.min_duration * frames_per_second) as u32,
            max_duration_frames: (pattern.max_duration * frames_per_second) as u32,
            config: config.clone(),
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
            let seconds_per_frame =
                self.config.hop_length_samples as f32 / self.config.sample_rate as f32;
            let start_time = self.start_frame as f32 * seconds_per_frame;
            let end_time = end_frame as f32 * seconds_per_frame;
            println!(
                "\n✅ DETECTED: '{}' (start: {:.2}s, end: {:.2}s)",
                self.phoneme, start_time, end_time
            );
        }
    }
}

static SAMPLES_RECEIVED: AtomicUsize = AtomicUsize::new(0);

fn push_samples(
    data: &[f32],
    channels: usize,
    input_sr: u32,
    target_sr: u32,
    buffer: &Arc<Mutex<Vec<f32>>>,
) {
    let samples_count = data.len() / channels;
    SAMPLES_RECEIVED.fetch_add(samples_count, Ordering::Relaxed);

    let step = input_sr as f32 / target_sr as f32;
    let mut i: f32 = 0.0;
    let mut buf = buffer.lock().unwrap();

    while (i.floor() as usize) * channels < data.len() {
        let index = (i.floor() as usize) * channels;
        let mut sample = 0.0;
        if index < data.len() {
            if channels > 1 {
                let mut sum = 0.0;
                for c in 0..channels {
                    sum += data.get(index + c).copied().unwrap_or(0.0);
                }
                sample = sum / channels as f32;
            } else {
                sample = data[index];
            }
        }
        buf.push(sample);
        i += step;
    }
}

fn build_input_stream(
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

    match config.sample_format() {
        cpal::SampleFormat::F32 => device
            .build_input_stream(
                &config.into(),
                move |data: &[f32], _| {
                    push_samples(data, channels, input_sr, target_sr, &audio_buffer);
                },
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::I16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let float_data: Vec<f32> =
                        data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                    push_samples(&float_data, channels, input_sr, target_sr, &audio_buffer);
                },
                err_fn,
                None,
            )
            .unwrap(),
        cpal::SampleFormat::U16 => device
            .build_input_stream(
                &config.into(),
                move |data: &[u16], _| {
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
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting Dhvani Real-Time Engine...");

    let config_str = fs::read_to_string("config/dhvani_config.toml")?;
    let config: Config = toml::from_str(&config_str)?;

    let active_profile = match config.active_profile.as_str() {
        "standard" => config.profiles.standard,
        "high_detail" => config.profiles.high_detail,
        "ultra_detail" => config.profiles.ultra_detail,
        _ => panic!(
            "Invalid active_profile '{}' in config file!",
            config.active_profile
        ),
    };

    println!("✅ Running with profile: '{}'", config.active_profile);

    let app_config = AppConfig {
        pattern_size_bytes: active_profile.pattern_size_kb * 1024,
        sample_rate: config.features.sample_rate,
        frame_length_samples: config.features.frame_length_samples,
        hop_length_samples: config.features.hop_length_samples,
        n_fft: config.features.n_fft,
        n_mfcc: active_profile.n_mfcc,
        gmm_components: active_profile.gmm_components,
        n_mfcc_used: active_profile.n_mfcc - 1,
        n_mels: 128,
    };

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
                let file_bytes = fs::read(&path)?;
                match parse_dbp_file(&file_bytes, phoneme.clone(), &app_config) {
                    Ok(pattern) => loaded_patterns.push(pattern),
                    Err(e) => eprintln!("Failed to load pattern {}: {}", phoneme, e),
                }
            }
        }
        println!("✅ {} patterns loaded successfully.", loaded_patterns.len());
    }

    let mut state_machines: Vec<MatcherStateMachine> = {
        let patterns_guard = patterns.lock().unwrap();
        patterns_guard
            .iter()
            .map(|p| MatcherStateMachine::new(p, &app_config))
            .collect()
    };

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    println!(
        "🎤 Using input device: {}",
        device.name().unwrap_or_else(|_| "Unknown".to_string())
    );

    let stream = build_input_stream(&device, audio_buffer.clone(), app_config.sample_rate);
    stream.play().expect("Failed to play stream");

    println!("\n🔊 Listening... (Press Ctrl+C to stop)");

    let extractor = MfccExtractor::new(&app_config);
    let mut frame_counter: u64 = 0;

    loop {
        let mut buffer = audio_buffer.lock().unwrap();
        while buffer.len() >= app_config.frame_length_samples {
            let frame: Vec<f32> = buffer[..app_config.frame_length_samples].to_vec();
            let frame_energy: f32 = frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32;

            let mfccs = extractor.extract(&frame);
            let mfccs_dvec = DVector::from_vec(mfccs.iter().map(|&x| x as f64).collect());

            let patterns_guard = patterns.lock().unwrap();
            for (i, pattern) in patterns_guard.iter().enumerate() {
                let detection_threshold = -90.2; // May need tuning

                // Inside the main loop, around line 550
                let mut normalized_mfccs =
                    (&mfccs_dvec - &pattern.mfcc_mean).component_div(&pattern.mfcc_std);

                // --- ADD THIS ONE LINE TO TEST ---
                normalized_mfccs[0] = 0.0; // Manually ignore the first coefficient (C0) for this test.

                let start_coeff = app_config.n_mfcc - app_config.n_mfcc_used;
                let mfccs_used = normalized_mfccs
                    .rows(start_coeff, app_config.n_mfcc_used)
                    .clone_owned();

                // --- ADD THIS DEBUG BLOCK ---
                if pattern.phoneme == "अ" && frame_energy > 0.05 {
                    // Only print when there's significant sound
                    println!("\n--- Normalized MFCCs for 'अ' ---");
                    println!(
                        "Vector (first 13): {:.2}",
                        mfccs_used.rows(0, 13).transpose()
                    );
                }
                // --- END DEBUG BLOCK ---

                let log_prob = score_gmm(&mfccs_used, pattern, &app_config);

                if pattern.phoneme == "अ" {
                    print!(
                        "\rLive score for 'अ': {:.2} | Energy: {:.6}      ",
                        log_prob, frame_energy
                    );
                    io::stdout().flush().unwrap();
                }

                state_machines[i].update(log_prob, detection_threshold, frame_counter);
            }

            buffer.drain(..app_config.hop_length_samples);
            frame_counter += 1;
        }
        drop(buffer);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
}
