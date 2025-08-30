// src/config.rs

use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

// ... (CliConfig and ModelType enums remain the same) ...
#[derive(Debug, Clone, ValueEnum)]
pub enum ModelType {
    Gmm,
    Onnx,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct CliConfig {
    #[arg(long, value_enum)]
    pub model_type: ModelType,
    #[arg(short, long)]
    pub model_path: PathBuf,
    #[arg(short, long, default_value_t = 0)]
    pub device_id: u32,
}

// --- Structs for deserializing the new TOML format ---
#[derive(Deserialize, Debug)]
struct TomlConfig {
    features: Features,
    gmm_settings: GmmSettings,
    onnx_settings: OnnxSettings,
}

#[derive(Deserialize, Debug)]
struct Features {
    sample_rate: u32,
    frame_length_samples: usize,
    hop_length_samples: usize,
    n_fft: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct GmmSettings {
    pattern_size_kb: u64,
    n_mfcc: usize,
    n_mfcc_used: usize,
    gmm_components: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct OnnxSettings {
    n_mels: usize,
    input_width: usize,
}

// --- The final AppConfig now holds parameters for ALL possible models ---
#[derive(Debug, Clone)]
pub struct AppConfig {
    // Shared features
    pub sample_rate: u32,
    pub frame_length_samples: usize,
    pub hop_length_samples: usize,
    pub n_fft: usize,

    // GMM-specific
    pub pattern_size_bytes: u64,
    pub n_mfcc: usize,
    pub n_mfcc_used: usize,
    pub gmm_components: usize,

    // ONNX-specific
    pub n_mels: usize,
    pub onnx_input_width: usize,
}

/// The main public function for this module.
pub fn load_config(toml_path: &str) -> Result<(CliConfig, AppConfig)> {
    let cli_config = CliConfig::parse();
    let config_str = fs::read_to_string(toml_path)?;
    let toml_config: TomlConfig = toml::from_str(&config_str)?;

    // Create the final AppConfig by combining values from the parsed TOML.
    let app_config = AppConfig {
        // Shared
        sample_rate: toml_config.features.sample_rate,
        frame_length_samples: toml_config.features.frame_length_samples,
        hop_length_samples: toml_config.features.hop_length_samples,
        n_fft: toml_config.features.n_fft,

        // From GMM block
        pattern_size_bytes: toml_config.gmm_settings.pattern_size_kb * 1024,
        n_mfcc: toml_config.gmm_settings.n_mfcc,
        n_mfcc_used: toml_config.gmm_settings.n_mfcc_used,
        gmm_components: toml_config.gmm_settings.gmm_components,

        // From ONNX block
        n_mels: toml_config.onnx_settings.n_mels,
        onnx_input_width: toml_config.onnx_settings.input_width,
    };

    Ok((cli_config, app_config))
}
