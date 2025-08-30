// src/main.rs

mod audio;
mod config;
mod models;
mod state_machine;

use crate::models::gmm::GmmModel;
use crate::models::onnx::OnnxModel;
use crate::models::Model;
use anyhow::Result;
use cpal::traits::{HostTrait, StreamTrait};
use std::sync::{Arc, Mutex};

fn main() -> Result<()> {
    // ... (Initialization is the same) ...
    println!("🚀 Starting Dhvani Real-Time Engine...");
    let (cli_config, app_config) = config::load_config("config/dhvani_config.toml")?;

    let model: Box<dyn Model> = match cli_config.model_type {
        config::ModelType::Gmm => {
            println!("🧠 Initializing GMM model...");
            Box::new(GmmModel::new(&cli_config.model_path, &app_config)?)
        }
        config::ModelType::Onnx => {
            println!("🧠 Initializing ONNX model...");
            Box::new(OnnxModel::new(&cli_config.model_path, &app_config)?)
        }
    };

    let mut state_machines: Vec<state_machine::MatcherStateMachine> = Vec::new();
    if let config::ModelType::Gmm = cli_config.model_type {
        let gmm_patterns = models::gmm::load_patterns(&cli_config.model_path, &app_config)?;
        state_machines = gmm_patterns
            .iter()
            .map(|p| state_machine::MatcherStateMachine::new(p, &app_config))
            .collect();
    }

    let audio_buffer = Arc::new(Mutex::new(Vec::<f32>::new()));
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .expect("No input device available");
    let stream = audio::build_input_stream(&device, audio_buffer.clone(), app_config.sample_rate);
    stream.play().expect("Failed to play stream");

    println!("\n🔊 Listening... (Press Ctrl+C to stop)");
    let mut frame_counter: u64 = 0;

    loop {
        let mut buffer = audio_buffer.lock().unwrap();
        if buffer.len() >= app_config.frame_length_samples {
            let frame: Vec<f32> = buffer[..app_config.frame_length_samples].to_vec();

            let feature_tensor =
                audio::preprocess_audio_frame(&frame, &app_config, &cli_config.model_type)?;
            let predictions = model.predict(&feature_tensor)?;

            for (i, prediction) in predictions.iter().enumerate() {
                if let Some(machine) = state_machines.get_mut(i) {
                    if let Some(event) = machine.update(prediction.score, -45.0, frame_counter) {
                        println!("{}", event);
                    }
                }
            }

            buffer.drain(..app_config.hop_length_samples);
            frame_counter += 1;
        }
        drop(buffer);
        // --- Corrected typo here ---
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}
