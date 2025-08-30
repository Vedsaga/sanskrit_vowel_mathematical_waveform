// src/models/onnx.rs

use crate::config::AppConfig;
use crate::models::{Model, Prediction};
use anyhow::Result;
use std::path::Path;
use tract_onnx::prelude::*;

// A type alias to make the model type signature cleaner.
type OnnxModelType = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// The public struct for our ONNX inference engine.
pub struct OnnxModel {
    model: OnnxModelType,
    phonemes: [&'static str; 16],
}

/// Implementation of the universal `Model` trait for our ONNX model.
impl Model for OnnxModel {
    /// Creates a new OnnxModel by loading the .onnx file from the given path.
    fn new(path: &Path, _config: &AppConfig) -> Result<Self> {
        // Load the ONNX model file using tract.
        let model = tract_onnx::onnx()
            .model_for_path(path)?
            // We must explicitly define the shape of the input tensor.
            // This must match the shape you used when exporting from PyTorch.
            // Shape: (batch_size, channels, height, width)
            .with_input_fact(0, f32::fact([1, 1, 40, 70]).into())?
            .into_optimized()?
            .into_runnable()?;

        // The labels must be in the exact same order as the model's output layer.
        let phonemes = [
            "अ", "आ", "इ", "ई", "उ", "ऊ", "ऋ", "ॠ", "ऌ", "ॡ", "ए", "ऐ", "ओ", "औ", "अं", "अः",
        ];

        Ok(Self { model, phonemes })
    }

    /// Runs inference on a pre-processed Mel Spectrogram tensor.
    fn predict(&self, features: &Tensor) -> Result<Vec<Prediction>> {
        let result = self.model.run(tvec!(features.clone().into()))?;
        let output_view = result[0].to_array_view::<f32>()?;

        // --- MODIFIED: Map all 16 scores to the Prediction struct ---
        let predictions = self
            .phonemes
            .iter()
            .zip(output_view.iter())
            .map(|(&phoneme, &score)| Prediction {
                phoneme: phoneme.to_string(),
                score: score as f64,
            })
            .collect();

        Ok(predictions)
    }
}
