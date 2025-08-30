// src/models/mod.rs

use crate::config::AppConfig;
use anyhow::Result;
use std::path::Path;
use tract_onnx::prelude::Tensor;

// 1. Declare the sub-modules that will exist in this directory.
//    This tells Rust to look for `gmm.rs` and `onnx.rs` files.
pub mod gmm;
pub mod onnx;

// 2. Define a standardized output struct for any model's prediction.
//    This ensures that no matter which model we use, the result
//    always has the same, predictable format.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub phoneme: String,
    /// Can be log-probability for GMM or confidence for ONNX.
    pub score: f64,
}

// 3. Define the universal `Model` trait (the "contract").
//    Any struct that wants to be a usable model in our application
//    MUST provide implementations for these two functions.
//
//    The `Send + Sync` part is important for ensuring thread-safety,
//    which is crucial for real-time applications like this.
pub trait Model: Send + Sync {
    fn new(path: &Path, config: &AppConfig) -> Result<Self>
    where
        Self: Sized;

    fn predict(&self, features: &Tensor) -> Result<Vec<Prediction>>;
}
