// src/models/gmm.rs

use crate::config::AppConfig;
use crate::models::{Model, Prediction};
use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use nalgebra::{DMatrix, DVector};
use std::f64::consts::PI;
use std::fs;
use std::io::Cursor;
use std::path::Path;
use tract_onnx::prelude::Tensor;

//================================================
// 1. The Public GMM Model Struct
//================================================
pub struct GmmModel {
    patterns: Vec<MasterPattern>,
    config: AppConfig,
}

//================================================
// 2. Implementation of the Universal `Model` Trait
//================================================
impl Model for GmmModel {
    /// Creates a new GmmModel by loading all .dbp patterns from a directory.
    fn new(path: &Path, config: &AppConfig) -> Result<Self> {
        let patterns = load_patterns(path, config)?;
        if patterns.is_empty() {
            return Err(anyhow!("No GMM patterns (.dbp files) found in {:?}", path));
        }
        Ok(Self {
            patterns,
            config: config.clone(),
        })
    }

    /// Scores an MFCC feature tensor against all loaded GMM patterns.
    fn predict(&self, features: &Tensor) -> Result<Vec<Prediction>> {
        let mfccs_vec: Vec<f64> = features
            .as_slice::<f32>()?
            .iter()
            .map(|x| *x as f64)
            .collect();
        let mfccs_dvec = DVector::from_vec(mfccs_vec);

        // Step 1: Calculate the raw log-likelihood score for every pattern.
        let mut raw_scores: Vec<(String, f64)> = self
            .patterns
            .iter()
            .map(|pattern| {
                let mut normalized_mfccs =
                    (&mfccs_dvec - &pattern.mfcc_mean).component_div(&pattern.mfcc_std);
                normalized_mfccs[0] = 0.0; // Ignore C0

                let start_coeff = self.config.n_mfcc - self.config.n_mfcc_used;
                let mfccs_used = normalized_mfccs
                    .rows(start_coeff, self.config.n_mfcc_used)
                    .clone_owned();

                let score = score_gmm(&mfccs_used, pattern, &self.config);
                (pattern.phoneme.clone(), score)
            })
            .collect();

        // Step 2: Apply the Softmax function to convert log-scores into probabilities.
        // This is for numerical stability and consistent visualization.
        let max_score = raw_scores
            .iter()
            .map(|(_, score)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = raw_scores
            .iter()
            .map(|(_, score)| (score - max_score).exp())
            .collect();
        let sum_exps: f64 = exps.iter().sum();

        let final_predictions = raw_scores
            .into_iter()
            .zip(exps.into_iter())
            .map(|((phoneme, _), exp_score)| Prediction {
                phoneme,
                score: exp_score / sum_exps,
            })
            .collect();

        Ok(final_predictions)
    }
}

//================================================
// 3. Internal Helper Structs and Functions
//================================================
// (These are the functions moved directly from your old main.rs)

#[derive(Debug, Clone)]
pub struct MasterPattern {
    pub phoneme: String,
    gmm_weights: DVector<f64>,
    gmm_means: Vec<DVector<f64>>,
    gmm_cov_invs: Vec<DMatrix<f64>>,
    gmm_cov_dets: Vec<f64>,
    pub mfcc_mean: DVector<f64>,
    pub mfcc_std: DVector<f64>,
    pub min_duration: f32,
    pub max_duration: f32,
}

/// Loads all .dbp pattern files from a given directory.
pub fn load_patterns(dir_path: &Path, config: &AppConfig) -> Result<Vec<MasterPattern>> {
    let mut patterns = Vec::new();
    for entry in fs::read_dir(dir_path)? {
        let path = entry?.path();
        if path.extension().map_or(false, |s| s == "dbp") {
            let phoneme = path.file_stem().unwrap().to_str().unwrap().to_string();
            println!("Loading pattern: {}", phoneme);
            let file_bytes = fs::read(&path)?;
            match parse_dbp_file(&file_bytes, phoneme.clone(), config) {
                Ok(pattern) => patterns.push(pattern),
                Err(e) => eprintln!("Failed to load pattern {}: {}", phoneme, e),
            }
        }
    }
    println!("✅ {} patterns loaded successfully.", patterns.len());
    Ok(patterns)
}

// src/models/gmm.rs

// ... (imports and other structs) ...

/// Scores a given MFCC vector against a pattern's GMM using a numerically stable log-likelihood method.
fn score_gmm(mfccs_used: &DVector<f64>, pattern: &MasterPattern, config: &AppConfig) -> f64 {
    let k = config.n_mfcc_used as f64;

    // Pre-calculate the constant part of the Gaussian formula
    let log_sqrt_2_pi_k = 0.5 * k * (2.0 * PI).ln();

    let mut log_likelihoods = Vec::with_capacity(config.gmm_components);

    for i in 0..config.gmm_components {
        let weight = pattern.gmm_weights[i];
        if weight < 1e-7 {
            continue; // Skip components with negligible weight
        }

        let start_coeff = config.n_mfcc - config.n_mfcc_used;
        let mean_used = pattern.gmm_means[i].rows(start_coeff, config.n_mfcc_used);
        let cov_inv = &pattern.gmm_cov_invs[i];
        let det = pattern.gmm_cov_dets[i];

        // Ensure the determinant is positive before taking the log
        if det <= 0.0 {
            continue;
        }

        let diff = mfccs_used - mean_used;
        let mahalanobis_dist_sq = (diff.transpose() * cov_inv * &diff).x;

        // This is the complete log-likelihood for a single Gaussian component
        let log_likelihood = -0.5 * (log_sqrt_2_pi_k + det.ln() + mahalanobis_dist_sq);

        log_likelihoods.push(weight.ln() + log_likelihood);
    }

    if log_likelihoods.is_empty() {
        return -1000.0; // Return a very low score
    }

    // Log-Sum-Exp trick to safely sum probabilities in the log domain
    let max_log_prob = log_likelihoods
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let sum_exp = log_likelihoods
        .iter()
        .map(|&lp| (lp - max_log_prob).exp())
        .sum::<f64>();

    // The final score is the total log-likelihood for the GMM
    max_log_prob + sum_exp.ln()
}

/// Parses a single .dbp file into a MasterPattern struct.
fn parse_dbp_file(bytes: &[u8], phoneme: String, config: &AppConfig) -> Result<MasterPattern> {
    // ... (Your exact, full parse_dbp_file function code goes here) ...
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
    mfcc_std.add_scalar_mut(1e-8);
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
