// src/state_machine.rs

use crate::config::AppConfig;
use crate::models::gmm::MasterPattern; // GMM-specific for now
use std::io::{self, Write};

#[derive(PartialEq, Debug, Clone)]
pub enum State {
    Idle,
    Matching,
    Mismatch,
}

pub struct MatcherStateMachine {
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
    pub fn new(pattern: &MasterPattern, config: &AppConfig) -> Self {
        let frames_per_second = config.sample_rate as f32 / config.hop_length_samples as f32;
        Self {
            state: State::Idle,
            phoneme: pattern.phoneme.clone(),
            start_frame: 0,
            mismatch_counter: 5,
            max_mismatch_frames: 5,
            min_duration_frames: (pattern.min_duration * frames_per_second) as u32,
            max_duration_frames: (pattern.max_duration * frames_per_second) as u32,
            config: config.clone(),
        }
    }

    pub fn update(&mut self, score: f64, threshold: f64, current_frame: u64) -> Option<String> {
        let mut detection_event = None;
        let is_match = score > threshold;

        match self.state {
            State::Idle => {
                if is_match {
                    self.state = State::Matching;
                    self.start_frame = current_frame;
                    self.mismatch_counter = 0;
                }
            }
            State::Matching => {
                if !is_match {
                    self.state = State::Mismatch;
                    self.mismatch_counter = 1;
                }
            }
            State::Mismatch => {
                if is_match {
                    self.state = State::Matching;
                    self.mismatch_counter = 0;
                } else {
                    self.mismatch_counter += 1;
                    if self.mismatch_counter > self.max_mismatch_frames {
                        detection_event = self.fire_event(current_frame);
                        self.state = State::Idle;
                    }
                }
            }
        }
        detection_event
    }

    fn fire_event(&self, current_frame: u64) -> Option<String> {
        let end_frame = current_frame - self.mismatch_counter as u64;
        let duration_frames = end_frame.saturating_sub(self.start_frame);

        if duration_frames >= self.min_duration_frames as u64
            && duration_frames <= self.max_duration_frames as u64
        {
            let seconds_per_frame =
                self.config.hop_length_samples as f32 / self.config.sample_rate as f32;
            let start_time = self.start_frame as f32 * seconds_per_frame;
            let end_time = end_frame as f32 * seconds_per_frame;
            return Some(format!(
                "✅ DETECTED: '{}' (start: {:.2}s, end: {:.2}s)",
                self.phoneme, start_time, end_time
            ));
        }
        None
    }
}
