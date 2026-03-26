//! Safe Rust wrapper around the qwen-asr C library.
#![allow(dead_code)]

use crate::ffi;
use std::ffi::{CStr, CString};
use std::sync::Mutex;

extern "C" {
    fn free(ptr: *mut std::ffi::c_void);
}

/// A loaded ASR model context. Thread-safe via internal mutex.
pub struct AsrModel {
    ctx: Mutex<*mut ffi::QwenCtx>,
}

// The C library uses a global thread pool and the ctx is used sequentially.
// We protect access with a Mutex so only one transcription runs at a time.
unsafe impl Send for AsrModel {}
unsafe impl Sync for AsrModel {}

impl AsrModel {
    /// Load the ASR model from the given directory.
    ///
    /// `model_dir` should contain `*.safetensors`, `vocab.json`, etc.
    pub fn load(model_dir: &str) -> Result<Self, String> {
        let c_dir = CString::new(model_dir)
            .map_err(|e| format!("Invalid model_dir path: {}", e))?;

        // Initialize thread pool with all CPUs
        unsafe {
            ffi::qwen_verbose = 1;
            let n_cpus = ffi::qwen_get_num_cpus();
            ffi::qwen_set_threads(n_cpus);
        }

        let ctx = unsafe { ffi::qwen_load(c_dir.as_ptr()) };
        if ctx.is_null() {
            return Err(format!("Failed to load model from {}", model_dir));
        }

        // Disable token callback (we only want the final string)
        unsafe {
            ffi::qwen_set_token_callback(ctx, None, std::ptr::null_mut());
        }

        Ok(AsrModel {
            ctx: Mutex::new(ctx),
        })
    }

    /// Transcribe raw PCM audio samples.
    ///
    /// `samples` must be mono float32 at 16kHz, values in [-1.0, 1.0].
    pub fn transcribe_audio(&self, samples: &[f32]) -> Result<String, String> {
        let ctx_guard = self.ctx.lock().map_err(|e| format!("Mutex poisoned: {}", e))?;
        let ctx = *ctx_guard;

        let text_ptr = unsafe {
            ffi::qwen_transcribe_audio(ctx, samples.as_ptr(), samples.len() as i32)
        };

        if text_ptr.is_null() {
            return Err("Transcription failed".to_string());
        }

        let text = unsafe { CStr::from_ptr(text_ptr) }
            .to_string_lossy()
            .into_owned();

        // Free the C-allocated string
        unsafe {
            free(text_ptr as *mut std::ffi::c_void);
        }

        Ok(text)
    }

    /// Transcribe raw PCM s16le audio data.
    ///
    /// Converts signed 16-bit little-endian integer samples to float32.
    /// Input must be 16kHz mono.
    pub fn transcribe_pcm_s16le(&self, pcm_data: &[u8]) -> Result<String, String> {
        if pcm_data.len() % 2 != 0 {
            return Err("PCM data length must be even (16-bit samples)".to_string());
        }

        let n_samples = pcm_data.len() / 2;
        let mut samples = Vec::with_capacity(n_samples);

        for chunk in pcm_data.chunks_exact(2) {
            let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push(sample_i16 as f32 / 32768.0);
        }

        self.transcribe_audio(&samples)
    }
}

impl Drop for AsrModel {
    fn drop(&mut self) {
        if let Ok(ctx) = self.ctx.lock() {
            if !ctx.is_null() {
                unsafe {
                    ffi::qwen_free(*ctx);
                }
            }
        }
    }
}
