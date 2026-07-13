//! Transcription backend selection.
//!
//! The server can transcribe either through the in-process C library
//! (`--backend c`, via [`crate::asr::AsrModel`]) or by proxying to the Qwen3-ASR
//! Python sidecar (`--backend python`, the default, via
//! [`crate::python_client::PythonClient`]). Both expose the same
//! `transcribe_audio(&[f32]) -> Result<String, String>` contract on mono f32
//! samples at 16kHz, so the HTTP handler in `main.rs` is backend-agnostic.

use crate::asr::AsrModel;
use crate::python_client::PythonClient;

/// A loaded transcription backend.
pub enum Backend {
    /// In-process antirez qwen-asr C library.
    C(AsrModel),
    /// Proxy to the Python Qwen3-ASR sidecar.
    Python(PythonClient),
}

impl Backend {
    /// Transcribe mono f32 samples at 16kHz, values in [-1.0, 1.0].
    pub fn transcribe_audio(&self, samples: &[f32]) -> Result<String, String> {
        match self {
            Backend::C(model) => model.transcribe_audio(samples),
            Backend::Python(client) => client.transcribe_audio(samples),
        }
    }

    /// Human-readable name for logging.
    pub fn name(&self) -> &'static str {
        match self {
            Backend::C(_) => "c",
            Backend::Python(_) => "python",
        }
    }
}
