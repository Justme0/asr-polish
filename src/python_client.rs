//! HTTP client for the Qwen3-ASR Python sidecar (see `python_sidecar/server.py`).
//!
//! Used when the server runs with `--backend python`. Sends the already-decoded,
//! already-resampled f32 mono @16kHz samples to the sidecar's `POST /transcribe`
//! endpoint as raw little-endian f32 bytes and returns the transcribed text.
#![allow(dead_code)]

use serde::Deserialize;
use std::time::{Duration, Instant};

/// Client for the Python sidecar.
pub struct PythonClient {
    agent: ureq::Agent,
    transcribe_url: String,
    health_url: String,
}

#[derive(Deserialize)]
struct TranscribeResponse {
    text: String,
    #[serde(default)]
    language: String,
}

#[derive(Deserialize)]
struct ErrorResponse {
    error: String,
}

impl PythonClient {
    /// Create a client targeting the sidecar at `base_url` (e.g. `http://127.0.0.1:8090`).
    pub fn new(base_url: &str) -> Self {
        let base = base_url.trim_end_matches('/');
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            // Generous read timeout: long audio can take a while to decode on CPU/GPU.
            .timeout_read(Duration::from_secs(300))
            .build();
        PythonClient {
            agent,
            transcribe_url: format!("{}/transcribe", base),
            health_url: format!("{}/health", base),
        }
    }

    /// Best-effort readiness probe: succeeds only if the sidecar answers and
    /// reports the model as loaded.
    pub fn health_check(&self) -> Result<(), String> {
        let resp = self
            .agent
            .get(&self.health_url)
            .call()
            .map_err(|e| format!("cannot reach sidecar: {}", e))?;
        let body = resp.into_string().unwrap_or_default();
        if body.contains("\"ready\":true") || body.contains("\"ready\": true") {
            Ok(())
        } else {
            Err(format!("sidecar reachable but model not ready: {}", body))
        }
    }

    /// Transcribe mono f32 samples at 16kHz. Matches `AsrModel::transcribe_audio`.
    pub fn transcribe_audio(&self, samples: &[f32]) -> Result<String, String> {
        // Serialize as little-endian f32 bytes; the sidecar decodes format=f32le.
        let t0 = Instant::now();
        let mut body = Vec::with_capacity(samples.len() * 4);
        for &s in samples {
            body.extend_from_slice(&s.to_le_bytes());
        }
        let serialize_ms = t0.elapsed().as_millis();

        let t1 = Instant::now();
        let result = self
            .agent
            .post(&self.transcribe_url)
            .query("format", "f32le")
            .query("sample_rate", "16000")
            .set("Content-Type", "application/octet-stream")
            .send_bytes(&body);
        let http_ms = t1.elapsed().as_millis();

        match result {
            Ok(resp) => {
                let t2 = Instant::now();
                let text = resp
                    .into_string()
                    .map_err(|e| format!("failed to read sidecar response: {}", e))?;
                let parsed: TranscribeResponse = serde_json::from_str(&text)
                    .map_err(|e| format!("invalid sidecar JSON: {} (body: {})", e, text))?;
                let parse_ms = t2.elapsed().as_millis();
                log::info!(
                    "[timing][py-client] serialize={}ms http_roundtrip={}ms parse={}ms ({} bytes sent)",
                    serialize_ms, http_ms, parse_ms, body.len()
                );
                Ok(parsed.text)
            }
            // The sidecar returns 4xx/5xx with a JSON {"error": "..."} body.
            Err(ureq::Error::Status(code, resp)) => {
                let body = resp.into_string().unwrap_or_default();
                let msg = serde_json::from_str::<ErrorResponse>(&body)
                    .map(|e| e.error)
                    .unwrap_or(body);
                Err(format!("sidecar returned HTTP {}: {}", code, msg))
            }
            Err(e) => Err(format!("sidecar request failed: {}", e)),
        }
    }
}
