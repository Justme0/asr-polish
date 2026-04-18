//! ASR HTTP Server
//!
//! Receives audio PCM data via HTTP POST and returns transcribed text
//! using the Qwen3-ASR 0.6B model.
//!
//! # API
//!
//! ## POST /asr
//!
//! **Content-Type**: `application/octet-stream`
//!
//! Request body: raw PCM audio data (signed 16-bit little-endian, 16kHz, mono).
//!
//! Query parameters:
//! - `sample_rate` (optional, default: 16000) — input sample rate in Hz.
//!   If not 16000, the server will resample.
//! - `format` (optional, default: "s16le") — input format.
//!   Supported: "s16le" (signed 16-bit LE), "f32le" (float32 LE).
//!
//! Response: JSON `{ "text": "transcribed text", "duration_ms": 123 }`
//!
//! ## GET /health
//!
//! Returns `{ "status": "ok" }`

mod asr;
mod ffi;

use actix_web::{web, App, HttpServer, HttpResponse, HttpRequest};
use serde::Serialize;
use std::sync::Arc;
use std::time::Instant;
use std::io::Write;

/// JSON response for ASR transcription.
#[derive(Serialize)]
struct AsrResponse {
    text: String,
    duration_ms: u64,
}

/// JSON response for errors.
#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// JSON response for health check.
#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

/// Shared application state.
struct AppState {
    model: asr::AsrModel,
}

/// Query parameters for the /asr endpoint.
#[derive(serde::Deserialize)]
struct AsrQuery {
    sample_rate: Option<u32>,
    format: Option<String>,
}

/// Health check endpoint.
async fn health(req: HttpRequest) -> HttpResponse {
    log::info!("Request: {} {} from {}",
        req.method(), req.uri(),
        req.peer_addr().map(|a| a.to_string()).unwrap_or_default());
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
    })
}

/// ASR transcription endpoint.
///
/// Accepts raw PCM audio in the request body and returns the transcribed text.
async fn transcribe(
    req: HttpRequest,
    data: web::Data<Arc<AppState>>,
    body: web::Bytes,
    query: web::Query<AsrQuery>,
) -> HttpResponse {
    let sample_rate = query.sample_rate.unwrap_or(16000);
    let format = query.format.as_deref().unwrap_or("s16le");

    log::info!("Request: {} {}?sample_rate={}&format={} from {} body_size={}",
        req.method(), req.path(),
        sample_rate, format,
        req.peer_addr().map(|a| a.to_string()).unwrap_or_default(),
        body.len());

    if body.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "Empty request body — send PCM audio data".to_string(),
        });
    }

    let start = Instant::now();
    let mut last = start;

    // Convert input to float32 samples at 16kHz
    let samples = match format {
        "s16le" => {
            if body.len() % 2 != 0 {
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: "PCM s16le data length must be even".to_string(),
                });
            }
            let mut samples = Vec::with_capacity(body.len() / 2);
            for chunk in body.chunks_exact(2) {
                let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                samples.push(sample_i16 as f32 / 32768.0);
            }
            samples
        }
        "f32le" => {
            if body.len() % 4 != 0 {
                return HttpResponse::BadRequest().json(ErrorResponse {
                    error: "PCM f32le data length must be multiple of 4".to_string(),
                });
            }
            let mut samples = Vec::with_capacity(body.len() / 4);
            for chunk in body.chunks_exact(4) {
                let sample = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                samples.push(sample);
            }
            samples
        }
        _ => {
            return HttpResponse::BadRequest().json(ErrorResponse {
                error: format!("Unsupported format '{}'. Use 's16le' or 'f32le'.", format),
            });
        }
    };

    if samples.is_empty() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error: "No audio samples in request body".to_string(),
        });
    }

    log::info!("[timing] PCM decode: {}ms", last.elapsed().as_millis());
    last = Instant::now();

    // Resample if needed (simple linear interpolation)
    let samples = if sample_rate != 16000 {
        resample(&samples, sample_rate, 16000)
    } else {
        samples
    };

    log::info!("[timing] Resample: {}ms", last.elapsed().as_millis());
    last = Instant::now();

    let audio_duration_s = samples.len() as f64 / 16000.0;
    log::info!(
        "Received {:.2}s of audio ({} samples, {}Hz {})",
        audio_duration_s,
        samples.len(),
        sample_rate,
        format
    );

    // Run transcription in a blocking thread to avoid blocking the async runtime
    let state = data.clone();
    let result = web::block(move || state.model.transcribe_audio(&samples)).await;

    let transcribe_ms = last.elapsed().as_millis();
    let elapsed = start.elapsed().as_millis() as u64;

    match result {
        Ok(Ok(text)) => {
            log::info!("[timing] Transcribe: {}ms | Total: {}ms", transcribe_ms, elapsed);
            log::info!("Result: {}", &text);
            HttpResponse::Ok().json(AsrResponse {
                text,
                duration_ms: elapsed,
            })
        }
        Ok(Err(e)) => {
            log::error!("Transcription error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Transcription failed: {}", e),
            })
        }
        Err(e) => {
            log::error!("Blocking task error: {}", e);
            HttpResponse::InternalServerError().json(ErrorResponse {
                error: format!("Internal error: {}", e),
            })
        }
    }
}

/// Simple linear interpolation resampler.
/// Have audio aliasing issues?
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }
    let ratio = from_rate as f64 / to_rate as f64;
    let out_len = (samples.len() as f64 / ratio) as usize;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;

        if idx + 1 < samples.len() {
            let val = samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac;
            output.push(val as f32);
        } else if idx < samples.len() {
            output.push(samples[idx]);
        }
    }

    output
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Log to both stderr and ./asr-server.log
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open("asr-server.log")
        .expect("Failed to open asr-server.log");
    let log_file = std::sync::Mutex::new(log_file);

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .format(move |buf, record| {
        let line = format!(
            "[{} {} {}:{} {}] {}",
            chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f"),
            record.level(),
            record.file().unwrap_or(""),
            record.line().unwrap_or(0),
            record.target(),
            record.args()
        );
        // Write to stderr (default)
        writeln!(buf, "{}", line)?;
        // Write to log file
        if let Ok(mut f) = log_file.lock() {
            let _ = writeln!(f, "{}", line);
        }
        Ok(())
    })
    .init();

    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();

    let mut model_dir = String::from("third_party/qwen-asr/qwen3-asr-0.6b");
    let mut host = String::from("0.0.0.0");
    let mut port: u16 = 8080;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "-d" | "--model-dir" => {
                i += 1;
                if i < args.len() {
                    model_dir = args[i].clone();
                }
            }
            "-h" | "--host" => {
                i += 1;
                if i < args.len() {
                    host = args[i].clone();
                }
            }
            "-p" | "--port" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().expect("Invalid port number");
                }
            }
            "--help" => {
                eprintln!("asr-server — Qwen3-ASR HTTP Server");
                eprintln!();
                eprintln!("Usage: asr-server [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -d, --model-dir <dir>   Model directory (default: third_party/qwen-asr/qwen3-asr-0.6b)");
                eprintln!("  -h, --host <addr>       Bind address (default: 0.0.0.0)");
                eprintln!("  -p, --port <port>       Port number (default: 8080)");
                eprintln!("  --help                  Show this help");
                eprintln!();
                eprintln!("API Endpoints:");
                eprintln!("  POST /asr               Transcribe PCM audio");
                eprintln!("    Body: raw PCM data (s16le, 16kHz, mono)");
                eprintln!("    Query: ?sample_rate=16000&format=s16le");
                eprintln!("    Response: {{ \"text\": \"...\", \"duration_ms\": 123 }}");
                eprintln!();
                eprintln!("  GET /health             Health check");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown option: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    log::info!("Loading model from: {}", model_dir);
    let load_start = Instant::now();
    let model = asr::AsrModel::load(&model_dir).expect("Failed to load ASR model");
    log::info!("Model loaded successfully in {}ms", load_start.elapsed().as_millis());

    let state = Arc::new(AppState { model });

    log::info!("Starting server on {}:{}", host, port);
    log::info!("API endpoints:");
    log::info!("  POST /asr      — Transcribe PCM audio");
    log::info!("  GET  /health   — Health check");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(state.clone()))
            .app_data(web::PayloadConfig::new(50 * 1024 * 1024)) // 50MB max payload
            .route("/health", web::get().to(health))
            .route("/asr", web::post().to(transcribe))
    })
    .bind((host.as_str(), port))?
    .run()
    .await
}
