# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**asr-polish** is a Rust HTTP server wrapping the [Qwen3-ASR](https://github.com/antirez/qwen-asr) C library for Chinese speech-to-text. It provides a production-ready REST API for audio transcription using the Qwen3-ASR 0.6B (or 1.7B) model. Licensed under MIT.

Planned use-case categories:
1. **Real-time Subtitles** — WebSocket-based protocol layer (not yet implemented in Rust; C library supports streaming)
2. **Offline Audio File Recognition** — Working via `POST /asr` endpoint

## Build & Run

Prerequisites: Rust toolchain, OpenBLAS (`libopenblas-dev`), C compiler

```bash
git submodule update --init --recursive
cargo build --release                    # Output: target/release/asr-server
cd third_party/qwen-asr && bash download_model.sh --model small  # ~1.8GB download
```

Run the server:
```bash
RUST_LOG=info ./target/release/asr-server -p 8080
```

Test:
```bash
curl http://localhost:8080/health
curl -s -X POST http://localhost:8080/asr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @file.pcm
```

CLI flags: `-d <model-dir>`, `-h <host>`, `-p <port>`, `--help`

## Architecture

```
HTTP Client (curl/app)
    │  POST /asr (raw PCM body)
    ▼
src/main.rs — Actix-web HTTP server
    │  Validates format, resamples if needed, runs inference in blocking thread
    ▼
src/asr.rs — Safe Rust wrapper (AsrModel)
    │  Mutex<*mut QwenCtx> serializes C calls; Drop frees C resources
    ▼
src/ffi.rs — Raw FFI bindings (extern "C")
    │  qwen_transcribe_audio(ctx, samples, n_samples)
    ▼
third_party/qwen-asr/ (C library, git submodule)
    Audio → mel spectrogram → Encoder (Conv2D + Transformer) → Decoder (LLM, KV cache) → Tokenizer → text
```

**Key design decisions:**
- C library compiled as static lib via `build.rs` using the `cc` crate with `-O3 -march=native -ffast-math -DUSE_OPENBLAS`
- `AsrModel` uses `Mutex` — only one transcription at a time (C library not fully thread-safe)
- Server uses `web::block()` to avoid blocking the async runtime during inference
- 50MB max payload; auto-resamples non-16kHz input via linear interpolation

## HTTP API

- `GET /health` → `{"status":"ok"}`
- `POST /asr` → `{"text":"...","duration_ms":123}`
  - Body: raw PCM bytes (`application/octet-stream`)
  - Query params: `sample_rate` (default 16000), `format` ("s16le" or "f32le")

## Testing

No automated tests yet. Manual testing via curl against a running server. Sample audio files in `third_party/qwen-asr/samples/`.

## C Library FFI Surface (src/ffi.rs)

Exposed C functions: `qwen_load`, `qwen_free`, `qwen_transcribe_audio`, `qwen_transcribe`, `qwen_set_prompt`, `qwen_set_force_language`, `qwen_set_token_callback`, `qwen_set_threads`, `qwen_get_num_cpus`. Globals: `qwen_verbose`, `qwen_monitor`.

The C library also supports streaming (`qwen_transcribe_stream`, `qwen_transcribe_stream_live`) and segmented transcription, but these are **not yet exposed** in the Rust wrapper.
