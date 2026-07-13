# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**asr-polish** is a Rust HTTP server for Chinese (and multilingual) speech-to-text. It exposes a production-ready REST API for audio transcription and supports two interchangeable inference backends selected at startup via `--backend`:

- **`python`** (default) — proxies audio to a Python sidecar (`python_sidecar/server.py`) that runs the official [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) PyTorch model (`third_party/Qwen3-ASR`, the `qwen_asr` package). GPU-accelerated; supports 52 languages and language auto-detection. The sidecar itself has two interchangeable inference engines, selected by its own `--backend` flag: **`vllm`** (default; CUDA graphs + paged attention, ~7× faster decode at batch=1) or **`transformers`** (eager HuggingFace `generate`).
- **`c`** — the in-process [antirez/qwen-asr](https://github.com/antirez/qwen-asr) C library (`third_party/qwen-asr`), statically linked via FFI. CPU-only, self-contained, no Python needed.

Both backends serve the same `POST /asr` HTTP contract. Licensed under MIT.

Planned use-case categories:
1. **Real-time Subtitles** — WebSocket-based protocol layer (not yet implemented in Rust; both backends have streaming support that is not yet exposed)
2. **Offline Audio File Recognition** — Working via `POST /asr` endpoint

## Build & Run

Prerequisites: Rust toolchain, OpenBLAS (`libopenblas-dev`), C compiler. For the Python backend also: a `qwen3-asr` conda env with the `third_party/Qwen3-ASR` package installed (see that submodule's CLAUDE.md). The sidecar's default `vllm` engine additionally needs vLLM (`pip install -e ".[vllm]"` in that submodule); use `ASR_BACKEND=transformers` if vLLM is unavailable.

```bash
git submodule update --init --recursive
cargo build --release                    # Output: target/release/asr-server
```

### Run with the Python backend (default)

Two processes. Start the sidecar first, then the Rust server:

```bash
# 1) Python sidecar — loads the model once, serves on :8090.
#    Its own --backend defaults to vllm; override to transformers if desired.
./python_sidecar/run.sh                              # or: python python_sidecar/server.py
ASR_BACKEND=transformers ./python_sidecar/run.sh     # eager HF backend instead of vLLM

# 2) Rust server — proxies /asr to the sidecar
RUST_LOG=info ./target/release/asr-server -p 8080    # --backend python is the default
```

> Two `--backend` flags at different layers: the **Rust server** picks `python|c`; the **sidecar** picks `vllm|transformers`. They are independent.

### Run with the C backend

Single process, no Python. Requires the C-library model weights:

```bash
cd third_party/qwen-asr && bash download_model.sh --model small  # ~1.8GB download
RUST_LOG=info ./target/release/asr-server --backend c -p 8080
```

Test (identical for both backends):
```bash
curl http://localhost:8080/health
curl -s -X POST http://localhost:8080/asr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @file.pcm
```

CLI flags: `--backend <c|python>`, `--sidecar-url <url>` (default `http://127.0.0.1:8090`), `-d <model-dir>` (C backend), `-h <host>`, `-p <port>`, `--help`

## Architecture

```
HTTP Client (curl/app)
    │  POST /asr (raw PCM body)
    ▼
src/main.rs — Actix-web HTTP server
    │  Validates format, resamples to 16kHz f32 if needed, runs inference in blocking thread
    ▼
src/backend.rs — enum Backend { C(AsrModel), Python(PythonClient) }
    │  Unified transcribe_audio(&[f32]) -> Result<String, String>; selected by --backend
    ├──────────────────────────────┐
    ▼ (--backend c)                 ▼ (--backend python, default)
src/asr.rs — AsrModel               src/python_client.rs — PythonClient
    │  Mutex<*mut QwenCtx>               │  ureq POST f32le bytes → sidecar /transcribe
    ▼                                    ▼
src/ffi.rs — extern "C"             python_sidecar/server.py — FastAPI/uvicorn
    ▼                                    │  Loads model once, warms up, serializes inference
third_party/qwen-asr/ (C lib)            ▼  (sidecar --backend: vllm default | transformers)
    Audio → mel → Encoder → …        third_party/Qwen3-ASR/ — qwen_asr (Qwen3ASRModel)
                                         vLLM (CUDA graphs) ⟷ or ⟷ HF transformers (eager)
                                         Qwen3ASRModel.transcribe((samples, 16000)) → text + language
```

**Key design decisions:**
- Backend abstraction (`src/backend.rs`): both paths expose the same `transcribe_audio(&[f32])` contract, so the HTTP handler is backend-agnostic. Default is `python`.
- **Python sidecar** is a separate process (a Python torch+CUDA stack can't be statically linked into the Rust binary). Rust sends already-decoded, already-resampled f32 mono @16kHz samples as raw `f32le` bytes to `POST /transcribe`. The sidecar loads the model once and serializes inference with a lock (one at a time, like the C backend).
  - **Sidecar inference engine** (`--backend` / `ASR_BACKEND`, default `vllm`):
    - **`vllm`** — uses CUDA graphs + paged attention. On **Tesla T4 (sm_75)** it auto-selects the **FlashInfer** attention backend for the decoder and **Torch-SDPA** for the audio encoder (FA2 needs sm_80+; rejected cleanly). ~7× faster decode than transformers at batch=1 (~158ms vs ~1121ms for a 12.76s clip), because it eliminates the ~4k eager kernel launches/token. Runs its heavy work in a separate `EngineCore` child process. vLLM-only knobs: `ASR_GPU_MEM_UTIL` (0.85), `ASR_MAX_MODEL_LEN` (16384), `ASR_ENFORCE_EAGER` (false → keeps CUDA graphs).
    - **`transformers`** — eager HuggingFace `generate` with `float16` + `attn_implementation="sdpa"` (bf16 and FlashAttention-2 require sm_80+). Slower (decode is kernel-launch-bound at batch=1) but simplest / fewest deps.
  - The sidecar **warms up** with one throwaway inference at startup (absorbs the ~4s FlashInfer/CUDA-graph JIT so the first real request is fast) and **reaps its child processes on shutdown** (the vLLM `EngineCore` otherwise orphans and keeps holding GPU memory).
  - The sidecar prepends `third_party/Qwen3-ASR` to `sys.path` so it always uses the in-repo submodule, regardless of CWD or any editable `qwen-asr` install elsewhere in the env.
- C library compiled as static lib via `build.rs` using the `cc` crate; `AsrModel` uses `Mutex` — only one transcription at a time (C library not fully thread-safe).
- Server uses `web::block()` to avoid blocking the async runtime during inference.
- 50MB max payload; auto-resamples non-16kHz input via linear interpolation.

## HTTP API

- `GET /health` → `{"status":"ok"}` (Rust server; always ok once up)
- `POST /asr` → `{"text":"...","duration_ms":123}`
  - Body: raw PCM bytes (`application/octet-stream`)
  - Query params: `sample_rate` (default 16000), `format` ("s16le" or "f32le")

### Python sidecar API (internal, `python_sidecar/server.py`)

- `GET /health` → `{"status":"ok","ready":true,"backend":...,"model":...,"device":...,"dtype":...}`
- `POST /transcribe` → `{"text":"...","language":"..."}`
  - Body: raw PCM bytes; Query: `format` (f32le/s16le), `sample_rate`, `language` (optional forced language)

## Testing

No automated tests yet. Manual testing via curl against a running server. Sample audio files in `third_party/qwen-asr/samples/`.

## C Library FFI Surface (src/ffi.rs)

Exposed C functions: `qwen_load`, `qwen_free`, `qwen_transcribe_audio`, `qwen_transcribe`, `qwen_set_prompt`, `qwen_set_force_language`, `qwen_set_token_callback`, `qwen_set_threads`, `qwen_get_num_cpus`. Globals: `qwen_verbose`, `qwen_monitor`.

The C library also supports streaming (`qwen_transcribe_stream`, `qwen_transcribe_stream_live`) and segmented transcription, but these are **not yet exposed** in the Rust wrapper.
