#!/usr/bin/env python3
# coding=utf-8
"""
Qwen3-ASR Python sidecar
========================

A small HTTP service that loads the official Qwen3-ASR PyTorch model
(third_party/Qwen3-ASR, the `qwen_asr` package) once and serves transcription
requests. The Rust `asr-server` proxies audio to this process when started with
`--backend python` (the default).

Wire contract (kept deliberately simple so Rust can talk to it with zero extra
dependencies):

    POST /transcribe
        body : raw PCM samples, little-endian, mono
        query:
            format      "f32le" (default) or "s16le"
            sample_rate  int, default 16000  (audio is resampled to 16k if needed)
            language     optional forced language, e.g. "English" / "Chinese"
        200 -> {"text": "...", "language": "..."}
        4xx -> {"error": "..."}

    GET /health -> {"status":"ok","ready":true,"model":"...","device":"...","dtype":"..."}

The Rust proxy always sends f32le @ 16 kHz mono (it already decodes and
resamples), but the sidecar accepts s16le and other sample rates too so it can
be exercised directly with curl.

Run:
    conda activate qwen3-asr
    python python_sidecar/server.py --model-dir third_party/Qwen3-ASR/Qwen3-ASR-0.6B --port 8090

Configuration (CLI flag > env var > default):
    --backend   / ASR_BACKEND     default: vllm  ("vllm" or "transformers")
                                  vllm: CUDA graphs + paged attn, ~7x faster
                                  decode at batch=1; transformers: eager HF generate
    --model-dir / ASR_MODEL_DIR   default: third_party/Qwen3-ASR/Qwen3-ASR-0.6B
    --device    / ASR_DEVICE      default: cuda:0 (falls back to cpu if no CUDA;
                                  vllm requires CUDA; the GPU index is honored)
    --dtype     / ASR_DTYPE       default: float16 (bf16 unsupported on Tesla T4 / sm_75)
    --host      / ASR_HOST        default: 0.0.0.0
    --port      / ASR_PORT        default: 8090
    --max-new-tokens / ASR_MAX_NEW_TOKENS   default: 512
  vLLM-only (ignored by transformers backend):
    --gpu-memory-utilization / ASR_GPU_MEM_UTIL   default: 0.85
    --max-model-len          / ASR_MAX_MODEL_LEN  default: 16384
    --enforce-eager          / ASR_ENFORCE_EAGER  default: false (keeps CUDA graphs)
"""

import argparse
import atexit
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Use the in-repo Qwen3-ASR submodule regardless of CWD or any (possibly stale)
# editable `qwen-asr` install elsewhere in the env. This file lives at
# <repo>/python_sidecar/server.py, so the submodule is ../third_party/Qwen3-ASR.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_QWEN_PKG_ROOT = _REPO_ROOT / "third_party" / "Qwen3-ASR"
if (_QWEN_PKG_ROOT / "qwen_asr").is_dir():
    sys.path.insert(0, str(_QWEN_PKG_ROOT))

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from qwen_asr import Qwen3ASRModel
# The profiler instance the model uses internally; we read its per-call records
# after each transcribe() to log a CUDA-synchronized phase breakdown.
from qwen_asr.inference.qwen3_asr import _profiler

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("qwen-asr-sidecar")

# Dedicated timing log (in addition to stdout), so the profiling breakdown is
# persisted next to the Rust server's asr-server.log.
_TIMING_LOG_PATH = os.environ.get(
    "ASR_TIMING_LOG", str(_REPO_ROOT / "sidecar-timing.log"))
_timing_fh = open(_TIMING_LOG_PATH, "a", buffering=1)  # line-buffered

def _timing(msg: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    log.info(msg)
    _timing_fh.write(line + "\n")

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}

# vLLM wants a canonical dtype string ("fp16"/"half" are not accepted).
_VLLM_DTYPE = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def _cuda_index(device: str) -> Optional[int]:
    """Extract the GPU index from a 'cuda:N' string, else None (cpu / bare 'cuda')."""
    if device and device.startswith("cuda") and ":" in device:
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


# ─── Model holder ────────────────────────────────────────────────────────────

class ModelHolder:
    """Loads the model once and serializes inference (the HF model + CUDA
    context are not safe to drive from multiple threads at once, and serializing
    also bounds GPU memory — same one-at-a-time policy as the C backend).

    Two inference backends, selected by `backend`:
      - "vllm"        : Qwen3ASRModel.LLM() — CUDA graphs + paged attention.
                        Far lower per-token latency at batch=1 (default).
      - "transformers": Qwen3ASRModel.from_pretrained() — eager HF generate.
    Both expose the same .transcribe() contract, so nothing downstream changes.
    """

    def __init__(self, backend: str, model_dir: str, device: str, dtype: torch.dtype,
                 max_new_tokens: int, gpu_memory_utilization: float,
                 max_model_len: int, enforce_eager: bool):
        self.backend = backend
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.enforce_eager = enforce_eager
        self._lock = threading.Lock()
        self._model: Optional[Qwen3ASRModel] = None

    def load(self):
        log.info("Loading Qwen3-ASR from %s (backend=%s, device=%s, dtype=%s)...",
                 self.model_dir, self.backend, self.device, self.dtype)
        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "transformers":
            self._load_transformers()
        else:
            raise SystemExit(
                f"unknown backend {self.backend!r} (use 'vllm' or 'transformers')")
        self._warmup()
        log.info("Model ready (backend=%s).", self.backend)

    def _warmup(self):
        """Run one throwaway inference so the first real request is fast.

        The first inference triggers lazy setup (FlashInfer JIT / CUDA-graph
        replay for vllm, cuDNN autotune for transformers) that costs several
        seconds. Absorbing it here keeps p99 low and honors the "load before
        serving" contract."""
        try:
            t = time.perf_counter()
            self.transcribe(np.zeros(16000, dtype=np.float32), 16000, None)
            log.info("Warmup inference done in %.1fs", time.perf_counter() - t)
        except Exception as e:  # non-fatal: serving can still proceed
            log.warning("warmup inference failed (non-fatal): %s", e)

    def _load_transformers(self):
        # device_map takes a concrete device; attn_implementation="sdpa" avoids
        # FlashAttention-2 (unsupported on sm_75 / Tesla T4).
        self._model = Qwen3ASRModel.from_pretrained(
            self.model_dir,
            dtype=self.dtype,
            device_map=self.device,
            attn_implementation="sdpa",
            max_new_tokens=self.max_new_tokens,
        )

    def _load_vllm(self):
        # vLLM selects the GPU via CUDA_VISIBLE_DEVICES (it takes no device_map).
        # Set it before constructing the engine so the spawned V1 engine-core
        # child (spawn) inherits the correct device. vLLM auto-picks FLASHINFER
        # attention on sm_75 (FA2 needs sm_80+); enforce_eager=False keeps the
        # CUDA graphs that make vLLM fast at batch=1.
        idx = _cuda_index(self.device)
        if idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
        vllm_dtype = _VLLM_DTYPE.get(self.dtype, "float16")
        self._model = Qwen3ASRModel.LLM(
            model=self.model_dir,
            dtype=vllm_dtype,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            enforce_eager=self.enforce_eager,
            max_new_tokens=self.max_new_tokens,
        )

    @property
    def ready(self) -> bool:
        return self._model is not None

    def transcribe(self, samples: np.ndarray, sample_rate: int, language: Optional[str]):
        """Blocking; call from a worker thread. Returns (text, language)."""
        if self._model is None:
            raise RuntimeError("model not loaded")
        with self._lock:
            results = self._model.transcribe(
                audio=(samples, sample_rate),
                language=language,
            )
        r = results[0]
        return (r.text or ""), (r.language or "")


holder: Optional[ModelHolder] = None
app = FastAPI(title="Qwen3-ASR sidecar")


# ─── Audio decode ────────────────────────────────────────────────────────────

def decode_samples(raw: bytes, fmt: str) -> np.ndarray:
    """Bytes -> float32 mono waveform in [-1, 1]."""
    if fmt == "f32le":
        if len(raw) % 4 != 0:
            raise ValueError("f32le body length must be a multiple of 4")
        return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)
    if fmt == "s16le":
        if len(raw) % 2 != 0:
            raise ValueError("s16le body length must be even")
        ints = np.frombuffer(raw, dtype="<i2").astype(np.float32)
        return ints / 32768.0
    raise ValueError(f"unsupported format '{fmt}' (use 'f32le' or 's16le')")


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": bool(holder and holder.ready),
        "backend": holder.backend if holder else None,
        "model": holder.model_dir if holder else None,
        "device": holder.device if holder else None,
        "dtype": str(holder.dtype).replace("torch.", "") if holder else None,
    }


@app.post("/transcribe")
async def transcribe(
    request: Request,
    format: str = Query("f32le"),
    sample_rate: int = Query(16000, ge=1),
    language: Optional[str] = Query(None),
):
    if holder is None or not holder.ready:
        return JSONResponse({"error": "model not loaded"}, status_code=503)

    t_req = time.perf_counter()
    raw = await request.body()
    t_body = time.perf_counter()
    if not raw:
        return JSONResponse({"error": "empty body — send raw PCM samples"}, status_code=400)

    try:
        samples = decode_samples(raw, format)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    t_decode = time.perf_counter()

    if samples.size == 0:
        return JSONResponse({"error": "no samples decoded from body"}, status_code=400)

    lang = language if (language and language.strip()) else None
    try:
        text, detected = await run_in_threadpool(
            holder.transcribe, samples, sample_rate, lang
        )
    except Exception as e:  # surface model/runtime errors as 500 JSON
        log.exception("transcription failed")
        return JSONResponse({"error": f"transcription failed: {e}"}, status_code=500)
    t_done = time.perf_counter()

    dur_s = samples.size / float(sample_rate)

    # Pull the CUDA-synchronized phase breakdown captured during transcribe().
    try:
        _text, records, counters, model_total = _profiler.report()
    except Exception:
        records, counters, model_total = [], {}, 0.0

    body_ms = (t_body - t_req) * 1000.0
    decode_ms = (t_decode - t_body) * 1000.0
    # Wall time spent in run_in_threadpool: model compute + threadpool scheduling.
    threadpool_ms = (t_done - t_decode) * 1000.0
    model_ms = model_total * 1000.0
    overhead_ms = threadpool_ms - model_ms  # threadpool dispatch + GIL + report

    phase_str = " | ".join(f"{p.split(' [')[0].split(' (')[0]}={t*1000:.0f}ms"
                           for p, t in records)
    gen_tok = next((v for k, v in counters.items() if k.startswith("generated_tokens")), None)
    in_tok = next((v for k, v in counters.items() if k.startswith("input_tokens")), None)
    ms_per_tok = (model_ms / gen_tok) if gen_tok else float("nan")

    _timing(
        f"[transcribe] audio={dur_s:.2f}s bytes={len(raw)} | "
        f"body_read={body_ms:.1f}ms decode={decode_ms:.1f}ms "
        f"threadpool={threadpool_ms:.1f}ms (model={model_ms:.1f}ms overhead={overhead_ms:.1f}ms) | "
        f"in_tok={in_tok} gen_tok={gen_tok} ms/tok={ms_per_tok:.1f} | phases: {phase_str}"
    )

    log.info("transcribed %.2fs (%d samples @ %dHz, lang=%s) -> %r",
             dur_s, samples.size, sample_rate, detected, text[:80])
    return {"text": text, "language": detected}


# ─── Shutdown / cleanup ──────────────────────────────────────────────────────

def _terminate_children(timeout: float = 8.0):
    """Reap child processes on shutdown.

    The vLLM V1 engine runs in a separate `EngineCore` process (spawned by
    vLLM), which does NOT die automatically when this parent is killed — it
    orphans and keeps holding GPU memory. On any exit we terminate our child
    processes so the GPU is released. Harmless no-op for the transformers
    backend (no child processes)."""
    try:
        import psutil
    except Exception:
        return
    try:
        children = psutil.Process().children(recursive=True)
    except Exception:
        return
    if not children:
        return
    log.info("Reaping %d child process(es) on shutdown (vLLM engine core, etc.)",
             len(children))
    for c in children:
        try:
            c.terminate()
        except Exception:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for c in alive:
        try:
            c.kill()
        except Exception:
            pass


@app.on_event("shutdown")
def _on_shutdown():
    # Fires on uvicorn graceful shutdown (including SIGTERM/SIGINT).
    _terminate_children()


# Backstop for exit paths that skip the FastAPI shutdown event.
atexit.register(_terminate_children)


# ─── Entrypoint ──────────────────────────────────────────────────────────────

def build_holder(args) -> ModelHolder:
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA not available; falling back to cpu")
        device = "cpu"
    dtype = _DTYPES.get(args.dtype.lower())
    if dtype is None:
        raise SystemExit(f"unknown --dtype {args.dtype!r} (use float16/bfloat16/float32)")
    backend = args.backend.lower()
    if backend not in ("vllm", "transformers"):
        raise SystemExit(f"unknown --backend {args.backend!r} (use 'vllm' or 'transformers')")
    if backend == "vllm" and device == "cpu":
        raise SystemExit("--backend vllm requires a CUDA device; got cpu")
    return ModelHolder(
        backend=backend,
        model_dir=args.model_dir,
        device=device,
        dtype=dtype,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
    )


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main(argv=None):
    p = argparse.ArgumentParser(description="Qwen3-ASR Python sidecar")
    # Inference backend: vllm (default, CUDA graphs, ~7x faster decode at batch=1)
    # or transformers (eager HF generate). Both serve the same /transcribe contract.
    p.add_argument("--backend", default=os.environ.get("ASR_BACKEND", "vllm"),
                   choices=["vllm", "transformers"])
    p.add_argument("--model-dir", default=os.environ.get(
        "ASR_MODEL_DIR", "third_party/Qwen3-ASR/Qwen3-ASR-0.6B"))
    p.add_argument("--device", default=os.environ.get("ASR_DEVICE", "cuda:0"))
    p.add_argument("--dtype", default=os.environ.get("ASR_DTYPE", "float16"))
    p.add_argument("--host", default=os.environ.get("ASR_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("ASR_PORT", "8090")))
    p.add_argument("--max-new-tokens", type=int,
                   default=int(os.environ.get("ASR_MAX_NEW_TOKENS", "512")))
    # vLLM-only knobs (ignored by the transformers backend).
    p.add_argument("--gpu-memory-utilization", type=float,
                   default=float(os.environ.get("ASR_GPU_MEM_UTIL", "0.85")),
                   help="[vllm] fraction of GPU memory for weights + KV cache")
    p.add_argument("--max-model-len", type=int,
                   default=int(os.environ.get("ASR_MAX_MODEL_LEN", "16384")),
                   help="[vllm] max sequence length (audio+prompt+output tokens)")
    p.add_argument("--enforce-eager", action="store_true",
                   default=_env_bool("ASR_ENFORCE_EAGER", False),
                   help="[vllm] disable CUDA graphs (slower; for debugging)")
    args = p.parse_args(argv)

    global holder
    holder = build_holder(args)
    holder.load()  # load before serving so /health is meaningful and first request is fast

    log.info("Serving on %s:%d (backend=%s)", args.host, args.port, args.backend)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    sys.exit(main())
