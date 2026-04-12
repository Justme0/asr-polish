# Qwen3-ASR Pure C Inference Engine — Architecture Deep Dive

## 1. High-Level System Architecture

```mermaid
graph TD
    CLI["main.c<br><i>CLI entry point</i>"]
    ORC["qwen_asr.c<br><i>Orchestrator</i>"]
    CLI --> ORC

    ORC --> audio.c & encoder.c & decoder.c & tokenizer.c & safetensors.c

    audio.c & encoder.c & decoder.c --> K

    subgraph K["qwen_asr_kernels.c + Thread pool + BLAS"]
        _generic.c
        _neon.c
        _avx.c
    end
```

## 2. The Complete Inference Pipeline

This is the **exact data flow** for a single audio → text conversion:

```mermaid
flowchart TD
    IN(["WAV / stdin / raw PCM"])

    subgraph A["qwen_asr_audio.c"]
        A1["Parse WAV → s16→f32 → Resample 16kHz → Skip silence"]
    end

    subgraph M["Mel Spectrogram"]
        M1["Reflect-pad → Hann → DFT 400pt → 128 mel bins<br>Dynamic-max clamp → log10 normalize → <b>[128, T]</b>"]
    end

    subgraph E["qwen_asr_encoder.c"]
        E1["<b>Conv2D Stem</b> (per-chunk)<br>3×Conv2D 3x3 s=2 GELU → Reshape → Linear<br>+ Sinusoidal PE (per-chunk, resets pos=0)"]
        E2["Concat chunks → <b>[total_tok, d_model]</b>"]
        E3["<b>Windowed Transformer</b> ×L<br>LayerNorm → QKV (biased) → Bidir attn (win=104)<br>→ residual → LayerNorm → FFN GELU → residual"]
        E4["LayerNorm → proj1 → GELU → proj2<br>→ <b>[T', output_dim]</b>  T'≈T/8"]
        E1 --> E2 --> E3 --> E4
    end

    subgraph P["Prompt Assembly — qwen_asr.c"]
        P1["‹im_start›system [prompt] ‹im_end›<br>‹im_start›user ‹audio_start›<br><b>[ENCODER EMBEDDINGS]</b><br>‹audio_end›‹im_end› ‹im_start›assistant<br>→ embed all via bf16 tok_embeddings"]
    end

    subgraph D["qwen_asr_decoder.c"]
        D1["<b>Prefill</b>: full prompt → KV cache"]
        D2["<b>Autoregressive Loop</b><br>RMSNorm → QKV bf16 → per-head Q/K RMSNorm<br>→ NeoX RoPE → GQA causal attn (2:1)<br>→ SwiGLU MLP bf16 (fused gate+up)<br>→ streaming argmax (no logits buffer, tied weights)<br>→ greedy token → callback → until EOS"]
        D1 --> D2
    end

    subgraph T["qwen_asr_tokenizer.c"]
        T1["GPT-2 BPE decode → text"]
    end

    IN --> A1 --> M1 --> E1
    E4 --> P1 --> D1
    D2 --> T1
```

## 3. Encoder vs Decoder — The Critical Differences

The encoder and decoder are **architecturally very different**, even though both are "transformers":

| | **Audio Encoder** | **LLM Decoder (Qwen3)** |
|---|---|---|
| **Norm** | LayerNorm (with bias) | RMSNorm (no bias) |
| **QKV** | f32, with biases | bf16, no biases + per-head Q/K RMSNorm |
| **Attention** | Bidirectional windowed | Causal GQA (2:1 heads:kv_heads) |
| **Position** | Sinusoidal (per chunk) | NeoX RoPE (split-half) |
| **FFN** | fc1 → GELU → fc2 (biased) | SwiGLU (bf16, no biases) |
| **Weights** | f32 (converted at load) | bf16 (mmap'd direct) |
| **Layers** | 24 or 18 | 28 |
| **Hidden** | 1024 or 896 | 2048 or 1024 |

## 4. Memory Layout & Weight Storage Strategy

```mermaid
graph TD
    subgraph S["SAFETENSORS — mmap'd"]
        direction TB
        SH["0.6B: 1 shard · 1.7B: 2 shards"]
        ENC["<b>Encoder</b>: bf16 on disk → <b>f32 malloc</b> at load<br><i>BLAS needs f32 for batch matmul</i>"]
        DEC["<b>Decoder</b>: bf16 stays <b>mmap'd</b><br>SIMD kernels consume bf16 directly<br><i>gate+up weights fused at load for single matvec</i>"]
        TOK["<b>tok_embeddings</b>: bf16 mmap'd, <b>tied with lm_head</b><br><i>streaming argmax — never materializes logits</i>"]
    end
```

## 5. The Three Transcription Modes

```mermaid
flowchart LR
    subgraph M1["MODE 1: OFFLINE"]
        direction LR
        a1([Audio]) --> b1[Mel] --> c1[Encoder] --> d1[Decoder] --> e1([Text])
    end
```

> Simplest path. Best quality for < 60s audio.

```mermaid
flowchart TD
    A0(["Audio split at silence"]) --> S1 & S2 & S3
    subgraph M2["MODE 2: SEGMENTED  ·  -S seconds"]
        S1[Seg 1] --> E1[Encode] --> D1[Decode] --> R1["Hello"]
        S2[Seg 2] --> E2[Encode] --> D2[Decode] --> R2["world"]
        S3[Seg 3] --> E3[Encode] --> D3[Decode] --> R3["today"]
    end
```

> `--past-text yes`: each segment conditions on prior text. Anti-collapse retry if output too short.

```mermaid
flowchart LR
    subgraph M3["MODE 3: STREAMING  ·  --stream"]
        direction TB
        C1["Chunk 1 ▸ 'The'"]
        C2["Chunk 2 ▸ 'The quick' <i>(rollback 5)</i>"]
        C3["Chunk 3 ▸ 'The quick brown'"]
        C1 --> C2 --> C3
    end
```

> **Encoder cache** (only re-encode tail window) · **Prefill reuse** (skip unchanged KV) · **Rollback** (last 5 tokens unfixed) · **Monotonic commit** (never retract).

## 6. Kernel Dispatch Architecture

```mermaid
graph TD
    API["kernels.h — Public API"]
    IMPL["kernels.c — Thread pool + BLAS"]
    DISPATCH["kernels_impl.h — compile-time dispatch"]

    API --> IMPL --> DISPATCH
    DISPATCH --> NEON["<b>ARM NEON</b><br>bf16 matvec, f32 dot"]
    DISPATCH --> AVX512["<b>AVX-512</b><br>16-wide bf16→f32 matvec<br>4 rows/iter"]
    DISPATCH --> AVX2["<b>AVX2+FMA</b><br>8-wide bf16→f32 matvec<br>2 rows/iter"]
    DISPATCH --> GEN["<b>Generic</b><br>Scalar fallback"]
```

> **Decode bottleneck:** `qwen_bf16_matvec_fused` (memory-bound bf16×f32) + `qwen_argmax_bf16_range` (streaming argmax, no 600KB logits buffer).
> **Encode bottleneck:** `cblas_sgemm` (f32 batch matmul via BLAS).

## 7. KV Cache Design

```mermaid
graph LR
    subgraph KV["KV Cache  ·  shape: [28 layers, max_seq, 1024]"]
        L0["Layer 0: pos0 pos1 … posN | unused"]
        L1["Layer 1: pos0 pos1 … posN | unused"]
        LX["⋮"]
        L27["Layer 27: pos0 pos1 … posN | unused"]
    end
```

> Grows by doubling · Reset between segments · Partially reused in streaming · GQA: 8 KV heads serve 16 Q heads.

## 8. The Prompt Token Layout (Decoder Input)

```mermaid
flowchart LR
    subgraph PF["Prefill"]
        direction LR
        t1["‹im_start› system prompt ‹im_end›"]
        t2["‹im_start› user ‹audio_start›"]
        t3["<b>ENCODER OUTPUTS</b>"]
        t4["‹audio_end› ‹im_end›"]
        t5["‹im_start› assistant"]
        t1 --> t2 --> t3 --> t4 --> t5
    end
    subgraph AR["Autoregressive"]
        t6["language / past-text / ‹asr_text›"]
        t7["generated tokens …"]
        t6 --> t7
    end
    t5 --> t6
```

## 9. Model Variant Auto-Detection

```mermaid
flowchart TD
    L["qwen_load()"] --> O["Open safetensors"]
    O --> C{"layer.18.q_proj<br>exists?"}
    C -- YES --> B17["<b>1.7B</b><br>enc: d=1024, L=24, out=2048<br>dec: hidden=2048, L=28<br>2 shards ~3.4GB"]
    C -- NO --> B06["<b>0.6B</b><br>enc: d=896, L=18, out=1024<br>dec: hidden=1024, L=28<br>1 shard ~1.2GB"]
```

## 10. Threading Model

```mermaid
graph TD
    subgraph TP["Persistent Thread Pool  ·  max 16 threads"]
        M["Main (tid=0)"] --- W1["Worker 1"] & W2["Worker 2"] & WN["Worker N-1"]
    end
```

> `qwen_parallel_for`: broadcast → main runs tid=0 → wait all workers.
> Parallelized: bf16 matvec (row-range) · argmax (range) · QKV (fused) · attention (head partitioning).
> BLAS uses its own separate OpenBLAS thread pool.

## 11. Audio Processing Details

```mermaid
flowchart TD
    A["PCM f32 @ 16kHz"] --> B["Reflect-pad 200 samples"]
    B --> C["Hann window"]
    C --> D["400-pt DFT → 201 bins<br><i>brute-force, not FFT</i>"]
    D --> E["Power spectrum → 128 mel bins<br><i>Slaney filterbank</i>"]
    E --> F{"More frames?<br><i>hop=160, win=400</i>"}
    F -- yes --> C
    F -- no --> G["log10 → dynamic-max clamp → normalize<br>→ <b>[128, n_frames]</b>"]
```

> **vs Whisper:** Whisper uses fixed `log_mel_max=1.5`. Qwen3-ASR uses dynamic maximum per utterance.

## 12. Streaming Internals (the complex part)

```mermaid
flowchart TD
    INIT["chunk_idx=0, committed=∅"]
    MORE{"More audio?"}
    GROW["audio_end += 2s"]

    ENC_CACHE{"Encoder<br>cache?"}
    ENC_Y["Encode tail window only<br>concat with cached"]
    ENC_N["Encode full mel"]

    PREFILL{"Prefill<br>reuse?"}
    PRE_Y["Skip unchanged prefix<br>prefill new suffix"]
    PRE_N["Full prefill"]
    DECODE["Autoregressive decode → candidates"]

    COLD{"Cold start?"}
    SKIP["Skip commit"]
    COMMIT["Commit fixed tokens via callback"]

    INC["chunk_idx++"]
    FINAL(["Emit remaining tokens"])

    INIT --> MORE
    MORE -- yes --> GROW --> ENC_CACHE
    ENC_CACHE -- yes --> ENC_Y --> PREFILL
    ENC_CACHE -- no --> ENC_N --> PREFILL
    PREFILL -- yes --> PRE_Y --> DECODE
    PREFILL -- no --> PRE_N --> DECODE
    DECODE --> COLD
    COLD -- yes --> SKIP --> INC
    COLD -- no --> COMMIT --> INC
    INC --> MORE
    MORE -- no --> FINAL

    style COMMIT fill:#c8e6c9
```

> **Invariant:** committed text is monotonic — never retract emitted text.

## 13. File-Level Dependency Graph

```mermaid
graph TD
    main.c --> qwen_asr.h & kernels.h

    qwen_asr.h --> qwen_asr.c & audio.h
    qwen_asr.c --> encoder.c & decoder.c & tokenizer.h & kernels.h
    audio.h --> audio.c
    encoder.c & decoder.c --> safetensors.h
    safetensors.h --> safetensors.c
    tokenizer.h --> tokenizer.c

    kernels.h --> kernels.c --> kernels_impl.h
    kernels_impl.h --> _generic.c & _neon.c & _avx.c
```

## 14. Key Design Decisions & Why

| Decision | Why |
|----------|-----|
| **bf16 mmap decoder, f32 malloc encoder** | Decoder does matvecs (SIMD bf16). Encoder does batch matmul (BLAS needs f32). |
| **Fused gate+up weights** | One matvec instead of two for SwiGLU. Halves memory traffic. |
| **Streaming argmax** | No 600KB logits buffer. Argmax while scanning rows. |
| **Tied embeddings** | `lm_head = tok_embeddings^T`. Saves ~300MB. |
| **Per-chunk sinusoidal PE** | Chunks are independent — each starts at pos=0. |
| **Brute-force DFT** | N=400 is small enough. Simpler than FFT. |
| **NeoX RoPE (split-half)** | `[x[:h]*cos - x[h:]*sin, x[:h]*sin + x[h:]*cos]` |
| **Persistent thread pool** | Avoids pthread_create/join per op. Workers sleep on condvar. |
