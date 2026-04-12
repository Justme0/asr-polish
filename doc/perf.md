# Performance Analysis

Test audio: `16k_zh_en_twocutHeadChar16k.real.pcm` — 12.76s, 16kHz mono s16le

Model: Qwen3-ASR-0.6B, CPU-only (OpenBLAS), 32-core machine

## End-to-End Breakdown (5568ms total)

| Phase | Time | % |
|-------|------|---|
| PCM decode (s16le→f32) | 0ms | 0% |
| Resample | 0ms | 0% |
| Tokenizer load (vocab.json) | 76ms | 1.4% |
| Mel spectrogram (1276 frames) | 30ms | 0.5% |
| **Encoder** (166 tokens) | **2370ms** | **42.6%** |
| **Prefill** (181 tokens into KV cache) | **2352ms** | **42.2%** |
| **Decode** (21 tokens, 34.3ms/tok) | **720ms** | **12.9%** |
| Overhead | ~20ms | 0.4% |

Real-time factor: 5.57s / 12.76s = **0.44x** (inference slower than audio duration)

## Encoder Deep Dive (2370ms)

| Stage | Time | % of Encoder |
|-------|------|---|
| **Conv2D stem** (13 chunks) | **1203ms** | **51%** |
| **Transformer** (18 layers) | **1155ms** | **49%** |
| Final LN + projection | 12ms | <1% |

### Per Transformer Layer (~62ms avg)

| Op | Time | % of Layer |
|----|------|---|
| Attention (LayerNorm + QKV linear + bidir attn + output proj) | ~21ms | 34% |
| FFN (LayerNorm + fc1 + GELU + fc2) | ~42ms | 66% |

All 18 layers are uniform — no outlier layers.

### Per-Layer Data

```
Layer  1/18: attn=21ms ffn=36ms total=57ms
Layer  2/18: attn=21ms ffn=37ms total=58ms
Layer  3/18: attn=21ms ffn=42ms total=63ms
Layer  4/18: attn=20ms ffn=42ms total=62ms
Layer  5/18: attn=20ms ffn=42ms total=62ms
Layer  6/18: attn=21ms ffn=43ms total=64ms
Layer  7/18: attn=20ms ffn=41ms total=62ms
Layer  8/18: attn=20ms ffn=42ms total=62ms
Layer  9/18: attn=20ms ffn=41ms total=60ms
Layer 10/18: attn=20ms ffn=43ms total=63ms
Layer 11/18: attn=21ms ffn=42ms total=63ms
Layer 12/18: attn=21ms ffn=42ms total=63ms
Layer 13/18: attn=22ms ffn=44ms total=65ms
Layer 14/18: attn=23ms ffn=46ms total=69ms
Layer 15/18: attn=23ms ffn=47ms total=70ms
Layer 16/18: attn=23ms ffn=48ms total=71ms
Layer 17/18: attn=23ms ffn=46ms total=70ms
Layer 18/18: attn=21ms ffn=48ms total=69ms
```

## Bottleneck Summary

1. **Conv2D stem (1203ms, 22% of total)** — 3 layers of Conv2D 3×3 stride 2 over 13 chunks. Each chunk processes mel [128, ~100] through three convolutions.

2. **Encoder Transformer FFN (756ms, 14% of total)** — `qwen_linear` doing f32 BLAS matmul: [166, 896] × [896, 3584] and [166, 3584] × [3584, 896]. Memory-bandwidth bound.

3. **Decoder Prefill (2352ms, 42% of total)** — 28 decoder layers processing 181 tokens. Batch bf16 matmul through all layers to populate KV cache.

4. **Decoder Autoregressive (720ms, 13% of total)** — 21 tokens at 34.3ms/tok. Single-token matvec, memory-bound on bf16 weight reads.

## Model Load Time

| Phase | Time |
|-------|------|
| Load encoder weights (bf16→f32 convert) | ~200ms |
| Load decoder weights (bf16 mmap) | ~200ms |
| **Total model load** | **419ms** |

Model load is fast — weights are mmap'd, only encoder weights need bf16→f32 conversion.

## Model Weight Loading — Code Analysis

Three-layer loading architecture: safetensors file → weight loaders → encoder/decoder structs.

### Layer 1: Safetensors File Open (`qwen_asr_safetensors.c`)

```c
safetensors_open(path):
    fd = open(path, O_RDONLY);
    data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);  // no data read yet!
    parse_header(sf);  // parse JSON index: tensor name → {offset, shape, dtype}
```

`mmap` maps the 1.8GB file into virtual address space **without reading any data**. The OS loads pages on demand when actually accessed. This is why open is ~1ms.

### Layer 2: Two Different Loading Strategies

**Encoder weights — `load_bf16_as_f32()` (`qwen_asr_encoder.c`)**

```c
// Find tensor pointer in mmap'd region (no copy)
const safetensor_t *t = multi_safetensors_find(ms, name, &sf);
uint16_t *bf16 = safetensors_get_bf16_direct(sf, t);

// Allocate new f32 buffer + convert every element
float *f32 = malloc(n * sizeof(float));
for (size_t i = 0; i < n; i++)
    d[i] = ((uint32_t)bf16[i]) << 16;   // bf16 → f32: shift 16 bits left
```

Reads every byte from mmap, allocates new memory, converts bf16→f32. Slow but necessary — BLAS (`cblas_sgemm`) needs f32 for batch matmul in encoder.

**Decoder weights — `load_bf16_direct()` (`qwen_asr_decoder.c`)**

```c
// Just return a pointer into the mmap'd file
uint16_t *safetensors_get_bf16_direct(sf, t) {
    return (uint16_t *)((char *)sf->data + 8 + sf->header_size + t->data_offset);
}
```

NO copy, NO conversion, NO malloc. Just a pointer. Custom SIMD kernels (AVX/NEON) consume bf16 directly during inference. OS loads pages when accessed.

### Layer 3: Gate+Up Fusion (decoder only)

```c
// For each of 28 decoder layers:
// gate_weight: [3072, 1024] bf16 (mmap'd pointer)
// up_weight:   [3072, 1024] bf16 (mmap'd pointer)
//
// Fuse into interleaved layout:
l->gate_up_fused_bf16 = malloc(2 * inter * hidden * sizeof(uint16_t));
for (int r = 0; r < inter; r++) {
    memcpy(fused + (2*r)   * hidden, gate + r * hidden, row_bytes);  // gate row
    memcpy(fused + (2*r+1) * hidden, up   + r * hidden, row_bytes);  // up row
}
// Result: [gate_row0, up_row0, gate_row1, up_row1, ...]
```

Real copy + interleave. One fused matvec instead of two during decode — halves memory traffic in the SwiGLU MLP bottleneck.

### Memory Map Summary

```
mmap'd (1.8GB virtual, loaded on-demand by OS):
  model.safetensors
  ├── Decoder QKV weights (bf16)        ← pointer only, 0ms
  ├── Decoder output proj (bf16)        ← pointer only, 0ms
  ├── Decoder gate/up/down (bf16)       ← read during fusion
  ├── Decoder tok_embeddings (bf16)     ← pointer only, 0ms
  ├── Encoder weights (bf16)            ← read during bf16→f32 convert
  └── Norm weights (f32)                ← read during memcpy

malloc'd (actual RAM):
  Encoder weights as f32:      ~285MB  (conv stem + 18 layers + projection)
  Decoder fused gate+up bf16:  ~344MB  (28 layers interleaved)
  Decoder norms f32:           ~1MB
  KV cache:                    grows during inference
  ──────────────────────────────────
  Total malloc'd:              ~630MB
```

### Load Time Breakdown (419ms total)

| Phase | Time | What happens |
|-------|------|-------------|
| mmap + parse header | ~1ms | Map file to virtual memory, parse JSON tensor index |
| Encoder bf16→f32 | ~200ms | Read mmap'd data, convert, malloc ~285MB |
| Decoder gate+up fusion | ~200ms | Read mmap'd gate/up, interleave, malloc ~344MB |
| Decoder direct pointers | ~0ms | Just pointer assignments into mmap region |
| Decoder norms (f32 copy) | ~18ms | Small tensors, memcpy from mmap |
