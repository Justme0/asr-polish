# asr-polish

现在有很多开源的ASR语音识别模型，在落地商用时往往碰到各种bad case，需要从工程上解决，比如不断调prompt，这些是开发者们的共性需求，本项目旨在合力共建，目标可落地商用。

按需求场景分两大类：
### 1. 实时字幕

基于 WebSocket 设计一套标准的业务层协议，屏蔽底层各模型的实现。有的模型不支持流式，将从工程层面适配解决。

### 2. 音频文件离线识别

## Quick Start

### 编译

需要先安装 [Rust](https://rustup.rs/) 和 OpenBLAS

```bash
# 初始化子模块并编译
git submodule update --init --recursive
cargo build --release
```

编译产物：`target/release/asr-server`

### 下载模型

```bash
cd third_party/qwen-asr && bash download_model.sh --model small
```

### 启动 HTTP 服务

本服务支持两种识别后端，通过 `--backend` 选择：

#### 方式一：Python 后端（默认，GPU 加速，支持 52 种语言）

使用 `third_party/Qwen3-ASR` 官方 PyTorch 模型。需先准备 `qwen3-asr` conda 环境（见该子模块说明）。分两个进程：

```bash
# 1) 先启动 Python sidecar（加载模型，监听 :8090）
./python_sidecar/run.sh

# 2) 再启动 Rust 服务（默认 --backend python，把 /asr 转发给 sidecar）
RUST_LOG=info ./target/release/asr-server -p 8080
```

sidecar 内部有两种推理引擎，通过它自己的 `--backend`（或 `ASR_BACKEND` 环境变量）选择，二者与 Rust 服务的 `--backend` 相互独立：

- **`vllm`**（默认）：CUDA graphs + paged attention，batch=1 下解码约快 7×（12.76s 音频约 158ms vs transformers 约 1121ms）。需在子模块中安装 vLLM：`pip install -e ".[vllm]"`。
- **`transformers`**：原生 HuggingFace `generate`（eager），依赖最少，但较慢。vLLM 不可用时可回退：

```bash
ASR_BACKEND=transformers ./python_sidecar/run.sh
```

> Tesla T4（sm_75）上 vLLM 会自动选用 FlashInfer（解码）+ Torch-SDPA（音频编码器）注意力后端（FA2 需 sm_80+）。sidecar 启动时会做一次预热推理，并在退出时回收 vLLM 的 `EngineCore` 子进程以释放显存。

#### 方式二：C 后端（单进程，纯 CPU，无需 Python）

使用 `third_party/qwen-asr` C 库，需先下载对应模型：

```bash
cd third_party/qwen-asr && bash download_model.sh --model small
RUST_LOG=info ./target/release/asr-server --backend c -p 8080
```

CLI 参数：`--backend <c|python>`、`--sidecar-url <url>`（默认 `http://127.0.0.1:8090`）、`-d <model-dir>`（C 后端）、`-h <host>`、`-p <port>`、`--help`

### 测试

```bash
# 健康检查
curl http://localhost:8080/health

# 发送 PCM 文件（s16le, 16kHz, mono）
curl -s -X POST http://localhost:8080/asr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @16k_zh_en_twocutHeadChar16k.real.pcm
```
