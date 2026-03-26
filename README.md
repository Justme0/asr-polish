# asr-polish

现在有很多开源的ASR语音识别模型，在落地商用时往往碰到各种bad case，需要从工程上解决，比如不断调prompt，这些是开发者们的共性需求，本项目旨在合力共建，目标可落地商用。

按需求场景分两大类：
## 1. 实时字幕

基于 WebSocket 设计一套标准的业务层协议，屏蔽底层各模型的实现。有的模型不支持流式，将从工程层面适配解决。

## 2. 音频文件离线识别。

## Quick Start

### 下载模型

```bash
cd third_party/qwen-asr && bash download_model.sh --model small
```

### 启动 HTTP 服务

```bash
RUST_LOG=info ./target/release/asr-server -p 8080
```

### 测试

```bash
# 健康检查
curl http://localhost:8080/health

# 发送 PCM 文件（s16le, 16kHz, mono）
curl -s -X POST http://localhost:8080/asr \
  -H "Content-Type: application/octet-stream" \
  --data-binary @16k_zh_en_twocutHeadChar16k.real.pcm | python3 -m json.tool
```
