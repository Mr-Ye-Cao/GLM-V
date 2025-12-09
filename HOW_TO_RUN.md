# How to Run GLM-4.6V-Flash with vLLM

This guide explains how to self-host GLM-4.6V-Flash on an NVIDIA GPU using vLLM.

## Requirements

- NVIDIA GPU with at least 24GB VRAM (tested on RTX 5090 32GB)
- CUDA 12.x compatible drivers
- Conda or Miniconda installed

## Setup

### 1. Create Conda Environment

```bash
conda create -n glm-v python=3.11 -y
conda activate glm-v
```

### 2. Install PyTorch with CUDA Support

```bash
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install vLLM and Dependencies

```bash
pip install vllm==0.12.0
pip install transformers==5.0.0rc0  # Required for GLM-4.6V processor support
pip install "numpy<2.3"  # Fix numba compatibility
```

> **Note**: vLLM 0.12.0 ships with transformers 4.x, but GLM-4.6V-Flash requires transformers 5.0.0rc0 for proper processor support. The force reinstall is necessary.

## Running the Server

### Basic Command

```bash
conda activate glm-v

vllm serve zai-org/GLM-4.6V-Flash \
    --tensor-parallel-size 1 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --served-model-name glm-4.6v-flash \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --port 8000 \
    --host 0.0.0.0
```

### Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--tool-call-parser` | glm45 | Parser for function/tool calls |
| `--reasoning-parser` | glm45 | Parser for reasoning/thinking output |
| `--served-model-name` | glm-4.6v-flash | Model name exposed via API |
| `--max-model-len` | 32768 | Maximum context length (reduce if OOM) |
| `--gpu-memory-utilization` | 0.85 | GPU memory fraction to use |
| `--port` | 8000 | Server port |
| `--host` | 0.0.0.0 | Listen on all interfaces |

### Memory Tuning

For GPUs with less VRAM, adjust these parameters:
- Reduce `--max-model-len` (e.g., 16384 or 8192)
- Lower `--gpu-memory-utilization` (e.g., 0.8)

## Testing the Server

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat Completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6v-flash",
    "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
    "max_tokens": 100
  }'
```

### With Image Input

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.6v-flash",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
        {"type": "text", "text": "What is in this image?"}
      ]
    }],
    "max_tokens": 512
  }'
```

## API Endpoints

The server provides OpenAI-compatible endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion API |
| `POST /v1/completions` | Text completion API |
| `GET /docs` | OpenAPI documentation |

## Response Format

The model includes reasoning/thinking in responses:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "2+2 is 4.",
      "reasoning": "The user is asking a simple arithmetic question..."
    }
  }]
}
```

## Troubleshooting

### OOM (Out of Memory) Errors

Reduce context length:
```bash
--max-model-len 16384
```

### Slow First Request

The first request takes longer due to CUDA graph compilation. Subsequent requests are faster.

### NumPy Version Error

If you see "Numba needs NumPy 2.2 or less":
```bash
pip install "numpy<2.3"
```

### Processor Type Error

If you see "Invalid type of HuggingFace processor":
```bash
pip install transformers==5.0.0rc0
```
