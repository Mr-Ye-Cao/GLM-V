#!/bin/bash
# Start GLM-4.6V-Flash vLLM Server
# Usage: ./start_server.sh [--background]

set -e

# Configuration
MODEL="zai-org/GLM-4.6V-Flash"
MODEL_NAME="glm-4.6v-flash"
PORT=8000
HOST="0.0.0.0"
MAX_MODEL_LEN=32768
GPU_MEMORY_UTILIZATION=0.85

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate glm-v

echo "Starting GLM-4.6V-Flash server..."
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Context length: $MAX_MODEL_LEN"
echo "  GPU memory: ${GPU_MEMORY_UTILIZATION}%"
echo ""

CMD="vllm serve $MODEL \
    --tensor-parallel-size 1 \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --served-model-name $MODEL_NAME \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --port $PORT \
    --host $HOST"

if [[ "$1" == "--background" ]]; then
    echo "Running in background..."
    $CMD &
    echo "Server PID: $!"
    echo "Test with: curl http://localhost:$PORT/health"
else
    echo "Running in foreground (Ctrl+C to stop)..."
    $CMD
fi
