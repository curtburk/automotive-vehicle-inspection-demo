#!/bin/bash
# =============================================================================
# Entrypoint for Vehicle Inspection Demo container
#
# 1. Start vLLM serving Qwen3-VL-8B-Instruct-FP8 on internal :8090
# 2. Poll /health until vLLM is ready
# 3. Launch FastAPI app on :8000 (mapped to host :8095)
# =============================================================================
set -e

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-VL-8B-Instruct-FP8}"
VLLM_PORT="${VLLM_PORT:-8090}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
APP_PORT="${PORT:-8000}"

echo "=============================================="
echo "Vehicle Inspection Demo - Container Start"
echo "=============================================="
echo "Model path: ${MODEL_PATH}"
echo "vLLM port (internal): ${VLLM_PORT}"
echo "App port (internal):  ${APP_PORT}"
echo "Max model len:        ${VLLM_MAX_MODEL_LEN}"
echo "=============================================="

if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model not found at ${MODEL_PATH}"
    echo "Run ./download_models.sh on the host before starting the container."
    exit 1
fi

# -- Start vLLM in the background ---------------------------------------------
echo "[entrypoint] Starting vLLM..."
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 1}' \
    > /var/log/vllm.log 2>&1 &

VLLM_PID=$!
echo "[entrypoint] vLLM PID: ${VLLM_PID}"

# -- Wait for vLLM ready ------------------------------------------------------
echo "[entrypoint] Waiting for vLLM to become ready (this can take 2-3 min on first load)..."
MAX_WAIT=600
ELAPSED=0
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "[entrypoint] vLLM is ready after ${ELAPSED}s"
        break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[entrypoint] ERROR: vLLM process died. Tail of log:"
        tail -n 80 /var/log/vllm.log
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "[entrypoint] still waiting... (${ELAPSED}s)"
    fi
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "[entrypoint] ERROR: vLLM did not become ready within ${MAX_WAIT}s"
    tail -n 80 /var/log/vllm.log
    exit 1
fi

# -- Launch FastAPI app -------------------------------------------------------
echo "[entrypoint] Starting FastAPI app on :${APP_PORT}"
cd /app
exec uvicorn backend.main:app --host 0.0.0.0 --port "${APP_PORT}" --log-level info
