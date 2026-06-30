#!/bin/bash
# =============================================================================
# Port-of-Entry Vehicle Inspection Analyst - Start Script
#
# Performs:
#   1. Port preflight (8095 must be free)
#   2. Standing-rule check: warn / abort if a vLLM instance is already running
#      (we do not run two vLLMs concurrently on the GB10)
#   3. Model presence check
#   4. Host IP discovery (so the in-container banner prints a usable URL)
#   5. docker compose up -d
#   6. Poll /api/health until ready, print connection banner
# =============================================================================
set -e

HOST_PORT=8095
APP_NAME="vehicle-inspection-analyst"
MODEL_DIR="./models/Qwen3-VL-8B-Instruct-FP8"

echo ""
echo "=============================================="
echo "  Vehicle Inspection Demo - Start"
echo "=============================================="

# ---- 1. Port preflight ------------------------------------------------------
if ss -ltn "sport = :${HOST_PORT}" 2>/dev/null | grep -q LISTEN; then
    echo "ERROR: Port ${HOST_PORT} is already in use."
    echo "       Run 'sudo ss -ltnp | grep ${HOST_PORT}' to find the owner."
    exit 1
fi
echo "[preflight] Host port ${HOST_PORT}: free"

# ---- 2. Standing-rule check: only one vLLM at a time on the GB10 -----------
# Look for the shared text vLLM on :8091 (competitive-intel-vllm) which is the
# most common conflict.
if ss -ltn "sport = :8091" 2>/dev/null | grep -q LISTEN; then
    echo ""
    echo "WARNING: A vLLM instance appears to be running on :8091."
    echo "         Standing rule: never run two vLLMs concurrently on the GB10."
    echo "         Stop the shared vLLM first (e.g., 'docker stop competitive-intel-vllm')."
    echo ""
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# ---- 3. Model presence check -----------------------------------------------
if [ ! -d "${MODEL_DIR}" ]; then
    echo ""
    echo "ERROR: Model directory not found at ${MODEL_DIR}"
    echo "       Run ./download_models.sh first."
    exit 1
fi
echo "[preflight] Model directory: present"

# ---- 4. Host IP discovery --------------------------------------------------
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
if [ -z "${HOST_IP}" ]; then
    HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')
fi
if [ -z "${HOST_IP}" ]; then
    HOST_IP="localhost"
    echo "[preflight] Could not auto-detect host IP, banner will say 'localhost'"
else
    echo "[preflight] Host IP detected: ${HOST_IP}"
fi
export HOST_IP

# ---- 5. Build + launch -----------------------------------------------------
echo ""
echo "[launch] docker compose up -d ..."
docker compose up -d --build

# ---- 6. Health poll --------------------------------------------------------
echo ""
echo "[health] Waiting for app to become ready (vLLM cold start: 2-4 min)..."
MAX_WAIT=600
ELAPSED=0
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -sf "http://localhost:${HOST_PORT}/api/health" 2>/dev/null | grep -q '"status":"healthy"'; then
        echo ""
        echo "=============================================="
        echo "  READY"
        echo "=============================================="
        echo ""
        echo "  Click link here:  http://${HOST_IP}:${HOST_PORT}"
        echo ""
        echo "  Logs:    docker logs -f ${APP_NAME}"
        echo "  Stop:    docker compose down"
        echo "=============================================="
        exit 0
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "[health] still waiting... (${ELAPSED}s) -- tail with: docker logs ${APP_NAME}"
    fi
done

echo ""
echo "ERROR: app did not become ready within ${MAX_WAIT}s"
echo "       Inspect logs: docker logs ${APP_NAME}"
exit 1
