#!/bin/bash
# =============================================================================
# Download Qwen3-VL-8B-Instruct-FP8 for the vehicle inspection demo.
#
# If you already have this model downloaded for the USDA crop demo (or any
# other project), set EXISTING_MODEL_PATH and this script will symlink it
# instead of re-downloading.
#
# Examples:
#   ./download_models.sh
#   EXISTING_MODEL_PATH=/home/curtis/usda-demo/models/Qwen3-VL-8B-Instruct-FP8 ./download_models.sh
# =============================================================================
set -e

MODEL_ID="Qwen/Qwen3-VL-8B-Instruct-FP8"
MODEL_DIR_NAME="Qwen3-VL-8B-Instruct-FP8"
TARGET_DIR="./models/${MODEL_DIR_NAME}"

mkdir -p ./models

# -- Already present? --------------------------------------------------------
if [ -d "${TARGET_DIR}" ] && [ -n "$(ls -A "${TARGET_DIR}" 2>/dev/null)" ]; then
    echo "Model already present at ${TARGET_DIR}; skipping."
    exit 0
fi

# -- Symlink path -----------------------------------------------------------
if [ -n "${EXISTING_MODEL_PATH}" ]; then
    if [ ! -d "${EXISTING_MODEL_PATH}" ]; then
        echo "ERROR: EXISTING_MODEL_PATH set but directory does not exist: ${EXISTING_MODEL_PATH}"
        exit 1
    fi
    echo "Symlinking ${EXISTING_MODEL_PATH} -> ${TARGET_DIR}"
    ln -s "$(readlink -f "${EXISTING_MODEL_PATH}")" "${TARGET_DIR}"
    echo "Done."
    exit 0
fi

# -- Fresh download via huggingface-cli + hf_transfer -----------------------
echo "Downloading ${MODEL_ID} via huggingface-cli..."
echo "(uses hf_transfer for max throughput; resumable on interruption)"

# Install deps if not present
if ! command -v huggingface-cli >/dev/null 2>&1; then
    pip install --upgrade "huggingface_hub[cli]" hf_transfer
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

huggingface-cli download "${MODEL_ID}" \
    --local-dir "${TARGET_DIR}" \
    --local-dir-use-symlinks False

echo ""
echo "Verifying cache integrity..."
huggingface-cli scan-cache || true

echo ""
echo "Done. Model at: ${TARGET_DIR}"
