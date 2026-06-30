# =============================================================================
# Port-of-Entry Vehicle Inspection Analyst
# Powered by HP ZGX Nano AI Station (NVIDIA GB10 Grace Blackwell, sm_121)
#
# Single container: vLLM (internal :8090) + FastAPI app (:8000)
# Host port mapping: 8095 (host) -> 8000 (container)
#
# Prerequisites:
#   - NVIDIA Container Toolkit installed
#   - Model downloaded to ./models/ via ./download_models.sh
#   - Verified on ZGX Nano with driver 580.95.05
#     (do NOT upgrade base image past :26.01 without confirming driver)
# =============================================================================
FROM nvcr.io/nvidia/vllm:26.01-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Additional Python deps for the FastAPI app
COPY backend/requirements-docker.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY sample-images/ /app/sample-images/

COPY backend/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Models are mounted at runtime from the host
VOLUME /models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD curl -f http://localhost:8000/api/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
