# Port-of-Entry Vehicle Inspection Analyst

VLM-powered damage triage for finished vehicles offloaded from RoRo carriers at a port of destination. One photo in, one structured inspection report out, with a recommended terminal disposition (release, touch-up, body shop hold, carrier claim, reject).

Built on **HP ZGX Nano AI Station** (NVIDIA GB10 Grace Blackwell, sm_121) using **Qwen3-VL-8B-Instruct-FP8** served via vLLM. All inference is on-prem. Vehicle photos (potentially commercially sensitive pre-launch units) never leave the terminal's network.

> **Compliance by Architecture:** imagery, inference, and inspector output stay on the ZGX Nano. Zero cloud dependency.

---

## Demo narrative

Today, RoRo terminal damage triage is a clipboard, a phone camera, and a paper exception report. The dealer learns about damage when the truck shows up. This demo replaces that with a VLM-driven inspection station: yard inspector snaps a photo, gets a structured report and a disposition in seconds, the system writes the exception report and notifies downstream stakeholders.

Audience: OEM finished-vehicle logistics, RoRo terminal operators, dealer network logistics.

---

## Architecture

```
[browser]
   v  https
[FastAPI :8000  (host :8095)] -- single multimodal call --> [vLLM :8090]
   |                                                              |
   +--> labeled-line parser                                        +--> Qwen3-VL-8B-Instruct-FP8
   +--> severity + disposition resolver
   +--> structured report (JSON)
```

Single container. vLLM runs on internal port 8090, FastAPI app on internal port 8000, host port 8095.

**Standing rule:** never run two vLLM instances concurrently on the GB10. Stop the shared text vLLM on :8091 before launching this demo. `start.sh` checks for the conflict and prompts.

---

## Files

```
.
+-- backend/
|   +-- main.py                  # FastAPI app, VLM call, parser, pipeline
|   +-- entrypoint.sh             # Starts vLLM in-container then app
|   +-- requirements-docker.txt
+-- frontend/
|   +-- index.html                # Single-page UI (dark theme, MB + HP header)
|   +-- mb-star.jpeg
|   +-- hp-logo.png
+-- sample-images/
|   +-- mb_eclass_clean.jpg
|   +-- mb_headlight_damage.jpeg
|   +-- mb_paint_damage.jpg
+-- tasks/
|   +-- todo.md
|   +-- lessons.md
+-- Dockerfile
+-- docker-compose.yml
+-- start.sh
+-- download_models.sh
+-- README.md
+-- .gitignore
```

---

## Prerequisites

- HP ZGX Nano with NVIDIA Container Toolkit
- Driver 580.95.05 (do not upgrade base image past `nvcr.io/nvidia/vllm:26.01-py3` without confirming driver compatibility)
- Disk: ~16 GB free for the model
- Host port 8095 free
- No other vLLM instance running on the Nano

---

## Quick start

```bash
# 1. Pull the model (reuses an existing copy if you point at one)
./download_models.sh
# or:
EXISTING_MODEL_PATH=/path/to/existing/Qwen3-VL-8B-Instruct-FP8 ./download_models.sh

# 2. Start
./start.sh
```

`start.sh` performs port + model preflight, builds the container, brings it up, and polls `/api/health` until vLLM is ready. First start takes 2-4 minutes (vLLM CUDA graph compilation). Subsequent starts are faster because `~/.cache/vllm` is mounted.

When ready, open `http://<YOUR_ZGX_NANO_IP>:8095`.

---

## Endpoints

- `GET  /` -- frontend
- `GET  /api/health` -- liveness + vLLM probe
- `POST /api/inspect` -- multipart form: `image`, `port`, `vessel`, `vin`
- `POST /api/inspect_sample` -- form: `sample` (filename in `sample-images/`), `port`, `vessel`, `vin`

---

## Report schema (abbrev.)

```json
{
  "report_id": "VPI-20260630-4821",
  "port": { "name": "Port of Baltimore", "code": "USBAL", ... },
  "vessel": "M/V Hoegh Trapper",
  "vin": "WDDZF4JB8KA123456",
  "vehicle": { "type": "...", "view_angle": "REAR_QUARTER" },
  "assessment": {
    "severity_key": "MODERATE",
    "severity_label": "MODERATE DAMAGE",
    "severity_color": "#E8A317",
    "affected_panels": "rear bumper",
    "damage_observations": "...",
    "affected_area_percent": 12,
    "inspector_summary": "..."
  },
  "disposition": {
    "key": "BODY_SHOP_HOLD",
    "label": "HOLD FOR BODY SHOP",
    "actions": ["...", "..."]
  },
  "raw_vlm_output": "...",
  "token_usage": { "prompt_tokens": ..., "completion_tokens": ... },
  "pipeline_seconds": 4.18
}
```

---

## Brand assets

- Mercedes-Benz star: used for demo storytelling (Curtis previously worked for Mercedes-Benz). Replace with the OEM relevant to the customer when showing.
- HP logo: rendered in a white chip so the HP Blue logo displays correctly against the dark theme.

---

## License

Apache 2.0.
