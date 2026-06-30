# Port-of-Entry Vehicle Inspection Analyst

VLM-powered damage triage for finished vehicles offloaded from RoRo carriers at a port of destination. One photo in, one structured inspection report out, with a recommended terminal disposition (release, touch-up, body shop hold, carrier claim, reject).

Built on **HP ZGX Nano AI Station** (NVIDIA GB10 Grace Blackwell, sm_121) using **Qwen3-VL-8B-Instruct-FP8** served via vLLM. All inference is on-prem. Vehicle photos (potentially commercially sensitive pre-launch units) never leave the terminal's network.

> **Compliance by Architecture:** imagery, inference, and inspector output stay on the ZGX Nano. Zero cloud dependency.

---

## Demo narrative

Today, RoRo (roll on-roll off) terminal damage triage is a clipboard, a phone camera, and a paper exception report. The dealer learns about damage when the vehicle shows up. This demo replaces that with a VLM-driven inspection station: yard inspector snaps a photo, gets a structured report and a disposition in seconds, the system writes the exception report and notifies downstream stakeholders.

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

Single container. vLLM runs on internal port 8090, FastAPI app on internal port 8000, host port **8095**.

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
|   +-- index.html                # SPA: dark theme, MB + HP header, 2x2 sample grid
|   +-- mb-star.jpeg
|   +-- hp-logo.png
+-- sample-images/
|   +-- mb_eclass_clean.jpg       # RELEASE baseline
|   +-- mb_headlight_damage.jpeg  # MAJOR -> BODY_SHOP_HOLD
|   +-- mb_paint_damage.jpg       # MODERATE -> BODY_SHOP_HOLD
|   +-- mb_eclass_totaled.jpeg    # CRITICAL -> REJECT
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
# or, reuse the USDA crop demo's copy:
EXISTING_MODEL_PATH=/path/to/existing/Qwen3-VL-8B-Instruct-FP8 ./download_models.sh

# 2. Start
./start.sh
```

`start.sh` performs port + model preflight, builds the container, brings it up, and polls `/api/health` until vLLM is ready. First start takes 2-4 minutes (vLLM CUDA graph compilation). Subsequent starts are faster because `~/.cache/vllm` is mounted.

When ready, open `http://<YOUR_ZGX_NANO_IP>:8095`.

---

## Volume mounts (live-edit setup)

Both `frontend/` and `sample-images/` are bind-mounted into the container, so edits to HTML/CSS/JS or new sample images take effect immediately — no rebuild required. The mount block in `docker-compose.yml`:

```yaml
    volumes:
      - ./models:/models:ro
      - ./sample-images:/app/sample-images:ro
      - ./frontend:/app/frontend:ro
      - ~/.cache/vllm:/root/.cache/vllm
```

| Change                            | Action needed                       |
|-----------------------------------|-------------------------------------|
| Edit `frontend/index.html`        | Hard-refresh browser (Ctrl+Shift+R) |
| Add/replace a `sample-images/*`   | Hard-refresh browser                |
| Edit `backend/main.py` (prompt, parser, etc.) | `docker compose up -d --build` |
| Edit `Dockerfile` or `entrypoint.sh` | `docker compose up -d --build`   |
| Quick service bounce, no rebuild  | `docker compose restart`            |

The `~/.cache/vllm` mount persists CUDA graph compilation between restarts, so warm restarts come up in ~30s instead of the 2-4 min cold start.

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
  "vehicle": { "type": "...", "view_angle": "FRONT_QUARTER" },
  "assessment": {
    "severity_key": "MAJOR",
    "severity_label": "MAJOR DAMAGE",
    "severity_color": "#E66B2A",
    "affected_panels": "front bumper, headlight assembly, driver door",
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

## Prompt-engineering notes

The VLM prompt in `backend/main.py` (`INSPECTION_PROMPT`) carries three load-bearing rules to keep the model honest:

1. **Severity tiers list explicit examples per tier**, including "ANY cracked or broken lighting assembly (headlight, taillight, fog light, turn signal lens)" under MAJOR. Without explicit examples the model would observe cracked headlights, call them out, and still grade MODERATE.
2. **A `SEVERITY ASSIGNMENT RULES` block** turns the rubric into a hard constraint: if you observe damage matching an example at tier X, your output severity must be at least X. Do not downgrade. A single severe finding outranks many minor ones.
3. **The `INSPECTOR_SUMMARY` instructions forbid boilerplate negative findings** ("no broken glass observed" etc.) that contradict what the model itself just wrote above. VLMs trained on inspection-report data default to appending these tails; the prompt explicitly bans them.

Output is **labeled plain-text lines**, not JSON schema. Labeled lines are more reliable for VLM output and the parser in `backend/main.py` (`parse_vlm_output`) is tolerant of casing and ordering drift.

---

## Known gotchas

- **vLLM 26.01 argument format changes.** `--limit-mm-per-prompt` no longer accepts the `image=2` shorthand; it parses with `json.loads` and requires `'{"image": 1}'`. For single-image workloads, omit the flag entirely (default is fine). See `backend/entrypoint.sh`.
- **Driver pinning.** Do not upgrade the base image past `nvcr.io/nvidia/vllm:26.01-py3` without confirming the Nano driver (currently 580.95.05) supports it. 26.01 nightly variants require 590.48+.
- **Two vLLMs at once will OOM the GB10.** The shared text vLLM on :8091 and this demo's in-container vLLM cannot coexist. Stop the other before launching.

---

## Brand assets

- **Mercedes-Benz star** in the header: Curtis previously worked for Mercedes-Benz, so MB anchors the storytelling. For other customer presentations, swap `frontend/mb-star.jpeg` for the relevant OEM mark and update the sample images accordingly.
- **HP logo** rendered in a white chip on the right side of the header so the HP Blue mark displays correctly against the dark theme.

---

## License

Apache 2.0.
