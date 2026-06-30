"""
Port-of-Entry Vehicle Inspection Analyst
Powered by HP ZGX Nano AI Station

A Vision Language Model demo for on-prem damage triage of finished vehicles
offloaded from RoRo carriers at a port of destination. One image in, one
structured inspection report out, with a recommended terminal disposition.

Uses Qwen3-VL-8B-Instruct-FP8 served via vLLM. Single multimodal call -- the
VLM emits labeled plain-text lines that we parse into a structured report
(this pattern is more reliable than JSON schema for VLM output).

Compliance by Architecture: imagery (potentially commercially sensitive
pre-launch vehicles), inference, and inspector output all stay on the
ZGX Nano. Zero cloud dependency.
"""

import os
import io
import base64
import random
import string
import time
import logging
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import httpx


# -- Logging setup -----------------------------------------------------------
# Robust logging from the start: every stage logs INFO with timing,
# failures get full tracebacks. No silent failures.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vehicle-inspection")

# -- Configuration -----------------------------------------------------------
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8000))
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8090/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "/models/Qwen3-VL-8B-Instruct-FP8")
VLLM_TIMEOUT = float(os.environ.get("VLLM_TIMEOUT", "300"))

# -- Ports of entry ----------------------------------------------------------
# Major US RoRo (roll-on / roll-off) terminals for finished-vehicle imports.
# Lean structure -- just enough context to make the report look real.
PORTS = {
    "baltimore": {
        "name": "Port of Baltimore",
        "code": "USBAL",
        "terminal": "Atlantic Terminal (Tradepoint)",
        "state": "Maryland",
    },
    "brunswick": {
        "name": "Port of Brunswick",
        "code": "USBQK",
        "terminal": "Colonel's Island Terminal",
        "state": "Georgia",
    },
    "long_beach": {
        "name": "Port of Long Beach",
        "code": "USLGB",
        "terminal": "Pier T RoRo Terminal",
        "state": "California",
    },
    "jacksonville": {
        "name": "Port of Jacksonville (JAXPORT)",
        "code": "USJAX",
        "terminal": "Blount Island Marine Terminal",
        "state": "Florida",
    },
}

# -- Damage severity tiers ---------------------------------------------------
SEVERITY_TIERS = {
    "NONE": {
        "label": "NO DAMAGE OBSERVED",
        "color_hex": "#3DBD5D",
        "description": "Vehicle exterior is in acceptable condition; no inspector action required.",
    },
    "MINOR": {
        "label": "MINOR COSMETIC",
        "color_hex": "#95C73E",
        "description": "Surface-level scratches, light scuffs, or paint transfer. Touch-up bay can resolve.",
    },
    "MODERATE": {
        "label": "MODERATE DAMAGE",
        "color_hex": "#E8A317",
        "description": "Paint chips, small dents, or edge rash beyond touch-up scope. Body shop repair likely.",
    },
    "MAJOR": {
        "label": "MAJOR DAMAGE",
        "color_hex": "#E66B2A",
        "description": "Panel damage, broken lighting assemblies, large dents, or deep gouges. Significant repair required.",
    },
    "CRITICAL": {
        "label": "CRITICAL DAMAGE",
        "color_hex": "#D94040",
        "description": "Structural concerns, broken glass, cracked frame elements, or multi-panel damage. Vehicle should not be accepted.",
    },
}

# -- Disposition playbook ----------------------------------------------------
# Replaces the USDA program-recommendation lookup. Two concrete actions per
# disposition -- enough to read as a real terminal-ops playbook without
# overbuilding.
DISPOSITION_PLAYBOOK = {
    "RELEASE": {
        "label": "RELEASE TO OUTBOUND YARD",
        "actions": [
            "Release vehicle to outbound yard / dealer transport queue",
            "No exception report required; routine acceptance",
        ],
    },
    "TOUCH_UP": {
        "label": "ROUTE TO TOUCH-UP BAY",
        "actions": [
            "Route to on-site touch-up bay for cosmetic correction",
            "Log entry in daily exception report; release post-repair",
        ],
    },
    "BODY_SHOP_HOLD": {
        "label": "HOLD FOR BODY SHOP",
        "actions": [
            "Hold in damaged-vehicle holding area pending body shop estimate",
            "Notify receiving dealer of estimated delivery delay (5-10 business days)",
        ],
    },
    "CARRIER_CLAIM": {
        "label": "FILE CARRIER DAMAGE CLAIM",
        "actions": [
            "Document damage with photographs and panel diagram for carrier claim",
            "Hold pending claim settlement; coordinate with carrier loss adjuster",
        ],
    },
    "REJECT": {
        "label": "REJECT ACCEPTANCE",
        "actions": [
            "Reject acceptance; initiate return-to-origin (RTO) process",
            "Notify OEM logistics, quality assurance, and receiving dealer",
        ],
    },
}


# -- FastAPI app + lifespan --------------------------------------------------
# Use a lifespan handler rather than the deprecated @app.on_event hooks.

FRONTEND_DIR = os.environ.get("FRONTEND_DIR", "/app/frontend")
SAMPLES_DIR = os.environ.get("SAMPLES_DIR", "/app/sample-images")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    host_ip = os.environ.get("HOST_IP", "")
    banner = "\n" + "=" * 66 + "\n"
    banner += "  Port-of-Entry Vehicle Inspection Analyst\n"
    banner += "  Qwen3-VL-8B-Instruct-FP8 | HP ZGX Nano | vLLM\n"
    banner += "  Compliance by Architecture: 100% on-prem inference\n"
    banner += "=" * 66 + "\n"
    if host_ip:
        banner += f"\n  \u27a1  http://{host_ip}:{PORT}\n"
    else:
        banner += f"\n  \u27a1  http://localhost:{PORT}\n"
    banner += "=" * 66 + "\n"
    print(banner)
    logger.info("Service started | model=%s | vllm=%s", VLLM_MODEL, VLLM_BASE_URL)
    app.state.http_client = httpx.AsyncClient(timeout=VLLM_TIMEOUT)
    try:
        yield
    finally:
        # Shutdown
        await app.state.http_client.aclose()
        logger.info("Service shutting down")


app = FastAPI(
    title="Port-of-Entry Vehicle Inspection Analyst",
    description="VLM-powered damage triage for RoRo vehicle imports",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(FRONTEND_DIR)
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
else:
    logger.warning("Frontend directory not found at %s", FRONTEND_DIR)

samples_path = Path(SAMPLES_DIR)
if samples_path.exists():
    app.mount("/samples", StaticFiles(directory=samples_path), name="samples")
else:
    logger.warning("Samples directory not found at %s", SAMPLES_DIR)


# -- VLM prompt --------------------------------------------------------------
# Single-call prompt, labeled-line output. The persona is a port inspector;
# the tiers and disposition options are spelled out so the model maps
# observations onto canonical categories.
INSPECTION_PROMPT = """You are a senior vehicle damage inspector at {port_name} ({port_code}), {terminal}. You assess finished vehicles offloaded from RoRo (roll-on/roll-off) carriers and triage exterior damage so terminal operations can release, repair, or reject each unit.

IMAGE
You are looking at one photograph submitted by a yard inspector. It may show a full vehicle, a partial view, or a close-up of a panel or area.

DAMAGE SEVERITY TIERS
- NONE: No visible exterior damage
- MINOR: Surface-level scratches, light scuffs, paint transfer; touch-up bay can resolve
- MODERATE: Paint chips, small dents, edge rash, scuffed bumpers; body shop repair likely
- MAJOR: Panel damage, ANY cracked or broken lighting assembly (headlight, taillight, fog light, turn signal lens), large dents, deep gouges, torn trim, damaged grille
- CRITICAL: Visibly crumpled, crushed, or buckled sheet metal (hood, fender, quarter panel) consistent with impact; deformed/misaligned bumper assembly from collision (not surface dent); frontend or rear-end collision damage; cracked windshield or window glass; cracked frame or structural members; missing safety equipment; three or more panels with damage from the same impact event; estimated affected area greater than 25 percent

SEVERITY ASSIGNMENT RULES
- If you observe damage that matches an example in MAJOR or CRITICAL, your DAMAGE_SEVERITY must be at least that tier. Do not downgrade.
- Cracked or fractured headlight/taillight lenses are MAJOR even if the rest of the vehicle is clean. They allow water ingress and require full assembly replacement.
- A crumpled, crushed, or deformed hood, fender, or bumper consistent with impact is CRITICAL, not MAJOR. "Significant denting" or "crumpling" of sheet metal is collision damage, not body shop damage.
- If three or more panels show damage from a single impact event, severity is CRITICAL.
- If estimated affected area is greater than 25 percent, severity is CRITICAL.
- A single severe finding outranks many minor findings. Use the highest tier any single observation reaches.

DISPOSITION OPTIONS
- RELEASE: No damage observed; release to outbound yard
- TOUCH_UP: Minor cosmetic only; route to on-site touch-up bay
- BODY_SHOP_HOLD: Moderate damage; hold for body shop estimate
- CARRIER_CLAIM: Damage attributable to ocean transport or carrier handling; file carrier damage claim
- REJECT: Collision damage, crumpled or crushed sheet metal, or damage where likely repair cost approaches or exceeds vehicle value; vehicle should not be accepted; initiate return-to-origin. CRITICAL severity should always map to REJECT.

OUTPUT FORMAT
Answer each prompt on its own line in exactly this format. Be specific to what you actually observe in the image -- do not generalize.

VEHICLE_TYPE: [Describe what you see, e.g., "Mercedes-Benz E-Class sedan, silver" or "Partial view of vehicle rear quarter, silver"]
VIEW_ANGLE: [One of: FRONT, REAR, DRIVER_SIDE, PASSENGER_SIDE, FRONT_QUARTER, REAR_QUARTER, CLOSE_UP_PANEL]
DAMAGE_SEVERITY: [One of: NONE, MINOR, MODERATE, MAJOR, CRITICAL]
AFFECTED_PANELS: [Comma-separated list using standard panel terms (hood, front bumper, headlight assembly, grille, driver door, rear bumper, tail light, trunk lid, etc.). If no damage, write NONE.]
DAMAGE_OBSERVATIONS: [Specific damage description per affected area: type (scratch, dent, paint chip, crack, missing trim, paint peeling, etc.), location, approximate size. If no damage, write "No exterior damage observed".]
ESTIMATED_AFFECTED_AREA_PERCENT: [Approximate percentage of visible vehicle surface showing damage. Single number followed by %. Use 0% if none.]
DISPOSITION: [One of: RELEASE, TOUCH_UP, BODY_SHOP_HOLD, CARRIER_CLAIM, REJECT]
INSPECTOR_SUMMARY: [Two to three sentence summary of vehicle condition, damage, and recommended action, written for a terminal operations supervisor. Describe only what you actually observed. Do not append boilerplate negative findings (e.g., "no other damage observed", "no broken glass observed") unless you have specifically inspected for them. If you noted a cracked lighting assembly above, your summary must reflect that as a significant finding, not minimize it.]"""


# -- vLLM interaction --------------------------------------------------------

async def query_vlm(image_b64: str, prompt: str, max_tokens: int = 1024) -> tuple[str, dict]:
    """Send a single image + prompt to Qwen3-VL. Returns (text, usage_dict)."""
    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "chat_template_kwargs": {"enable_thinking": False},
    }

    t0 = time.time()
    try:
        logger.info("Submitting VLM call (max_tokens=%d)...", max_tokens)
        response = await app.state.http_client.post(
            f"{VLLM_BASE_URL}/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        elapsed = time.time() - t0
        logger.info(
            "vLLM call complete in %.2fs (prompt=%d, completion=%d, total=%d)",
            elapsed,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )
        return content, usage
    except httpx.HTTPStatusError as e:
        logger.error("vLLM HTTP error %d: %s", e.response.status_code, e.response.text[:500])
        raise
    except Exception as e:
        logger.error("vLLM call failed after %.2fs: %s", time.time() - t0, e)
        logger.debug(traceback.format_exc())
        raise


# -- VLM output parsing ------------------------------------------------------

PARSED_FIELDS = [
    "vehicle_type",
    "view_angle",
    "damage_severity",
    "affected_panels",
    "damage_observations",
    "estimated_affected_area_percent",
    "disposition",
    "inspector_summary",
]


def parse_vlm_output(raw_response: str) -> dict:
    """Parse labeled-line VLM output into a dict.

    Tolerant of minor formatting drift -- matches by label prefix only,
    case-insensitive. Missing fields get explicit "Not provided" sentinels.
    """
    parsed = {field: "Not provided by model" for field in PARSED_FIELDS}

    for line in raw_response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for field in PARSED_FIELDS:
            tag = field.upper() + ":"
            if line.upper().startswith(tag):
                value = line[len(tag):].strip()
                if value:
                    parsed[field] = value
                break

    logger.info(
        "Parsed VLM output: %s",
        {k: (v[:60] + "..." if len(v) > 60 else v) for k, v in parsed.items()},
    )
    return parsed


# -- Classification ----------------------------------------------------------

def classify_severity(parsed: dict) -> tuple[str, dict]:
    """Map the VLM's DAMAGE_SEVERITY output to one of the canonical tiers.

    Match in order from most-severe to least so ambiguous responses bias
    toward more-conservative (more-damage) classification.
    """
    raw = (parsed.get("damage_severity") or "").upper()
    for tier in ["CRITICAL", "MAJOR", "MODERATE", "MINOR", "NONE"]:
        if tier in raw:
            return tier, SEVERITY_TIERS[tier]
    logger.warning("Could not classify severity from: '%s', defaulting to MINOR", raw)
    return "MINOR", SEVERITY_TIERS["MINOR"]


def resolve_disposition(parsed: dict, severity_key: str) -> dict:
    """Resolve disposition. Trust the model first; on miss, default by severity."""
    raw = (parsed.get("disposition") or "").upper()
    for key in DISPOSITION_PLAYBOOK:
        if key in raw:
            return {
                "key": key,
                "label": DISPOSITION_PLAYBOOK[key]["label"],
                "actions": DISPOSITION_PLAYBOOK[key]["actions"],
            }

    # Fallback: severity-driven default
    fallback_map = {
        "NONE": "RELEASE",
        "MINOR": "TOUCH_UP",
        "MODERATE": "BODY_SHOP_HOLD",
        "MAJOR": "BODY_SHOP_HOLD",
        "CRITICAL": "REJECT",
    }
    key = fallback_map.get(severity_key, "BODY_SHOP_HOLD")
    logger.info("Disposition defaulted to %s based on severity %s", key, severity_key)
    return {
        "key": key,
        "label": DISPOSITION_PLAYBOOK[key]["label"],
        "actions": DISPOSITION_PLAYBOOK[key]["actions"],
    }


def parse_area_percent(parsed: dict) -> float | None:
    """Extract a numeric percent from the affected-area field."""
    raw = parsed.get("estimated_affected_area_percent", "")
    if not raw or raw == "Not provided by model":
        return None
    digits = "".join(c for c in raw if c.isdigit() or c == ".")
    if not digits:
        return None
    try:
        value = float(digits)
        if 0 <= value <= 100:
            return value
    except ValueError:
        pass
    return None


# -- Helpers -----------------------------------------------------------------

def generate_report_id() -> str:
    """Generate a Vehicle Port Inspection (VPI) report ID."""
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    seq = "".join(random.choices(string.digits, k=4))
    return f"VPI-{date_str}-{seq}"


def prepare_image_for_vlm(image: Image.Image, max_dim: int = 1024) -> str:
    """Convert PIL image to base64 JPEG, resizing if needed.

    Qwen3-VL tokenizes vision input at roughly (pixels)/(14^2 * 4) tokens.
    A 1024x1024 image is ~1280 vision tokens, fitting comfortably with the
    prompt and output in an 8192-token context.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info("Resized image to %s", new_size)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -- Main pipeline -----------------------------------------------------------

async def run_inspection(
    image: Image.Image,
    port_key: str,
    vessel: str,
    vin: str,
) -> dict:
    """Full inspection pipeline.

    Steps:
      1. Preprocess image to base64.
      2. Build port context and assemble final prompt.
      3. Single VLM call.
      4. Parse labeled output into structured fields.
      5. Classify severity and resolve disposition.
      6. Assemble report.
    """
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("Starting inspection | port=%s | vessel=%s | vin=%s", port_key, vessel or "n/a", vin or "n/a")

    # 1. Preprocess
    image_b64 = prepare_image_for_vlm(image)

    # 2. Build prompt
    port = PORTS.get(port_key, PORTS["baltimore"])
    final_prompt = INSPECTION_PROMPT.format(
        port_name=port["name"],
        port_code=port["code"],
        terminal=port["terminal"],
    )

    # 3. VLM call
    raw_output, usage = await query_vlm(image_b64, final_prompt, max_tokens=1024)
    logger.debug("Raw VLM output:\n%s", raw_output)

    # 4. Parse
    parsed = parse_vlm_output(raw_output)

    # 5. Classify + disposition
    severity_key, severity_data = classify_severity(parsed)
    area_pct = parse_area_percent(parsed)
    disposition = resolve_disposition(parsed, severity_key)

    # 6. Assemble report
    report = {
        "report_id": generate_report_id(),
        "classification": "FOR DEMONSTRATION PURPOSES ONLY",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%d %b %Y %H%MZ").upper(),
        "port": {
            "key": port_key,
            "name": port["name"],
            "code": port["code"],
            "terminal": port["terminal"],
            "state": port["state"],
        },
        "vessel": vessel or "Not provided",
        "vin": vin or "Not provided",
        "vehicle": {
            "type": parsed["vehicle_type"],
            "view_angle": parsed["view_angle"],
        },
        "assessment": {
            "severity_key": severity_key,
            "severity_label": severity_data["label"],
            "severity_color": severity_data["color_hex"],
            "severity_description": severity_data["description"],
            "affected_panels": parsed["affected_panels"],
            "damage_observations": parsed["damage_observations"],
            "affected_area_percent": area_pct,
            "inspector_summary": parsed["inspector_summary"],
        },
        "disposition": disposition,
        "raw_vlm_output": raw_output,
        "token_usage": usage,
        "pipeline_seconds": round(time.time() - t0, 2),
    }

    logger.info(
        "Inspection complete in %.2fs | severity=%s | disposition=%s | area=%s%%",
        report["pipeline_seconds"],
        severity_key,
        disposition["key"],
        area_pct if area_pct is not None else "n/a",
    )
    logger.info("=" * 60)
    return report


# -- API routes --------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = Path(FRONTEND_DIR) / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>Vehicle Inspection Analyst</h1><p>index.html not found.</p>")


@app.get("/api/health")
async def health_check():
    """Health check. Also probes vLLM."""
    vllm_healthy = False
    vllm_error = None
    try:
        resp = await app.state.http_client.get(
            f"{VLLM_BASE_URL.replace('/v1', '')}/health",
            timeout=5.0,
        )
        vllm_healthy = resp.status_code == 200
    except Exception as e:
        vllm_error = str(e)

    return {
        "status": "healthy" if vllm_healthy else "degraded",
        "vllm_server": "ready" if vllm_healthy else "not ready",
        "vllm_error": vllm_error,
        "model": "Qwen3-VL-8B-Instruct-FP8",
        "inference_engine": "vLLM",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/inspect")
async def inspect_endpoint(
    image: UploadFile = File(..., description="Vehicle photograph"),
    port: str = Form("baltimore"),
    vessel: str = Form(""),
    vin: str = Form(""),
):
    """Run inspection on a single vehicle photograph."""
    if port not in PORTS:
        logger.warning("Unknown port '%s', defaulting to baltimore", port)
        port = "baltimore"

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload must be an image")

    try:
        data = await image.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large (max 20MB)")
        pil_image = Image.open(io.BytesIO(data))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image loading failed: %s", e)
        raise HTTPException(status_code=400, detail=f"Could not load image: {e}")

    try:
        report = await run_inspection(
            image=pil_image,
            port_key=port,
            vessel=vessel.strip(),
            vin=vin.strip(),
        )
        return JSONResponse(content=report)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


@app.post("/api/inspect_sample")
async def inspect_sample_endpoint(
    sample: str = Form(..., description="Filename of a sample image under SAMPLES_DIR"),
    port: str = Form("baltimore"),
    vessel: str = Form(""),
    vin: str = Form(""),
):
    """Run inspection against a bundled sample image (demo convenience)."""
    if port not in PORTS:
        port = "baltimore"

    sample_path = samples_path / sample
    # Guard against path traversal
    try:
        sample_path = sample_path.resolve()
        samples_path_resolved = samples_path.resolve()
        if not str(sample_path).startswith(str(samples_path_resolved)):
            raise HTTPException(status_code=400, detail="Invalid sample path")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sample path")

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample not found: {sample}")

    try:
        pil_image = Image.open(sample_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open sample: {e}")

    try:
        report = await run_inspection(
            image=pil_image,
            port_key=port,
            vessel=vessel.strip(),
            vin=vin.strip(),
        )
        return JSONResponse(content=report)
    except Exception as e:
        logger.error("Pipeline failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Inspection failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
