# Vehicle Inspection Demo - Lessons

## Simplifications carried over from the USDA crop demo review

- **Two-image comparison plumbing** is not needed for inspection; single-image pipeline shaves ~150 lines.
- **Synthetic DMS centroid** (random lat/lon + degrees/minutes/seconds formatter) was pure theater; replaced with real vessel/VIN metadata that the user actually has.
- **6-region dict with bounding boxes** collapsed to a 4-port dict with just name/code/terminal/state. The prompt only injects 3 lines so the extra structure was unused.
- **4 hardcoded program actions per category** trimmed to 2 actions per disposition. Still reads as a real playbook, less to maintain.
- **Defensive fallback in `build_program_recommendations`** that re-infers from severity was kept (it's cheap and clean) but as a simple dict lookup, not nested if/elif.
- **`parse_stressed_area_percent`** stayed as a separate helper because it's reused in both the structured report and the log line.
- **Deprecated `@app.on_event` startup/shutdown** replaced with a `lifespan` context manager. Cleaner, single source of truth for the http_client.
- **Custom guidance form field** dropped. Adds UI surface and prompt branching for a feature no one uses in a 5-min demo.
- **`/api/regions` endpoint** dropped; frontend hardcodes the 4-port dropdown.

## Things explicitly preserved

- Labeled-line VLM output + tolerant prefix parser (rule: "labeled plain-text lines for VLM outputs more reliable than JSON schema").
- Robust logging from line 1, with timing on every stage.
- Severity-tier dict with hex colors for frontend rendering.
- Port preflight + shared-vLLM conflict check in start.sh.
- Reusing `~/.cache/vllm` so CUDA kernel compilation persists between restarts.

## Open questions to revisit after smoke test

- Does Qwen3-VL-8B reliably emit the canonical disposition keys (RELEASE / TOUCH_UP / BODY_SHOP_HOLD / CARRIER_CLAIM / REJECT), or is the fallback firing often? If it fires often, tighten the prompt with an explicit "Pick exactly one" line.
- Headlight assembly damage tends to be a hot button -- does the model bias too low on severity (calling it MODERATE when it should be MAJOR)? Tune the tier descriptions if so.
- Bumper paint peeling -- does the model recognize it as paint damage, or just "scuff"? The MB paint damage sample is a good probe.

## vLLM 26.01 container - argument format changes
- `--limit-mm-per-prompt` no longer accepts `image=2` shorthand; it parses with `json.loads` and requires `'{"image": 1}'` (note: must be quoted for the shell).
- For single-image inspection workloads, omit the flag entirely; default behavior is sufficient.

## VLM prompt: severity rubric must be self-enforcing
- Listing "cracked lighting assembly" as a MAJOR example wasn't enough; the model would call out the crack and still grade MODERATE.
- Fix: add an explicit "if you observe X, severity must be at least Y; do not downgrade" rule. Phrase it as a hard constraint, not a hint.
- Also fix: VLMs default to appending boilerplate negative findings ("no broken glass observed"). Forbid this in the summary instructions or it produces self-contradictory reports.

## VLM prompt: CRITICAL tier needs photo-visible anchors, not invisible ones
- Original CRITICAL examples ("cracked frame members", "structural concerns") are invisible in a vehicle photo; model had no way to map observations to the tier.
- Fix: add visible-from-outside anchors -- crumpled sheet metal, deformed bumper from impact, crushed hood -- plus quantitative escalation thresholds (>25% area, 3+ panels from one event).
- Causal language ("consistent with impact", "from a single impact event") shifts the model from describing surfaces to reasoning about cause. This matters more than the tier list itself for collision cases.
