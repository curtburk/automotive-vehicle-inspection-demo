# Vehicle Inspection Demo - Todo

## Phase 0: Scaffolding (complete)
- [x] Spec written + reviewed against USDA crop reference
- [x] Backend slimmed: single-image pipeline, lifespan handler, simpler disposition lookup
- [x] Frontend: dark theme with MB-luxury gradient, MB star + HP logo header
- [x] Sample image quick-select (clean / headlight damage / paint damage)
- [x] Docker + compose on host port 8095
- [x] start.sh: port preflight, shared-vLLM conflict check, health poll

## Phase 1: Deploy + smoke test on the Nano
- [ ] Stop shared text vLLM on :8091 (per standing rule)
- [ ] `EXISTING_MODEL_PATH=<usda-demo path> ./download_models.sh` to symlink Qwen3-VL
- [ ] `./start.sh`, verify banner prints reachable URL
- [ ] Smoke test all three sample images, confirm severity + disposition look right
- [ ] Smoke test custom upload (Curtis's own photo if available)
- [ ] Verify pipeline timing under 10s per inspection

## Phase 2: Narrative polish
- [ ] Add a fourth sample: scratched door / minor cosmetic, to demonstrate TOUCH_UP path
- [ ] Replace placeholder vessel/VIN defaults in UI with plausible MB-style examples
- [ ] Capture a 60-second screen recording for the LinkedIn dev-facing post

## Phase 3: Extensions (if time)
- [ ] Multi-image inspection: accept 2-4 photos of the same vehicle (front, sides, rear), fold into one report
- [ ] Damage location overlay: ask VLM for bbox coords, draw on the image client-side
- [ ] PDF export of the inspection report (python-pptx pattern in reverse - use reportlab)
- [ ] Phoenix by Arize.ai eval integration (-2 to +2 scoring, same as competitive intel)

## Phase 4: LinkedIn post variants
- [ ] Developer-facing: "Single-call VLM inspection pipeline on the Nano, 8s end-to-end"
- [ ] Enterprise/buyer-facing: "Why finished-vehicle import photos shouldn't go to the cloud"
