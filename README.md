# MRI Evaluation Agent

An agent that reads MRI DICOM (`.dcm`) files and works toward analysis / diagnosis.
Built step by step so the design stays understandable and owned by me.

## Data

`mri_test_data/` holds 16 series folders, each containing numbered `.dcm` slices.
One patient study = many series = many slices. Scanner: GE Signa Pioneer, 3 T.
The data is anonymized (no patient info, blank SeriesDescription).

**Body part: BRAIN.** Identified sequences (from TE/TR/scanning-sequence + images):

| Series | Sequence (inferred)        | Plane    | Notes                         |
|--------|----------------------------|----------|-------------------------------|
| Seri1  | DWI (diffusion)            | Axial    | 54 = ~2 b-values stacked      |
| Seri2  | T1 (inversion recovery)    | Axial    |                               |
| Seri3  | T2                         | Axial    |                               |
| Seri4  | T2                         | Coronal  |                               |
| Seri5  | T1                         | Sagittal |                               |
| Seri6  | T2 FLAIR                   | Axial    | CSF suppressed (verified)     |
| Seri7  | 3D T1 (likely post-contrast)| Axial   | 1.39 mm, 152 slices           |
| Seri8  | DWI b-value / trace        | Axial    |                               |
| Seri9  | DWI b-value / trace        | Axial    |                               |
| Seri10 | stray single GR slice      | Axial    | ignore                        |
| Seri11 | 3D T1 volume               | Axial    | 2.31 mm, 158 slices           |
| Seri12–16 | Reformats (MPR) of 3D vol| Ax/Cor  | thickness 0.0 mm, derived     |

These labels are inferred and get confirmed/encoded as rules in Step 3.

## Code layout

`mri_read` is a proper Python package, split into a service layer and thin CLI
entry points so the agent-loop orchestrator (Step 5) can import the service
layer directly instead of shelling out to scripts:

```
src/
  mri_read/    # service logic — importable, no argparse/__main__ here
    paths/, mri/, dwi/, engine/, manifest/, qc/, agent/, analyze/,
    ollama_vision/, ollama_client/, claude_vision/, visualize/, explore/,
    config/
    # each is a subpackage (folder + __init__.py), split by responsibility
    # into files inside it — e.g. mri/ separates disk-reading (loading.py)
    # from tag normalization (tags.py) from windowing (windowing.py). Each
    # __init__.py re-exports the subpackage's public names, so callers
    # always do `from mri_read.mri import load_series`, never reach into
    # the internal files directly.
  cmd/         # thin CLI wrappers, one per subpackage, e.g.:
    explore.py, visualize.py, manifest.py, qc.py, analyze.py, agent.py, dwi.py
tests/
  mri_read/    # mirrors src/mri_read/'s layout 1:1, one test_*.py per source file
```

`pyproject.toml` makes `mri_read` pip-installable (`pip install -e .`), so both
the `cmd/` scripts and any future orchestrator can `import mri_read.x` from
anywhere without path hacks.

## Roadmap

Each step produces something runnable before moving on. Don't skip ahead.

- [x] **Step 1 — Explore.** Read every DICOM header, group by series, understand
  what's in the data (modality, sequence, dimensions, slice count, geometry).
  Deliverable: `src/mri_read/explore.py` (CLI: `src/cmd/explore.py`).
- [x] **Step 2 — Load & visualize.** Turn a series into a clean 3D numpy volume
  and export slices as PNG. Deliverables: `src/mri_read/mri.py`,
  `src/mri_read/visualize.py` (CLI: `src/cmd/visualize.py`).
- [x] **Step 3 — Sequence classifier + manifest.** Rule-based labeling of each
  series (T1/T2/FLAIR/DWI/3D-T1/reformat) from TE/TR/scanning-sequence, emitted
  as `output/manifest.json`. Deliverable: `src/mri_read/manifest.py`
  (CLI: `src/cmd/manifest.py`).
- [x] **Step 3b — QC checks (deterministic).** Per-series flags — missing/uneven
  slices, low contrast, low SNR, empty slices — written into the manifest.
  Deliverable: `src/mri_read/qc.py` (CLI: `src/cmd/qc.py`).
- [x] **Step 4 — Analysis (local).** Swappable engine interface (`engine.py`);
  default is a fully local Ollama vision engine (`ollama_vision.py`). Fixed
  orchestrator (`analyze.py`) reads the manifest, selects slices, writes
  `output/report.md` + `report.json` — now a development/debugging tool, see
  Step 5. A specialized model can slot in behind the same interface later.
  CLI: `src/cmd/analyze.py`.
- [x] **Step 5 — Agent: deterministic analyze + LLM synthesis.** The primary
  way to run the project. No model decides what to look at anymore — the
  deterministic manifest -> qc -> analyze sequence always runs in full (one
  series per sequence type, every series QC'd), then a text-reasoning model
  reads that structured output and writes the final concise report. Builds
  the manifest itself and runs in Docker (`docker-compose.yml`) by default.
  Deliverable: `src/mri_read/agent.py` (CLI: `src/cmd/agent.py`).

## Running it (Step 5 — the agent)

`src/cmd/agent.py` is the one command to run for a report. It always: builds
the manifest, runs QC on every series, sends every primary series (one per
sequence type) to the vision engine in a single call, then hands the
resulting manifest + QC + vision findings — never the images — to a
text-reasoning model that synthesizes the final concise impression.

```bash
ollama serve
python src/cmd/agent.py
python src/cmd/agent.py --model meditron:7b --engine ollama --vision-model llava:13b
```

Two different local models are involved, each doing only what it's suited
for: a **vision** model (`OLLAMA_MODEL`, default `qwen2.5vl:7b` — switched
from `llava:13b`; see the Performance section below for why) reads the
images and reports structured per-sequence findings; a separate **text**
model (`OLLAMA_AGENT_MODEL`, default `meditron:7b` — a local medical-domain
fine-tune) reads the deterministic manifest/QC data plus those findings and
writes the final report. Neither model needs tool-calling support — the text
model's job is a single one-shot completion over data that's already fully
assembled, not deciding which series exist or what to inspect. `OLLAMA_HOST`
(default `http://localhost:11434`) is shared by both. All models auto-pull on
first use. Swap `OLLAMA_AGENT_MODEL` for another local medical fine-tune
(`medllama2`) or a stronger general model (`llama3.1:8b`, `qwen2.5:7b`) if
you have the RAM/GPU for it — this machine is CPU-only, which is why the
defaults here are 7-8B-class rather than 70B.

Hardening in the analysis path: volume-level windowing (comparable slices),
content-aware slice selection (skips near-empty slices), and DWI handling
(high-b stack + computed ADC map).

To add a specialized brain-MRI model later, implement `AnalysisEngine.analyze()`
in a new file and register it in `get_engine()` — nothing else changes.

### Performance: CPU-only vision inference is slow — read this if a run seems stuck

**If `agent.py` appears to hang or times out, this is very likely why, and it's
not a bug.** Measured on this project's own CPU-only dev machine with the
*original* default, `llava:13b`: a single per-series vision call (4 images)
took **434.6 seconds** — over 7 minutes — for one sequence. `run_agent()`
makes one such call per selected sequence type (typically 5: DWI, T1(IR), T2,
T2 FLAIR, 3D T1), so a full run's vision-analysis phase alone could cost
**30-40+ minutes** before text-synthesis even starts, and the default
`--vision-timeout` (600s) was close enough to that measured number that
ordinary variance (more images, system load, a longer reply) genuinely
exceeded it. This is inherent to running a vision-language model on hardware
with no GPU — not something a code fix alone resolves.

Two more things confirmed by digging into a real stuck run: (1) `llava:13b`'s
vision tower was `CLIP-ViT-L/14-336` — a **fixed 336×336** input resolution,
architecturally incapable of using anything beyond that no matter what size
image you send it (this project sends full-resolution slices, e.g.
1024×1024, which were being silently downsampled by the model anyway — pure
wasted transfer/decode cost, not a source of lost detail, since the model was
never going to see more than 336×336 in the first place). (2) Neither vision
nor synthesis call originally bounded reply length (`num_predict`); a
generation that never reaches a natural stop token has no cap other than the
context window, and since Ollama here runs a single processing slot (`-np
1`), one stuck generation blocks every subsequent series from ever being
served — explaining a run where ALL FIVE series time out, not just a slow
one. The `num_predict` cap is now fixed in code regardless of model (see
`CLAUDE.md`'s Performance section); the fixed-336-resolution problem is
addressed by switching the default to `qwen2.5vl:7b`, below.

**Current default is `qwen2.5vl:7b`, not `llava:13b`.** `llava:13b`'s
CLIP-336 ceiling meant this project was sending 1024×1024 images for zero
benefit; `qwen2.5vl` uses native dynamic resolution instead of a fixed crop,
so it can actually use more of what's sent — but that cuts both ways:
unlike `llava`, oversized input to `qwen2.5vl` isn't free-and-discarded, it's
*more compute*. Measured directly on this machine: a **single** real image
took **169-232s** depending on size — dominated by Ollama's
`--image-min-tokens 1024` floor (a 256×256 image already cost ~1049 tokens;
a 1024×1024 one cost ~1394, not a proportional 16x increase) rather than
actual resolution. A real call sends 4 images in one combined prompt, so
this is *worse* per-image than `llava:13b` was, not better —
`OllamaVisionEngine`'s default `--vision-timeout` is 1500s (up from 600s)
specifically to give this model's real measured cost some margin, not a
guess. If you don't want to resize/otherwise touch the image data (this
project's own preference, since `--image-min-tokens` makes resizing
unlikely to help much anyway — see `CLAUDE.md`), the two levers that don't
touch image content are the timeout (already raised) and `--slices` (item 4
below).

What actually helps, roughly in order of impact:

1. **Use a GPU if you have one.** By far the biggest lever — CPU inference on
   a model this size is the whole problem. If Ollama can see a GPU, point
   `OLLAMA_HOST` at that instance instead of a CPU-only one. With a GPU, much
   larger/more capable models become practical — worth trying instead of a
   small CPU-oriented one:
   ```bash
   python src/cmd/agent.py --vision-model qwen2.5vl:32b     # bigger version of the current default
   python src/cmd/agent.py --vision-model llama3.2-vision:90b
   python src/cmd/agent.py --vision-model llava:34b         # same CLIP-336 ceiling as llava:13b, just a bigger LLM half
   ```
   (Availability/exact tags depend on what's current in Ollama's library —
   check `ollama pull <name>` works before wiring it in.)
2. **If `qwen2.5vl:7b` is still too slow on CPU**, `llava:7b`/`bakllava`/
   `llava-phi3` are all CLIP-336 under the hood (same "resolution above 336
   is wasted" property `llava:13b` had — smaller only helps the
   language-model half, not vision encoding) but meaningfully smaller;
   `minicpm-v` is architecturally closer to `qwen2.5vl` (adapts to the
   image rather than the image adapting to a fixed crop) and built
   specifically for efficient/edge deployment:
   ```bash
   python src/cmd/agent.py --vision-model llava:7b        # smaller, fixed-336 tradeoff
   python src/cmd/agent.py --vision-model minicpm-v        # built for efficient/edge (CPU) deployment, adaptive high-res handling
   python src/cmd/agent.py --vision-model moondream       # smallest/fastest of the bunch, but back to a fairly fixed internal resolution
   ```
   None of these are pulled by default in this project's dev environment, and
   compatibility with a given Ollama version isn't guaranteed —
   `llama3.2-vision` (`mllama` architecture) is already known NOT to work on
   the Ollama version this was developed against; verify a new model
   actually loads and responds before relying on it. Smaller/different models
   also trade off analysis quality for speed — re-check the
   `flags`/`observations` in `output/report.md` before assuming a faster
   model is "good enough."
3. **Raise the timeouts further if 1500s/900s still isn't enough.** Those are
   the current defaults (vision/synthesis respectively) — already raised
   once based on real measurement, not a starting guess — but a slower
   machine or a heavier model swap may still need more:
   ```bash
   python src/cmd/agent.py --vision-timeout 2400 --synth-timeout 1200
   ```
4. **Reduce `--slices`** (default 4) to cut per-call image count, e.g.
   `--slices 2` — fewer images per call means less compute per call, at the
   cost of a coarser look at each sequence.

See `CLAUDE.md`'s Performance section for the full measured cost breakdown
and code-level fixes (deduped windowing computation, stall timeout on model
pulls, reply-length cap, and the 336×336 finding above).

## Running pipeline stages individually (development / debugging)

`manifest.py`, `qc.py`, and `analyze.py` (Steps 3, 3b, 4) are the deterministic
building blocks the agent loop calls into. They're no longer the primary way
to run the project, but each is still runnable standalone for testing a stage
in isolation — e.g. checking classification or QC output without spending time
on a full agent run:

```bash
ollama serve                          # local model server
python src/cmd/manifest.py            # classify series -> manifest.json
python src/cmd/qc.py                  # add quality flags to the manifest
python src/cmd/analyze.py             # fixed pipeline, no LLM orchestrator, one series per sequence type
python src/cmd/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

`analyze.py` builds the manifest/QC itself if they're missing, so it's runnable
on its own too — pipeline order manifest -> qc -> analyze is just how it fills
in the gaps, not a required sequence of commands.

## Privacy: everything runs locally

This is sensitive imaging data, so **no third-party LLM is used**. Analysis runs
against a local [Ollama](https://ollama.com) vision model; nothing leaves the
machine. The Claude engine in the repo is a non-local option kept only to show
the swappable interface — don't use it on this data.

## Running with Docker (recommended)

```bash
docker compose up --build
```

Two local services start: `ollama` (model server; weights persist in a named
volume, pulled at runtime — not baked into any image) and `app` (our pipeline,
runs the agent loop by default), writing reports to `./output`. First run
downloads both the orchestrator and vision models once; later runs reuse them.

Ad-hoc commands:

```bash
docker compose run --rm app python src/cmd/visualize.py --deep
docker compose run --rm app python src/cmd/analyze.py     # fixed pipeline, no agent
```

## Important disclaimer

This is a research / engineering project. Output is **not** a medical diagnosis
and must not be used for clinical decisions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps    # registers the mri_read package for import
```

## Tests

```bash
pip install -r requirements-dev.txt   # adds pytest on top of the runtime deps
pytest
```

Fast (~4s), safe to run anywhere: no disk, no network by default. Two
marker-gated categories opt in to more: `@pytest.mark.data` (needs
`mri_test_data/` on disk, auto-skipped if absent) and `@pytest.mark.ollama`
(real CPU inference against a live local Ollama server — **minutes** per
test, always skipped unless you pass `pytest --run-ollama`). See
`CLAUDE.md`'s Architecture section for more.

## Running step 1

```bash
python src/cmd/explore.py
```
