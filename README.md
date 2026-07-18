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
    paths.py, mri.py, dwi.py, engine.py, manifest.py, qc.py, agent.py,
    analyze.py, ollama_vision.py, ollama_client.py, claude_vision.py,
    visualize.py, explore.py
  cmd/         # thin CLI wrappers, one per script, e.g.:
    explore.py, visualize.py, manifest.py, qc.py, analyze.py, agent.py, dwi.py
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
for: a **vision** model (`OLLAMA_MODEL`, default `llava:13b`) reads the
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

## Running step 1

```bash
python src/cmd/explore.py
```
