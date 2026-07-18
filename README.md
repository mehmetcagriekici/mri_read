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
entry points so a future orchestrator (Step 5) can import the service layer
directly instead of shelling out to scripts:

```
src/
  mri_read/    # service logic — importable, no argparse/__main__ here
    paths.py, mri.py, dwi.py, engine.py, manifest.py, qc.py,
    analyze.py, ollama_vision.py, claude_vision.py, visualize.py, explore.py
  cmd/         # thin CLI wrappers, one per script, e.g.:
    explore.py, visualize.py, manifest.py, qc.py, analyze.py, dwi.py
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
  default is a fully local Ollama vision engine (`ollama_vision.py`). Orchestrator
  (`analyze.py`) reads the manifest, selects slices, writes `output/report.md` +
  `report.json`. Runs in Docker (`docker-compose.yml`). A specialized model can
  slot in behind the same interface later. CLI: `src/cmd/analyze.py`.
- [x] **Step 5 — Agent loop.** A tool-calling orchestrator model decides which
  series to inspect, QC, and analyze — replacing the fixed manifest -> qc ->
  analyze sequence with tools it can call in whatever order it judges useful.
  Deliverable: `src/mri_read/agent.py` (CLI: `src/cmd/agent.py`).

## Analysis engine (Step 4)

The orchestrator is engine-agnostic. Default engine is **local Ollama**.

Without Docker (needs Ollama installed and running locally):

```bash
ollama serve                          # local model server
python src/cmd/manifest.py            # classify series -> manifest.json
python src/cmd/qc.py                  # add quality flags to the manifest
python src/cmd/analyze.py             # local ollama, one series per sequence type
python src/cmd/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

Pipeline order is manifest -> qc -> analyze. Hardening in the analysis path:
volume-level windowing (comparable slices), content-aware slice selection
(skips near-empty slices), and DWI handling (high-b stack + computed ADC map).

Config via env: `OLLAMA_HOST` (default `http://localhost:11434`),
`OLLAMA_MODEL` (default `llama3.2-vision`). The model auto-pulls on first use.

To add a specialized brain-MRI model later, implement `AnalysisEngine.analyze()`
in a new file and register it in `get_engine()` — nothing else changes.

## Agent loop (Step 5)

An alternative to the fixed manifest -> qc -> analyze pipeline: a tool-calling
orchestrator model decides what to do. It never sees the images itself — it
calls `list_series`, `get_manifest`, `run_qc`, `analyze_series` (which hands
the chosen series to the vision engine from Step 4), and `write_report`, in
whatever order and repetition it judges useful, then summarizes.

```bash
ollama serve
python src/cmd/agent.py
python src/cmd/agent.py --model qwen2.5 --engine ollama --vision-model llama3.2-vision
```

The orchestrator model is deliberately a *different, smaller, text-only* model
from the vision model — it needs Ollama tool-calling support (e.g. `qwen2.5`,
`llama3.1`), which most vision models don't have. Config via env:
`OLLAMA_AGENT_MODEL` (default `qwen2.5`), `OLLAMA_HOST` (shared with Step 4).

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
volume, pulled at runtime — not baked into any image) and `app` (our pipeline).
The app builds the manifest and runs the analysis, writing reports to `./output`.
First run downloads the vision model once; later runs reuse it.

Ad-hoc commands:

```bash
docker compose run --rm app python src/cmd/visualize.py --deep
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
