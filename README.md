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

## Roadmap

Each step produces something runnable before moving on. Don't skip ahead.

- [x] **Step 1 — Explore.** Read every DICOM header, group by series, understand
  what's in the data (modality, sequence, dimensions, slice count, geometry).
  Deliverable: `src/explore.py`.
- [x] **Step 2 — Load & visualize.** Turn a series into a clean 3D numpy volume
  and export slices as PNG. Deliverables: `src/mri.py`, `src/visualize.py`.
- [x] **Step 3 — Sequence classifier + manifest.** Rule-based labeling of each
  series (T1/T2/FLAIR/DWI/3D-T1/reformat) from TE/TR/scanning-sequence, emitted
  as `output/manifest.json`. Deliverable: `src/manifest.py`.
- [x] **Step 3b — QC checks (deterministic).** Per-series flags — missing/uneven
  slices, low contrast, low SNR, empty slices — written into the manifest.
  Deliverable: `src/qc.py`.
- [x] **Step 4 — Analysis (local).** Swappable engine interface (`engine.py`);
  default is a fully local Ollama vision engine (`ollama_vision.py`). Orchestrator
  (`analyze.py`) reads the manifest, selects slices, writes `output/report.md` +
  `report.json`. Runs in Docker (`docker-compose.yml`). A specialized model can
  slot in behind the same interface later.
- [ ] **Step 5 — Agent loop.** Wrap the pieces as tools an LLM orchestrates
  (pick series → load → analyze → write report).

## Analysis engine (Step 4)

The orchestrator is engine-agnostic. Default engine is **local Ollama**.

Without Docker (needs Ollama installed and running locally):

```bash
ollama serve                          # local model server
python src/manifest.py                # classify series -> manifest.json
python src/qc.py                       # add quality flags to the manifest
python src/analyze.py                 # local ollama, one series per sequence type
python src/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

Pipeline order is manifest -> qc -> analyze. Hardening in the analysis path:
volume-level windowing (comparable slices), content-aware slice selection
(skips near-empty slices), and DWI handling (high-b stack + computed ADC map).

Config via env: `OLLAMA_HOST` (default `http://localhost:11434`),
`OLLAMA_MODEL` (default `llama3.2-vision`). The model auto-pulls on first use.

To add a specialized brain-MRI model later, implement `AnalysisEngine.analyze()`
in a new file and register it in `get_engine()` — nothing else changes.

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
docker compose run --rm app python src/visualize.py --deep
```

## Important disclaimer

This is a research / engineering project. Output is **not** a medical diagnosis
and must not be used for clinical decisions.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running step 1

```bash
python src/explore.py
```
