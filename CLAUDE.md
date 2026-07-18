# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An agent that reads MRI DICOM (`.dcm`) files and works toward local, non-diagnostic analysis. Built step by step (see Roadmap in README.md) so the design stays understandable. **This is a research/engineering prototype — output is never a medical diagnosis and must not be used for clinical decisions.**

## Setup & commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps    # registers the mri_read package; required before running anything in src/cmd/
```

Run the fixed pipeline (needs `ollama serve` running locally):

```bash
python src/cmd/manifest.py            # classify series -> output/manifest.json
python src/cmd/qc.py                  # add quality flags to the manifest
python src/cmd/analyze.py             # local ollama, one series per sequence type
python src/cmd/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

Run the agent loop instead (a tool-calling LLM drives the pipeline itself):

```bash
python src/cmd/agent.py
python src/cmd/agent.py --model qwen2.5 --engine ollama --vision-model llama3.2-vision
```

Docker (recommended for the full stack — starts an `ollama` service too):

```bash
docker compose up --build
docker compose run --rm app python src/cmd/visualize.py --deep   # ad-hoc commands
```

Lint: `ruff check src/` (no ruff config file — defaults only).

There is currently no automated test suite. Verify changes by running the relevant `src/cmd/*.py` script against `mri_test_data/` and inspecting the printed output / `output/manifest.json` / `output/report.md`.

## Architecture

**Package split is the load-bearing structural decision in this repo:**
- `src/mri_read/` — the importable service layer. No `argparse`/`__main__` here, ever. This is what a future orchestrator (or the agent loop) imports directly.
- `src/cmd/` — thin CLI wrappers, one per service module. All argparse wiring, `print()` calls, and `output/*.json` file I/O live here, not in `mri_read/`.

`pyproject.toml` (`[tool.setuptools.packages.find]`, `where=["src"]`, `include=["mri_read*"]`) makes `mri_read` pip-installable in editable mode so `import mri_read.x` resolves from anywhere without path hacks — this is why `pip install -e . --no-deps` is a required setup step and is also run inside the Dockerfile after `COPY src/`.

**Single-source-of-truth modules** — when touching path resolution, Ollama HTTP calls, or DICOM reading, extend these rather than reimplementing locally:
- `mri_read/paths.py` — `ROOT`/`DATA_DIR`/`OUT`, computed once from `paths.py`'s own location. Every other module imports from here instead of recomputing `Path(__file__).resolve()...`.
- `mri_read/ollama_client.py` — the only place that talks HTTP (stdlib `urllib`) to a local Ollama server (`post`, `model_present`, `ensure_model`). Both `ollama_vision.py` (image analysis) and `agent.py` (tool-calling orchestration) depend on it.
- `mri_read/mri.py` — the only implementation of DICOM reading, geometric slice ordering, rescale-slope/intercept, and windowing. Every other pipeline module (`explore`, `visualize`, `manifest`, `qc`, `dwi`, `analyze`) imports from here. `explore.py` is the one deliberate exception — it predates `mri.py` and does its own light header-only reading so it works standalone on a fresh copy of the data.

**Engine abstraction** (`mri_read/engine.py`): `AnalysisEngine` is an ABC with one method, `analyze(study_meta, series) -> AnalysisResult`. `get_engine(name)` is a factory with **lazy, per-branch imports** — choosing `"ollama"` never imports the `anthropic` SDK and vice versa. Adding a new engine means implementing `AnalysisEngine` in a new file and adding one branch to `get_engine()`; nothing else changes. **Local Ollama is the default and only sanctioned engine for real data** — the Claude engine (`claude_vision.py`) exists only to prove the interface is swappable and must not be pointed at real imaging data (see README "Privacy" section). Preserve this default when adding engines or CLI flags.

**Two ways the pipeline runs, sharing the same building blocks:**
1. Fixed pipeline: `manifest.py` (classify series from TE/TR/ScanningSequence physics) → `qc.py` (deterministic quality flags, folded into the manifest) → `analyze.py` (`select_series` filters by `use_for_analysis`/QC status, `build_series_images` does content-aware slice selection + volume-level windowing, hands slices to the engine, `write_report` persists `output/report.md` + `report.json`). `output/manifest.json` is the JSON contract between stages.
2. Agent loop (`mri_read/agent.py`): a tool-calling LLM decides which of `list_series`, `get_manifest`, `run_qc`, `analyze_series`, `write_report` to call and in what order/repetition, instead of the fixed sequence above. `analyze_series` routes through `select_named_series` → the same `build_series_images` the fixed pipeline uses, so slice-selection logic isn't duplicated between the two modes. The orchestrator model is deliberately a **different, smaller, text/tool-calling model** (`OLLAMA_AGENT_MODEL`, default `qwen2.5`) from the vision model used inside `analyze_series` (`OLLAMA_MODEL`, default `llama3.2-vision`) — most vision models lack Ollama tool-calling support.

**DWI is special-cased throughout**: a DWI series stacks multiple b-values in one folder, so `dwi.py` splits by b-value and computes an ADC map when two b-values are available. `build_series_images` branches on `label == "DWI"` to route through `dwi.diffusion_views` instead of a plain `load_series` call.

**Config is env-var driven**: `OLLAMA_HOST` (default `http://localhost:11434`, shared by the vision engine and the agent), `OLLAMA_MODEL` (vision engine), `OLLAMA_AGENT_MODEL` (agent orchestrator). Docker sets `OLLAMA_HOST=http://ollama:11434` to reach the sibling `ollama` compose service; model weights are pulled at runtime into a persistent named volume, never baked into the image.

**Data**: `mri_test_data/` (real anonymized DICOM data) and `output/` (generated reports/montages) are both gitignored — never assume they're present or committed. The dataset is a single brain MRI study, GE Signa Pioneer 3T, 16 series folders (see README.md's series table for the currently-known layout: DWI, T1, T2, T2 FLAIR, 3D T1, and MPR reformats).
