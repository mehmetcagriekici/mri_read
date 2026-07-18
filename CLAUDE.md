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

Run the agent loop — this is the primary entry point (needs `ollama serve` running locally):

```bash
python src/cmd/agent.py
python src/cmd/agent.py --model meditron:7b --engine ollama --vision-model llava:13b
```

Run the fixed pipeline instead — development/debugging only, for testing a stage in isolation without an LLM orchestrator in the loop (`analyze.py` builds manifest/QC itself if missing, so each is independently runnable):

```bash
python src/cmd/manifest.py            # classify series -> output/manifest.json
python src/cmd/qc.py                  # add quality flags to the manifest
python src/cmd/analyze.py             # local ollama, one series per sequence type
python src/cmd/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

Docker (recommended for the full stack — starts an `ollama` service too; runs the agent loop by default):

```bash
docker compose up --build
docker compose run --rm app python src/cmd/visualize.py --deep   # ad-hoc commands
docker compose run --rm app python src/cmd/analyze.py            # fixed pipeline, no agent
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
- `mri_read/ollama_client.py` — the only place that talks HTTP (stdlib `urllib`) to a local Ollama server (`post`, `model_present`, `ensure_model`, `parse_json_reply`). Both `ollama_vision.py` (image analysis) and `agent.py` (the text-reasoning synthesis pass) depend on it.
- `mri_read/mri.py` — the only implementation of DICOM reading, geometric slice ordering, rescale-slope/intercept, and windowing. Every other pipeline module (`explore`, `visualize`, `manifest`, `qc`, `dwi`, `analyze`) imports from here. `explore.py` is the one deliberate exception — it predates `mri.py` and does its own light header-only reading so it works standalone on a fresh copy of the data.

**Engine abstraction** (`mri_read/engine.py`): `AnalysisEngine` is an ABC with one method, `analyze(study_meta, series) -> AnalysisResult`. `get_engine(name)` is a factory with **lazy, per-branch imports** — choosing `"ollama"` never imports the `anthropic` SDK and vice versa. Adding a new engine means implementing `AnalysisEngine` in a new file and adding one branch to `get_engine()`; nothing else changes. **Local Ollama is the default and only sanctioned engine for real data** — the Claude engine (`claude_vision.py`) exists only to prove the interface is swappable and must not be pointed at real imaging data (see README "Privacy" section). Preserve this default when adding engines or CLI flags.

**Two ways the pipeline runs, sharing the same building blocks. The agent is the primary entry point (`src/cmd/agent.py`); the fixed pipeline is a development/debugging tool, not user-facing:**
1. Agent (`mri_read/agent.py`, run via `src/cmd/agent.py`): no model decides what to inspect anymore — `run_agent()` always builds the manifest, runs `run_qc()` on every series (folded into the manifest, mirroring `qc.py`), and calls `select_series()` (one series per sequence type, same as the fixed pipeline) followed by the vision engine's `.analyze()` call, which internally issues **one `/api/chat` request per series** (see Performance below) rather than one bundled call. Only once that deterministic output is fully assembled does an LLM get involved: a separate **text-reasoning model** (`OLLAMA_AGENT_MODEL`, default `meditron:7b`, a local medical-domain fine-tune) reads the manifest + QC + vision findings — never the images — and writes the final concise impression in one one-shot call (`_synthesize()`). This model needs no tool-calling support, since it isn't deciding anything, just summarizing data that's already complete. `write_report()` always runs, so every invocation produces `output/report.md`/`report.json`; `src/cmd/agent.py` also persists the qc-augmented manifest to `output/manifest.json` afterward. The vision model (`OLLAMA_MODEL`, default `llava:13b`) stays separate from the text model, same separation of concerns as before, just no longer for tool-calling reasons.
2. Fixed pipeline (dev/debug only): `manifest.py` (classify series from TE/TR/ScanningSequence physics) → `qc.py` (deterministic quality flags, folded into the manifest) → `analyze.py` (`select_series` filters by `use_for_analysis`/QC status, `build_series_images` does content-aware slice selection + volume-level windowing, hands slices to the engine, `write_report` persists `output/report.md` + `report.json`). `output/manifest.json` is the JSON contract between stages. `mri_read.agent` imports `select_series`/`write_report` straight from `mri_read.analyze` — the agent's deterministic stage and the fixed pipeline are the same code path, just wrapped differently; only the final synthesis pass is agent-specific.

**Performance — this machine is CPU-only, no GPU (see `mri_read/ollama_client.py` callers and `ollama ps`), so vision inference is the bottleneck. Two things matter here:**
- `OllamaVisionEngine.analyze()` (`mri_read/ollama_vision.py`) issues **one `/api/chat` call per series**, not one call bundling every selected series' slices together. The earlier bundled version sent every sequence type (DWI, T1, T2, T2 FLAIR, 3D T1 — 20+ images total, since DWI alone contributes high-b + ADC) in a single request; on CPU that single forward pass routinely exceeded even a generous timeout, with zero progress feedback (`stream=False`) — indistinguishable from a hang. Splitting per series keeps each call's image count small (2-4 slices) and means one slow/failed sequence degrades the report (a flag noting the failure) instead of sinking the whole run. `--vision-timeout` (default 600) is now a **per-series** budget, not a whole-study budget.
- `mri_read.mri.load_series` is `functools.lru_cache`'d per process. In the agent's single-process run, `run_qc()` and `select_series()`/`build_series_images()` both need the same series' decoded pixels (QC for contrast/SNR, analysis for slice selection) — without the cache a 150+-slice series (e.g. 3D T1) gets read and decoded from disk twice back to back. The fixed-pipeline scripts (`manifest.py`/`qc.py`/`analyze.py`) don't benefit from this since each runs as a separate process, but that's expected — they're dev/debug tools, not the hot path. Cached `Series` objects share their numpy array across callers, so treat them as read-only.
- If you add a new engine or touch `select_series`/`build_series_images`, preserve the per-series call boundary — re-bundling multiple sequence types into one vision-model request is the specific thing that caused the original timeouts.

**DWI is special-cased throughout**: a DWI series stacks multiple b-values in one folder, so `dwi.py` splits by b-value and computes an ADC map when two b-values are available. `build_series_images` branches on `label == "DWI"` to route through `dwi.diffusion_views` instead of a plain `load_series` call.

**Config is env-var driven**: `OLLAMA_HOST` (default `http://localhost:11434`, shared by the vision engine and the agent), `OLLAMA_MODEL` (vision engine, default `llava:13b`), `OLLAMA_AGENT_MODEL` (agent's text-reasoning/synthesis model, default `meditron:7b` — swap for another local medical fine-tune like `medllama2`, or a stronger general model like `llama3.1:8b`/`qwen2.5:7b` if you have the RAM/GPU; this machine is CPU-only, which is why the defaults are 7-8B-class, not 70B). Docker sets `OLLAMA_HOST=http://ollama:11434` to reach the sibling `ollama` compose service; model weights are pulled at runtime into a persistent named volume, never baked into the image.

**Data**: `mri_test_data/` (real anonymized DICOM data) and `output/` (generated reports/montages) are both gitignored — never assume they're present or committed. The dataset is a single brain MRI study, GE Signa Pioneer 3T, 16 series folders (see README.md's series table for the currently-known layout: DWI, T1, T2, T2 FLAIR, 3D T1, and MPR reformats).
