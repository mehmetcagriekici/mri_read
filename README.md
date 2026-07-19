# MRI Read

An agent that reads MRI DICOM (`.dcm`) files and works toward local,
non-diagnostic analysis. Everything runs on-device against a local
[Ollama](https://ollama.com) server — no imaging data ever leaves the
machine. **This is a research/engineering prototype. Output is never a
medical diagnosis and must not be used for clinical decisions.**

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps    # registers the mri_read package for import

ollama serve                  # in another terminal
python src/cmd/agent.py
```

That's the whole loop: it builds a manifest of your `mri_test_data/` series,
runs deterministic QC, sends each sequence type to a local vision model, and
writes `output/report.md` + `output/report.json`. A first run also pulls the
default models, so expect a wait; see [Performance](#performance) below for
what to expect on CPU-only hardware.

## How it works

`src/cmd/agent.py` is the primary entry point — an agent loop, but a
deliberately narrow one: no model decides *what* to inspect. The
deterministic pipeline (`manifest` → `qc` → series selection) always runs in
full, and only once that structured output is fully assembled do two local
models get involved, each doing only what it's suited for:

- A **vision** model (`OLLAMA_MODEL`, default `qwen2.5vl:7b`) gets one
  `/api/chat` call per sequence type (DWI, T1, T2, T2 FLAIR, 3D T1 — not one
  bundled call for everything) and reports structured, schema-checked
  findings per series.
- A **text** model (`OLLAMA_AGENT_MODEL`, default `meditron:7b`, a local
  medical-domain fine-tune) never sees the images — it reads the manifest +
  QC flags + vision findings and writes the final concise impression in one
  one-shot call.

Before anything reaches the report, a deterministic hallucination guard
(no LLM involved — see `CLAUDE.md`) checks both models' output: unsupported
diagnostic-sounding claims that no other sequence corroborates get
suppressed (never silently upgraded to a confident claim), and DWI findings
lacking a structural (FLAIR/T1) correlate get flagged rather than dropped.
The design stance throughout: it's fine for the report to say nothing about
a finding, it is not fine for it to state an unsupported claim as
established.

```bash
python src/cmd/agent.py
python src/cmd/agent.py --model meditron:7b --engine ollama --vision-model llava:13b
python src/cmd/agent.py --vision-timeout 2400 --synth-timeout 1200 --slices 2
```

`OLLAMA_HOST` (default `http://localhost:11434`) is shared by both models;
all models auto-pull on first use. Swap `OLLAMA_AGENT_MODEL` for another
local medical fine-tune (`medllama2`) or a stronger general model
(`llama3.1:8b`, `qwen2.5:7b`) if you have the RAM/GPU for it — this project's
own dev machine is CPU-only, which is why the defaults are 7-8B-class rather
than 70B.

To add a new analysis engine, implement `AnalysisEngine.analyze()` in a new
subpackage and register it in `mri_read.engine.get_engine()` — nothing else
changes (see `CLAUDE.md`'s Engine abstraction section).

### Running pipeline stages individually (development / debugging)

`manifest.py`, `qc.py`, and `analyze.py` are the deterministic building
blocks the agent calls into. They're not the primary way to run the
project, but each is runnable standalone for testing a stage in isolation
without spending time on a full agent run:

```bash
python src/cmd/manifest.py            # classify series -> output/manifest.json
python src/cmd/qc.py                  # add quality flags to the manifest
python src/cmd/analyze.py             # fixed pipeline, no LLM orchestrator, one series per sequence type
python src/cmd/analyze.py --slices 5 --skip-qc-warn --model qwen2.5vl
```

`analyze.py` builds the manifest/QC itself if they're missing, so it's
runnable on its own — the manifest → qc → analyze order is just how it
fills in the gaps, not a required sequence of commands.

## Data

`mri_test_data/` (gitignored, not shipped with the repo) holds 16 series
folders, each containing numbered `.dcm` slices — one brain MRI study, GE
Signa Pioneer 3T, anonymized (no patient info, blank `SeriesDescription`).
Sequence types are inferred from TE/TR/scanning-sequence physics, not
trusted metadata:

| Series | Sequence (inferred)          | Plane    | Notes                      |
|--------|-------------------------------|----------|----------------------------|
| Seri1  | DWI (diffusion)                | Axial    | ~2 b-values stacked        |
| Seri2  | T1 (inversion recovery)        | Axial    |                            |
| Seri3  | T2                              | Axial    |                            |
| Seri4  | T2                              | Coronal  |                            |
| Seri5  | T1                              | Sagittal |                            |
| Seri6  | T2 FLAIR                        | Axial    | CSF suppressed (verified)  |
| Seri7  | 3D T1 (likely post-contrast)    | Axial    | 1.39 mm, 152 slices        |
| Seri8  | DWI b-value / trace             | Axial    |                            |
| Seri9  | DWI b-value / trace             | Axial    |                            |
| Seri10 | stray single GR slice           | Axial    | ignore                     |
| Seri11 | 3D T1 volume                    | Axial    | 2.31 mm, 158 slices        |
| Seri12–16 | Reformats (MPR) of 3D volume | Ax/Cor   | thickness 0.0 mm, derived  |

## Performance

**This is CPU-only hardware, no GPU — vision inference is genuinely slow,
not stuck.** A single per-series vision call has been measured at several
minutes on this project's own dev machine; a full 5-series run's
vision-analysis phase alone can take 30-60+ minutes. `--vision-timeout`
(default 1500s) and `--synth-timeout` are per-call budgets, already raised
once based on real measurement — if a run is timing out, that's the
dominant cause, not a bug. See `CLAUDE.md`'s Performance section for the
full measured cost breakdown and every code-level fix that came out of it
(reply-length caps, stall timeouts, deduped windowing computation, the
`llava:13b` → `qwen2.5vl:7b` model switch and why).

What actually helps, roughly in order of impact:

1. **Use a GPU if you have one.** By far the biggest lever. Point
   `OLLAMA_HOST` at a GPU-backed Ollama instance and a larger, more capable
   model becomes practical:
   ```bash
   python src/cmd/agent.py --vision-model qwen2.5vl:32b
   python src/cmd/agent.py --vision-model llama3.2-vision:90b
   python src/cmd/agent.py --vision-model llava:34b
   ```
   (Availability/exact tags depend on what's current in Ollama's library —
   check `ollama pull <name>` works before wiring it in.)
2. **If `qwen2.5vl:7b` is still too slow on CPU**, try a smaller model:
   `llava:7b`/`bakllava`/`llava-phi3` (fixed 336×336 CLIP vision tower —
   smaller only shrinks the language-model half), `minicpm-v` (adapts to the
   image like `qwen2.5vl` does, built for efficient/edge deployment), or
   `moondream` (smallest/fastest, back to a fairly fixed internal
   resolution). None of these are pulled by default; `llama3.2-vision`
   (`mllama` architecture) is already known not to work on the Ollama
   version this was developed against — verify a new model actually loads
   before relying on it, and re-check `output/report.md`'s
   `flags`/`observations` before assuming a faster model is "good enough."
3. **Raise `--vision-timeout`/`--synth-timeout` further** if the current
   defaults (1500s/900s) still aren't enough on a slower machine.
4. **Reduce `--slices`** (default 4) to cut per-call image count, at the
   cost of a coarser look at each sequence.

This project deliberately does not resize images to fit a smaller model's
input size — see `CLAUDE.md` for why (`--image-min-tokens` makes it
unlikely to help, and the project's own stance is to adapt code to data,
not the other way around).

## Running with Docker (recommended)

```bash
docker compose up --build
```

Two local services start: `ollama` (model server; weights persist in a
named volume, pulled at runtime, never baked into an image) and `app` (this
project, runs the agent loop by default), writing reports to `./output`.
First run downloads both models once; later runs reuse them.

```bash
docker compose run --rm app python src/cmd/visualize.py --deep
docker compose run --rm app python src/cmd/analyze.py     # fixed pipeline, no agent
```

## Privacy: everything runs locally

This is sensitive imaging data, so **no third-party LLM is used** for real
analysis. The default and only sanctioned engine is a local Ollama vision
model — nothing leaves the machine. A Claude-backed engine exists in the
repo only to prove the engine interface is swappable; it must not be
pointed at real imaging data.

## Tests

```bash
pip install -r requirements-dev.txt   # adds pytest on top of the runtime deps
pytest
```

Fast (~4s), safe to run anywhere — no disk, no network by default. Two
marker-gated categories opt in to more: `@pytest.mark.data` (needs
`mri_test_data/` on disk, auto-skipped if absent) and `@pytest.mark.ollama`
(real CPU inference against a live local Ollama server — minutes per test,
always skipped unless you pass `pytest --run-ollama`). See `CLAUDE.md`'s
Architecture section for the full package layout and design rationale.

## Disclaimer

This is a research/engineering project. Output is **not** a medical
diagnosis and must not be used for clinical decisions.
