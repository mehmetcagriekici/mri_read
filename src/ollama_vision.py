"""
Ollama vision engine — fully LOCAL AnalysisEngine implementation.

No data leaves the machine: slices are sent to a local Ollama server running a
local vision model (e.g. llama3.2-vision, qwen2.5vl). This is the default engine
for this project because the imaging data is sensitive.

Config (env or CLI):
  OLLAMA_HOST   default http://localhost:11434
  OLLAMA_MODEL  default llama3.2-vision

Only the standard library is used (urllib) so the container stays small.

NOT a diagnostic tool. Research/engineering prototype only.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request

from engine import AnalysisEngine, AnalysisResult, SeriesImages

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2-vision")

SYSTEM = """You are assisting a software engineer prototyping a LOCAL MRI-reading
pipeline. You are shown selected slices from a brain MRI, grouped and labeled by
sequence (T1, T2, FLAIR, DWI, 3D T1). Describe what is visible in a structured,
neutral, radiology-style way.

Rules:
- Research/engineering prototype, NOT clinical care. Frame everything as
  "visible on these images". Do not give patient instructions.
- Only comment on what the provided slices actually show. If slices are too few
  to judge, say so.
- Note asymmetries, signal abnormalities, mass effect, midline shift, restricted
  diffusion (DWI), or fluid-suppressed lesions (FLAIR) if visible.
- Be explicit about uncertainty.

Return ONLY valid JSON in this exact shape:
{
  "sequences_reviewed": ["T2 FLAIR", "DWI"],
  "observations": [
    {"sequence": "T2 FLAIR", "finding": "...", "location": "...", "confidence": "low|moderate|high"}
  ],
  "impression": "1-3 sentence overall summary",
  "flags": ["short phrases for anything notable or limiting"]
}"""


def _extract_json(text: str) -> dict:
    """Pull a JSON object out of the model's reply, tolerating extra text.

    Local models don't reliably return pure JSON — they add prose or wrap it in
    ```json fences. We strip a leading fence, then slice from the first "{" to
    the last "}" and parse that.
    """
    t = text.strip()
    if t.startswith("```"):                          # drop a ```json ... ``` fence
        t = t.split("```", 2)[1]
        if t.startswith("json"):
            t = t[4:]
    start, end = t.find("{"), t.rfind("}")           # outermost braces
    return json.loads(t[start:end + 1])


class OllamaVisionEngine(AnalysisEngine):
    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                 timeout: int = 900, auto_pull: bool = True):
        # host normalized (no trailing slash); timeout is generous because local
        # vision inference on CPU can take minutes.
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        if auto_pull:
            self._ensure_model()                     # fetch weights if missing

    # --- low-level HTTP helpers (stdlib urllib only, no 'requests' dependency) ---
    def _post(self, path: str, payload: dict, timeout: int | None = None) -> dict:
        """POST JSON to the Ollama server and return the parsed JSON reply."""
        req = urllib.request.Request(
            f"{self.host}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def _model_present(self) -> bool:
        """Is our model already downloaded? (GET /api/tags lists local models.)

        Also doubles as the connectivity check — if Ollama is unreachable we
        raise a clear error here rather than failing cryptically later.
        """
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=15) as r:
                tags = json.loads(r.read().decode()).get("models", [])
            # Compare on the base name (before any ":tag") so "llama3.2-vision"
            # matches "llama3.2-vision:latest".
            names = {m.get("name", "").split(":")[0] for m in tags}
            return self.model.split(":")[0] in names
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host} ({e}). "
                "Is the ollama server running?"
            ) from e

    def _ensure_model(self) -> None:
        """Pull the model into the local Ollama store if it's not there yet.

        This is why the Docker image stays small: weights are pulled at runtime
        into a persistent volume, not baked into the image.
        """
        if self._model_present():
            return
        print(f"Pulling local model '{self.model}' (one-time)...")
        # /api/pull streams progress lines; read to completion.
        req = urllib.request.Request(
            f"{self.host}/api/pull",
            data=json.dumps({"name": self.model, "stream": True}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=None) as r:
            for line in r:
                try:
                    status = json.loads(line).get("status", "")
                except json.JSONDecodeError:
                    continue
                if status:
                    print(f"  {status}", end="\r")
        print("\n  done.")

    # --- engine interface (implements AnalysisEngine.analyze) ---
    def analyze(self, study_meta: dict,
                series: list[SeriesImages]) -> AnalysisResult:
        # Build the chat: system rules, a study intro, then ONE message per
        # series carrying that sequence's label + its slice images. Keeping each
        # sequence's images in its own labeled message is how the model knows
        # which pictures are FLAIR vs DWI (Ollama attaches images per-message).
        messages = [{"role": "system", "content": SYSTEM}]
        messages.append({
            "role": "user",
            "content": (f"Study: {study_meta.get('body_part')} MRI on "
                        f"{study_meta.get('model')} at "
                        f"{study_meta.get('field_T')}T. "
                        f"{len(series)} sequences follow."),
        })
        for s in series:
            messages.append({
                "role": "user",
                "content": f"=== {s.label} ({s.plane}, {s.series}) — "
                           f"slice indices {s.slice_indices} ===",
                # Ollama wants base64 strings (no data-URI prefix) in "images".
                "images": [base64.b64encode(p).decode() for p in s.slice_pngs],
            })
        messages.append({
            "role": "user",
            "content": "Now return ONLY the JSON described in the system prompt.",
        })

        # stream=False -> one complete reply; low temperature -> steadier JSON.
        resp = self._post("/api/chat", {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2},
        })
        text = resp.get("message", {}).get("content", "")

        # If the model didn't return parseable JSON, keep its raw text as the
        # impression and flag it, rather than crashing.
        try:
            data = _extract_json(text)
        except Exception:                            # noqa: BLE001
            data = {"impression": text, "observations": [], "flags": ["unparsed"]}

        return AnalysisResult(
            engine=f"{self.name}:{self.model}",
            sequences_reviewed=data.get("sequences_reviewed",
                                        [s.label for s in series]),
            observations=data.get("observations", []),
            impression=data.get("impression", ""),
            flags=data.get("flags", []),
            disclaimer=("Local research/engineering prototype. NOT a medical "
                        "diagnosis. Not validated for clinical use."),
            raw=data,
        )
