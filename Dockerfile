# App image only — small on purpose.
# The vision model is NOT baked in here; it is pulled at runtime by the local
# Ollama service into a persistent volume (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# numpy + pillow (DICOM/image handling). Ollama is reached over HTTP (stdlib),
# so no LLM SDK is installed.
COPY requirements.txt pyproject.toml .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Register the mri_read package (src/mri_read/) so `import mri_read.x` resolves
# from anywhere, e.g. the src/cmd/ scripts below. --no-deps: deps already came
# from requirements.txt above; pyproject.toml declares none of its own.
RUN pip install --no-cache-dir -e . --no-deps

# Talk to the Ollama service by its compose hostname; override if needed.
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=llava:13b

# Default run: the agent loop (see src/cmd/agent.py). It builds the manifest
# itself and drives QC/analysis/report via tool calls — manifest.py, qc.py,
# and analyze.py are still available individually for development/debugging
# (`docker compose run --rm app python src/cmd/analyze.py`), but are no
# longer the primary entry point.
CMD ["python", "src/cmd/agent.py"]
