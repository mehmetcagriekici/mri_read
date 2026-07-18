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
ENV OLLAMA_MODEL=llama3.2-vision

# Default run: manifest -> QC -> local analysis.
CMD ["sh", "-c", "python src/cmd/manifest.py && python src/cmd/qc.py && python src/cmd/analyze.py"]
