# App image only — small on purpose.
# The vision model is NOT baked in here; it is pulled at runtime by the local
# Ollama service into a persistent volume (see docker-compose.yml).
FROM python:3.11-slim

WORKDIR /app

# numpy + pillow (DICOM/image handling). Ollama is reached over HTTP (stdlib),
# so no LLM SDK is installed.
COPY requirements.txt .
RUN pip install --no-cache-dir pydicom numpy pillow

COPY src/ ./src/

# Talk to the Ollama service by its compose hostname; override if needed.
ENV OLLAMA_HOST=http://ollama:11434
ENV OLLAMA_MODEL=llama3.2-vision

# Default run: manifest -> QC -> local analysis.
CMD ["sh", "-c", "python src/manifest.py && python src/qc.py && python src/analyze.py"]
