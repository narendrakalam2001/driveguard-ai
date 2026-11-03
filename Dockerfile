# DriveGuard AI - FastAPI serving image (CPU).
# Build:  docker build -t driveguard-ai .
# Run:    docker run -p 8000:8000 --env-file .env driveguard-ai
# NOTE: not built in the authoring environment - run the build once locally before relying on it.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (the default PyPI wheel drags in ~2 GB of CUDA libraries).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p models data/knowledge_base violation_logs evidence_frames chromadb_store logs \
    && useradd --create-home app && chown -R app:app /app
USER app

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app API_HOST=0.0.0.0 API_PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Fetch weights (if HF_MODEL_REPO is set), build the RAG store if missing, then serve.
CMD ["sh", "-c", "python scripts/download_models.py; [ -f chromadb_store/chroma.sqlite3 ] || python scripts/ingest_knowledge_base.py || true; python scripts/run_api.py"]
