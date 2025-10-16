# ---- Base image ----
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr and creating .pyc
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STANZA_RESOURCES_DIR=/usr/local/stanza_resources

WORKDIR /app

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (layer-cached) ----
# Copy only requirements first to leverage Docker layer cache
COPY requirements.txt /app/requirements.txt

# 1) Install CPU-only PyTorch explicitly (prevents CUDA/nvidia-* deps)
#    Adjust torch version as needed; this line is the key change.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu

# 2) Install the rest of your deps (stanza will now see torch CPU and won't pull CUDA)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download Stanza model for Indonesian into $STANZA_RESOURCES_DIR
RUN python - <<'PY'
import os, stanza
os.makedirs(os.environ.get("STANZA_RESOURCES_DIR","/usr/local/stanza_resources"), exist_ok=True)
stanza.download('id', model_dir=os.environ["STANZA_RESOURCES_DIR"])
PY

# ---- App code ----
COPY . /app

# Expose Flask/Gunicorn port
EXPOSE 3000

# Healthcheck (adjust path if your health endpoint differs)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:3000/ || exit 1

# ---- Run with Gunicorn ----
CMD ["gunicorn", "apiflask:app", \
     "--bind", "0.0.0.0:3000", \
     "--workers", "1", \
     "--threads", "1", \
     "--worker-class", "sync", \
     "--timeout", "120", \
     "--log-level", "info"]
