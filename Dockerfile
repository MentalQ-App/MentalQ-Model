# ---- Base image ----
FROM python:3.9-slim

# Prevent Python from buffering stdout/stderr and creating .pyc
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Put Stanza resources in a fixed path so they’re cached in the image
    STANZA_RESOURCES_DIR=/usr/local/stanza_resources

WORKDIR /app

# System deps (only what we need)
# - gcc, g++ often NOT needed if wheels exist; remove if your deps all have wheels
# - libgomp1 sometimes required by numpy/scipy wheels; keep if you see runtime errors
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Python deps (layer-cached) ----
# Copy only requirements first, to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

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

# Optional: basic healthcheck pinging your /predict route (change if needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:3000/ || exit 1

# ---- Run with Gunicorn ----
# Use one worker and a couple of threads for a small VPS.
# If your inference is CPU-bound, threads=1 may be better.
CMD ["gunicorn", "apiflask:app", \
     "--bind", "0.0.0.0:3000", \
     "--workers", "1", \
     "--threads", "2", \
     "--worker-class", "gthread", \
     "--timeout", "120", \
     "--log-level", "info"]
