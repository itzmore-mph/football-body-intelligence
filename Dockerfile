# Dockerfile — Football Body Intelligence SageMaker Processing Container
#
# Build:  docker build -t football-bi-processing .
# Push:   see pipelines/build_and_push.sh
#
# SageMaker Processing expects:
#   - Entry-point scripts at /opt/ml/code/
#   - Source code importable via PYTHONPATH
#   - Output written to /opt/ml/processing/output/

FROM python:3.11-slim-bookworm

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system deps (libgomp for numpy parallelism, ca-certs for TLS)
# Upgrade all system packages to pick up security patches, then install deps
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements-processing.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools "wheel>=0.46.2" \
    && pip install --no-cache-dir -r requirements-processing.txt

# Copy source package and entry-point scripts
COPY src/ ./src/
COPY scripts/ ./scripts/

# Make src importable without install (mirrors local dev setup)
ENV PYTHONPATH=/opt/ml/code

# SageMaker Processing overrides CMD at runtime with the script path,
# but setting a default makes local testing easier.
CMD ["python3", "scripts/run_awi_job.py"]
