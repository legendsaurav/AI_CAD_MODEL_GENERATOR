# ──────────────────────────────────────────────────────────────────────────
#  desktop-agent  —  Dockerfile
# ──────────────────────────────────────────────────────────────────────────
#  NOTE: The desktop-agent normally runs directly on the host to access
#  the native GUI toolkit.  This image is provided for headless / CI use.
#
#  Build:  docker build -t desktop-agent .
#  Run:    docker run desktop-agent
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

LABEL maintainer="ai-cad-os" \
      description="Desktop Agent — headless/CI container (normally runs on host)"

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Install shared-schemas first (editable so imports resolve)
COPY --chown=appuser:appuser ../shared-schemas /app/shared-schemas
RUN pip install --no-cache-dir -e /app/shared-schemas || true

# Copy application code
COPY --chown=appuser:appuser . .

# Install any local dependencies
RUN pip install --no-cache-dir pydantic>=2.0.0 pyyaml>=6.0 || true

USER appuser

CMD ["python", "pipeline.py"]