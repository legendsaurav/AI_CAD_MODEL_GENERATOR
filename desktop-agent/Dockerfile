# ============================================================
# Desktop Agent — Production Dockerfile
# ============================================================
# NOTE: The desktop-agent typically runs on the host machine
# (not in a container) because it needs GUI access to CAD software.
# This Dockerfile is for CI testing and headless execution.
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="ai-cad-os" \
      description="Desktop Agent — CAL execution on CAD software"

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shared-schemas dependency
COPY --chown=appuser:appuser . .

USER appuser

# Desktop agent does not expose an HTTP server by default
# but can be started as a service for headless execution
CMD ["python", "pipeline.py"]