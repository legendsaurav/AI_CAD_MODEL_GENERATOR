# ──────────────────────────────────────────────────────────────────────────
#  cad-planner  —  Production Dockerfile  (multi-stage)
# ──────────────────────────────────────────────────────────────────────────
#  Build:  docker build -t cad-planner .
#  Run:    docker run -p 8001:8001 cad-planner
# ──────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="ai-cad-os" \
      description="CAD Planner — GGL-to-CAL beam-search planning engine"

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8001/health"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]