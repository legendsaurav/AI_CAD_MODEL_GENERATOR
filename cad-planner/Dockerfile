# ============================================================
# CAD Planner — Production Dockerfile
# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="ai-cad-os" \
      description="CAD Planner — GGL to CAL conversion with beam search"

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd --gid 1000 appuser && \
    useradd  --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8001/health"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]