# ============================================================
# Geometry Engine — Production Dockerfile
# ============================================================
# Stage 1: Builder
# ============================================================
FROM python:3.11-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# Install PyTorch CPU index (swap URL for cu121 etc. in GPU images)
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ============================================================
# Stage 2: Runtime
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="ai-cad-os" \
      description="Geometry Engine — GGL extraction from DiT hidden states"

# System deps for numpy / scipy compiled extensions
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
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

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8000/health"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]