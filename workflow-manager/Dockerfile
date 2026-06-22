# ============================================================
# Stage 1 — Builder: install Python deps into a venv
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY . .
RUN /opt/venv/bin/pip install --no-cache-dir .

# ============================================================
# Stage 2 — Runtime: minimal image, non-root user
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="AI CAD OS Team"
LABEL description="Workflow Manager — Central pipeline orchestrator"

# Security: run as non-root
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=builder /build/app ./app
COPY --from=builder /build/workflow_manager ./workflow_manager

# Create checkpoint directory owned by appuser
RUN mkdir -p /app/checkpoints && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8080/health'); exit(0 if r.status_code == 200 else 1)"

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
