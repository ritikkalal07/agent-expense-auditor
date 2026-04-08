# Multi-stage build for expense_audit_env
# Works for both in-repo and standalone deployments

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies
RUN pip install --no-cache-dir -r server/requirements.txt
RUN pip install --no-cache-dir -e .

# Final runtime stage
FROM python:3.11-slim

# Create user with UID 1000 and the app directory
RUN useradd -m -u 1000 user && mkdir -p /app/env && chown -R user:user /app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Switch to the user
USER user

# Copy the environment code
COPY --chown=user:user . /app/env

WORKDIR /app/env

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
