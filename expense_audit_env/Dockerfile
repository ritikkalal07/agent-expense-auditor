# Single-stage build for Expense Audit Environment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create user with UID 1000 and the app directory
RUN useradd -m -u 1000 user
WORKDIR /app/env

# Set ownership to the non-root user
RUN chown -R user:user /app

# Copy the environment code
COPY --chown=user:user . .

# Install dependencies and the package
USER user
RUN pip install --user --no-cache-dir -r server/requirements.txt
RUN pip install --user --no-cache-dir .

# Add user bin to path
ENV PATH="/home/user/.local/bin:${PATH}"
ENV PYTHONPATH="/app/env:${PYTHONPATH}"

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:7860/health || exit 1

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
