FROM python:3.13-slim

# Prevent Python from writing pyc files & enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install uv (from official image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy only dependency files first (better caching)
COPY pyproject.toml uv.lock* /app/

# Create virtual environment & install deps
RUN uv sync --no-dev

# Copy the rest of the app
COPY . /app

# Create non-root user
RUN adduser --disabled-password --gecos "" --uid 5678 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Run using the venv directly (faster than `uv uvicorn`)
CMD ["/app/.venv/bin/uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]