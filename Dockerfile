# Reproducible environment for the research pipeline.
# Runs the offline demo by default; needs no credentials and no network.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    KALSHI_ALPHA_MODE=offline

WORKDIR /app

# Dependency layer first so source edits do not invalidate the install.
COPY pyproject.toml README.md ./
COPY src/kalshi_alpha/__init__.py src/kalshi_alpha/__init__.py
RUN pip install --upgrade pip && pip install -e ".[plots]"

COPY src/ src/
COPY tests/ tests/
COPY Makefile ./
RUN pip install -e ".[dev,plots]"

# Run as a non-root user: this image has no business writing outside /app.
RUN useradd --create-home --uid 10001 research \
    && mkdir -p /app/artifacts /app/data \
    && chown -R research:research /app
USER research

ENTRYPOINT ["python", "-m", "kalshi_alpha.cli"]
CMD ["demo", "--out", "artifacts"]
