FROM python:3.12-slim

# Minimal system dependencies for sklearn/xgboost.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Thread limits consistent with the project's resource policy.
ENV OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    VECLIB_MAXIMUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    CARDIO_N_JOBS=4 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_TRACKING_URI=file:///app/mlruns

WORKDIR /app

# Layer cache: install dependencies before source code.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# Runtime artifacts (versioned models) are mounted via volume.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "cardio_ml.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
