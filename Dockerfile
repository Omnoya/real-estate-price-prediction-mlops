FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    REAL_ESTATE_MODEL_BUNDLE_DIR=/app/model

WORKDIR /app

RUN groupadd --system app && useradd --system --gid app --home-dir /app app

COPY requirements-serving.txt /app/requirements-serving.txt
RUN python -m pip install --no-cache-dir -r /app/requirements-serving.txt

COPY src/real_estate/__init__.py /app/src/real_estate/__init__.py
COPY src/real_estate/api /app/src/real_estate/api
COPY src/real_estate/ml/__init__.py /app/src/real_estate/ml/__init__.py
COPY src/real_estate/ml/baselines.py /app/src/real_estate/ml/baselines.py
COPY src/real_estate/ml/catboost_model.py /app/src/real_estate/ml/catboost_model.py
COPY src/real_estate/ml/dataset.py /app/src/real_estate/ml/dataset.py
COPY src/real_estate/ml/serving_bundle.py /app/src/real_estate/ml/serving_bundle.py

RUN mkdir -p /app/model && chown app:app /app/model

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()"]

CMD ["python", "-m", "uvicorn", "real_estate.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
