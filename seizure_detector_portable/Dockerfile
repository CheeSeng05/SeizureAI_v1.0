FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements.txt

EXPOSE 8888

CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
