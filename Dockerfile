FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY data ./data
COPY docs ./docs
COPY tests ./tests

RUN pip install --upgrade pip==25.0.1 && pip install '.[dev]'

ENTRYPOINT ["jssp-yafs"]
CMD ["smoke", "--config", "configs/smoke.yaml"]
