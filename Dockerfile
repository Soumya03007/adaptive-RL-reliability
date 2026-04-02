FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml requirements.txt uv.lock ./
COPY __init__.py ./
COPY client.py ./
COPY inference.py ./
COPY models.py ./
COPY openenv_models.py ./
COPY openenv_tasks.py ./
COPY configs ./configs
COPY envs ./envs
COPY server ./server
COPY utils ./utils
COPY models ./models
COPY rl ./rl
COPY scripts ./scripts
COPY training ./training
COPY agents ./agents

RUN mkdir -p outputs && \
    python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "fastapi>=0.135.0" \
        "huggingface_hub>=0.36.0" \
        "numpy>=2.0.0" \
        "openai>=2.30.0" \
        "openenv-core>=0.2.3" \
        "pydantic>=2.12.0" \
        "torch>=2.11.0" \
        "tensordict>=0.11.0" \
        "torchrl>=0.11.1" \
        "uvicorn>=0.40.0"

EXPOSE 8000

CMD ["python", "-m", "server.app"]
