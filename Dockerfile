FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY __init__.py ./
COPY client.py ./
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

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
