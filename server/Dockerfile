FROM python:3.12-slim

RUN useradd -m -u 1000 user

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ENABLE_WEB_INTERFACE=true \
    PORT=8000 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user pyproject.toml README.md openenv.yaml requirements.txt uv.lock ./
COPY --chown=user __init__.py ./
COPY --chown=user client.py ./
COPY --chown=user inference.py ./
COPY --chown=user models.py ./
COPY --chown=user openenv_models.py ./
COPY --chown=user openenv_tasks.py ./
COPY --chown=user configs ./configs
COPY --chown=user envs ./envs
COPY --chown=user server ./server
COPY --chown=user utils ./utils
COPY --chown=user models ./models
COPY --chown=user rl ./rl
COPY --chown=user scripts ./scripts
COPY --chown=user training ./training
COPY --chown=user agents ./agents

RUN mkdir -p outputs && chown -R user:user $HOME/app

USER user

RUN python -m pip install --user --upgrade pip setuptools wheel && \
    python -m pip install --user --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
