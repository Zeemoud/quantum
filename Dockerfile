FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
RUN uv sync --no-dev --frozen

COPY model/ model/
COPY api/ api/
COPY training/ training/
COPY .env.example .env

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]