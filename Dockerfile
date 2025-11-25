FROM python:3.13-slim

# Install uv.
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the application dependencies.
WORKDIR /app

# Install CPU-only dependencies
COPY requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

# COPY pyproject.toml uv.lock ./
# ENV UV_PYTORCH_WITHOUT_CUDA=1
# RUN uv sync --frozen --no-cache --no-dev

COPY src/ ./src/
COPY data/ ./data/
COPY pyproject.toml ./

RUN pip install --no-cache-dir --no-deps -e .

# Run the application.
EXPOSE 8000
# CMD [".venv/bin/uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["uvicorn", "src.app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]