FROM python:3.13-slim

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy the application into the container.
COPY . /app

# Install the application dependencies.
WORKDIR /app
ENV UV_PYTORCH_WITHOUT_CUDA=1
RUN uv sync --no-cache

# Run the application.
EXPOSE 8000
CMD [".venv/bin/uvicorn", "app.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]