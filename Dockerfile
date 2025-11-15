FROM python:3.13-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies (optional but recommended for ML libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies from pyproject.toml via pip
# (requires pip >= 23.1)
RUN pip install --no-cache-dir .

# Expose backend port
EXPOSE 8000

# Start FastAPI backend
CMD ["uvicorn", "src.gradio_demo.app.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
