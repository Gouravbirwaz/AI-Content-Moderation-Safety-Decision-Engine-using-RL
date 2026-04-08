FROM python:3.11-slim

WORKDIR /app

# Install uv for fast, deterministic builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install dependencies using uv.lock
COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache .

# Copy application files
COPY . .

# Set environment variables
ENV MODEL_NAME="gemini-2.5-flash"

# Run the FastAPI server on port 8000 using the console script
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
