FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables (for Docker execution)
ENV MODEL_NAME="gemini-2.5-flash"

# Run the inference script by default
CMD ["python", "-m", "agents.inference"]
