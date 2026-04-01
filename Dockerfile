FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables (for Docker execution)
ENV GEMINI_API_KEY=""
ENV MODEL_NAME="gemini-2.5-flash"

# Run the inference script by default
CMD ["python", "inference.py"]
