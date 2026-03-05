# ============================================================
# Dockerfile
# Credit Risk ML API
# ============================================================

# Base Python image
FROM python:3.11-slim

# Working directory inside container
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Start API service
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]