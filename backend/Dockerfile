# backend/Dockerfile
FROM python:3.11-slim

# Prevent .pyc files, and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app
# Create a writable cache directory
ENV TRANSFORMERS_CACHE=/app/cache
RUN mkdir -p /app/cache

# system deps (for sentence-transformers and pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ffmpeg libsndfile1 libjpeg-dev libpng-dev \
 && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt faiss-cpu sentence-transformers


# copy app code
COPY . .

# create models & data dir if not present
RUN mkdir -p /data /models

# expose port used by uvicorn
EXPOSE 8000

# create a user (optional safer)
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER root

# Command to run the app
# Use --host 0.0.0.0 so it's reachable from other containers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
