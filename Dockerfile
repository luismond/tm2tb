# Use a recent Python with manylinux wheels available
FROM python:3.11-slim

# System deps (build tools rarely needed, but safe)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git && \
    rm -rf /var/lib/apt/lists/*

# Avoid pip cache bloat
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir
WORKDIR /app

# Copy only dependency files first to leverage Docker layer caching
COPY requirements*.txt /app/

# Install Python deps
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the code
COPY . /app

# Expose Flask port (adjust if different)
EXPOSE 5000

# Default start command
CMD ["python", "app.py"]
