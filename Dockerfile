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
# (adjust if you use pyproject.toml/poetry)
COPY requirements*.txt /app/
# If you don't have a requirements lock yet, create a quick one:
# echo -e "flask\nspacy\nsentence-transformers\npandas\n" > requirements.txt

# Install Python deps
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Now copy the rest of the code
COPY . /app

# Expose Flask port (adjust if different)
EXPOSE 5000

# Default start command (adjust if your entrypoint differs)
# If your app is "app.py" running a Flask server:
CMD ["python", "app.py"]
