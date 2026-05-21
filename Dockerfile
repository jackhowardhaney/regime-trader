FROM python:3.11-slim

WORKDIR /app

# System deps for hmmlearn / scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway injects PORT; default to 8080 locally
ENV PORT=8080
ENV DB_PATH=/data/trader.db
ENV RESULTS_DIR=/data/results

# /data is where Railway mounts the persistent volume
RUN mkdir -p /data/results

EXPOSE 8080

CMD ["/bin/sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
