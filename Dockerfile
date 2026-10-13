# Production image: CPU-only torch, static files collected at build time, runs as a non-root user.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them (Docker layer cache)
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .
# collectstatic only needs settings to load; this key never reaches the running container
RUN DJANGO_SECRET_KEY=build-only-placeholder python manage.py collectstatic --noinput

RUN useradd --create-home --uid 10001 appuser && mkdir -p /data && chown appuser /data
USER appuser

ENV SQLITE_PATH=/data/db.sqlite3 \
    PORT=8000
EXPOSE 8000
VOLUME ["/data"]

# Forecasts train models inside the request, so allow up to 120 s per request
CMD python manage.py migrate --noinput && \
    exec gunicorn core.wsgi:application --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 120 --access-logfile -
