release: python manage.py migrate --noinput
web: gunicorn core.wsgi:application --workers 2 --threads 4 --timeout 120 --log-file -
