"""
Django settings for core project.

Configuration comes from environment variables (see .env.example), so no secret lives in
the repository and the same code runs in development, CI and production:

  DJANGO_DEBUG                 "1" for local development; off by default (fail safe)
  DJANGO_SECRET_KEY            required whenever DEBUG is off
  DJANGO_ALLOWED_HOSTS         comma-separated host names
  DJANGO_CSRF_TRUSTED_ORIGINS  comma-separated origins, e.g. https://investiq.example.com
  DJANGO_SECURE_SSL_REDIRECT   "0" to stop redirecting HTTP to HTTPS (on by default in production)
  RATE_LIMIT_ENABLED           "0" to turn rate limiting off
  RATE_LIMIT_TRUSTED_PROXIES   number of reverse proxies in front of the app (0 = trust none)
  SQLITE_PATH                  database file

For the full list of settings and their values, see
https://docs.djangoproject.com/en/4.2/ref/settings/
"""

from pathlib import Path
import os


def env_bool(name, default=False):
    return os.environ.get(name, '1' if default else '0').strip().lower() in ('1', 'true', 'yes', 'on')


def env_list(name, default=''):
    return [item.strip() for item in os.environ.get(name, default).split(',') if item.strip()]


# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = env_bool('DJANGO_DEBUG')

SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', '')
if not SECRET_KEY:
    if not DEBUG:
        # RuntimeError, not ImproperlyConfigured: manage.py swallows the latter and reports an unrelated error
        raise RuntimeError('Set DJANGO_SECRET_KEY, or DJANGO_DEBUG=1 for local development.')
    SECRET_KEY = 'django-insecure-local-development-only'

ALLOWED_HOSTS = env_list('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1')
CSRF_TRUSTED_ORIGINS = env_list('DJANGO_CSRF_TRUSTED_ORIGINS')


# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'app',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'app.security.RateLimitMiddleware',  # early, so refused requests cost almost nothing
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'core.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
            'libraries' : {
                'staticfiles': 'django.templatetags.static',
            }
        },
    },
]

WSGI_APPLICATION = 'core.wsgi.application'


# Database
# https://docs.djangoproject.com/en/4.2/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.environ.get('SQLITE_PATH', Path.home() / 'sqlitedb' / 'stock_predict.db'),
    }
}


# Passwords
# Argon2id is memory-hard, so guessing passwords on GPUs or ASICs costs far more than with
# PBKDF2. Existing PBKDF2 hashes still verify and are re-hashed with Argon2 at the next login.
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.ScryptPasswordHasher',
]

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {'min_length': 10},
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Rate limiting (app/security.py): token buckets per client and policy, first match wins.
# `requests` per `per_seconds`, with bursts of up to `requests`.
RATE_LIMIT_ENABLED = env_bool('RATE_LIMIT_ENABLED', default=True)
RATE_LIMIT_TRUSTED_PROXIES = int(os.environ.get('RATE_LIMIT_TRUSTED_PROXIES', '0'))
RATE_LIMITS = [
    {'name': 'auth', 'pattern': r'^/accounts/(login|signup|password_reset)/$', 'methods': ('POST',),
     'requests': 5, 'per_seconds': 60},
    {'name': 'forecast', 'pattern': r'^/company/[^/]+/forecast/$', 'requests': 6, 'per_seconds': 60},
    {'name': 'peers', 'pattern': r'^/company/[^/]+/peers/$', 'requests': 10, 'per_seconds': 60},
    {'name': 'api', 'pattern': r'^/api/', 'requests': 60, 'per_seconds': 60},
    {'name': 'pages', 'pattern': r'^/(?!static/)', 'requests': 120, 'per_seconds': 60},
]


# Security headers and cookies

SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = 'same-origin'
SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin'
X_FRAME_OPTIONS = 'DENY'
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Lax'
CSRF_COOKIE_SAMESITE = 'Lax'

if not DEBUG:
    # HTTPS only: cookies never travel in clear text, and browsers remember to use HTTPS (HSTS)
    SECURE_SSL_REDIRECT = env_bool('DJANGO_SECURE_SSL_REDIRECT', default=True)
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')  # set by the platform's proxy
    SESSION_COOKIE_SECURE = True
    CSRF_COOKIE_SECURE = True
    SECURE_HSTS_SECONDS = 60 * 60 * 24 * 365
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True


# Internationalization
# https://docs.djangoproject.com/en/4.2/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images), served by WhiteNoise in production
# https://docs.djangoproject.com/en/4.2/howto/static-files/

STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STORAGES = {
    'default': {'BACKEND': 'django.core.files.storage.FileSystemStorage'},
    'staticfiles': {'BACKEND': 'whitenoise.storage.CompressedStaticFilesStorage'},  # gzip + brotli copies
}
WHITENOISE_USE_FINDERS = DEBUG  # serve straight from app/static while developing, no collectstatic needed

# Accounts: where to go after logging in or out
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'


# Default primary key field type
# https://docs.djangoproject.com/en/4.2/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {'console': {'class': 'logging.StreamHandler'}},
    'loggers': {'django.security': {'handlers': ['console'], 'level': 'WARNING'}},
}
