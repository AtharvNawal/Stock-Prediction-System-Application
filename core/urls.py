"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include

from app.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # login, logout, password change/reset
    path('accounts/signup/', signup, name='signup'),
    path('', search),
    path('market/', index),
    path('search/', search_redirect),
    path('api/companies/', company_suggest),
    path('predict/<str:ticker_value>/<str:number_of_days>/', predict),
    path('ticker/', ticker),
    path('history/', history),
    path('company/', company_search),
    path('company/<str:ticker_value>/', company),
    path('company/<str:ticker_value>/peers/', company_peers),
    path('company/<str:ticker_value>/forecast/', company_forecast),
]