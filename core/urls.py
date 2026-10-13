"""core URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', views.Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path

from app import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('accounts/', include('django.contrib.auth.urls')),  # login, logout, password change/reset
    path('accounts/signup/', views.signup, name='signup'),
    path('', views.search),
    path('market/', views.index),
    path('search/', views.search_redirect),
    path('api/companies/', views.company_suggest),
    path('predict/<str:ticker_value>/<str:number_of_days>/', views.predict),
    path('ticker/', views.ticker),
    path('history/', views.history),
    path('company/', views.company_search),
    path('company/<str:ticker_value>/', views.company),
    path('company/<str:ticker_value>/peers/', views.company_peers),
    path('company/<str:ticker_value>/forecast/', views.company_forecast),
]