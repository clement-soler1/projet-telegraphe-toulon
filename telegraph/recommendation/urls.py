import os
from django.urls import path

from . import views

app_name = 'recommendation'

urlpatterns = [
    path('', views.home, name='home'),
    path('pair_wine', views.pair_wine, name='pair_wine'),
]