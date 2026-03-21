from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health),
    path("health-check/", views.health_check)
]