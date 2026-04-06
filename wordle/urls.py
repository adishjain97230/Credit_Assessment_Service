from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health),
    path("health-check/", views.health_check),
    path("get-word/", views.get_word),
    path("check-word/", views.check_word),
]