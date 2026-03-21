from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health),
    path("health-check/", views.health_check),
    path("logistic-regression/predict/", views.logistic_regression_predict),
]