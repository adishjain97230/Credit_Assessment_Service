import json
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from config import logging_config, switch_properties, constants, utils
from .serializers import LogisticRegressionPredictSerializer
import pandas as pd
from data_cleaning import dataCleaning
from models import logistic_regression
from . import services
from django_ratelimit.decorators import ratelimit

# Create your views here.

logger = logging_config.get_logger(__name__)

@require_http_methods(["GET"])
@ratelimit(key="ip", rate="10/m")
def health(request):
    return JsonResponse({"status": "ok"})

@require_http_methods(["POST"])
@csrf_exempt
def health_check(request):
    data = json.loads(request.body)
    headers = request.headers
    logger.info("printing body: %s", data)
    logger.info("printing headers: %s", dict(headers))
    return JsonResponse({"status": "ok", "data": data, "headers": dict(headers)}, status=200)

@require_http_methods(["POST"])
@csrf_exempt
def logistic_regression_predict(request):

    df, error = services.validateLogisticRegressionRequest(request)
    if error != None:
        return JsonResponse({"status": "error", "error": utils.getErrorJsonObject(error)}, status=400)

    p, threshold, error = logistic_regression.predict(df)

    if error != None:
        return JsonResponse({"status": "error", "error": utils.getErrorJsonObject(error)})


    return JsonResponse({"status": "ok", "threshold": float(threshold), "p": float(p[0])}, status=200)