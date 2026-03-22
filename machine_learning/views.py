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

# Create your views here.

logger = logging_config.get_logger(__name__)

@require_http_methods(["GET"])
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

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        logger.warning("invalid JSON: %s", e)
        return JsonResponse({"status": "error", "errors": "Invalid JSON body"}, status=400)

    serializer = LogisticRegressionPredictSerializer(data=data)

    if not(serializer.is_valid()):
        logger.error("Request is not valid: %s", serializer.errors)
        return JsonResponse({"status": "error", "errors": serializer.errors}, status=400)
    
    df = pd.DataFrame([serializer.validated_data["data"]])

    model = joblib.load(switch_properties.SWITCH_PROPERTIES[constants.models][constants.logistic_regression][constants.model_path])

    p, threshold, error = logistic_regression.predict(df, model)

    if error != None:
        return JsonResponse({"status": "not ok", "error": utils.getErrorJsonObject(error)})


    return JsonResponse({"status": "ok", "threshold": float(threshold), "p": float(p[0])}, status=200)