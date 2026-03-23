import json
from config import logging_config
from django.http import JsonResponse
from .serializers import LogisticRegressionPredictSerializer
import pandas as pd
from rest_framework.exceptions import ValidationError

logger = logging_config.get_logger(__name__)

def validateLogisticRegressionRequest(request):
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        logger.warning("invalid JSON: %s", e)
        return None, ValueError("Invalid JSON body")

    serializer = LogisticRegressionPredictSerializer(data=data)

    if not(serializer.is_valid()):
        logger.error("Request is not valid: %s", serializer.errors)
        return None, ValidationError(serializer.errors)

    return pd.DataFrame([serializer.validated_data["data"]]), None