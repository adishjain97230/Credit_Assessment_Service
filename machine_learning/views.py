import json
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from config import logging_config, switch_properties, constants
from .serializers import LogisticRegressionPredictSerializer
import pandas as pd
from data_cleaning import dataCleaning

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
    data = json.loads(request.body)
    serializer = LogisticRegressionPredictSerializer(data=data)

    if not(serializer.is_valid()):
        logger.error("Request is not valid: %s", serializer.errors)
        return JsonResponse({"status": "error", "errors": serializer.errors}, status=400)
    
    df = pd.DataFrame([serializer.validated_data["data"]])
    df = dataCleaning.convertStringColumnsToNumeric(df)

    model = joblib.load(switch_properties.SWITCH_PROPERTIES[constants.models][constants.logistic_regression][constants.model_path])

    ohe_df = model[constants.ohe_enc].transform(df[model[constants.ohe_cols]].astype(str))
    
    ohe_cols_names = model[constants.ohe_enc].get_feature_names_out(model[constants.ohe_cols])
    ohe_df = pd.DataFrame(ohe_df, columns=ohe_cols_names, index=df.index)
    df = pd.concat([df.drop(columns=model[constants.ohe_cols]), ohe_df], axis=1).copy()

    for col in model[constants.me_cols]:
        df[f"{col}_enc"] = df[col].map(model[constants.me_mappings][col]).fillna(model[constants.global_mean]).astype(float)
        df = df.drop(columns=col)

    df = df[model[constants.all_cols]]
    
    df = df[model[constants.all_cols]]
    df = model[constants.scaler].transform(df)

    p = model[constants.model].predict_proba(df)[:, 1]
    y_pred = (p >= model[constants.p_threshold]).astype(int)


    return JsonResponse({"status": "ok", "data": "success", "y_pred": int(y_pred[0]), "p": float(p[0])}, status=200)