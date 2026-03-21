import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

# Create your views here.


@require_http_methods(["GET"])
def health(request):
    return JsonResponse({"status": "ok"})

@require_http_methods(["POST"])
@csrf_exempt
def health_check(request):
    data = json.loads(request.body)
    headers = request.headers
    return JsonResponse({"status": "ok", "data": data, "headers": dict(headers)}, status=200)