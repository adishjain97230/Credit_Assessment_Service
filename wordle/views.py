from config import logging_config
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

# Create your views here.
logger = logging_config.get_logger(__name__)

def get_real_ip(group, request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        # The list looks like: "spoofed_ip, real_ip"
        # We split by comma and take the last one (strip spaces just in case)
        ip = x_forwarded_for.split(',')[-1].strip()
    else:
        # No proxy? Fallback to standard remote addr
        ip = request.META.get('REMOTE_ADDR')
    return ip

@require_http_methods(["GET"])
@ratelimit(key=lambda g, r: "global", rate="2000/m", block=False)
@ratelimit(key=get_real_ip, rate="20/m", block=False)
def health(request):
    if getattr(request, 'limited', False):
        return JsonResponse({"status": "error", "message": "Too many requests"}, status=403)
    return JsonResponse({"status": "ok"})


@require_http_methods(["POST"])
@ratelimit(key=lambda g, r: "global", rate="2000/m", block=False)
@ratelimit(key=get_real_ip, rate="5/m", block=False)
@csrf_exempt
def health_check(request):
    if getattr(request, 'limited', False):
        return JsonResponse({"status": "error", "message": "Too many requests"}, status=403)
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        logger.warning("health_check: invalid JSON: %s", e)
        return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)

    headers = request.headers
    logger.info("printing body: %s", data)
    logger.info("printing headers: %s", dict(headers))
    return JsonResponse({"status": "ok", "data": data, "headers": dict(headers)}, status=200)

