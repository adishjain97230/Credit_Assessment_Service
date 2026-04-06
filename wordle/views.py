from config import logging_config
from django.views.decorators.http import require_http_methods
from django_ratelimit.decorators import ratelimit
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from wordle import wordle, services
import json
from wordle.serializers import CheckWordRequest
from pydantic import ValidationError

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

@require_http_methods(["GET"])
@ratelimit(key=lambda g, r: "global", rate="2000/m", block=False)
@ratelimit(key=get_real_ip, rate="5/m", block=False)
@csrf_exempt
def get_word(request):
    if getattr(request, 'limited', False):
        return JsonResponse({"status": "error", "message": "Too many requests"}, status=403)
    
    word = wordle.get_word()
    return JsonResponse({"status": "ok", "word_id": services.saveWord(word)}, status=200)

@require_http_methods(["POST"])
@ratelimit(key=lambda g, r: "global", rate="2000/m", block=False)
@ratelimit(key=get_real_ip, rate="20/m", block=False)
@csrf_exempt
def check_word(request):
    if getattr(request, 'limited', False):
        return JsonResponse({"status": "error", "message": "Too many requests"}, status=403)
    
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as e:
        logger.error("check_word: invalid JSON: %s", e)
        return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)
    
    try:
        data_object = CheckWordRequest.model_validate(data)
    except ValidationError as e:
        logger.error("check_word: invalid data: %s", e)
        return JsonResponse({"status": "error", "message": "Invalid data"}, status=400)
    
    word, err = services.getWord(data_object.word_id)
    if err is not None:
        logger.error("check_word: error getting word: %s", err)
        return JsonResponse({"status": "error", "message": f"Error getting word: {str(err)}"}, status=400)
    
    feedback, err = wordle.get_feedback(word, data_object.guess)
    if err is not None:
        logger.error("check_word: error getting feedback: %s", err)
        return JsonResponse({"status": "error", "message": f"Error getting feedback: {str(err)}"}, status=400)
    
    return JsonResponse({"status": "ok", "feedback": feedback}, status=200)