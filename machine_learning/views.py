from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods

# Create your views here.


@require_http_methods(["GET"])
def health(request):
    print(request)
    return JsonResponse({"status": "ok"})