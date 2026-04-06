from datetime import timedelta
from django.utils import timezone
from wordle.models import WordEntry


def saveWord(word):
    entry = WordEntry.objects.create(
        word=word,
        expires_at=timezone.now() + timedelta(hours=1),
    )

    return entry.id