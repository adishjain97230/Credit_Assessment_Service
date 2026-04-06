from datetime import timedelta
from django.utils import timezone
from wordle.models import WordEntry


def saveWord(word):
    entry = WordEntry.objects.create(
        word=word,
        expires_at=timezone.now() + timedelta(hours=1),
    )

    return entry.id

def getWord(id: int):
    entry = WordEntry.objects.filter(pk=id).first()
    if entry is None:
        return None, ValueError("Word not found")
    if entry.expires_at < timezone.now():
        return None, ValueError("Word has expired")
    return entry.word, None