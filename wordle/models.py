from django.db import models
from django.db import models
from django.utils import timezone
from datetime import timedelta

# Create your models here.

def default_expiry():
    return timezone.now() + timedelta(hours=1)

class WordEntry(models.Model):
    word = models.CharField(max_length=5)
    expires_at = models.DateTimeField(default=default_expiry, db_index=True)

    class Meta:
        db_table = "words"
    
    def __str__(self):
        return f"{self.id}: {self.word}"