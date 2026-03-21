from rest_framework import serializers

class LogisticRegressionPredictSerializer(serializers.Serializer):
    data = serializers.JSONField()