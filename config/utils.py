def getErrorJsonObject(error):
    return {
        "type": type(error).__name__,
        "message": str(error)
    }