from pydantic import BaseModel

class CheckWordRequest(BaseModel):
    word_id: int
    guess: str
    guess_number: int