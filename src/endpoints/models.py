from pydantic import BaseModel
from typing import List

class RankedRecommendation(BaseModel):
    user_id: str
    candidate_movies: List[str]