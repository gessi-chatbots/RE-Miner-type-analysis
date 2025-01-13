from pydantic import BaseModel
from typing import List

class ReviewItem(BaseModel):
    reviewId: str
    text: str
    type: str | None = None

class TypeRequest(BaseModel):
    reviews: List[ReviewItem] 

class TrainModelRequest(BaseModel):
    reviews: List[ReviewItem]

class SingleReviewRequest(BaseModel):
    text : str