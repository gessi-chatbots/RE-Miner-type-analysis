from models.models import ReviewItem
from typing import List
from textblob import TextBlob

# If you need a base class, you could keep it here:
class TypeService:
    def analyze_type(self, reviews: List[ReviewItem]) -> List[ReviewItem]:
        raise NotImplementedError

    def train_model(self, data: List[ReviewItem]):
        raise NotImplementedError