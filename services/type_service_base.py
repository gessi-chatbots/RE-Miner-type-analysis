from abc import ABC, abstractmethod
from typing import List
from models.models import ReviewItem

class TypeService(ABC):
    @abstractmethod
    def analyze_type(self, reviews: List[ReviewItem]) -> List[ReviewItem]:
        pass 
    
    @abstractmethod
    def train_model(self, data: List[ReviewItem]):
        pass 