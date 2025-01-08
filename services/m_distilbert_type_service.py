from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
from models.models import ReviewItem
from services.type_service_base import TypeService
import logging

class MDistilbertTypeService(TypeService):
    def __init__(self):
        self.model = None
        self.vectorizer = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_type(self, reviews: List[ReviewItem], threshold: float = None) -> List[ReviewItem]:
        #TODO: Implement type analysis
        pass
    
    def train_model(self, data: List[ReviewItem]):
        #TODO: Implement model training
        pass