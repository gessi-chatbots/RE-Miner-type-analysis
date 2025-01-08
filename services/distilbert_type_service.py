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
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from torch.optim import AdamW
from tqdm import tqdm

class DistilbertTypeService(TypeService):
    def __init__(self):
        self.model = None
        self.vectorizer = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_type(self, reviews: List[ReviewItem]) -> List[ReviewItem]:
        # Ensure model is initialized
        if self.model is None or self.tokenizer is None:
            self.logger.error("Model or tokenizer not initialized. Please train the model first.")
            return reviews

        # Prepare device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()

        # Prepare texts for inference
        texts = [review.text for review in reviews]
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=100,
            return_tensors='pt'
        )

        # Create dataloader for batch processing
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

        # Process batches
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = [b.to(device) for b in batch]
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                logits = outputs.logits
                pred_labels = torch.argmax(logits, dim=1)
                predictions.extend(pred_labels.cpu().numpy())

        # Map predictions back to labels and update reviews
        for review, pred_id in zip(reviews, predictions):
            review.type = self.id2label[pred_id]

        return reviews
    
    def train_model(self, data: List[ReviewItem]):
        # Initialize tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        num_labels = len(set(item.type for item in data))
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', 
            num_labels=num_labels
        )
        
        # Prepare data
        texts = [item.text for item in data]
        labels = [item.type for item in data]
        
        # Create label mapping
        self.label2id = {label: i for i, label in enumerate(set(labels))}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
        # Convert labels to ids
        label_ids = [self.label2id[label] for label in labels]
        
        # Tokenize texts
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=100,
            return_tensors='pt'
        )
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(label_ids)
        )
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Setup training parameters
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(15):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
            
            for batch in progress_bar:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            avg_loss = total_loss / len(dataloader)
            self.logger.info(f'Epoch {epoch + 1} - Average loss: {avg_loss:.4f}')
        
        self.logger.info('Training completed')