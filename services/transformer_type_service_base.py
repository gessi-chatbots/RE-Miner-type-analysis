from typing import List
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import logging
from models.models import ReviewItem
from services.type_service_base import TypeService
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TransformerTypeServiceBase(TypeService):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_type(self, reviews: List[ReviewItem]) -> List[ReviewItem]:
        # Load model if not initialized
        if self.model is None or self.tokenizer is None:
            save_directory = f"saved_models/{self.model_name.replace('/', '_')}"
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
                self.model = AutoModelForSequenceClassification.from_pretrained(save_directory)
                # Load label mappings
                mappings = torch.load(f"{save_directory}/label_mappings.pt")
                self.label2id = mappings['label2id']
                self.id2label = mappings['id2label']
                self.logger.info("Model, tokenizer, and mappings loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model: {str(e)}")
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
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        num_labels = len(set(item.type for item in data))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
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
        for epoch in range(3):
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
        
        # Create save directory if it doesn't exist
        save_directory = f"saved_models/{self.model_name.replace('/', '_')}"
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        # Save label mappings
        torch.save({
            'label2id': self.label2id,
            'id2label': self.id2label
        }, f"{save_directory}/label_mappings.pt")
        
        self.logger.info(f'Model and tokenizer saved to {save_directory}')
        self.logger.info('Training completed')