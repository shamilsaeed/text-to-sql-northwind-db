from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import DataLoader
from src.finetuner.dataset import SQLDataset
from rouge_score import rouge_scorer
import os
import numpy as np
import torch
from tqdm import tqdm

class SQLTrainer:
    def __init__(
        self,
        model_name: str = 't5-small',  # distillation on small language model for faster training
        max_input_length: int = 128,
        max_output_length: int = 256,
        schema_context: str = None
        
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.schema_context = schema_context
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def prepare_datasets(self, train_file: str, val_file: str, batch_size: int = 4):
        """Prepare training and validation datasets."""
        self.train_dataset = SQLDataset(
            train_file, self.tokenizer, self.max_input_length, 
            self.max_output_length, self.schema_context
        )
        self.val_dataset = SQLDataset(
            val_file, self.tokenizer, self.max_input_length, 
            self.max_output_length, self.schema_context
        )
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size)

    def train_epoch(self, optimizer, scheduler):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
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
            scheduler.step()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Evaluate every 50 steps
            if step > 0 and step % 50 == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Switch back to train mode
                
        return total_loss / len(self.train_loader)

    def evaluate(self):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Generate predictions
                preds = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.max_output_length,
                    early_stopping=True,
                )
                
                # Decode predictions and labels
                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
                decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                print(len(decoded_preds), len(decoded_labels))
                print('Prediction: ', decoded_preds)
                print('Label: ', decoded_labels)
                all_preds.extend(decoded_preds)
                all_labels.extend(decoded_labels)
        
        # Calculate metrics
        rouge_scores = [
            self.scorer.score(pred, label)
            for pred, label in zip(all_preds, all_labels)
        ]
        
        metrics = {
            'loss': total_loss / len(self.val_loader),
            'rouge1_f': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
            'rouge2_f': np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
            'rougeL_f': np.mean([s['rougeL'].fmeasure for s in rouge_scores]),
            'exact_match': sum(p.strip() == l.strip() for p, l in zip(all_preds, all_labels)) / len(all_preds)
        }
        
        print("\nEvaluation Results:")
        print(f"Loss: {metrics['loss']:.4f}")
        print(f"ROUGE-1: {metrics['rouge1_f']:.2%}")
        print(f"ROUGE-2: {metrics['rouge2_f']:.2%}")
        print(f"ROUGE-L: {metrics['rougeL_f']:.2%}")
        print(f"Exact Match: {metrics['exact_match']:.2%}")
        print("-" * 50)
        
        return metrics

    def train(self, num_epochs: int = 5, learning_rate: float = 3e-4):
        """Train the model."""
        # Prepare optimizer and scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(self.train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=100,
            num_training_steps=total_steps
        )
        
        print(f"Starting training on {self.device}")
        best_exact_match = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            avg_loss = self.train_epoch(optimizer, scheduler)
            print(f"\nAverage training loss: {avg_loss:.4f}")
            
            # Evaluate
            metrics = self.evaluate()
            
            # # Save best model
            # if metrics['exact_match'] > best_exact_match:
            #     best_exact_match = metrics['exact_match']
            #     self.save_model('text2sql-t5-small-best')

    def save_model(self, output_dir: str):
        """Save the model and tokenizer."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"\nModel saved to {output_dir}")

if __name__ == "__main__":
    # Schema context
    SCHEMA_PROMPT = """
    Schema:
    * Customers(CustomerID, CompanyName, ContactName, ContactTitle, Address, City, Region, PostalCode, Country, Phone, Fax)
    * Orders(OrderID, CustomerID, EmployeeID, OrderDate, RequiredDate, ShippedDate, ShipVia, Freight, ShipName, ShipAddress, ShipCity, ShipRegion, ShipPostalCode, ShipCountry)
    * OrderDetails(OrderID, ProductID, UnitPrice, Quantity, Discount)
    * Products(ProductID, ProductName, SupplierID, CategoryID, QuantityPerUnit, UnitPrice, UnitsInStock, UnitsOnOrder, ReorderLevel, Discontinued)
    * Categories(CategoryID, CategoryName, Description, Picture)

    Foreign Keys:
    * Orders.CustomerID -> Customers.CustomerID
    * OrderDetails.OrderID -> Orders.OrderID
    * OrderDetails.ProductID -> Products.ProductID
    * Products.CategoryID -> Categories.CategoryID
    """

    # Initialize trainer
    trainer = SQLTrainer(schema_context=SCHEMA_PROMPT)
    
    # Prepare datasets
    trainer.prepare_datasets(
        train_file='data/processed/train.jsonl',
        val_file='data/processed/val.jsonl'
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model('text2sql-t5-small') 