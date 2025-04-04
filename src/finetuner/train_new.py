import json
import torch
import numpy as np
from datasets import Dataset
from rouge_score import rouge_scorer
from transformers import (
    AutoTokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import pandas as pd


SCHEMA_CONTEXT = """
DATABASE SCHEMA:
* Customers(CustomerID text PK, CompanyName text, ContactName text, ContactTitle text, Address text, City text, Region text, PostalCode text, Country text, Phone text, Fax text)
* Orders(OrderID integer PK, CustomerID text FK->Customers.CustomerID, EmployeeID integer, OrderDate date, RequiredDate date, ShippedDate date, ShipVia integer, Freight numeric, ShipName text, ShipAddress text, ShipCity text, ShipRegion text, ShipPostalCode text, ShipCountry text)
* OrderDetails(OrderID integer FK->Orders.OrderID PK_PART, ProductID integer FK->Products.ProductID PK_PART, UnitPrice numeric, Quantity integer, Discount real)
* Products(ProductID integer PK, ProductName text, SupplierID integer, CategoryID integer FK->Categories.CategoryID, QuantityPerUnit text, UnitPrice numeric, UnitsInStock integer, UnitsOnOrder integer, ReorderLevel integer, Discontinued integer)
* Categories(CategoryID integer PK, CategoryName text, Description text, Picture binary)

RELATIONSHIPS:
* Customers.CustomerID -> Orders.CustomerID
* Orders.OrderID -> OrderDetails.OrderID
* Products.ProductID -> OrderDetails.ProductID
* Categories.CategoryID -> Products.CategoryID
"""

# Step 1: Load and prepare the dataset
def load_jsonl_to_dataset(file_path):
    """Load JSONL file and convert to HuggingFace Dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                data.append(item)
    
    # Convert to pandas DataFrame first for easier processing
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)

# Load your datasets
train_dataset = load_jsonl_to_dataset("data/processed/train.jsonl")
val_dataset = load_jsonl_to_dataset("data/processed/val.jsonl")

# Step 2: Format the dataset with task prefix
def format_dataset(example):
    """Add task prefix to input."""
    return {
        'input': f"translate to SQL: \nSchema: {SCHEMA_CONTEXT}\nQuestion: {example['input']}", 
        #'input': f"translate to SQL: {example['input']}", 
        'target': example['output']
    }
    
    
# Apply formatting
train_dataset = train_dataset.map(format_dataset)
val_dataset = val_dataset.map(format_dataset)

# Step 3: Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('suriya7/t5-base-text-to-sql')
model = T5ForConditionalGeneration.from_pretrained("suriya7/t5-base-text-to-sql")

#Step 5: Tokenize the dataset
def preprocess_function(examples):
    """Tokenize inputs and targets."""
    model_inputs = tokenizer(
        examples["input"],
        max_length=256,
        padding="max_length",
        truncation=True
    )
    
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target"],
            max_length=128,
            padding="max_length",
            truncation=True
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # Replace padding token id with -100 for loss calculation
    for i in range(len(model_inputs["labels"])):
        model_inputs["labels"][i] = [
            -100 if token == tokenizer.pad_token_id else token 
            for token in model_inputs["labels"][i]
        ]
    
    return model_inputs

# Apply preprocessing
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Step 6: Setup evaluation metrics
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_metrics(pred):
    """Compute ROUGE metrics for evaluation."""
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    
    # Replace -100 with pad token id
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    
    # Decode predictions and references
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    
    print('pred_str: ', pred_str)
    print('=================================')
    print('label_str: ', label_str)
    print('=================================')
    
    scores = []
    for pred, ref in zip(pred_str, label_str):
        score = scorer.score(ref, pred)
        scores.append(score)
    
    # Calculate average scores
    results = {
        "rouge1_precision": np.mean([s['rouge1'].precision for s in scores]),
        "rouge1_recall": np.mean([s['rouge1'].recall for s in scores]),
        "rouge1_fmeasure": np.mean([s['rouge1'].fmeasure for s in scores]),
        "rouge2_precision": np.mean([s['rouge2'].precision for s in scores]),
        "rouge2_recall": np.mean([s['rouge2'].recall for s in scores]),
        "rouge2_fmeasure": np.mean([s['rouge2'].fmeasure for s in scores]),
        "rougeL_precision": np.mean([s['rougeL'].precision for s in scores]),
        "rougeL_recall": np.mean([s['rougeL'].recall for s in scores]),
        "rougeL_fmeasure": np.mean([s['rougeL'].fmeasure for s in scores]),
    }

    # Round values for readability
    results = {k: round(v, 4) for k, v in results.items()}
    
    # Add exact match score
    exact_matches = sum(pred.strip() == ref.strip() for pred, ref in zip(pred_str, label_str))
    results["exact_match"] = round(exact_matches / len(pred_str), 4)
    
    return results


# Step 8: Setup training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="exact_match",
)

# Step 9: Initialize data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8 if training_args.fp16 else None,  #this is helpful for memory optimization
)

# Step 10: Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Step 11: Train the model
print("Starting training...")
trainer.train()

# Step 12: Save the model
model_path = "text2sql-t5-small"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
print(f"Model saved to {model_path}")