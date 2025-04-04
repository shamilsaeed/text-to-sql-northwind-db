import os
import torch
import json
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from rouge_score import rouge_scorer
import gc

# Memory optimization
gc.collect()

# Define paths
data_path = "./data"
model_name_or_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./output/tinyllama_sql_lora"

# Define schema context
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

# Load and prepare the dataset
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

# Format dataset for instruction fine-tuning
def format_for_instruction_tuning(example):
    """Format examples for instruction tuning with TinyLlama chat format."""
    # Format for TinyLlama chat format
    formatted_text = f"<|user|>\nI have the following database schema:\n{SCHEMA_CONTEXT}\n\nQuestion: {example['input']}\n\nWrite the SQL query to answer this question.<|endoftext|>\n<|assistant|>\n{example['output']}<|endoftext|>"
    
    return {"text": formatted_text}

# Load datasets
print("Loading datasets...")
train_dataset = load_jsonl_to_dataset(f"{data_path}/processed/train.jsonl")
val_dataset = load_jsonl_to_dataset(f"{data_path}/processed/val.jsonl")

# Format datasets
print("Formatting datasets...")
train_dataset = train_dataset.map(format_for_instruction_tuning, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(format_for_instruction_tuning, remove_columns=val_dataset.column_names)

# Initialize tokenizer
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Tokenize datasets
def tokenize_function(examples):
    """Tokenize the examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,  # Reduced length for CPU training
        padding="max_length",
        return_tensors="pt"
    )

print("Tokenizing datasets...")
# Process in smaller batches to save memory
tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True, 
    batch_size=4,
    remove_columns=["text"]
)
tokenized_val = val_dataset.map(
    tokenize_function, 
    batched=True, 
    batch_size=4,
    remove_columns=["text"]
)

# Set format for PyTorch
tokenized_train.set_format("torch")
tokenized_val.set_format("torch")

# Memory optimization
gc.collect()

# Load model (CPU only)
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    device_map="cpu",  # Force CPU
    torch_dtype=torch.float32,  # Use float32 for CPU
    low_cpu_mem_usage=True
)

# Define LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Increased rank for better performance
    lora_alpha=32,
    lora_dropout=0.05,
    # Target more modules for TinyLlama
    target_modules=["q_proj", "v_proj"]
)

# Get PEFT model
print("Applying LoRA adapters...")
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Custom metrics computation
class MetricsComputer:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def compute_rouge(self, predictions, references):
        """Compute ROUGE scores between predictions and references."""
        scores = []
        for pred, ref in zip(predictions, references):
            score = self.rouge_scorer.score(ref, pred)
            scores.append(score)
        
        # Calculate average ROUGE scores
        rouge_metrics = {
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
        
        return rouge_metrics
    
    def compute_exact_match(self, predictions, references):
        """Compute exact match score."""
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        return exact_matches / len(predictions) if predictions else 0
    
    def compute_metrics(self, eval_preds):
        """Compute all metrics for evaluation."""
        predictions, references = eval_preds
        
        # Clean predictions and references
        predictions = [pred.strip() for pred in predictions]
        references = [ref.strip() for ref in references]
        
        # Compute ROUGE scores
        rouge_scores = self.compute_rouge(predictions, references)
        
        # Compute exact match
        exact_match = self.compute_exact_match(predictions, references)
        
        # Combine all metrics
        metrics = {
            "exact_match": exact_match,
            **rouge_scores
        }
        
        return metrics

# Initialize metrics computer
metrics_computer = MetricsComputer()

# Custom evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Get predictions
    predictions = []
    references = []
    
    for i in range(len(logits)):
        # Get the predicted tokens
        pred_tokens = torch.argmax(torch.tensor(logits[i]), dim=-1)
        
        # Get the reference tokens (ignoring -100 padding)
        ref_tokens = [token for token in labels[i] if token != -100]
        
        # Decode tokens
        pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
        
        # Extract SQL query from the text (after "<|assistant|>")
        try:
            pred_sql = pred_text.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()
        except IndexError:
            pred_sql = pred_text
        
        try:
            ref_sql = ref_text.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()
        except IndexError:
            ref_sql = ref_text
        
        predictions.append(pred_sql)
        references.append(ref_sql)
    
    # Compute metrics
    return metrics_computer.compute_metrics((predictions, references))

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments - Optimized for CPU
training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=3,  # Reduced for CPU training
    weight_decay=0.01,
    save_strategy="epoch",
    save_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=False,  # Disable fp16 for CPU
    logging_steps=10,
    report_to="none",
    optim="adamw_torch",
    max_grad_norm=0.3,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
   # ompute_metrics=compute_metrics,
)

# Train model
print("Starting training...")
trainer.train()

# Save model
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")

# Function for inference
def generate_sql(question, model, tokenizer, schema_context=SCHEMA_CONTEXT):
    """Generate SQL from natural language question."""
    # Format input for TinyLlama chat format
    prompt = f"<|user|>\nI have the following database schema:\n{schema_context}\n\nQuestion: {question}\n\nWrite the SQL query to answer this question.<|endoftext|>\n<|assistant|>"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=128,
            temperature=0.1,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>")
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract the SQL part (between <|assistant|> and <|endoftext|>)
    try:
        sql = response.split("<|assistant|>")[1].split("<|endoftext|>")[0].strip()
    except IndexError:
        sql = response
    
    return sql

# Test the model after training
print("\nTesting the fine-tuned model...")
# Load the original datasets to get the questions and expected outputs
test_dataset = load_jsonl_to_dataset(f"{data_path}/processed/val.jsonl")

# Test on validation set
predictions = []
references = []

print("\nValidation Results:")
for i, example in enumerate(test_dataset[:5]):  # Test on first 5 examples to save time on CPU
    question = example["input"]
    expected_sql = example["output"]
    
    # Generate SQL
    predicted_sql = generate_sql(question, model, tokenizer)
    
    predictions.append(predicted_sql)
    references.append(expected_sql)
    
    print(f"Example {i+1}:")
    print(f"Question: {question}")
    print(f"Predicted: {predicted_sql}")
    print(f"Reference: {expected_sql}")
    print("-" * 50)

# Compute final metrics
final_metrics = metrics_computer.compute_metrics((predictions, references))

print("\nFinal Metrics:")
for metric_name, metric_value in final_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")