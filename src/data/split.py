import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Dict, List

SEED = 42

def split_dataset(input_file: str, train_size: float = 0.7, val_size: float = 0.15):
    """
    Split dataset into train/validation/test sets.
    
    Args:
        input_file: Path to input JSONL file
        train_size: Proportion for training (default 0.7 = 70%)
        val_size: Proportion for validation (default 0.15 = 15%)
        # test_size will be the remainder (0.15 = 15%)
    """
    # Read all examples
    examples = []
    seen = set()
    with open(input_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            example_str = json.dumps(example, sort_keys=True)
            if example_str not in seen:
                examples.append(example)
                seen.add(example_str)
    
    print(f"Total examples after removing duplicates: {len(examples)}")
    
    # First split into train and temp (validation + test)
    train_data, temp_data = train_test_split(
        examples, 
        train_size=train_size,
        random_state=SEED  # for reproducibility
    )
    
    # Split temp into validation and test
    val_ratio = val_size / (1 - train_size)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_ratio,
        random_state=SEED
    )
    
    # Verify no overlap between splits
    train_set = {json.dumps(ex, sort_keys=True) for ex in train_data}
    val_set = {json.dumps(ex, sort_keys=True) for ex in val_data}
    test_set = {json.dumps(ex, sort_keys=True) for ex in test_data}
    
    print(f"\nSplit sizes:")
    print(f"Train set: {len(train_set)} examples")
    print(f"Val set: {len(val_set)} examples")
    print(f"Test set: {len(test_set)} examples")
    
    assert not (train_set & val_set), "Overlap found between train and validation sets"
    assert not (train_set & test_set), "Overlap found between train and test sets"
    assert not (val_set & test_set), "Overlap found between validation and test sets"
    
    # Create output directory if it doesn't exist
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    # Write splits to files
    for split_name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data)
    ]:
        output_file = f"data/processed/{split_name}.jsonl"
        with open(output_file, 'w') as f:
            for example in split_data:
                f.write(json.dumps(example) + '\n')
        
        print(f"{split_name} split: {len(split_data)} examples")

if __name__ == "__main__":
    split_dataset("data/synthetic_examples.jsonl")
