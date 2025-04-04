from torch.utils.data import Dataset
import json
from typing import Optional, Dict, List

class SQLDataset(Dataset):
    def __init__(
        self, 
        file_path: str, 
        tokenizer, 
        max_input_length: int = 128,
        max_output_length: int = 256,
        schema_context: Optional[str] = None 
    ):
        """Initialize SQL Dataset.
        
        Args:
            file_path: Path to JSONL file containing examples
            tokenizer: HuggingFace tokenizer
            max_input_length: Maximum length for input sequences
            max_output_length: Maximum length for output sequences
            schema_context: Optional database schema context
        """
        self.data = self._load_data(file_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.schema_context = schema_context
        
    def _load_data(self, file_path: str) -> List[Dict]:
        """Load examples from JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    data.append(item)
        return data
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a single example with tokenized inputs and labels.
           Will be used by Dataloader to get a batch of tokenized examples.
        """
        question = self.data[idx]['input']
        sql_query = self.data[idx]['output']
        
        # Prepare input text with schema context
        input_text = f"translate to SQL: {question}"
        
        # print('Input text: ', input_text)
        # print('SQL query: ', sql_query)
        
        # Tokenize input
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target SQL query
        labels = self.tokenizer(
            sql_query,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels['input_ids'].squeeze()
        } 