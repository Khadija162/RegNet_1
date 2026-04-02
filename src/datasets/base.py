from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetSpec:
    name: str
    file_path: str
    textual_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_model: str = "distilbert-base-uncased"
    target_column: Optional[str] = None
