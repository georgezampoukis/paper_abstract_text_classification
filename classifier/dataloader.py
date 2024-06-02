from torch import Tensor, LongTensor, from_numpy
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from transformers import BertTokenizer





class DriveDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(dataset)}")
        
        self.abstract_texts: list[str] = dataset['abstractText'].tolist()
        self.labels: np.ndarray = dataset[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']].values
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        # Extract Sample Abstract Text
        abstract_text: str = self.abstract_texts[index]

        # Get BERT Inputs
        bert_inputs: dict[str, Tensor] = self.tokenizer(abstract_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512)

        # Prepare Classifier Inputs
        input_ids: Tensor = bert_inputs['input_ids'].squeeze()
        attention_mask: Tensor = bert_inputs['attention_mask'].squeeze()
        token_type_ids: Tensor = bert_inputs['token_type_ids'].squeeze()

        # Extract Corresponding Sample Label
        text_label: np.ndarray = np.float32(self.labels[index])

        return LongTensor(input_ids), LongTensor(attention_mask), LongTensor(token_type_ids), from_numpy(text_label)


    def __len__(self) -> int:
        return len(self.abstract_texts)
    