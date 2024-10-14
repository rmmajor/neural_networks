from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
import torch
import pandas as pd


class DruglibDataset(Dataset):
    def __init__(self, reviews: pd.Series, targets: pd.Series, tokenizer: PreTrainedTokenizer, max_len: int):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.reviews)

    def __getitem__(self, idx: int) -> dict:
        review_text = str(self.reviews.iloc[idx])
        target = self.targets.iloc[idx]

        encoding = self.tokenizer.encode_plus(
            review_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'review_text': review_text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def gen_data_loader(reviews, targets, tokenizer, max_len, batch_size):
    ds = DruglibDataset(reviews, targets, tokenizer, max_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=8)
