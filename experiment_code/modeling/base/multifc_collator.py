from dataclasses import dataclass
from typing import Union, Optional

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


@dataclass
class DataCollatorMultiFC:
    """
    Collator to read MultiFC data for training.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features]
        texts = [feature['text'] for feature in features]
        batch = self.tokenizer.pad(
            texts,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch
