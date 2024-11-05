import os
from torch.utils.data import Dataset
import torch


class SpeechesClassificationDataset(Dataset):
    """
    Dataset class for text classification task.
    This the dataset you will use to train your encoder, and classifier jointly, 
    end-to-end for the text classification task.

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode the text.
        file_path (str): The path to the file containing the speech classification data.

    """

    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.samples = []

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                label, text = line.strip().split('\t')
                if label not in ('0', '1', '2'):
                    raise ValueError(f"Invalid label: {label}")
                if len(text.strip()) == 0:
                    continue
                self.samples.append((int(label), text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, text = self.samples[index]
        input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return input_ids, label_tensor
    
    


class LanguageModelingDataset(torch.utils.data.Dataset):
    """
    Dataset class for language modeling task. This is the dataset you will use to train your encoder for the language modeling task. 

    Args:
        tokenizer (Tokenizer): The tokenizer used to encode the text.
        text (str): The text data.
        block_size (int): The size of each block of text.
    """

    def __init__(self, tokenizer, text, block_size):
        self.tokenizer = tokenizer
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y