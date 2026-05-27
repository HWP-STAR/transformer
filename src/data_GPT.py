from datasets import load_from_disk
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, ids, block_size):
        self.input_ids = []
        self.labels = []
        for i in range(0, len(ids) - block_size, block_size // 2):
            chunk = ids[i:i + block_size + 1]
            if len(chunk) < block_size + 1:
                break
            self.input_ids.append(torch.tensor(chunk[:-1]))
            self.labels.append(torch.tensor(chunk[1:]))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def get_gpt_dataloaders(batch_size=32, block_size=128, stride=None):
    dataset = load_from_disk("../../data")

    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    if stride is None:
        stride = block_size // 2

    def tokenize_texts(texts):
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text)
            all_ids.extend(ids + [tokenizer.eos_token_id])
        return all_ids

    train_ids = tokenize_texts(dataset['train']['text'])
    val_ids = tokenize_texts(dataset['validation']['text'])
    test_ids = tokenize_texts(dataset['test']['text'])

    train_dataset = GPTDataset(train_ids, block_size)
    val_dataset = GPTDataset(val_ids, block_size)
    test_dataset = GPTDataset(test_ids, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer.vocab_size,tokenizer


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab_size = get_gpt_dataloaders(batch_size=4, block_size=64)
    print(f"vocab_size: {vocab_size}")
    print(f"train batches: {len(train_loader)}, val batches: {len(val_loader)}, test batches: {len(test_loader)}")
    for x, y in train_loader:
        print(f"input shape: {x.shape}, labels shape: {y.shape}")
        print(f"input sample: {x[0][:10].tolist()}")
        print(f"label sample:  {y[0][:10].tolist()}")
        break
