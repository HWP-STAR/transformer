import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, MarianTokenizer
from datasets import load_from_disk


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    disk_path = "../../data_disk/opus-mt-en-de"
    print('Loading tokenizer and model...')
    tokenizer = MarianTokenizer.from_pretrained(disk_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(disk_path).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    print('Loading dataset...')
    dataset = load_from_disk("../../data_disk/multi30k")

    def tokenize_batch(batch):
        inputs = tokenizer(batch['en'], max_length=128, truncation=True, padding='max_length')
        targets = tokenizer(batch['de'], max_length=128, truncation=True, padding='max_length')
        labels = np.array(targets['input_ids'])
        labels[labels == tokenizer.pad_token_id] = -100
        batch['input_ids'] = inputs['input_ids']
        batch['attention_mask'] = inputs['attention_mask']
        batch['labels'] = labels.tolist()
        return batch

    print('Tokenizing training data...')
    train_dataset = dataset['train'].map(tokenize_batch, batched=True, batch_size=1000)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 3

    print(f'\nTraining for {num_epochs} epochs...')
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f'Epoch {epoch + 1}, batch {num_batches}: loss = {loss.item():.4f}')

        print(f'Epoch {epoch + 1}/{num_epochs} | avg loss: {total_loss / num_batches:.4f}')

    print('Done!')


if __name__ == '__main__':
    main()
