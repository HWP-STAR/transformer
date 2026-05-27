import os
import time
import torch
import torch.nn as nn
from data_GPT import get_gpt_dataloaders
from models_GPT import GPT
from utils import NoamLR,generate_text


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None, clip=1.0):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train(model, train_loader, val_loader, epochs, optimizer, criterion, device,
          scheduler=None, save_path='checkpoints'):
    os.makedirs(save_path, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss = evaluate(model, val_loader, criterion, device)

        elapsed = time.time() - start
        print(f'Epoch {epoch:2d}/{epochs} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | time: {elapsed:.1f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best_gpt.pt')
            print(f'  -> saved best model (val_loss: {val_loss:.4f})')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Hyperparameters
    batch_size = 32
    block_size = 128
    d_model = 512
    n_head = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    epochs = 3
    warmup_steps = 4000

    print('Loading data...')
    train_loader, val_loader, _, vocab_size ,tokenizer= get_gpt_dataloaders(
        batch_size=batch_size, block_size=block_size
    )
    print(f'Vocab size: {vocab_size}')
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')

    model = GPT(
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_layers=num_layers,
        d_ff=d_ff,
        max_len=block_size,
        dropout=dropout,
        device=device,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')

    # 加载 checkpoint 权重
    checkpoint_path = 'checkpoints/best_gpt.pt'
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f'Loaded checkpoint from {checkpoint_path}')
    else:
        print(f'Checkpoint {checkpoint_path} not found, training from scratch...')
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamLR(optimizer, d_model=d_model, warmup_steps=warmup_steps)
        train(model, train_loader, val_loader, epochs, optimizer, criterion, device, scheduler=scheduler)

    print("\nGenerating sample text...")
    prompts = ["The meaning of life is", "Artificial intelligence will", "In the future"]
    for prompt in prompts:
        output = generate_text(model, tokenizer, prompt, max_new_tokens=50, device=device)
        print(f"Prompt: {prompt}")
        print(f"Output: {output}\n")

    print("Done!")



if __name__ == '__main__':
    main()
