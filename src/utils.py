import os
import time
import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR


class NoamLR(LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        def lr_lambda(step):
            step = max(step, 1)
            return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
        super().__init__(optimizer, lr_lambda)


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0
    for src, tgt_input, tgt_output in dataloader:
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)

            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
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

        if val_loss < best_val_loss and epoch+1 >5:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'{save_path}/best_model_{epoch+1}.pt')
            print(f'  -> saved best model (val_loss: {val_loss:.4f})')



@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    device: str = "cpu",
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    model.eval()
    prompt_ids = tokenizer.encode(prompt, truncation=True, max_length=512)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id or 0]
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = model.generate(
        idx, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k,
    )
    output_ids = generated[0].tolist()
    return tokenizer.decode(output_ids)