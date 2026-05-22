import torch
import torch.nn as nn
from data import get_dataloaders
from models import Transformer
from utils import train, NoamLR


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    batch_size = 32
    epochs = 3

    print('Loading data and building vocab...')
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders(batch_size=batch_size)
    print(f'Source vocab size: {len(src_vocab)}')
    print(f'Target vocab size: {len(tgt_vocab)}')

    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=512,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=200,
        dropout=0.1,
        device=device,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=512, warmup_steps=4000)

    train(model, train_loader, val_loader, epochs, optimizer, criterion, device, scheduler=scheduler)


if __name__ == '__main__':
    main()
