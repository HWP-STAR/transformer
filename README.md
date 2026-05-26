# Transformer

A from-scratch implementation of the Transformer model (Vaswani et al., 2017) for English-German machine translation, trained on the Multi30k dataset.

## Project Structure

```
├── src/
│   ├── models.py          # Transformer (encoder-decoder) & MiniTransformer (decoder-only)
│   ├── data.py            # Data loading, tokenization (spaCy), vocab building
│   ├── utils.py           # Training loop, evaluation, NoamLR scheduler
│   ├── main.py            # Train custom Transformer on Multi30k
│   ├── main_compare.py    # Fine-tune MarianMT (HuggingFace) for comparison
│   └── checkpoints/       # Saved model weights
├── temp/
│   └── download.py        # Utility to load pre-trained models
└── tests/
```

## Model Architecture

The full `Transformer` follows the original paper with 6-layer encoder and 6-layer decoder, each containing:
- **Multi-Head Self-Attention** (8 heads, d_model=512)
- **Positionwise Feed-Forward** (d_ff=2048, ReLU activation)
- **Layer Normalization** & residual connections
- **Sinusoidal Positional Encoding**

## Usage

```bash
# Train the custom Transformer
python src/main.py
```

## Requirements

- PyTorch
- spaCy (`en_core_web_sm`, `de_core_news_sm`)
- 🤗 Datasets
- 🤗 Transformers (for comparison script)
