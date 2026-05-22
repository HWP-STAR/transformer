from collections import Counter
from datasets import load_from_disk
import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def get_tokenizers():
    import spacy
    nlp_en = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
    nlp_de = spacy.load('de_core_news_sm', disable=['parser', 'ner', 'tagger'])
    return nlp_en, nlp_de


def tokenize(text, nlp):
    return [token.text.lower() for token in nlp(text)]


class Vocab:
    def __init__(self, tokenized_sentences, min_freq=2):
        counter = Counter()
        for tokens in tokenized_sentences:
            counter.update(tokens)

        specials = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
        self.itos = list(specials)
        self.stoi = {s: i for i, s in enumerate(specials)}

        for token, freq in counter.items():
            if freq >= min_freq:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [self.stoi.get(t, UNK_IDX) for t in tokens]


class Multi30kDataset(Dataset):
    def __init__(self, src_tokens, tgt_tokens, src_vocab, tgt_vocab):
        self.src = []
        self.tgt = []
        for s, t in zip(src_tokens, tgt_tokens):
            self.src.append(torch.tensor([SOS_IDX] + src_vocab.encode(s) + [EOS_IDX]))
            self.tgt.append(torch.tensor([SOS_IDX] + tgt_vocab.encode(t) + [EOS_IDX]))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]


def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_IDX)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_IDX)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    return src, tgt_input, tgt_output


def get_dataloaders(batch_size=32, min_freq=2):
    dataset = load_from_disk("../../data_disk/multi30k")

    nlp_en, nlp_de = get_tokenizers()

    src_tokens = []
    tgt_tokens = []
    for item in dataset['train']:
        src_tokens.append(tokenize(item['en'], nlp_en))
        tgt_tokens.append(tokenize(item['de'], nlp_de))

    src_vocab = Vocab(src_tokens, min_freq)
    tgt_vocab = Vocab(tgt_tokens, min_freq)

    train_dataset = Multi30kDataset(src_tokens, tgt_tokens, src_vocab, tgt_vocab)

    val_src = [tokenize(item['en'], nlp_en) for item in dataset['validation']]
    val_tgt = [tokenize(item['de'], nlp_de) for item in dataset['validation']]
    val_dataset = Multi30kDataset(val_src, val_tgt, src_vocab, tgt_vocab)

    test_src = [tokenize(item['en'], nlp_en) for item in dataset['test']]
    test_tgt = [tokenize(item['de'], nlp_de) for item in dataset['test']]
    test_dataset = Multi30kDataset(test_src, test_tgt, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
