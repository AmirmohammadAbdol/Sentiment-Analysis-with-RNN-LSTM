import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from collections import Counter
import re


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def tokenize(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return text.split()

class Vocabulary:
    def __init__(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {idx: word for word, idx in word2idx.items()}
        self.pad_idx = word2idx[PAD_TOKEN]
        self.unk_idx = word2idx[UNK_TOKEN]
    def __len__(self):
        return len(self.word2idx)

class ImdbDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
    def __getitem__(self, idx):
        text_tokens = self.texts[idx]
        label = self.labels[idx]
        word_ids = [self.vocab.word2idx.get(token, self.vocab.unk_idx) for token in text_tokens]
        return torch.tensor(word_ids, dtype=torch.long), label    
    def __len__(self):
        return len(self.texts)

def load_imdb_data(data_dir):
    data_path = Path(data_dir)
    train_texts = []
    train_labels = []
    test_texts = []
    test_labels = []
    for sentiment, label in [('neg', 0), ('pos', 1)]:
        folder = data_path / 'train' / sentiment
        for file in folder.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                train_texts.append(f.read())
                train_labels.append(label)
    for sentiment, label in [('neg', 0), ('pos', 1)]:
        folder = data_path / 'test' / sentiment
        for file in folder.glob('*.txt'):
            with open(file, 'r', encoding='utf-8') as f:
                test_texts.append(f.read())
                test_labels.append(label)
    return train_texts, train_labels, test_texts, test_labels

def build_vocab(train_texts, vocab_size=20000, min_freq=5):
    all_tokens = []
    for text in train_texts:
        tokens = tokenize(text)
        all_tokens.extend(tokens)
    word_counts = Counter(all_tokens)
    word2idx = {PAD_TOKEN: 0,  UNK_TOKEN: 1}
    index = 2
    for word, count in word_counts.most_common(vocab_size-2):
        if count >= min_freq:
            word2idx[word] = index
            index += 1
    return Vocabulary(word2idx)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences = []
    labels = []
    for seq, label in batch:
        sequences.append(seq)
        labels.append(label)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(lengths), torch.tensor(labels)

def create_data_loader(train_texts, train_labels, test_texts, test_labels, vocab, batch_size=64):
    train_tokenized = [tokenize(text) for text in train_texts]
    test_tokenized = [tokenize(text) for text in test_texts]
    train_dataset = ImdbDataset(train_tokenized, train_labels, vocab)
    test_dataset = ImdbDataset(test_tokenized, test_labels, vocab)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, test_loader

def get_data_loaders_and_vocab(data_dir, batch_size=64, vocab_size=20000, min_freq=5):
    train_texts, train_labels, test_texts, test_labels = load_imdb_data(data_dir)
    vocab = build_vocab(train_texts, vocab_size, min_freq)
    train_loader, test_loader = create_data_loader(train_texts, train_labels, test_texts, test_labels, vocab, batch_size)
    return train_loader, test_loader, vocab