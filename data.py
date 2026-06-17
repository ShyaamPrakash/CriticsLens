import torch
import pandas as pd
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


def build_vocab_from_texts(texts, min_freq=5):
    """Build vocab from training texts."""
    word_counts = Counter()
    for text in texts:
        word_counts.update(preprocess_text(str(text)))

    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    print(f"Vocab size: {len(vocab)}")
    return vocab


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.unk_idx = vocab['<UNK>']

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        words = preprocess_text(str(self.texts[index]))
        label = self.labels[index]

        tokens = [self.vocab.get(word, self.unk_idx) for word in words]

        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return torch.LongTensor(tokens), torch.tensor(label, dtype=torch.long)


def make_dataloader(texts, labels, vocab, batch_size=32, shuffle=False):
    """Create a dataloader from texts and labels."""
    ds = ReviewDataset(texts, labels, vocab)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_and_split_data(csv_path, test_size=0.2, random_state=42):
    """Load CSV and return train/val/test splits."""
    df = pd.read_csv(csv_path)
    texts = df['review'].to_numpy()[:20000]
    labels = (df['sentiment'].to_numpy() == 'positive').astype(int)[:20000]

    train_t, temp_t, train_l, temp_l = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    val_t, test_t, val_l, test_l = train_test_split(
        temp_t, temp_l, test_size=0.5, random_state=random_state, stratify=temp_l
    )

    return (train_t, val_t, test_t), (train_l, val_l, test_l)