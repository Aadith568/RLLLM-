import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

MAX_VOCAB = 10000
MAX_LEN = 200


def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.lower().split())

    most_common = counter.most_common(MAX_VOCAB - 2)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for i, (word, _) in enumerate(most_common, start=2):
        vocab[word] = i

    return vocab


def encode_text(text, vocab):
    tokens = text.lower().split()
    encoded = [vocab.get(word, 1) for word in tokens]
    encoded = encoded[:MAX_LEN]
    encoded += [0] * (MAX_LEN - len(encoded))
    return encoded


def load_imdb_csv(path="data/cleaned_dataset.csv"):
    df = pd.read_csv(path)

    df["sentiment"] = df["sentiment"].map({
        "positive": 1,
        "negative": 0
    })

    texts = df["review"].astype(str).values
    labels = df["sentiment"].values.astype(np.float32)

    vocab = build_vocab(texts)

    encoded = np.array([encode_text(t, vocab) for t in texts])

    x_train, x_test, y_train, y_test = train_test_split(
        encoded,
        labels,
        test_size=0.2,
        random_state=42
    )

    return x_train, y_train, x_test, y_test, len(vocab)
