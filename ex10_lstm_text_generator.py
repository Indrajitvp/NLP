"""Ex 10: LSTM next-word text generator."""
import numpy as np
import nltk
import tensorflow as tf
from collections import Counter
from nltk.corpus import gutenberg
from tensorflow.keras import layers, models

nltk.download("gutenberg", quiet=True)
tf.keras.utils.set_random_seed(42)

NUM_SENTENCES, MAX_WORDS, EPOCHS = 3000, 25, 30

# ---- corpus + cleaning ----
sents = [[w.lower() for w in s if w.isalpha()] for s in gutenberg.sents("austen-emma.txt")[:NUM_SENTENCES]]
sents = [s for s in sents if 3 <= len(s) <= MAX_WORDS]

counts = Counter(w for s in sents for w in s)
vocab = ["<pad>"] + sorted(counts)
w2i = {w: i for i, w in enumerate(vocab)}
i2w = {i: w for w, i in w2i.items()}
V = len(vocab)

# ---- n-gram sequences ----
seqs = []
for s in sents:
    ids = [w2i[w] for w in s]
    seqs += [ids[: i + 1] for i in range(1, len(ids))]
max_len = max(map(len, seqs))
seqs = tf.keras.utils.pad_sequences(seqs, maxlen=max_len, padding="pre")
X, y = seqs[:, :-1], seqs[:, -1]
print(f"vocab={V}, sequences={len(X)}, max_len={max_len}")

# ---- model ----
model = models.Sequential([
    layers.Input(shape=(max_len - 1,)),
    layers.Embedding(V, 100, mask_zero=True),
    layers.LSTM(150),
    layers.Dense(V, activation="softmax"),
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=EPOCHS, batch_size=128, verbose=2)


def generate_text(seed, n_words, model, max_len, top_k=5):
    ids = [w2i[w] for w in seed.lower().split() if w in w2i]
    if not ids:
        return "(no known words in seed)"
    out = list(seed.lower().split())
    for _ in range(n_words):
        x = tf.keras.utils.pad_sequences([ids], maxlen=max_len - 1, padding="pre")
        p = model.predict(x, verbose=0)[0]
        p[0] = 0
        top = np.argsort(p)[-top_k:]
        nxt = int(np.random.choice(top, p=p[top] / p[top].sum()))
        ids.append(nxt)
        out.append(i2w[nxt])
    return " ".join(out).capitalize()


if __name__ == "__main__":
    for s in ["emma was", "mr knightley", "miss woodhouse"]:
        print(generate_text(s, 10, model, max_len))
    while True:
        seed = input("\nSeed words (blank to quit): ").strip()
        if not seed:
            break
        n = input("How many words [10]: ").strip()
        print(generate_text(seed, int(n) if n.isdigit() else 10, model, max_len))
