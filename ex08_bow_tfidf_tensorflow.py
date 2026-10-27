"""Ex 8: Text classification with Bag of Words and TF-IDF (TensorFlow)."""
import re
import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

for p in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(p, quiet=True)
tf.keras.utils.set_random_seed(42)

texts = [
    "I love machine learning and AI", "AI is the future of technology",
    "TensorFlow is a powerful tool", "I enjoy programming with TensorFlow",
    "Deep learning models are amazing", "This library is great and easy to use",
    "I hate bugs in my code", "The tool is slow and terrible",
    "This update is awful and broken", "I dislike confusing error messages",
    "The software crashed again, very frustrating", "Worst documentation I have ever read",
]
labels = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

# ---- cleaning ----
stop, lem = set(stopwords.words("english")) - {"not", "no"}, WordNetLemmatizer()
clean = lambda s: " ".join(lem.lemmatize(w) for w in re.findall(r"[a-z]+", s.lower()) if w not in stop)
texts = [clean(t) for t in texts]

X_tr, X_va, y_tr, y_va = train_test_split(texts, labels, test_size=0.25, stratify=labels, random_state=42)
train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(4)
val_ds = tf.data.Dataset.from_tensor_slices((X_va, y_va)).batch(4)

# ---- vectorizers ----
bow = layers.TextVectorization(output_mode="count")
tfidf = layers.TextVectorization(output_mode="tf_idf")
bow.adapt(train_ds.map(lambda x, y: x))
tfidf.adapt(train_ds.map(lambda x, y: x))
vocab = bow.get_vocabulary()
print("Vocabulary:", vocab)
print("\nBoW table:\n", pd.DataFrame(bow(tf.constant(X_tr)).numpy(), columns=vocab).round(3).to_string())
print("\nTF-IDF table:\n", pd.DataFrame(tfidf(tf.constant(X_tr)).numpy(), columns=vocab).round(3).to_string())


def build_and_train(vectorizer, name):
    tr = train_ds.map(lambda x, y: (vectorizer(x), y))
    va = val_ds.map(lambda x, y: (vectorizer(x), y))
    model = models.Sequential([
        layers.Input(shape=(vectorizer.vocabulary_size(),)),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(tr, validation_data=va, epochs=30, verbose=0)
    loss, acc = model.evaluate(va, verbose=0)
    print(f"\n{name}: validation accuracy = {acc:.2f}")
    return model


models_ = {"Bag of Words": (build_and_train(bow, "Bag of Words"), bow),
           "TF-IDF": (build_and_train(tfidf, "TF-IDF"), tfidf)}

new = [clean("I love this amazing tool"), clean("This is a terrible broken update")]
for name, (m, vec) in models_.items():
    pred = m.predict(vec(tf.constant(new)), verbose=0).ravel()
    print(name, "->", [f"{'positive' if p > .5 else 'negative'} ({p:.2f})" for p in pred])
