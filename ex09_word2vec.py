"""Ex 9: Word2Vec (CBOW and Skip-gram) and semantic similarity."""
import nltk
from gensim.models import Word2Vec
from nltk.corpus import gutenberg, stopwords
from nltk.stem import WordNetLemmatizer

for p in ["gutenberg", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(p, quiet=True)

stop, lem = set(stopwords.words("english")), WordNetLemmatizer()
sentences = [[lem.lemmatize(w.lower()) for w in s if w.isalpha() and w.lower() not in stop]
             for s in gutenberg.sents("carroll-alice.txt")]
sentences = [s for s in sentences if len(s) > 1]

params = dict(sentences=sentences, vector_size=100, window=5, min_count=3, workers=1, seed=42, epochs=50)
cbow = Word2Vec(sg=0, **params)
skip = Word2Vec(sg=1, **params)

pairs = [("alice", "queen"), ("king", "queen"), ("rabbit", "hatter"), ("cat", "dog"), ("alice", "rabbit")]
for a, b in pairs:
    for name, m in (("CBOW", cbow), ("Skip Gram", skip)):
        if a in m.wv and b in m.wv:
            print(f"Cosine similarity between '{a}' and '{b}' - {name} : {m.wv.similarity(a, b):.6f}")
        else:
            print(f"'{a}' or '{b}' not in vocabulary")

for w in ["queen", "alice"]:
    print(f"\nMost similar to '{w}' (Skip Gram):", [(x, round(s, 3)) for x, s in skip.wv.most_similar(w, topn=5)])
