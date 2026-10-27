"""Ex 2: t-test and chi-square test for collocations."""
import math
import nltk
from collections import Counter
from nltk.corpus import gutenberg, stopwords

nltk.download("gutenberg", quiet=True)
nltk.download("stopwords", quiet=True)

T_CRIT = 2.576     # alpha = 0.005 (one-tailed)
CHI_CRIT = 3.841   # alpha = 0.05, df = 1

tokens = [w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha()]
stop = set(stopwords.words("english"))
N = len(tokens)
uni = Counter(tokens)
bi = Counter(zip(tokens, tokens[1:]))


def t_score(w1, w2):
    c12 = bi[(w1, w2)]
    if c12 == 0:
        return 0.0
    x_bar = c12 / N                       # observed mean
    mu = (uni[w1] / N) * (uni[w2] / N)    # expected under independence
    return (x_bar - mu) / math.sqrt(x_bar / N)   # s^2 ~ x_bar for Bernoulli


def chi_square(w1, w2):
    o11 = bi[(w1, w2)]
    o12 = uni[w1] - o11
    o21 = uni[w2] - o11
    o22 = N - o11 - o12 - o21
    obs = [[o11, o12], [o21, o22]]
    row = [sum(r) for r in obs]
    col = [obs[0][j] + obs[1][j] for j in range(2)]
    chi = 0.0
    for i in range(2):
        for j in range(2):
            e = row[i] * col[j] / N
            if e:
                chi += (obs[i][j] - e) ** 2 / e
    return chi


def check(w1, w2):
    t, chi = t_score(w1, w2), chi_square(w1, w2)
    print(f"{w1!r:>10} {w2!r:<12} t={t:7.3f} -> {'collocation' if t > T_CRIT else 'not collocation':16s}"
          f" chi2={chi:10.3f} -> {'collocation' if chi > CHI_CRIT else 'not collocation'}")


# candidate bigrams: freq >= 5, no stop words
cands = [b for b, c in bi.items() if c >= 5 and b[0] not in stop and b[1] not in stop]

print("Top 10 bigrams by t-score")
for b in sorted(cands, key=lambda b: -t_score(*b))[:10]:
    check(*b)

print("\nTop 10 bigrams by chi-square")
for b in sorted(cands, key=lambda b: -chi_square(*b))[:10]:
    check(*b)

print("\nPMI (top 5, freq >= 5)")
pmi = lambda b: math.log2((bi[b] / N) / ((uni[b[0]] / N) * (uni[b[1]] / N)))
for b in sorted(cands, key=lambda b: -pmi(b))[:5]:
    print(f"  {b}  PMI={pmi(b):.3f}")

print("\nTest specific pairs")
for pair in [("mr", "knightley"), ("miss", "woodhouse"), ("good", "morning"), ("of", "the"), ("said", "emma")]:
    check(*pair)
