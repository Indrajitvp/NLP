"""Ex 4: Hindle & Rooth PP-attachment.
lambda(v,n,p) = log2[ P(VA_p=1|v) * P(NA_p=0|n) / P(NA_p=1|n) ];  lambda > 0 -> verb, else noun.
"""
import math
from collections import Counter

# (verb, noun, preposition, attachment) -- toy training corpus; V = verb attach, N = noun attach
data = [
    ("saw", "man", "with", "N"), ("saw", "star", "with", "V"), ("saw", "girl", "with", "V"),
    ("hit", "ball", "with", "V"), ("hit", "man", "with", "V"), ("hit", "ball", "with", "V"),
    ("bought", "book", "with", "N"), ("bought", "book", "with", "N"), ("bought", "car", "with", "V"),
    ("ate", "pizza", "with", "V"), ("ate", "pizza", "with", "N"), ("ate", "salad", "with", "V"),
    ("read", "book", "with", "N"), ("read", "story", "with", "V"), ("opened", "door", "with", "V"),
    ("opened", "box", "with", "V"), ("found", "ball", "with", "N"), ("found", "ball", "with", "N"),
    ("found", "key", "with", "N"), ("kicked", "ball", "with", "V"),
]

C_v, C_vp, C_n, C_np = Counter(), Counter(), Counter(), Counter()
for v, n, p, att in data:
    C_v[v] += 1
    C_n[n] += 1
    if att == "V":
        C_vp[(v, p)] += 1
    else:
        C_np[(n, p)] += 1


def prob(count, total, alpha=0.5):  # smoothed
    return (count + alpha) / (total + 2 * alpha)


def attach(v, n, p):
    p_va = prob(C_vp[(v, p)], C_v[v])         # P(VA_p = 1 | v)
    p_na = prob(C_np[(n, p)], C_n[n])         # P(NA_p = 1 | n)
    lam = math.log2(p_va * (1 - p_na) / p_na)
    return lam, ("Verb" if lam > 0 else "Noun")


if __name__ == "__main__":
    for v, n, p in [("hit", "ball", "with"), ("found", "ball", "with"),
                    ("saw", "man", "with"), ("opened", "door", "with")]:
        lam, res = attach(v, n, p)
        print(f"{v} {n} {p}: lambda={lam:+.3f} -> {res} attachment")

    print("\nTry your own:")
    v = input("Verb: ").strip().lower()
    n = input("Noun: ").strip().lower()
    p = input("Preposition [with]: ").strip().lower() or "with"
    lam, res = attach(v, n, p)
    print(f"lambda={lam:+.3f} -> {res}")
