"""Ex 3: Naive Bayes word sense disambiguation (word: 'chair')."""
import math
import re
from collections import Counter

STOP = {"the", "a", "an", "of", "on", "in", "to", "and", "was", "is", "for", "with", "he", "she", "his", "her", "by", "at"}
TARGET = "chair"

train = {
    "furniture": [
        "she sat on the wooden chair near the table",
        "the chair and the desk were in the office",
        "he bought a comfortable chair with soft cushion",
        "move the chair close to the dining table",
        "the old chair has a broken leg",
        "a leather chair stood beside the sofa",
    ],
    "position": [
        "the chair of the committee announced the meeting",
        "he was appointed chair of the department",
        "the chair presided over the board meeting",
        "she became chair of the university senate",
        "the chair resigned from the position yesterday",
        "the chair proposed a vote on the new policy",
    ],
}


def tokens(s):
    return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and w != TARGET]


# ---- Training ----
sense_count = {s: len(docs) for s, docs in train.items()}
total = sum(sense_count.values())
word_count = {s: Counter(w for d in docs for w in tokens(d)) for s, docs in train.items()}
vocab = {w for c in word_count.values() for w in c}
prior = {s: sense_count[s] / total for s in train}


def p_word(w, s):   # add-one smoothing
    return (word_count[s][w] + 1) / (sum(word_count[s].values()) + len(vocab))


# ---- Disambiguation ----
def disambiguate(sentence):
    ctx = tokens(sentence)
    scores = {s: math.log(prior[s]) + sum(math.log(p_word(w, s)) for w in ctx) for s in train}
    best = max(scores, key=scores.get)
    return best, scores


for test in ["the chair of the board spoke first",
             "she sat on the old chair near the table"]:
    sense, scores = disambiguate(test)
    print(f"Sentence: {test}")
    print(f"The sense of 'Chair' is {sense}. {scores[sense]}")
    print("Scores:", {k: round(v, 4) for k, v in scores.items()}, "\n")
