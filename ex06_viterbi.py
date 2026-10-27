"""Ex 6: Viterbi algorithm - best hidden state (tag) sequence and its probability."""


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{s: start_p[s] * emit_p[s][obs[0]] for s in states}]
    back = [{}]
    for t in range(1, len(obs)):
        V.append({})
        back.append({})
        for s in states:
            prob, prev = max((V[t - 1][r] * trans_p[r][s] * emit_p[s][obs[t]], r) for r in states)
            V[t][s], back[t][s] = prob, prev
    best_prob, last = max((V[-1][s], s) for s in states)
    path = [last]
    for t in range(len(obs) - 1, 0, -1):
        path.insert(0, back[t][path[0]])
    return round(best_prob, 6), path


if __name__ == "__main__":
    states = ("Rainy", "Sunny")
    observations = ("Walk", "Shop", "Clean")
    start = {"Rainy": 0.6, "Sunny": 0.4}
    trans = {"Rainy": {"Rainy": 0.7, "Sunny": 0.3},
             "Sunny": {"Rainy": 0.4, "Sunny": 0.6}}
    emit = {"Rainy": {"Walk": 0.1, "Shop": 0.4, "Clean": 0.5},
            "Sunny": {"Walk": 0.6, "Shop": 0.3, "Clean": 0.1}}
    print(viterbi(observations, states, start, trans, emit))   # (0.01344, ['Sunny','Rainy','Rainy'])

    # POS-tagging flavour
    tags = ("DET", "NOUN", "VERB")
    words = ("the", "dog", "barks")
    s = {"DET": 0.8, "NOUN": 0.15, "VERB": 0.05}
    tr = {"DET": {"DET": 0.01, "NOUN": 0.98, "VERB": 0.01},
          "NOUN": {"DET": 0.1, "NOUN": 0.2, "VERB": 0.7},
          "VERB": {"DET": 0.5, "NOUN": 0.4, "VERB": 0.1}}
    em = {"DET": {"the": 0.9, "dog": 0.0, "barks": 0.0},
          "NOUN": {"the": 0.0, "dog": 0.5, "barks": 0.1},
          "VERB": {"the": 0.0, "dog": 0.01, "barks": 0.6}}
    print(viterbi(words, tags, s, tr, em))
