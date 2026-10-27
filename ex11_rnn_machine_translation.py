"""Ex 11: Character-level LSTM seq2seq (encoder-decoder) English -> French.
Uses a small built-in dataset; if 'fra.txt' (Anki: en<TAB>fr<TAB>...) is present it is used instead.
"""
import os
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

LATENT, EPOCHS, BATCH, NUM_SAMPLES = 256, 150, 16, 10000

pairs = [("Go.", "Va !"), ("Run!", "Cours !"), ("Who?", "Qui ?"), ("Wow!", "Ça alors !"),
         ("Fire!", "Au feu !"), ("Help!", "À l'aide !"), ("Jump.", "Saute."), ("Stop!", "Ça suffit !"),
         ("Wait!", "Attends !"), ("Hello!", "Bonjour !"), ("I try.", "J'essaye."), ("I won!", "J'ai gagné !"),
         ("Oh no!", "Oh non !"), ("Attack!", "Attaque !"), ("Cheers!", "Santé !"), ("Get up.", "Lève-toi."),
         ("Go on.", "Poursuis."), ("I know.", "Je sais."), ("I left.", "Je suis parti."), ("Hug me.", "Serre-moi dans tes bras !")]
if os.path.exists("fra.txt"):
    with open("fra.txt", encoding="utf-8") as f:
        pairs = [l.rstrip("\n").split("\t")[:2] for l in f.read().split("\n")[:NUM_SAMPLES] if "\t" in l]
    EPOCHS = 50

src_texts = [e for e, _ in pairs]
tgt_texts = ["\t" + f + "\n" for _, f in pairs]          # \t = <start>, \n = <end>

src_chars = sorted({c for t in src_texts for c in t})
tgt_chars = sorted({c for t in tgt_texts for c in t})
s2i = {c: i for i, c in enumerate(src_chars)}
t2i = {c: i for i, c in enumerate(tgt_chars)}
i2t = {i: c for c, i in t2i.items()}
max_src, max_tgt = max(map(len, src_texts)), max(map(len, tgt_texts))

enc_in_data = np.zeros((len(pairs), max_src, len(src_chars)), dtype="float32")
dec_in_data = np.zeros((len(pairs), max_tgt, len(tgt_chars)), dtype="float32")
dec_out_data = np.zeros_like(dec_in_data)
for n, (s, t) in enumerate(zip(src_texts, tgt_texts)):
    for i, c in enumerate(s):
        enc_in_data[n, i, s2i[c]] = 1.0
    for i, c in enumerate(t):
        dec_in_data[n, i, t2i[c]] = 1.0
        if i > 0:
            dec_out_data[n, i - 1, t2i[c]] = 1.0        # teacher forcing: shifted by one

# ---- training model ----
enc_inputs = Input(shape=(None, len(src_chars)))
_, h, c = LSTM(LATENT, return_state=True)(enc_inputs)
dec_inputs = Input(shape=(None, len(tgt_chars)))
dec_lstm = LSTM(LATENT, return_sequences=True, return_state=True)
dec_seq, _, _ = dec_lstm(dec_inputs, initial_state=[h, c])
dec_dense = Dense(len(tgt_chars), activation="softmax")
model = Model([enc_inputs, dec_inputs], dec_dense(dec_seq))
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
model.fit([enc_in_data, dec_in_data], dec_out_data, batch_size=BATCH, epochs=EPOCHS, validation_split=0.1, verbose=2)

# ---- inference models ----
encoder_model = Model(enc_inputs, [h, c])
dh, dc = Input(shape=(LATENT,)), Input(shape=(LATENT,))
d_out, nh, nc = dec_lstm(dec_inputs, initial_state=[dh, dc])
decoder_model = Model([dec_inputs, dh, dc], [dec_dense(d_out), nh, nc])


def translate(sentence):
    x = np.zeros((1, max_src, len(src_chars)), dtype="float32")
    for i, ch in enumerate(sentence[:max_src]):
        if ch in s2i:
            x[0, i, s2i[ch]] = 1.0
    state = encoder_model.predict(x, verbose=0)
    tok = np.zeros((1, 1, len(tgt_chars)), dtype="float32")
    tok[0, 0, t2i["\t"]] = 1.0
    out = ""
    for _ in range(max_tgt):
        probs, hh, cc = decoder_model.predict([tok] + state, verbose=0)
        ch = i2t[int(np.argmax(probs[0, -1]))]
        if ch == "\n":
            break
        out += ch
        tok = np.zeros((1, 1, len(tgt_chars)), dtype="float32")
        tok[0, 0, t2i[ch]] = 1.0
        state = [hh, cc]
    return out


print("\nEnglish to French Translation")
for s in ["Go.", "Run!", "Hello!", "Help!", "I know."]:
    print(f"Input sentence: {s}\nDecoded sentence: {translate(s)}\n")
