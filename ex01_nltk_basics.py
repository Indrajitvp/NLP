"""Ex 1: Tokenize, POS-tag, NER, parse tree, parse-tree ambiguity (NLTK)."""
import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag, ne_chunk, CFG, ChartParser

for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng", "maxent_ne_chunker",
            "maxent_ne_chunker_tab", "words", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg, quiet=True)

text = ("Sundar Pichai is the CEO of Google. He visited Chennai in March 2025. "
        "The sun moves eastward.")

# 1. Tokenization
sentences = sent_tokenize(text)
print("Sentences:", sentences)
words = word_tokenize(text)
print("Words:", words)

# 2. POS tagging
tagged = pos_tag(words)
print("\nPOS tags:", tagged)

# 3. Named entity recognition
tree = ne_chunk(tagged)
print("\nNamed entities:")
for st in tree:
    if hasattr(st, "label"):
        print(f"  {st.label():13s} {' '.join(w for w, _ in st.leaves())}")

# 4. Parse tree from a grammar
grammar = CFG.fromstring("""
S  -> NP VP
NP -> A NN
VP -> V NN
A  -> 'The'
NN -> 'sun' | 'eastward'
V  -> 'moves'
""")
parser = ChartParser(grammar)
for t in parser.parse("The sun moves eastward".split()):
    print("\nParse tree:", t)
    t.pretty_print()

# 5. Ambiguity via multiple parse trees
amb_grammar = CFG.fromstring("""
S  -> NP VP
PP -> P NP
NP -> Det N | Det N PP | 'I'
VP -> V NP | VP PP
Det -> 'the'
N  -> 'man' | 'telescope'
V  -> 'saw'
P  -> 'with'
""")
sentence = "I saw the man with the telescope".split()
trees = list(ChartParser(amb_grammar).parse(sentence))
print(f"\n'{' '.join(sentence)}' has {len(trees)} parse(s)")
for i, t in enumerate(trees, 1):
    print(f"\nParse {i}:")
    t.pretty_print()
print("Sentence is AMBIGUOUS" if len(trees) > 1 else "Sentence is unambiguous")
