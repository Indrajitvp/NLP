"""Ex 12: LLM chatbot with multi-turn memory (Anthropic API).
Setup:  pip install anthropic   |   export ANTHROPIC_API_KEY="sk-ant-..."
(Swap-in for OpenAI: replace client.messages.create with client.chat.completions.create.)
"""
import os
import sys
import anthropic

MODEL = "claude-sonnet-5-5"
MAX_TURNS = 10      # session memory: keep the last N user/assistant exchanges
SYSTEM = ("You are a friendly assistant for a perfume shop. We sell floral, woody, citrus, "
          "oriental and fresh fragrances for men, women and unisex. Be concise and helpful; "
          "if you don't know something about the shop, say so.")

if not os.getenv("ANTHROPIC_API_KEY"):
    sys.exit("Set ANTHROPIC_API_KEY first.")
client = anthropic.Anthropic()
history = []


def chat(user_msg):
    history.append({"role": "user", "content": user_msg})
    del history[: max(0, len(history) - 2 * MAX_TURNS)]       # trim old context
    try:
        r = client.messages.create(model=MODEL, max_tokens=500, temperature=0.7,
                                   system=SYSTEM, messages=history)
        reply = r.content[0].text
    except anthropic.APIError as e:
        history.pop()
        return f"[API error: {e}]"
    history.append({"role": "assistant", "content": reply})
    return reply


if __name__ == "__main__":
    q = "What types of perfumes do you sell?"
    print(f"You: {q}\nBot: {chat(q)}\n")
    print("Chat (type 'quit' to exit)")
    while (msg := input("You: ").strip()).lower() not in {"quit", "exit"}:
        if msg:
            print("Bot:", chat(msg), "\n")
