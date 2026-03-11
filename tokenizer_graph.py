import matplotlib.pyplot as plt
from transformers import AutoTokenizer

models = ["bert-base-cased", "gpt2", "google/flan-t5-small"]

sentence = "Artificial intelligence is transforming the world"

counts = []

for model in models:
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokens = tokenizer.tokenize(sentence)
    counts.append(len(tokens))

plt.bar(models, counts)
plt.title("Token Count Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("Number of Tokens")
plt.show()