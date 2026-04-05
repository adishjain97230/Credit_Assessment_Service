import pandas as pd

df = pd.read_csv("unigram_freq.csv")

df = df[df["word"].str.len() == 5].reset_index(drop=True)

word_counts = df.set_index("word")["count"].to_dict()

words = list(word_counts.keys())

words.sort(key=lambda x: word_counts[x], reverse=True)

print(words[-10:])