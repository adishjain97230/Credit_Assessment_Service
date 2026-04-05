import pandas as pd
import pickle
from pathlib import Path

df = pd.read_csv("wordle/unigram_freq.csv")

df = df[df["word"].str.len() == 5].reset_index(drop=True)

word_counts = df.set_index("word")["count"].to_dict()

with open("wordle/wordle-allowed-guesses.txt", "r") as f:
    allowed_guesses = [line.strip() for line in f if line.strip()]

word_counts_v2 = {allowed_guess: word_counts[allowed_guess] if allowed_guess in word_counts else 1 for allowed_guess in allowed_guesses}

words = list(word_counts_v2.keys())
words.sort(key=lambda x: word_counts_v2[x], reverse=True)

word_counts_v3 = [[word, word_counts_v2[word]] for word in words]

path = Path("wordle/word_counts.pkl")

with path.open("wb") as f:
    pickle.dump(word_counts_v3, f)