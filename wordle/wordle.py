import string
import math
from itertools import product
import pickle


with open("valid-wordle-words.txt", "r", encoding="utf-8") as f:
    words = [line.strip() for line in f if line.strip()]

yellow_words = {}
green_words = {}

for input_word in words:
    for feedback in range(len(input_word)):
        if input_word[feedback] not in yellow_words:
            yellow_words[input_word[feedback]] = []
        if input_word[feedback] not in green_words:
            green_words[input_word[feedback]] = [[] for k in range(5)]
        yellow_words[input_word[feedback]].append(input_word)
        green_words[input_word[feedback]][feedback].append(input_word)

def found_grey(word_list, word, index):
    new_word_list = [word for word in word_list if word[index] not in word]

def rare_case(iter_word, input_word, feedback, index):
    n_char = 0
    for i in range(5):
        if (feedback[i] == '1' or feedback[i] == '2') and input_word[i] == input_word[index]:
            n_char += 1
    return iter_word.count(input_word[index]) == n_char

def filter_words(word_list, input_word, feedback):
    new_word_list = word_list.copy()
    for k in range(5):
        if feedback[k] == '2':
            new_word_list = list(set(new_word_list) & set(green_words[input_word[k]][k]))
    
    for k in range(5):
        if feedback[k] == '1':
            print("Filtering for yellow at position", k)
            new_word_list = list(set(new_word_list) & set(yellow_words[input_word[k]]))
            new_word_list = [word for word in new_word_list if word[k] != input_word[k]]
    
    for k in range(5):
        if feedback[k] == '0':
            new_word_list = [word for word in new_word_list if (input_word[k] not in word) or rare_case(word, input_word, feedback, k)]
    print("After filtering", len(new_word_list))
    return new_word_list

def findOutcome(correctWord, word):
    feedback = [0] * 5
    correctWordFreq = {}
    for i in range(5):
        if word[i] == correctWord[i]:
            feedback[i] = 2
            correctWordFreq[correctWord[i]] = correctWordFreq.get(correctWord[i], 0) + 1
    
    for i in range(5):
        if feedback[i] == 0 and word[i] in correctWord:
            if correctWordFreq.get(word[i], 0) < correctWord.count(word[i]):
                feedback[i] = 1
                correctWordFreq[word[i]] = correctWordFreq.get(word[i], 0) + 1
    return tuple(feedback)

def get_entropy(outcomes, total):
    entropy = 0
    for outcome, count in outcomes.items():
        p = count / total
        entropy += p * (-math.log2(p))
    return entropy


def get_entropies(word_list, all_words, frequency):
    entropy_values = {}
    print("Calculating entropies...")
    for i in range(len(all_words)):
        print("\r" + " " * 20 + "\r", end="")
        print(f"Calculated entropy for {i} words", end="")

        outcomes = {}

        for j in word_list:
            outcome = findOutcome(j, all_words[i])
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        entropy_values[all_words[i]] = get_entropy(outcomes, len(word_list))
    print()
    return entropy_values
            

def getCharacterFrequency(word_list):
    frequency = {char: 0 for char in string.ascii_lowercase}
    for word in word_list:
        for char in set(word):
            frequency[char] += 1
    return frequency

def play(word_list, all_words, n):
    if n == 0:
        with open('entropy_1.pkl', 'rb') as file:
            entropy_values = pickle.load(file) 
    else:
        frequency = getCharacterFrequency(word_list)
        entropy_values = get_entropies(word_list, all_words, frequency)
    new_all_words = all_words.copy()
    new_all_words.sort(key=lambda word: entropy_values[word], reverse=True)
    print("good guesses: ", new_all_words[0:10])
    # input_word = new_all_words[0]
    input_word = input("1) filter words by start:\n2) filter words by end:\n3) filter words by contain:\n4) filter words by not contain\nEnter your guess:\n")
    if input_word.isnumeric():
        if input_word == '1':
            letter = input("Enter the letter: ")
            for i in new_all_words:
                if i[0] == letter:
                    print("best guess: ", i)
                    if input("Is this correct guess? (y/n): ") == 'y':
                        break
        elif input_word == '2':
            letter = input("Enter the letter: ")
            for i in new_all_words:
                if i[4] == letter:
                    print("best guess: ", i)
                    if input("Is this correct guess? (y/n): ") == 'y':
                        break
        elif input_word == '3':
            letter = input("Enter the letter: ")
            max_count = 0
            for i in range(len(new_all_words)):
                if new_all_words[i].count(letter) > max_count:
                    max_count = new_all_words[i].count(letter)
                    print("best guess: ", new_all_words[i])
                    print("position: ", i)
        elif input_word == '4':
            letter = input("Enter the letter: ")
            for i in new_all_words:
                if letter not in i:
                    print("best guess: ", i)
                    if input("Is this correct guess? (y/n): ") == 'y':
                        break
        input_word = input("Enter your guess: ")
    feedback = input()
    return filter_words(word_list, input_word, feedback)
    # best_word = max(all_words, key=lambda word: sum(frequency[char] for char in set(word)))
    # print("Best word to guess:", best_word)
    # feedback = input("Enter feedback (g for green, y for yellow, b for black): ")
    # new_word_list = []
    # for word in word_list:
    #     match = True
    #     for i in range(5):
    #         if feedback[i] == 'g' and word[i] != best_word[i]:
    #             match = False
    #             break
    #         elif feedback[i] == 'y' and (best_word[i] not in word or word[i] == best_word[i]):
    #             match = False
    #             break
    #         elif feedback[i] == 'b' and best_word[i] in word:
    #             match = False
    #             break
    #     if match:
    #         new_word_list.append(word)
    # return new_word_list

# max_char = 'i'
# max_num = 0
# max_count_word = ""
# for i in words:
#     if i.count(max_char) > max_num:
#         max_count_word = i
#         max_num = i.count(max_char)
# print("max count word: ", max_count_word)



print(len(words), "valid words loaded.")
new_words = words.copy()
for i in range(6):
    new_words = play(new_words, words, i)
    print("Examples of remaining possible words:", new_words[:10])
    print("Remaining possible words:", len(new_words))