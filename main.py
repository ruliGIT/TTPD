import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def load_lyrics(directory):
    lyrics = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                lyrics.append(file.read())
    return lyrics

def construct_co_occurrence_matrix(corpus, window_size=2):
    words = set()
    for doc in corpus:
        words.update(doc.split())

    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_word = {i: word for word, i in word_to_index.items()}
    vocab_size = len(words)
    co_occurrence_matrix = np.zeros((vocab_size, vocab_size))

    for doc in corpus:
        tokens = doc.split()
        for i, target_word in enumerate(tokens):
            target_index = word_to_index[target_word]
            context = tokens[max(0, i - window_size):i] + tokens[i + 1:i + window_size + 1]
            for context_word in context:
                context_index = word_to_index[context_word]
                co_occurrence_matrix[target_index][context_index] += 1

    return co_occurrence_matrix, word_to_index, index_to_word

def construct_graph(co_occurrence_matrix, index_to_word):
    G = nx.Graph()
    for i in range(len(co_occurrence_matrix)):
        G.add_node(index_to_word[i])
    for i in range(len(co_occurrence_matrix)):
        for j in range(i + 1, len(co_occurrence_matrix)):
            if co_occurrence_matrix[i][j] > 0:
                G.add_edge(index_to_word[i], index_to_word[j])
    return G

def visualize_graph(graph):
    nx.draw(graph, with_labels=True)
    plt.show()

def main():
    lyrics_directory = "lyrics"
    lyrics = load_lyrics(lyrics_directory)
    co_occurrence_matrix, word_to_index, index_to_word = construct_co_occurrence_matrix(lyrics)
    graph = construct_graph(co_occurrence_matrix, index_to_word)
    visualize_graph(graph)

if __name__ == "__main__":
    main()
