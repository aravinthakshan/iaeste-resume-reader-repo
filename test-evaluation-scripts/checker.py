from resume_compatibility import calculate_similarity 
"""
query_text = "The candidate has skills in python, css, javascript"
query_text_2 = "The candidate has skills in python, css, javascript,tailwind"
query_text_3 = "The candidate has skills in swimming and talking"


large_text = "candidate should have css, python, javascript, html, typecast, tailwind, react"
similarity_score = calculate_similarity(query_text, large_text)
print(f"Similarity score: {similarity_score}")
similarity_score = calculate_similarity(query_text_2, large_text)
print(f"Similarity score: {similarity_score}")

similarity_score = calculate_similarity(query_text_3, large_text)
print(f"Similarity score: {similarity_score}")
"""
"""
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import numpy as np

def preprocess_text(text):
    # Tokenize and preprocess the text
    return simple_preprocess(text.lower())

def compute_similarity(text1, text2, model):
    tokens1 = preprocess_text(text1)
    tokens2 = preprocess_text(text2)

    # Compute average word vectors for each text
    vec1 = np.mean([model[word] for word in tokens1 if word in model], axis=0)
    vec2 = np.mean([model[word] for word in tokens2 if word in model], axis=0)

    # Compute cosine similarity between vectors
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return similarity

# Example texts
text1 = "The candidate has skills in python, css, javascript"
text2 = "This job requires skills in python, java, html, and css."

# Example Word2Vec training data (replace with your own corpus)
training_data = [
    "The candidate has skills in python css javascript",
    "This job requires skills in python java html and css"
]

# Tokenize and preprocess the training data
processed_data = [preprocess_text(text) for text in training_data]

# Train the Word2Vec model
model = Word2Vec(processed_data, vector_size=100, window=5, min_count=1, workers=4)

# Compute similarity between text1 and text2 using the trained model
similarity_score = compute_similarity(text1, text2, model)

print(f"Similarity score between text1 and text2: {similarity_score:.4f}")

"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords

# Sample sentence
sentence = "Windows | Scientific Computing | Quantum | Physics | Linux Interview can be required!"

# Tokenize the sentence into words
words = word_tokenize(sentence)

# Perform part-of-speech (POS) tagging
tagged_words = pos_tag(words)

# Define function to filter words based on POS tags (e.g., NN for nouns)
def filter_nouns(tagged_words, pos='NN'):
    """
    Filter words based on a specific part-of-speech tag.
    :param tagged_words: List of (word, tag) tuples
    :param pos: Part-of-speech tag to filter (default: 'NN' for nouns)
    :return: List of words filtered by the specified POS tag
    """
    filtered_words = []
    for word, tag in tagged_words:
        if tag.startswith(pos):
            filtered_words.append(word.lower())  # Convert to lowercase for consistency
    return set(filtered_words)

# Filter words based on nouns (or any other POS tag)
nouns = filter_nouns(tagged_words)

print("Nouns found:", nouns)
