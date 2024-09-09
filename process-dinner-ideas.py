"""
This script preprocesses recipe ideas from a text file by removing irrelevant information,
tokenizing, removing stopwords, and applying stemming and lemmatization. It then trains
a Word2Vec model on the preprocessed data to generate word embeddings. Finally, it ranks
similar recipe ideas based on cosine similarity and writes the results to an output file.
"""

import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download the stopwords resource if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')  # Download the punkt_tab resource for tokenization
nltk.download('wordnet')  # Download the WordNet resource for lemmatization

def preprocess_recipe(recipe_text):
    """
    Preprocess a single recipe text by:

    1. Removing irrelevant information (e.g., comments, metadata)
    2. Tokenizing the text into individual words and converting to lowercase
    3. Removing stopwords
    4. Applying stemming or lemmatization to the entire line

    Args:
        recipe_text (str): The text of a single recipe

    Returns:
        tuple: Two strings: the stemmed version of the recipe text and the lemmatized version
    """
    # Remove irrelevant information (e.g., comments, metadata)
    recipe_text = re.sub(r'[#$<>]', '', recipe_text)

    # Tokenize the text into individual words and convert to lowercase
    tokens = [word.lower() for word in recipe_text.split()]

    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    # Apply stemming or lemmatization to the entire line
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_line = ' '.join(stemmer.stem(word) for word in tokens)
    lemmatized_line = ' '.join(lemmatizer.lemmatize(word) for word in tokens)

    return stemmed_line, lemmatized_line

def preprocess_recipes_txt(file_path):
    """
    Preprocess a text file containing recipe ideas.

    The text file should contain one recipe idea per line.

    Args:
        file_path (str): The path to the text file

    Returns:
        tuple: A tuple of two lists: the initial list of recipe ideas and the list of preprocessed recipe ideas
    """
    with open(file_path, 'r') as file:
        initial_ideas = [line.strip() for line in file]

    preprocessed_recipes = [preprocess_recipe(initial_idea) for initial_idea in initial_ideas]

    return initial_ideas, preprocessed_recipes

def get_unique_ideas(preprocessed_recipes):
    """
    Get unique ideas after stemming and lemmatization.

    Args:
        preprocessed_recipes (list): A list of tuples, with each tuple containing a stemmed recipe idea and a lemmatized recipe idea

    Returns:
        tuple: A tuple of two sets, the first containing the unique stemmed ideas and the second containing the unique lemmatized ideas
    """
    unique_stemmed_ideas = {recipes[0] for recipes in preprocessed_recipes}
    unique_lemmatized_ideas = {recipes[1] for recipes in preprocessed_recipes}

    return unique_stemmed_ideas, unique_lemmatized_ideas

def train_word2vec_model(sentences):
    """
    Train a Word2Vec model on a list of sentences.

    Args:
        sentences (list): A list of sentences to train the model on. Each sentence should be a list of words.

    Returns:
        model (gensim.models.Word2Vec): The trained Word2Vec model
    """
    model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=5,
        sg=1,
        hs=0,
        negative=5,
        ns_exponent=0.75,
        compute_loss=True,
        batch_words=10000
    )
    return model

def get_sentence_vector(sentence, model):
    """
    Get the vector for a sentence by averaging the word vectors.

    Args:
        sentence (str): The sentence to compute the vector for.
        model (gensim.models.Word2Vec): The Word2Vec model to use.

    Returns:
        sentence_vector (numpy.ndarray): The vector for the sentence.
    """
    words = sentence.split()
    vocab = set(model.wv.key_to_index.keys())
    words = [word for word in words if word in vocab]
    vectors = [model.wv[word] for word in words]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

def rank_similar_ideas(unique_ideas, model, threshold=0.5):
    """
    Rank similar ideas by computing the cosine similarity between the vectors of each idea.

    Args:
        unique_ideas (set of str): The set of unique ideas to rank.
        model (gensim.models.Word2Vec): The Word2Vec model to use to get the vectors.
        threshold (float): The minimum cosine similarity to consider two ideas as similar.

    Returns:
        similar_ideas (dict): A dictionary mapping each idea to a list of similar ideas.
    """
    unique_ideas_list = list(unique_ideas)
    vector_dict = {idea: get_sentence_vector(idea, model) for idea in unique_ideas_list}
    cosine_sim_matrix = cosine_similarity(list(vector_dict.values()), list(vector_dict.values()))

    similar_ideas = {}
    for i, idea in enumerate(unique_ideas_list):
        similar_indices = [j for j in cosine_sim_matrix[i].argsort()[:-10:-1] if cosine_sim_matrix[i][j] > threshold and j != i]
        similar_ideas[idea] = [unique_ideas_list[j] for j in similar_indices]

    return similar_ideas

def write_similar_ideas_to_file(similar_ideas, output_file_path):
    """
    Write the similar ideas to an output file.

    Args:
        similar_ideas (dict): A dictionary mapping each idea to a list of similar ideas.
        output_file_path (str): The path to the output file.
    """
    with open(output_file_path, 'w') as file:
        for idea, similar in similar_ideas.items():
            file.write(f"Idea: {idea}\n")
            file.write("Similar Ideas:\n")
            for sim in similar:
                file.write(f"- {sim}\n")
            file.write("\n")

# Specify the path to the text file containing the recipe ideas
txt_file_path = 'family_dinner_ideas.txt'

# Preprocess the recipes in the text file
initial_ideas, preprocessed_recipes = preprocess_recipes_txt(txt_file_path)

# Get unique ideas after stemming and lemmatization
unique_stemmed_ideas, unique_lemmatized_ideas = get_unique_ideas(preprocessed_recipes)

# Train Word2Vec model on the unique lemmatized ideas
sentences = [idea.split() for idea in unique_lemmatized_ideas]
word2vec_model = train_word2vec_model(sentences)

# Rank similar ideas after lemmatization using Word2Vec
similar_ideas = rank_similar_ideas(unique_lemmatized_ideas, word2vec_model, threshold=0.5)

# Print the initial list of dinner ideas
print("Initial List of Dinner Ideas:")
for i, idea in enumerate(initial_ideas):
    print(f"{i+1}. {idea}")
print()

# Print the count and list of unique ideas after lemmatization
print("Unique Ideas after Lemmatization:")
print("Count:", len(unique_lemmatized_ideas))
print("List:", unique_lemmatized_ideas)
print()

# Print similar ideas
print("Similar Ideas:")
for idea, similar in similar_ideas.items():
    print(f"Idea: {idea}")
    print("Similar Ideas:", similar)
    print()

# Write similar ideas to an output file
output_file_path = 'similar_ideas.txt'
write_similar_ideas_to_file(similar_ideas, output_file_path)