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
nltk.download('wordnet')  # Download the WordNet resource for lemmatization

def preprocess_recipe(recipe_text):
    # Remove irrelevant information (e.g., comments, metadata)
    recipe_text = re.sub(r'#.*', '', recipe_text)  # Remove comments starting with '#'
    recipe_text = re.sub(r'<.*?>', '', recipe_text)  # Remove metadata enclosed in '<>'

    # Tokenize the text into individual words
    tokens = word_tokenize(recipe_text)

    # Convert all tokens to lowercase
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Apply stemming or lemmatization to the entire line
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    stemmed_line = ' '.join([stemmer.stem(token) for token in tokens])
    lemmatized_line = ' '.join([lemmatizer.lemmatize(token) for token in tokens])

    return stemmed_line, lemmatized_line

def preprocess_recipes_csv(file_path):
    initial_ideas = []
    preprocessed_recipes = []

    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            recipe_text = row[0]  # Assuming each line is a single idea
            initial_ideas.append(recipe_text)

            # Preprocess the recipe text
            stemmed_line, lemmatized_line = preprocess_recipe(recipe_text)
            preprocessed_recipes.append((stemmed_line, lemmatized_line))

    return initial_ideas, preprocessed_recipes

def get_unique_ideas(preprocessed_recipes):
    unique_stemmed_ideas = set()
    unique_lemmatized_ideas = set()

    for stemmed_line, lemmatized_line in preprocessed_recipes:
        unique_stemmed_ideas.add(stemmed_line)
        unique_lemmatized_ideas.add(lemmatized_line)

    return unique_stemmed_ideas, unique_lemmatized_ideas

def train_word2vec_model(sentences):
    # Train Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    return model

def get_sentence_vector(sentence, model):
    # Get the vector for a sentence by averaging the word vectors
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def rank_similar_ideas(unique_ideas, model, threshold=0.5):
    # Convert the set of unique ideas to a list
    unique_ideas_list = list(unique_ideas)

    # Get sentence vectors
    sentence_vectors = [get_sentence_vector(idea, model) for idea in unique_ideas_list]

    # Compute cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(sentence_vectors, sentence_vectors)

    # Rank similar ideas with threshold
    similar_ideas = {}
    for i in range(len(unique_ideas_list)):
        similar_indices = [j for j in cosine_sim_matrix[i].argsort()[:-10:-1] if cosine_sim_matrix[i][j] > threshold and j != i]
        similar_ideas[unique_ideas_list[i]] = [unique_ideas_list[j] for j in similar_indices]

    return similar_ideas

# Specify the path to the CSV file containing the recipe ideas
csv_file_path = 'family_dinner_ideas.csv'

# Preprocess the recipes in the CSV file
initial_ideas, preprocessed_recipes = preprocess_recipes_csv(csv_file_path)

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