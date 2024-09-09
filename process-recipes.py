import os
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download the stopwords resource if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

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
    
    # Apply stemming or lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return stemmed_tokens, lemmatized_tokens

def train_ner_model(labeled_data):
    # Load the spaCy English model
    nlp = spacy.load('en_core_web_sm')
    
    # Add the NER pipeline to the model if it doesn't exist
    if 'ner' not in nlp.pipe_names:
        ner = nlp.add_pipe('ner')
    else:
        ner = nlp.get_pipe('ner')
    
    # Add labels to the NER model
    for _, annotations in labeled_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    # Train the NER model
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(10):
            print(f"Starting iteration {itn}")
            for text, annotations in labeled_data:
                nlp.update([text], [annotations], sgd=optimizer)
    
    return nlp

def preprocess_recipes_directory(directory, ner_model):
    preprocessed_recipes = []
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # Read the recipe text file
            with open(file_path, 'r') as file:
                recipe_text = file.read()
            
            # Preprocess the recipe text
            stemmed_tokens, lemmatized_tokens = preprocess_recipe(recipe_text)
            
            # Apply NER to extract entities
            doc = ner_model(recipe_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            preprocessed_recipes.append((stemmed_tokens, lemmatized_tokens, entities))
    
    return preprocessed_recipes

# Specify the directory containing the recipe text files
recipes_directory = 'Recipes/'

# Prepare the labeled data for training the NER model
labeled_data = [
    ("Spaghetti Carbonara", {"entities": [(0, 9, "DISH"), (10, 19, "DISH")]}),
    ("Ingredients: spaghetti, eggs, cheese, bacon", {"entities": [(13, 22, "INGREDIENT"), (24, 28, "INGREDIENT"), (30, 36, "INGREDIENT"), (38, 43, "INGREDIENT")]}),
    # Add more labeled examples...
]

# Train the NER model
ner_model = train_ner_model(labeled_data)

# Preprocess the recipes in the directory
preprocessed_recipes = preprocess_recipes_directory(recipes_directory, ner_model)

# Print the preprocessed recipes
for i, (stemmed_tokens, lemmatized_tokens, entities) in enumerate(preprocessed_recipes):
    print(f"Recipe {i+1}:")
    print("Stemmed tokens:", stemmed_tokens)
    print("Lemmatized tokens:", lemmatized_tokens)
    print("Entities:", entities)
    print()