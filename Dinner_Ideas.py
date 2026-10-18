import os
import re
import sys
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import logging
from gensim.models.phrases import Phrases, Phraser

# Configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Flag to determine if GUI is available
GUI_AVAILABLE = False
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    logging.warning("Tkinter not available. Will use command-line interface.")

class DinnerIdeasProcessor:
    """
    A class to preprocess dinner ideas and generate similar recipe suggestions
    using natural language processing techniques.
    """
    
    def __init__(self):
        """Initialize the DinnerIdeasProcessor with required NLTK components."""
        # Download necessary NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logging.error(f"Error downloading NLTK resources: {e}")
        
        # Get standard stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add recipe-specific stopwords
        recipe_stopwords = {
            'cup', 'cups', 'teaspoon', 'teaspoons', 'tablespoon', 'tablespoons',
            'ounce', 'ounces', 'gram', 'grams', 'pound', 'pounds',
            'serve', 'serves', 'serving', 'servings', 'make', 'makes',
            'recipe', 'recipes', 'dish', 'dishes', 'meal', 'meals',
            'cook', 'cooks', 'cooking', 'cooked', 'preparation',
            'minute', 'minutes', 'hour', 'hours'
        }
        self.stop_words.update(recipe_stopwords)
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.dinner_ideas = []
        self.processed_ideas = []
        self.word2vec_model = None
        self.idea_vectors = None
        self.similarity_matrix = None
        
        # For n-gram processing
        self.bigram_model = None
        self.trigram_model = None
    
    def load_ideas_from_file(self, file_path):
        """
        Load dinner ideas from a text file.
        
        Args:
            file_path (str): Path to the text file containing dinner ideas.
            
        Returns:
            list: List of dinner ideas loaded from the file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                self.dinner_ideas = [line.strip() for line in file if line.strip()]
            logging.info(f"Loaded {len(self.dinner_ideas)} dinner ideas from {file_path}")
            return self.dinner_ideas
        except Exception as e:
            logging.error(f"Error loading ideas from file: {e}")
            return []
    
    def preprocess_text(self, text):
        """
        Clean and preprocess text by removing irrelevant characters,
        tokenizing, removing stopwords, and applying stemming and lemmatization.
        
        Args:
            text (str): The text to preprocess.
            
        Returns:
            list: List of preprocessed tokens.
        """
        # Improved text cleaning - preserve important characters like hyphens in ingredient names
        # but remove other punctuation
        text = re.sub(r'[^\w\s-]', '', text.lower())
        
        # Replace hyphens with spaces to properly tokenize hyphenated words
        text = text.replace('-', ' ')
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words (likely not meaningful)
        filtered_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 2]
        
        # Apply stemming and lemmatization
        processed_tokens = []
        for token in filtered_tokens:
            # Skip numbers as they're not meaningful for recipe similarity
            if token.isdigit():
                continue
                
            lemmatized = self.lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def detect_phrases(self):
        """
        Detect common phrases (bigrams and trigrams) in the preprocessed ideas.
        This helps capture multi-word concepts like "fried rice" or "tomato sauce".
        """
        # Train bigram model
        bigram = Phrases(self.processed_ideas, min_count=2, threshold=10)
        self.bigram_model = Phraser(bigram)
        
        # Train trigram model on top of bigrams
        trigram = Phrases(self.bigram_model[self.processed_ideas], min_count=2, threshold=10)
        self.trigram_model = Phraser(trigram)
        
        # Apply to all processed ideas
        self.processed_ideas = [
            self.trigram_model[self.bigram_model[tokens]] for tokens in self.processed_ideas
        ]
        
        logging.info("Detected and applied common phrases to processed ideas")
    
    def preprocess_all_ideas(self):
        """
        Preprocess all loaded dinner ideas.
        
        Returns:
            list: List of preprocessed dinner ideas (as token lists).
        """
        self.processed_ideas = [self.preprocess_text(idea) for idea in self.dinner_ideas]
        
        # Detect and apply common phrases
        self.detect_phrases()
        
        logging.info(f"Preprocessed {len(self.processed_ideas)} dinner ideas")
        return self.processed_ideas
    
    def train_word2vec_model(self):
        """
        Train a Word2Vec model on the preprocessed dinner ideas.
        
        Returns:
            Word2Vec: Trained Word2Vec model.
        """
        try:
            # Configure Word2Vec model parameters with improved settings
            self.word2vec_model = Word2Vec(
                sentences=self.processed_ideas,
                vector_size=200,    # Increased dimensions for better representation
                window=10,          # Wider context window to capture more relationships
                min_count=1,        # Keep even rare words since recipe corpus is small
                workers=4,          # Number of threads to run in parallel
                epochs=200,         # More training epochs
                sg=1                # Use skip-gram (more accurate for small datasets)
            )
            logging.info("Word2Vec model trained successfully")
            return self.word2vec_model
        except Exception as e:
            logging.error(f"Error training Word2Vec model: {e}")
            return None
    
    def get_idea_vector(self, idea_tokens):
        """
        Get vector representation of an idea by averaging word vectors.
        Uses TF-IDF-like weighting to emphasize important words.
        
        Args:
            idea_tokens (list): List of tokens representing a dinner idea.
            
        Returns:
            numpy.ndarray: Vector representation of the dinner idea.
        """
        if not self.word2vec_model:
            logging.error("Word2Vec model not trained yet")
            return None
        
        # Count all tokens across all ideas for IDF calculation
        if not hasattr(self, 'token_document_counts'):
            self.token_document_counts = {}
            for idea in self.processed_ideas:
                unique_tokens = set(idea)
                for token in unique_tokens:
                    self.token_document_counts[token] = self.token_document_counts.get(token, 0) + 1
        
        total_docs = len(self.processed_ideas)
        
        # Get word vectors with TF-IDF-like weighting
        word_vectors = []
        word_weights = []
        
        # Count tokens in this specific idea (for TF)
        token_counts = {}
        for token in idea_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        for token in idea_tokens:
            if token in self.word2vec_model.wv:
                # Calculate TF-IDF weight
                tf = token_counts[token] / len(idea_tokens)
                idf = np.log(total_docs / (1 + self.token_document_counts.get(token, 0)))
                weight = tf * idf
                
                word_vectors.append(self.word2vec_model.wv[token] * weight)
                word_weights.append(weight)
        
        if not word_vectors:
            return np.zeros(self.word2vec_model.vector_size)
        
        # Weighted average of word vectors
        if sum(word_weights) > 0:
            return np.sum(word_vectors, axis=0) / sum(word_weights)
        else:
            return np.mean(word_vectors, axis=0)
    
    def compute_idea_vectors(self):
        """
        Compute vector representations for all processed dinner ideas.
        
        Returns:
            numpy.ndarray: Matrix of idea vectors.
        """
        if not self.processed_ideas:
            logging.error("No processed ideas to compute vectors for")
            return None
        
        self.idea_vectors = np.array([self.get_idea_vector(idea) for idea in self.processed_ideas])
        logging.info(f"Computed vector representations for {len(self.idea_vectors)} ideas")
        return self.idea_vectors
    
    def compute_similarity_matrix(self):
        """
        Compute cosine similarity matrix between all dinner idea vectors.
        
        Returns:
            numpy.ndarray: Similarity matrix.
        """
        if self.idea_vectors is None:
            logging.error("Idea vectors not computed yet")
            return None
        
        self.similarity_matrix = cosine_similarity(self.idea_vectors)
        logging.info("Computed similarity matrix")
        return self.similarity_matrix
    
    def get_similar_ideas(self, idea_index, threshold=0.7):  # Higher default threshold
        """
        Get similar dinner ideas based on cosine similarity.
        
        Args:
            idea_index (int): Index of the reference dinner idea.
            threshold (float): Similarity threshold for considering ideas as similar.
            
        Returns:
            list: List of tuples containing (index, idea, similarity score) for similar ideas.
        """
        if self.similarity_matrix is None:
            logging.error("Similarity matrix not computed yet")
            return []
        
        similarities = self.similarity_matrix[idea_index]
        similar_ideas = []
        
        for i, score in enumerate(similarities):
            if i != idea_index and score >= threshold:  # Exclude the idea itself
                similar_ideas.append((i, self.dinner_ideas[i], score))
        
        # Sort by similarity score in descending order
        similar_ideas.sort(key=lambda x: x[2], reverse=True)
        
        return similar_ideas
    
    def process_and_analyze(self, file_path, threshold=0.7):  # Higher default threshold
        """
        Process dinner ideas and analyze similarity in one go.
        
        Args:
            file_path (str): Path to the text file containing dinner ideas.
            threshold (float): Similarity threshold for considering ideas as similar.
            
        Returns:
            dict: Dictionary containing the results of the analysis.
        """
        results = {
            'original_ideas': [],
            'unique_ideas_count': 0,
            'similar_ideas_by_index': {},
            'processed_tokens': {}
        }
        
        # Load ideas from file
        self.load_ideas_from_file(file_path)
        results['original_ideas'] = self.dinner_ideas
        
        if not self.dinner_ideas:
            return results
        
        # Preprocess ideas
        self.preprocess_all_ideas()
        
        # Store processed tokens for debugging
        results['processed_tokens'] = {
            idea: tokens for idea, tokens in zip(self.dinner_ideas, self.processed_ideas)
        }
        
        # Train Word2Vec model
        self.train_word2vec_model()
        
        # Compute idea vectors
        self.compute_idea_vectors()
        
        # Compute similarity matrix
        self.compute_similarity_matrix()
        
        # Get unique ideas count
        # Two ideas are considered non-unique if their similarity exceeds the threshold
        unique_ideas = set()
        for i in range(len(self.dinner_ideas)):
            is_unique = True
            for j in unique_ideas:
                if self.similarity_matrix[i, j] >= threshold and i != j:
                    is_unique = False
                    break
            if is_unique:
                unique_ideas.add(i)
        
        results['unique_ideas_count'] = len(unique_ideas)
        
        # Get similar ideas for each idea
        for i in range(len(self.dinner_ideas)):
            similar = self.get_similar_ideas(i, threshold)
            if similar:  # Only include ideas that have similar ones
                results['similar_ideas_by_index'][i] = {
                    'idea': self.dinner_ideas[i],
                    'similar': [(idx, idea, float(score)) for idx, idea, score in similar]
                }
        
        return results
    
    def export_results(self, results, output_file_path):
        """
        Export the analysis results to a text file.
        
        Args:
            results (dict): Dictionary containing the results of the analysis.
            output_file_path (str): Path for the output text file.
            
        Returns:
            bool: True if export was successful, False otherwise.
        """
        try:
            with open(output_file_path, 'w', encoding='utf-8') as file:
                file.write("# Dinner Ideas Analysis Results\n\n")
                
                file.write("## Original Ideas\n")
                for i, idea in enumerate(results['original_ideas']):
                    file.write(f"{i+1}. {idea}\n")
                
                file.write(f"\n## Unique Ideas Count: {results['unique_ideas_count']}\n")
                
                file.write("\n## Similar Ideas\n")
                for idx, data in results['similar_ideas_by_index'].items():
                    file.write(f"\n### Idea: {data['idea']}\n")
                    file.write("Similar ideas:\n")
                    for sim_idx, sim_idea, score in data['similar']:
                        file.write(f"- {sim_idea} (Similarity: {score:.4f})\n")
                
                # Include debugging information about tokenization
                file.write("\n## Debug Information\n")
                file.write("### Processed Tokens\n")
                for idea, tokens in results.get('processed_tokens', {}).items():
                    file.write(f"- {idea}: {', '.join(tokens)}\n")
            
            logging.info(f"Results exported to {output_file_path}")
            return True
        except Exception as e:
            logging.error(f"Error exporting results: {e}")
            return False


class DinnerIdeasProcessorGUI:
    """
    GUI for the Dinner Ideas Processor application.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI.
        
        Args:
            root (tk.Tk): The root window for the GUI.
        """
        self.root = root
        self.root.title("Dinner Ideas Processor")
        self.root.geometry("800x600")
        
        self.processor = DinnerIdeasProcessor()
        self.results = None
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Input")
        input_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Label(input_frame, text="Input File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.input_file_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_file_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)
        
        ttk.Label(input_frame, text="Similarity Threshold:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.threshold_var = tk.DoubleVar(value=0.7)  # Updated to higher default threshold
        threshold_spinner = ttk.Spinbox(input_frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.threshold_var, width=5)
        threshold_spinner.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Button(input_frame, text="Process", command=self.process_ideas).grid(row=2, column=0, columnspan=3, padx=5, pady=10)
        
        # Results frame
        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.results_notebook = ttk.Notebook(results_frame)
        self.results_notebook.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Tab for original ideas
        self.original_ideas_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.original_ideas_frame, text="Original Ideas")
        
        self.original_ideas_text = tk.Text(self.original_ideas_frame, wrap="word")
        self.original_ideas_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Tab for similar ideas
        self.similar_ideas_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.similar_ideas_frame, text="Similar Ideas")
        
        self.similar_ideas_text = tk.Text(self.similar_ideas_frame, wrap="word")
        self.similar_ideas_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Tab for debug information
        self.debug_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.debug_frame, text="Debug Info")
        
        self.debug_text = tk.Text(self.debug_frame, wrap="word")
        self.debug_text.pack(padx=5, pady=5, fill="both", expand=True)
        
        # Export button
        export_frame = ttk.Frame(self.root)
        export_frame.pack(padx=10, pady=10, fill="x")
        
        ttk.Button(export_frame, text="Export Results", command=self.export_results).pack(side="right", padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(self.root, textvariable=self.status_var, relief="sunken", anchor="w").pack(side="bottom", fill="x")
    
    def browse_input_file(self):
        """Open a file dialog to select the input file."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_file_var.set(file_path)
    
    def process_ideas(self):
        """Process dinner ideas and display results."""
        file_path = self.input_file_var.get()
        if not file_path:
            messagebox.showerror("Error", "Please select an input file.")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"File not found: {file_path}")
            return
        
        threshold = self.threshold_var.get()
        
        self.status_var.set("Processing dinner ideas...")
        self.root.update_idletasks()
        
        try:
            self.results = self.processor.process_and_analyze(file_path, threshold)
            
            # Display original ideas
            self.original_ideas_text.delete(1.0, tk.END)
            self.original_ideas_text.insert(tk.END, "Original Dinner Ideas:\n\n")
            for i, idea in enumerate(self.results['original_ideas']):
                self.original_ideas_text.insert(tk.END, f"{i+1}. {idea}\n")
            
            self.original_ideas_text.insert(tk.END, f"\nUnique Ideas Count: {self.results['unique_ideas_count']}\n")
            
            # Display similar ideas
            self.similar_ideas_text.delete(1.0, tk.END)
            if not self.results['similar_ideas_by_index']:
                self.similar_ideas_text.insert(tk.END, "No similar ideas found with current threshold.")
            else:
                for idx, data in self.results['similar_ideas_by_index'].items():
                    self.similar_ideas_text.insert(tk.END, f"\nIdea {idx+1}: {data['idea']}\n")
                    self.similar_ideas_text.insert(tk.END, "Similar ideas:\n")
                    for sim_idx, sim_idea, score in data['similar']:
                        self.similar_ideas_text.insert(tk.END, f"- {sim_idea} (Similarity: {score:.4f})\n")
            
            # Display debug information
            self.debug_text.delete(1.0, tk.END)
            self.debug_text.insert(tk.END, "Processed Tokens:\n\n")
            for idea, tokens in self.results.get('processed_tokens', {}).items():
                self.debug_text.insert(tk.END, f"- {idea}: {', '.join(tokens)}\n\n")
            
            self.status_var.set("Processing completed")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during processing: {str(e)}")
            self.status_var.set("Processing failed")
    
    def export_results(self):
        """Export results to a text file."""
        if not self.results:
            messagebox.showerror("Error", "No results to export. Please process dinner ideas first.")
            return
        
        output_file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if output_file_path:
            if self.processor.export_results(self.results, output_file_path):
                messagebox.showinfo("Success", f"Results exported to {output_file_path}")
                self.status_var.set(f"Results exported to {output_file_path}")
            else:
                messagebox.showerror("Error", "Failed to export results")


def main():
    """Main function to run the application."""
    if GUI_AVAILABLE:
        # Run GUI version
        root = tk.Tk()
        app = DinnerIdeasProcessorGUI(root)
        root.mainloop()
    else:
        # Run command-line version
        if len(sys.argv) < 2:
            print("Usage: python Dinner_Ideas.py <input_file> [threshold] [output_file]")
            print("Example: python Dinner_Ideas.py dinner_ideas.txt 0.7 results.txt")
            return
        
        input_file = sys.argv[1]
        threshold = 0.7  # Higher default threshold
        output_file = "dinner_ideas_results.txt"  # Default output file
        
        if len(sys.argv) >= 3:
            try:
                threshold = float(sys.argv[2])
            except ValueError:
                print(f"Invalid threshold value: {sys.argv[2]}. Using default: 0.7")
        
        if len(sys.argv) >= 4:
            output_file = sys.argv[3]
        
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return
        
        print(f"Processing dinner ideas from '{input_file}' with similarity threshold {threshold}...")
        
        processor = DinnerIdeasProcessor()
        results = processor.process_and_analyze(input_file, threshold)
        
        print("\nOriginal Dinner Ideas:")
        for i, idea in enumerate(results['original_ideas']):
            print(f"{i+1}. {idea}")
        
        print(f"\nUnique Ideas Count: {results['unique_ideas_count']}")
        
        print("\nSimilar Ideas:")
        if not results['similar_ideas_by_index']:
            print("No similar ideas found with current threshold.")
        else:
            for idx, data in results['similar_ideas_by_index'].items():
                print(f"\nIdea {idx+1}: {data['idea']}")
                print("Similar ideas:")
                for sim_idx, sim_idea, score in data['similar']:
                    print(f"- {sim_idea} (Similarity: {score:.4f})")
        
        # Export results to file
        if processor.export_results(results, output_file):
            print(f"\nResults exported to '{output_file}'")
        else:
            print(f"\nFailed to export results to '{output_file}'")


if __name__ == "__main__":
    main()
