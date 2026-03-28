"""
Text preprocessing module for customer support bot.
Author: Maheen Riaz
"""

import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class TextPreprocessor:
    """Handles text cleaning and preprocessing for the chatbot."""

    def __init__(self):
        """Initialize NLP tools."""
        for resource in ['punkt', 'wordnet', 'stopwords']:
            try:
                nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                nltk.download(resource, quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep important words for intent detection
        self.keep_words = {'not', 'no', 'nor', 'don', "don't", 'isn', "isn't",
                          'wasn', "wasn't", 'won', "won't", 'can', "can't",
                          'how', 'where', 'when', 'what', 'why', 'which'}
        self.stop_words -= self.keep_words

    def clean(self, text):
        """Full preprocessing pipeline."""
        text = str(text).lower().strip()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'[^\w\s?!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words]

        return ' '.join(tokens)

    def tokenize(self, text):
        """Tokenize and lemmatize text."""
        text = self.clean(text)
        return word_tokenize(text)
