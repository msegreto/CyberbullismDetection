import re
import emoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from sklearn.base import BaseEstimator, TransformerMixin

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r'\b[a-z]{2,}\b')  

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # URL removal
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Emoji removal
    text = emoji.replace_emoji(text, replace='')

    # tag & hashtag removal
    text = re.sub(r'@\w+|#', '', text)
    
    # punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # number removal
    text = re.sub(r'\d+', '', text)
    
    # Tokenization (regex-based)
    tokens = tokenizer.tokenize(text)
    
    # Stopword removal + stemming
    processed = [
        stemmer.stem(word) for word in tokens if word not in stop_words and word.isalpha()
    ]
    
    return ' '.join(processed)

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(preprocess_text)