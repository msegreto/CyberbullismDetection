import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def compute_w2v_features(texts, model, vector_size):
    features = []
    for sentence in texts:
        tokens = sentence.split()
        valid_tokens = [token for token in tokens if token in model.wv]
        if valid_tokens:
            vectors = [model.wv[token] for token in valid_tokens]
            avg_vector = np.mean(vectors, axis=0)
        else:
            avg_vector = np.zeros(vector_size)
        features.append(avg_vector)
    return np.array(features)

class W2VTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model
        self.dim = model.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return compute_w2v_features(X, self.model, self.dim)