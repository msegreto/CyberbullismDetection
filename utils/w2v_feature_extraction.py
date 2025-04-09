import numpy as np

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