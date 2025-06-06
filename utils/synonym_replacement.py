import random
from collections import Counter
import nlpaug.augmenter.word as naw
from sklearn.base import BaseEstimator

class SynonymAugmenterToBalance(BaseEstimator):

    def fit_resample(self, X, y):
        random.seed(42)

        label_counts = Counter(y)
        minority_class = min(label_counts, key=label_counts.get)
        majority_class = max(label_counts, key=label_counts.get)
        n_to_generate = label_counts[majority_class] - label_counts[minority_class]

        minority_texts = [text for text, label in zip(X, y) if label == minority_class]

        augmenter = naw.SynonymAug(aug_src='wordnet')

        selected_texts = random.choices(minority_texts, k=n_to_generate)

        X_aug = []
        y_aug = []

        for original in selected_texts:
            try:
                augmented = augmenter.augment(original) #no random seed for reproducibility since it is random lib based
                if isinstance(augmented, list):
                    augmented = augmented[0]
            except:
                augmented = original
            X_aug.append(augmented)
            y_aug.append(minority_class)

        X_final = list(X) + X_aug
        y_final = list(y) + y_aug
        return X_final, y_final
