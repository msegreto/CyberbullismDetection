from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class TwoStageClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, binary_pipeline, multiclass_pipeline, label_map):
        self.binary_pipeline = binary_pipeline
        self.multiclass_pipeline = multiclass_pipeline
        self.label_map = label_map

    def fit(self, X, y_binary=None, y_multi=None):
        return self  

    def predict(self, X):
        binary_preds = self.binary_pipeline.predict(X)
        results = []
        for i, binary_label in enumerate(binary_preds):
            if binary_label == 0:
                results.append("not_cyberbullying")
            else:
                class_pred = self.multiclass_pipeline.predict([X[i]])[0]
                label = self.label_map.get(class_pred, "Unknown")
                results.append(f"cyberbullying:{label}")
        return np.array(results)