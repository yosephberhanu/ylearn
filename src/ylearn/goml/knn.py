import numpy as np
from distance import Minkowski, Cosine  # , hamming


class KNN:
    def set_distance_measure(self, measure_name):
        if measure_name == "euclidean":
            self.distance = Minkowski(2)
        elif measure_name == "manhattan":
            self.distance = Minkowski(1)
        elif measure_name == "minkowski":
            self.distance = Minkowski(self.p)
        elif measure_name == "cosine":
            self.distance = Cosine()
        # elif measure_name == "hamming":
        # 	self.distance = Hamming()
        else:
            # Set default measure to euclidean
            self.distance = Minkowski(2)

    def __init__(self, k=2, distance="euclidean", p=0):
        self.k = k
        self.p = p
        self.set_distance_measure(distance)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, data):
        predictions = []
        for d in data:
            # Compute distances from this data point to every point in the training dataset
            distances = self.distance(self.X_train, d)
            # Find the k nearest neighbors (and their labels)
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[k_indices]

            # Majority vote for the prediction
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            majority_vote = unique[np.argmax(counts)]
            predictions.append(majority_vote)

        return np.array(predictions)

    def save(self, name="knn.pk"):
        pass

    @staticmethod
    def load(path):
        pass
