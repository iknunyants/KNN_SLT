import numpy as np
import math
from numba import jit
from scipy.spatial.distance import cdist


@jit(nopython=True)
def euclidean(vector1, vector2):
    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(vector1, vector2)]))
    return dist


@jit(nopython=True)
def euclidean_dataset(x, data):
    calcs = np.array([euclidean(x, point) for point in data])
    return calcs


class KnnClassifierNumba:
    def __init__(self, k=1, data=None, labels=None, distance_p=2) -> None:
        self.k = k
        self.data = data
        self.labels = labels
        self.distance_p = distance_p
        self.dist_matrix = None

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def set_k(self, k):
        self.k = k

    def predict(self, pred_data):
        result = []
        for x in pred_data:
            calcs = euclidean_dataset(x, self.data)
            top_k_labels = self.labels[np.argpartition(calcs, self.k)[:self.k]]
            result.append(np.argmax(np.bincount(top_k_labels.astype(int))))
        return np.array(result)


class KnnClassifierScipy(KnnClassifierNumba):
    def predict(self, pred_data):
        result = []
        dist_matrix = cdist(pred_data, self.data, metric='minkowski', p=self.distance_p)

        for x in dist_matrix:
            top_k_labels = self.labels[np.argpartition(x, self.k)[:self.k]]
            result.append(np.argmax(np.bincount(top_k_labels.astype(int))))
        return np.array(result)

    def loocv(self, recalculate_matrix=True):
        if recalculate_matrix:
            self.dist_matrix = cdist(self.data, self.data, metric='minkowski', p=self.distance_p)
        result = []
        for i, x in enumerate(self.dist_matrix):
            x = np.delete(x, i)
            top_k_labels = np.delete(self.labels, i)[np.argpartition(x, self.k)[:self.k]]
            result.append(np.argmax(np.bincount(top_k_labels.astype(int))))

        return np.mean(np.array(result) == self.labels)
