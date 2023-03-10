import numpy as np
import numba
from scipy.spatial.distance import cdist


@numba.jit(nopython=True)
def euclidean(vector1, vector2):
    dist = sum([(a - b) ** 2 for a, b in zip(vector1, vector2)]) ** (1 / 2)
    return dist


@numba.jit(nopython=True)
def minkowski(vector1, vector2, p=10):
    dist = sum([abs(a - b) ** p for a, b in zip(vector1, vector2)]) ** (1 / p)
    return dist


@numba.jit(nopython=True)
def cosine(vector1, vector2):
    dist = 1 - np.dot(vector1, vector2) / (sum(vector1 ** 2) ** (1 / 2) * sum(vector2 ** 2) ** (1 / 2))
    return dist


@numba.jit(nopython=True)
def correlation(vector1, vector2):
    v1 = vector1 - np.mean(vector1)
    v2 = vector2 - np.mean(vector2)
    dist = 1 - np.dot(v1, v2) / (sum(v1 ** 2) ** (1 / 2) * sum(v2 ** 2) ** (1 / 2))
    return dist


@numba.njit(fastmath=True, parallel=True)
def euclidean_dataset(xa, xb):
    dim1, dim2 = xa.shape[0], xb.shape[0]
    calcs = np.empty(shape=(dim1, dim2), dtype=np.float32)
    for i in numba.prange(dim1):
        for j in numba.prange(dim2):
            calcs[i][j] = euclidean(xa[i], xb[j])
    return calcs


@numba.njit(fastmath=True, parallel=True)
def minkowski_dataset(xa, xb, p=2):
    dim1, dim2 = xa.shape[0], xb.shape[0]
    calcs = np.empty(shape=(dim1, dim2), dtype=np.float32)

    for i in numba.prange(dim1):
        for j in numba.prange(dim2):
            calcs[i][j] = minkowski(xa[i], xb[j])
    return calcs


@numba.njit(fastmath=True, parallel=True)
def cosine_dataset(xa, xb):
    dim1, dim2 = xa.shape[0], xb.shape[0]
    calcs = np.empty(shape=(dim1, dim2), dtype=np.float32)

    for i in numba.prange(dim1):
        for j in numba.prange(dim2):
            calcs[i][j] = cosine(xa[i], xb[j])
    return calcs


@numba.njit(fastmath=True, parallel=True)
def correlation_dataset(xa, xb):
    dim1, dim2 = xa.shape[0], xb.shape[0]
    calcs = np.empty(shape=(dim1, dim2), dtype=np.float32)
    for i in numba.prange(dim1):
        for j in numba.prange(dim2):
            calcs[i][j] = correlation(xa[i], xb[j])
    return calcs


class KnnClassifierNumba:
    def __init__(self, k=1, data=None, labels=None, metric='minkowski', p=2) -> None:
        self.k = k
        self.data = data
        self.labels = labels
        self.dist_matrix = None
        if metric == 'minkowski':
            self.metric = {'metric': metric, 'p': p}
        else:
            self.metric = {'metric': metric}

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def set_k(self, k):
        self.k = k

    def calc_dist_matrix(self, pred_data):
        if self.metric['metric'] == 'minkowski':
            dist_matrix = minkowski_dataset(pred_data, self.data, p=self.metric['p'])
        elif self.metric['metric'] == 'cosine':
            dist_matrix = cosine_dataset(pred_data, self.data)
        elif self.metric['metric'] == 'correlation':
            dist_matrix = correlation_dataset(pred_data, self.data)
        else:
            dist_matrix = euclidean_dataset(pred_data, self.data)

        return dist_matrix

    @staticmethod
    def select_class(classes, dists):
        labels, label_cnt = np.unique(classes, return_counts=True)
        if sum(label_cnt == label_cnt.max()) > 1:
            top_labels = labels[label_cnt == label_cnt.max()]
            top_label_dists = []
            for top_label in top_labels:
                top_label_dists.append(dists[classes == top_label].sum())
            return top_labels[np.argmin(top_label_dists)]
        else:
            return labels[np.argmax(label_cnt)]

    def predict(self, pred_data):
        result = []

        self.dist_matrix = self.calc_dist_matrix(pred_data)

        for x in self.dist_matrix:
            inds = np.argpartition(x, self.k)[:self.k]
            result.append(self.select_class(self.labels[inds], x[inds]))
        return np.array(result)

    def loocv(self, recalculate_matrix=True):
        if recalculate_matrix:
            self.dist_matrix = self.calc_dist_matrix(self.data)
        result = []
        for i, x in enumerate(self.dist_matrix):
            x = np.delete(x, i)
            inds = np.argpartition(x, self.k)[:self.k]
            result.append(self.select_class(np.delete(self.labels, i)[inds], x[inds]))

        return np.mean(np.array(result) == self.labels)


class KnnClassifierScipy:

    def __init__(self, k=1, data=None, labels=None, metric='minkowski', p=2) -> None:
        self.k = k
        self.data = data
        self.labels = labels
        self.dist_matrix = None
        if metric == 'minkowski':
            self.metric = {'metric': metric, 'p': p}
        else:
            self.metric = {'metric': metric}

    def fit(self, data, labels):
        self.data = data
        self.labels = labels

    def set_k(self, k):
        self.k = k

    def predict(self, pred_data):
        result = []
        dist_matrix = cdist(pred_data, self.data, **self.metric)

        for x in dist_matrix:
            top_k_labels = self.labels[np.argpartition(x, self.k)[:self.k]]
            result.append(np.argmax(np.bincount(top_k_labels.astype(int))))
        return np.array(result)

    def loocv(self, recalculate_matrix=True):
        if recalculate_matrix:
            self.dist_matrix = cdist(self.data, self.data, **self.metric)
        result = []
        for i, x in enumerate(self.dist_matrix):
            x = np.delete(x, i)
            top_k_labels = np.delete(self.labels, i)[np.argpartition(x, self.k)[:self.k]]
            result.append(np.argmax(np.bincount(top_k_labels.astype(int))))

        return np.mean(np.array(result) == self.labels)
