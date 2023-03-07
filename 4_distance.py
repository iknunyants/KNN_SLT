from knn_models import KnnClassifierScipy
import pandas as pd
import numpy as np
import time
from skimage.feature import hog

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

test = np.genfromtxt('../MNIST_test_small.csv', delimiter=',')

x_test, y_test = test[:, 1:], test[:, 0]

x_train = np.array([hog(x.reshape(28,28), pixels_per_cell=(4, 4),
                           channel_axis=None) for x in x_train])

x_test = np.array([hog(x.reshape(28,28), pixels_per_cell=(4, 4),
                           channel_axis=None) for x in x_test])

metrics = [{'metric': 'minkowski', 'p': 2}, {'metric': 'cosine'}, {'metric': 'correlation'}]
for metric in metrics:
    print(metric)
    knn_model = KnnClassifierScipy(k=4, **metric)

    knn_model.fit(x_train, y_train)
    print(knn_model.loocv())
    print(np.mean(knn_model.predict(x_test) == y_test))
