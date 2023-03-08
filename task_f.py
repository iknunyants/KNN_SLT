from knn_models import KnnClassifierNumba, KnnClassifierScipy

import numpy as np
from skimage.feature import hog

train = np.genfromtxt('../MNIST_train.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

x_train = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                        channel_axis=None) for x in x_train])

test = np.genfromtxt('../MNIST_test.csv', delimiter=',')

x_test, y_test = test[:, 1:], test[:, 0]

x_test = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                       channel_axis=None) for x in x_test])

metrics = {'metric': 'minkowski', 'p': 11}

knn_model = KnnClassifierNumba(k=4, **metrics)

knn_model.fit(x_train, y_train)
print(np.mean(knn_model.predict(x_test) == y_test))
