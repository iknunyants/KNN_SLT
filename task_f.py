from knn_models import KnnClassifierNumba, KnnClassifierScipy

import numpy as np

train = np.genfromtxt('../MNIST_train.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

test = np.genfromtxt('../MNIST_test.csv', delimiter=',')

x_test, y_test = test[:, 1:], test[:, 0]

metrics = {'metric': 'minkowski', 'p': 11}

knn_model = KnnClassifierNumba(k=4, **metrics)

knn_model.fit(x_train, y_train)
print(np.mean(knn_model.predict(x_test) == y_test))
