from knn_models import KnnClassifierNumba
import time

import numpy as np
from skimage.feature import hog

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

x_train = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                        channel_axis=None) for x in x_train])

print(x_train[0].shape)

metrics = {'metric': 'minkowski', 'p': 2}

loocv_accuracy = []

knn_model = KnnClassifierNumba(**metrics)

start = time.time()

knn_model.fit(x_train, y_train)
for k in range(1, 21):
    knn_model.set_k(k)
    print('K =', k)

    loocv_accuracy.append((k, knn_model.loocv(recalculate_matrix=False if k > 1 else True)))
    print('Loocv accuracy:', loocv_accuracy[-1][-1])

print(time.time() - start)
best_set = sorted(loocv_accuracy, key=lambda x: x[1])[-1]
print("Best set: k = {k}, loocv accuracy = {loocv}".format(k=best_set[0], loocv=best_set[1]))
