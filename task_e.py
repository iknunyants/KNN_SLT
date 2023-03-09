from knn_models import KnnClassifierNumba
import time

import numpy as np
from skimage.feature import hog

train = np.genfromtxt('../MNIST_train.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

x_train = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                        channel_axis=None) for x in x_train])

metrics = {'metric': 'minkowski', 'p': 2}

knn_model = KnnClassifierNumba(**metrics)

start = time.time()

knn_model.fit(x_train, y_train)
best_k = (0, 0)
for k in range(1, 21):
    knn_model.set_k(k)
    print('K =', k)

    loocv_res = knn_model.loocv(recalculate_matrix=False if k > 1 else True)
    if best_k[1] < loocv_res:
        best_k = (k, loocv_res)
    print('Loocv accuracy:', loocv_res)

print('Time spent:', time.time() - start)
print("Best set: k = {k}, loocv accuracy = {loocv}".format(k=best_k[0], loocv=best_k[1]))
