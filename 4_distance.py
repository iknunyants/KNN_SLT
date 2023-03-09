from knn_models import KnnClassifierScipy, KnnClassifierNumba
import numpy as np
from skimage.feature import hog

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

test = np.genfromtxt('../MNIST_test_small.csv', delimiter=',')

x_test, y_test = test[:, 1:], test[:, 0]

x_train = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                        channel_axis=None) for x in x_train])

x_test = np.array([hog(x.reshape(28, 28), pixels_per_cell=(6, 6),
                       channel_axis=None) for x in x_test])

metrics = [{'metric': 'correlation'}, {'metric': 'cosine'}] + [{'metric': 'minkowski', 'p': p} for p in range(1, 10)]
loocv_accuracy = []
for metric in metrics:
    print(metric)
    knn_model = KnnClassifierNumba(k=5, **metric)

    knn_model.fit(x_train, y_train)

    for k in range(3, 7):
        knn_model.set_k(k)
        print('K =', k, 'Metric =', metric)

        loocv_accuracy.append((k, metric, knn_model.loocv(recalculate_matrix=False if k > 3 else True)))
        print('Loocv accuracy:', loocv_accuracy[-1][-1])

best_set = sorted(loocv_accuracy, key=lambda x: x[2])[-1]
print("Best set: k = {k}, p = {p}, loocv accuracy = {loocv}".format(k=best_set[0], p=best_set[1], loocv=best_set[2]))


