from knn_models import KnnClassifierNumba

import numpy as np

train = np.genfromtxt('../MNIST_train.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

metrics = {'metric': 'minkowski', 'p': 11}

loocv_accuracy = []

knn_model = KnnClassifierNumba(**metrics)

knn_model.fit(x_train, y_train)
for k in range(1, 21):
    knn_model.set_k(k)
    print('K =', k)

    loocv_accuracy.append((k, knn_model.loocv(recalculate_matrix=False if k > 1 else True)))
    print('Loocv accuracy:', loocv_accuracy[-1][-1])

best_set = sorted(loocv_accuracy, key=lambda x: x[1])[-1]
print("Best set: k = {k}, loocv accuracy = {loocv}".format(k=best_set[0], loocv=best_set[1]))
