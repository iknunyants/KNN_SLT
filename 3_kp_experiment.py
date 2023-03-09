from knn_models import KnnClassifierScipy, KnnClassifierNumba

import numpy as np

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

best_set = (0, 0, 0)
for p in range(1, 16):
    knn_model = KnnClassifierNumba(metric='minkowski', p=p)

    knn_model.fit(x_train, y_train)
    for k in range(1, 21):
        knn_model.set_k(k)
        print('K =', k, 'P =', p)
        loocv_res = knn_model.loocv(recalculate_matrix=False if k > 1 else True)
        if best_set[2] < loocv_res:
            best_set = (k, p, loocv_res)
        print('Loocv accuracy:', loocv_res)

print("Best set: k = {k}, p = {p}, loocv accuracy = {loocv}".format(k=best_set[0], p=best_set[1], loocv=best_set[2]))
