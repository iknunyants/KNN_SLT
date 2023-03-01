from knn_models import KnnClassifierScipy
import pandas as pd
import numpy as np
import time

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

result_table = pd.DataFrame()

times = []
loocv_accuracy = []

for p in range(1, 16):
    knn_model = KnnClassifierScipy(distance_p=p)

    knn_model.fit(x_train, y_train)
    for k in range(1, 21):
        knn_model.set_k(k)
        print('K =', k, 'P =', p)

        loocv_accuracy.append((k, p, knn_model.loocv(recalculate_matrix=False if k > 1 else True)))
        print('Loocv accuracy:', loocv_accuracy[-1][-1])

best_set = sorted(loocv_accuracy, key=lambda x: x[2])[-1]
print("Best set: k = {k}, p = {p}, loocv accuracy = {loocv}".format(k=best_set[0], p=best_set[1], loocv=best_set[2]))
