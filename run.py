from knn_models import KnnClassifierNumba, KnnClassifierScipy
import pandas as pd
import numpy as np
import time

train = np.genfromtxt('../MNIST_train_small.csv', delimiter=',')

x_train, y_train = train[:, 1:], train[:, 0]

test = np.genfromtxt('../MNIST_test_small.csv', delimiter=',')

x_test, y_test = test[:, 1:], test[:, 0]

result_table = pd.DataFrame()

times = []
test_accuracy = []
train_accuracy = []

for k in range(1, 21):
    start = time.time()
    print('K =', k)
    knn_model = KnnClassifierScipy(k)

    knn_model.fit(x_train, y_train)

    y_pred = knn_model.predict(x_train)
    train_accuracy.append(np.mean(y_pred == y_train))
    print('Train accuracy:', train_accuracy[-1])

    y_pred = knn_model.predict(x_test)
    test_accuracy.append(np.mean(y_pred == y_test))
    print('Test accuracy:', test_accuracy[-1])

    loop_time = time.time() - start
    times.append(loop_time)
    print('Seconds:', loop_time, end='\n\n')

print(pd.DataFrame({'K': list(range(1, 21)), 'Overall time': np.round(times, 3),
                    'Train accuracy': np.round(train_accuracy, 3),
                    'Test accuracy': np.round(test_accuracy, 3)}).set_index('K'))

# knn_model = KnnClassifierNumba(1)
#
# knn_model.fit(x_train, y_train)
#
# start = time.time()
# y_pred = knn_model.predict(x_test)
# print(time.time() - start)