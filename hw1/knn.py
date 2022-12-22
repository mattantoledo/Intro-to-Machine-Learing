from sklearn.datasets import fetch_openml
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np


# section a
def knn(train, train_labels, x_test, k):

    dist = np.array([np.linalg.norm(x_test-x_train) for x_train in train])
    dist_sorted = np.argsort(dist)[:k]
    k_labels = np.array([train_labels[dist_sorted[i]] for i in range(k)]).astype(int)
    return stats.mode(k_labels, keepdims=False)[0]


def accuracy(train, train_labels, test, test_labels, k):
    return np.sum([knn(train, train_labels, test[i], k) == test_labels[i] for i in range(len(test))]) / len(test)


if __name__ == "__main__":

    # loading data
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    # section b
    n = 1000
    k = 10
    print("(b) n = 1000, k = 10")
    print("Accuracy of prediction: %.3f \n" % accuracy(train[:n], train_labels[:n], test, test_labels.astype(int), k))

    # section c
    n = 1000
    arr_k = np.arange(1, 101)
    accuracy_vector = [accuracy(train[:n], train_labels[:n], test, test_labels.astype(int), k) for k in arr_k]

    # find the best k from the accuracy vector
    max_accur = np.amax(accuracy_vector)
    max_k = arr_k[np.argmax(accuracy_vector)]

    print("(c) n = 1000, best K found is k = ", max_k, " with accuracy of ", max_accur)

    plt.plot(arr_k, accuracy_vector)
    plt.title('(c) Prediction accuracy')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.show()

    # section d
    arr_n = np.arange(100, 5001, 100)
    k = 1
    accuracy_vector = [accuracy(train[:n], train_labels[:n], test, test_labels.astype(int), k) for n in arr_n]

    plt.plot(arr_n, accuracy_vector)
    plt.title('(d) Prediction accuracy')
    plt.xlabel('n')
    plt.ylabel('accuracy')
    plt.show()
