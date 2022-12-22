#################################
# Your name: Mattan Toledo
#################################


import numpy as np
import numpy.random
import scipy
from sklearn.datasets import fetch_openml
import sklearn.preprocessing

import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss.
    """

    w = np.zeros(data.shape[1])     # w_1 = 0

    for t in range(1, T + 1):

        eta_t = np.divide(eta_0, t)

        i = numpy.random.randint(0, data.shape[0])
        x_i = data[i]
        y_i = labels[i]

        if y_i * np.dot(w, x_i) < 1:
            w = (1 - eta_t) * w + eta_t * C * y_i * x_i
        else:
            w = (1 - eta_t) * w

    return w


def SGD_log(data, labels, eta_0, T):
    """
    Implements SGD for log loss.
    """
    w = np.zeros(data.shape[1])  # w_1 = 0

    norms_vec = []
    for t in range(1, T + 1):

        eta_t = np.divide(eta_0, t)

        i = numpy.random.randint(0, data.shape[0])
        x_i = data[i]
        y_i = labels[i]

        w = w - eta_t * gradient_log_loss(x_i, y_i, w)
        norms_vec.append(np.sqrt(np.dot(w, w)))
    return w, norms_vec

#################################

# Place for additional code

#################################


def gradient_log_loss(x, y, w):
    s = scipy.special.softmax([0, -y * np.dot(w, x)])
    return s[1] * (-y) * x


def calc_accuracy(x, y, w):
    return np.mean([y[i] * np.dot(w, x[i]) >= 0 for i in range(len(x))])


train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()


def q1a():

    T = 1000
    c = 1

    eta_0_vec = np.logspace(-6, 3, num=10)

    accur_avgs = []
    for eta_0 in eta_0_vec:
        accur_lst = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, eta_0, T)
            accur = calc_accuracy(validation_data, validation_labels, w)
            accur_lst.append(accur)
        accur_avgs.append(np.mean(accur_lst))

    title = '(1a) Averaged Accuracy as a function of eta (hinge-loss)'
    plt.title(title)
    plt.plot(eta_0_vec, accur_avgs, marker='x', linestyle='dashdot', color='r')
    plt.xlabel('eta')
    plt.xscale('log')
    plt.show()
    return


best_eta_hinge = 1.0


def q1b():

    T = 1000
    c_vec = np.logspace(-8, 3, num=12)
    accur_avgs = []

    for c in c_vec:
        accur_lst = []
        for i in range(10):
            w = SGD_hinge(train_data, train_labels, c, best_eta_hinge, T)
            accur = calc_accuracy(validation_data, validation_labels, w)
            accur_lst.append(accur)
        accur_avgs.append(np.mean(accur_lst))

    title = '(1b) Averaged Accuracy as a function of c (hinge-loss)'
    plt.title(title)
    plt.plot(c_vec, accur_avgs, marker='x', linestyle='dashdot', color='b')
    plt.xlabel('eta')
    plt.xscale('log')
    plt.show()
    return


best_c_hinge = 0.0001


def q1c():
    T = 20000
    title = '(1c) Optimal Classifier, eta=1.0, c=0.0001 (hinge-loss)'
    w = SGD_hinge(train_data, train_labels, best_c_hinge, best_eta_hinge, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.title(title)
    plt.show()
    return


def q1d():
    T = 20000
    w = SGD_hinge(train_data, train_labels, best_c_hinge, best_eta_hinge, T)
    test_accur = calc_accuracy(test_data, test_labels, w)
    print("The accuracy of the best classifier on the test set is " + str(test_accur))
    return


def q2a():

    T = 1000
    eta_0_vec = np.logspace(-6, 3, num=10)
    accur_avgs = []

    for eta_0 in eta_0_vec:
        accur_lst = []
        for i in range(10):
            w, _ = SGD_log(train_data, train_labels, eta_0, T)
            accur = calc_accuracy(validation_data, validation_labels, w)
            accur_lst.append(accur)
        accur_avgs.append(np.mean(accur_lst))

    title = '(2a) Averaged Accuracy as a function of eta (log-loss)'
    plt.title(title)
    plt.plot(eta_0_vec, accur_avgs, marker='x', linestyle='dashdot', color='r')
    plt.xlabel('eta')
    plt.xscale('log')
    plt.show()
    return


best_eta_log = 1e-06


def q2b():
    T = 20000
    w, _ = SGD_log(train_data, train_labels, best_eta_log, T)

    title = '(2b) Optimal classifier, eta=1e-6 (log-loss)'
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.title(title)
    plt.show()

    test_accur = calc_accuracy(test_data, test_labels, w)
    print("The accuracy of the best classifier on the test set is " + str(test_accur))

    return


def q2c():
    T = 20000
    w, norms_vec = SGD_log(train_data, train_labels, best_eta_log, T)
    t_vec = np.arange(1, T + 1)

    title = '(2c) Norm of w as a function of t (iteration) (log-loss)'
    plt.title(title)
    plt.plot(t_vec, norms_vec, marker='x', linestyle='--', color='b')
    plt.xlabel('t')
    plt.show()

    return


def main():

    q1a()
    q1b()
    q1c()
    q1d()

    q2a()
    q2b()
    q2c()

    return


if __name__ == "__main__":
    main()

