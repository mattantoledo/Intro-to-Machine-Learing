import numpy as np

import backprop_data

import backprop_network
import matplotlib.pyplot as plt


def a():

    training_data, test_data = backprop_data.load(train_size=10000,test_size=5000)

    net = backprop_network.Network([784, 40, 10])

    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# a()


def b():

    training_data, test_data = backprop_data.load(train_size=10000, test_size=5000)

    learning_rates = [10**i for i in range(-3,3)]
    train_accuracy = []
    train_loss = []
    test_accuracy = []

    for rate in learning_rates:
        # print("### Learning rate: " + str(rate) + "###")
        net = backprop_network.Network([784, 40, 10])

        tr, loss, te = net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=rate, test_data=test_data)

        train_accuracy.append(tr)
        train_loss.append(loss)
        test_accuracy.append(te)

    epochs = np.arange(30)

    for i in range(len(learning_rates)):
        plt.plot(epochs, train_accuracy[i], label="rate = {}".format(learning_rates[i]))
    plt.xlabel('epochs')
    plt.ylabel('training accuracy')
    plt.legend()
    plt.show()

    for i in range(len(learning_rates)):
        plt.plot(epochs, train_loss[i], label="rate = {}".format(learning_rates[i]))
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.legend()
    plt.show()

    for i in range(len(learning_rates)):
        plt.plot(epochs, test_accuracy[i], label="rate = {}".format(learning_rates[i]))
    plt.xlabel('epochs')
    plt.ylabel('test accuracy')
    plt.legend()
    plt.show()

# b()


def c():
    training_data, test_data = backprop_data.load(train_size=50000, test_size=10000)

    net = backprop_network.Network([784, 40, 10])

    net.SGD(training_data, epochs=30, mini_batch_size=10, learning_rate=0.1, test_data=test_data)

# c()