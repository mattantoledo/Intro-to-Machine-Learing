import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import math

if __name__ == "__main__":

    N = 200000
    n = 20

    x = bernoulli.rvs(p=0.5, size=(N, n))
    x_i = np.mean(x, axis=1)

    eps = np.linspace(0, 1, 50)

    emp_prob = [np.count_nonzero(abs(x_i - 0.5) > eps[i]) / len(x_i) for i in range(len(eps))]
    hoeff_bound = [2 * math.exp(-2*n*eps[i]*eps[i]) for i in range(len(eps))]

    plt.plot(eps, emp_prob, label='empirical prob')
    plt.plot(eps, hoeff_bound, label='hoeffding bound')
    plt.title("Hoeffding bound and empirical probability")
    plt.xlabel('epsilon')
    plt.ylabel('probability')
    plt.show()
