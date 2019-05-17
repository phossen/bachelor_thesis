import numpy as np
from numpy.random import uniform
import random
from wisard.wisard import WiSARD
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
    # ---- Create synthetic dataset ----
    """
    data_c1 = []
    data_c2 = []
    data_c3 = []
    data_c4 = []

    for i in range(1000):
        data_c1.append(
            (uniform(
                low=.15, high=.35), uniform(
                low=.15, high=.35)))
        data_c2.append(
            (uniform(
                low=.65, high=.85), uniform(
                low=.15, high=.35)))
        data_c3.append(
            (uniform(
                low=.15, high=.35), uniform(
                low=.65, high=.85)))
        data_c4.append(
            (uniform(
                low=.65, high=.85), uniform(
                low=.65, high=.85)))

    data = data_c1 + data_c2 + data_c3 + data_c4
    random.shuffle(data)
    datastream = zip(data, list(range(4000)))
    """

    def create_points(t, centers, radius):
        data = []
        for i in range(t):
            p = np.random.uniform(0.0, 2.0*np.pi, 1)
            r = radius * np.sqrt(np.random.uniform(0.0, 1.0, 1))
            data.append((r * np.cos(p) + centers[i][0], r * np.sin(p) + centers[i][1]))
        return data

    t = 1000

    # Custer 1
    radius = 0.1
    x = list(map(lambda x : .2+(.6/t)*x, range(t)))
    y = list(map(lambda x : .2+(.6/t)*x, range(t)))
    centers = list(zip(x, y))
    c1 = create_points(t, centers, radius)

    # Cluster 2
    radius = 0.1
    x = list(map(lambda x : .8-(.6/t)*x, range(t)))
    y = list(map(lambda x : .8+-(.6/t)*x, range(t)))
    centers = list(zip(x, y))
    c2 = create_points(t, centers, radius)

    data = [None]*(len(c1)+len(c2))
    data[::2] = c1
    data[1::2] = c2
    datastream = zip(data, range(2*t))

    # ---- Online Processing ----
    # beta * delta = gamma * len(x)
    OMEGA = 20
    BETA = 50
    GAMMA = 100
    EPSILON = 0.5
    DELTA = 4
    µ = 0

    classifier = WiSARD(
        omega=OMEGA,
        beta=BETA,
        gamma=GAMMA,
        epsilon=EPSILON,
        delta=DELTA,
        µ=µ)
    amount_discs = classifier.classify(datastream)

    plt.plot(amount_discs)
    plt.show()

    # Visualize dataset
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter([data[i][0] for i in range(2*t)], [data[i][1] for i in range(2*t)], range(2*t), c="r", marker="o")
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Time')
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


    # ---- Offline Processing ----
    """
    clustering = DBSCAN()
    """
