import numpy as np
from numpy.random import uniform
import random
from wisard import WiSARD
from sklearn.cluster import DBSCAN


if __name__ == "__main__":
    # ---- Create synthetic dataset ----
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

    # ---- Online Processing ----
    # beta * delta = gamma * len(x)
    OMEGA = 400
    BETA = 50
    GAMMA = 100
    EPSILON = 0.5
    DELTA = 4

    classifier = WiSARD(
        omega=OMEGA,
        beta=BETA,
        gamma=GAMMA,
        epsilon=EPSILON,
        delta=DELTA)
    classifier.classify(datastream)

    # ---- Offline Processing ----
    """
    clustering = DBSCAN()
    """
