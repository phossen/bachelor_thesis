import numpy as np
import math

class Clusterer(object):
    def __init__(self):
        pass


class AgglomerativeClustering(Clusterer):
    def __init__(self, diff_measure):
        self.diff_measure = diff_measure

    def fit(self, X):
        clusters = []

        for instance in X:
            clusters.append(Cluster(instance))

        proximity_matrix = np.full((len(clusters), len(clusters)), math.inf)

        # Calculate Proximity Matrix
        for i in len(clusters):
            for j in len(clusters):
                if j<i:
                    continue
                if i==j:
                    proximity_matrix[i][j] = 0
                proximity_matrix[i][j] = self.diff_measure(clusters[i], clusters[j])

        # Select smallest entry in Proximity Matrix
        min_i, min_j = np.

    def clear(self):
        pass
