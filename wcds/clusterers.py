import numpy as np
from sklearn.cluster import AgglomerativeClustering
import logging


class Clusterer(object):
    """
    Abstract class for different clustering
    variants based on the online clustering.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the clusterer.
        """
        raise NotImplementedError("This class is abstract. Derive it.")

    def fit(self, X):
        """
        Clusters based on the given dictionary of discriminators.

        Parameters:
        -----------
            X : Dictionary of discriminators to be clusterd.
            n_clusters : The number of clusters to find. It must be None if distance_threshold is not None.
            distance_threshold : The linkage distance threshold under which, clusters will not be merged. If not None, n_clusters must be None.

        Returns:
        --------
            clusters : Dictionary of discriminator ids and their offline cluster.
        """
        raise NotImplementedError("This method is abstract. Override it.")


class MinDistanceClustering(Clusterer):
    """
    Agglomerative clustering based on the
    intersection levels of WiSARD's discriminators
    without merging them. It resembles single linkage.
    """

    def __init__(self, n_clusters=None, distance_threshold=None):
        if (n_clusters is None and distance_threshold is None) or (
                n_clusters is not None and distance_threshold is not None):
            raise KeyError(
                "Check parameters n_clusters and distance_threshold!")
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def fit(self, X):
        # Creating lists needed throughout the clustering process
        discr = list(X.values())
        ids = [[x] for x in X.keys()]

        # Compute similarities between every pair of objects in the data set
        distances = np.zeros((len(discr), len(discr)))
        for i in range(len(discr)):
            for j in range(len(discr)):
                if i > j:
                    distances[i][j] = discr[i].intersection_level(discr[j])
                    distances[j][i] = distances[i][j]
        logging.info("Calculated distance matrix.")

        while True:
            # Stop conditions
            if len(distances) <= self.n_clusters:
                logging.info("Number of clusters reached.")
                break
            if self.distance_threshold is not None:
                if np.max(distances.flatten()) < self.distance_threshold:
                    logging.info(
                        "Nothing to merge anymore, discriminators too dissimilar.")
                    break

            # Get position of highest intersection
            pos_highest = np.unravel_index(distances.argmax(), distances.shape)
            first = pos_highest[0]
            second = pos_highest[1]
            if distances[first][second] == 0.0:
                logging.info(
                    "Can't cluster anymore, discriminators too dissimilar.")
                break
            logging.info(
                "Highest intersection level at {} with {}%".format(
                    pos_highest, distances[first][second]))

            # Combine discriminators and delete unnecessary one
            ids[first] = sorted(ids[first] + ids[second])
            del ids[second]
            del discr[second]

            # Recalculate distances
            for i in range(len(distances)):
                if i != first:
                    distances[i][first] = min(
                        distances[first][i], distances[second][i])
                    distances[first][i] = distances[i][first]

            # Reshape distance matrix by deleting merged rows and columns
            distances = np.delete(distances, second, 0)
            distances = np.delete(distances, second, 1)
            logging.info(
                "Recalculated distances and shrinked distance matrix to size: {}.".format(
                    distances.shape))

        # Calculate clustering labels
        clusters = dict()
        for i, group in enumerate(ids):
            for id_ in group:
                clusters[id_] = int(i)

        return clusters


class MergeClustering(MinDistanceClustering):
    """
    Agglomerative clustering based on the
    intersection levels of WiSARD's discriminators
    with merging them.
    """

    def __init__(self, n_clusters=None, distance_threshold=None):
        if (n_clusters is None and distance_threshold is None) or (
                n_clusters is not None and distance_threshold is not None):
            raise KeyError(
                "Check parameters n_clusters and distance_threshold!")
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def fit(self, X):
        # Creating lists needed throughout the clustering process
        discr = list(X.values())
        ids = [[x] for x in X.keys()]

        # Compute similarities between every pair of objects in the data set
        distances = np.zeros((len(discr), len(discr)))
        for i in range(len(discr)):
            for j in range(len(discr)):
                if i > j:
                    distances[i][j] = discr[i].intersection_level(discr[j])
                    distances[j][i] = distances[i][j]
        logging.info("Calculated distance matrix.")

        while True:
            # Stop conditions
            if len(distances) <= self.n_clusters:
                logging.info("Number of clusters reached.")
                break
            if self.distance_threshold is not None:
                if np.max(distances.flatten()) < self.distance_threshold:
                    logging.info(
                        "Nothing to merge anymore, discriminators too dissimilar.")
                    break

            # Get position of highest intersection
            pos_highest = np.unravel_index(distances.argmax(), distances.shape)
            first = pos_highest[0]
            second = pos_highest[1]
            if distances[first][second] == 0.0:
                logging.info(
                    "Can't cluster anymore, discriminators too dissimilar.")
                break
            logging.info(
                "Highest intersection level at {} with {}%".format(
                    pos_highest, distances[first][second]))

            # Merge discriminators and delete unnecessary one
            discr[first].merge(discr[second])
            ids[first] = sorted(ids[first] + ids[second])
            del ids[second]
            del discr[second]
            logging.info(
                "Merging discriminator {} into {} and deleting {}.".format(
                    second, first, second))

            # Reshape distance matrix by deleting merged rows and columns
            distances = np.delete(distances, second, 0)
            distances = np.delete(distances, second, 1)

            # Recalculate distances
            for i in range(len(distances)):
                if i != first:
                    distances[i][first] = discr[first].intersection_level(
                        discr[i])
                    distances[first][i] = distances[i][first]
            logging.info(
                "Recalculated distances and shrinked distance matrix to size: {}.".format(
                    distances.shape))

        # Calculate clustering labels
        clusters = dict()
        for i, group in enumerate(ids):
            for id_ in group:
                clusters[id_] = int(i)

        return clusters


class CentroidClustering(Clusterer):
    """
    This clusterer works based on the agglomerative
    clustering implementation from sklearn and the
    approximate centroids of the discriminators.
    """

    def __init__(self, n_clusters=None, distance_threshold=None):
        if (n_clusters is None and distance_threshold is None) or (
                n_clusters is not None and distance_threshold is not None):
            raise KeyError(
                "Check parameters n_clusters and distance_threshold!")
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def fit(self, X, centroids, linkage="ward"):
        clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            distance_threshold=self.distance_threshold,
            linkage=linkage)
        clusterer.fit(centroids)

        # Group discriminator ids into cluster groups
        clusters = dict()
        for x, c in zip(X.keys(), clusterer.labels_):
            clusters[x] = c
        return clusters
