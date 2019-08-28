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
        """
        raise NotImplementedError("This method is abstract. Override it.")


class MinDistanceClustering(Clusterer):
    """
    Agglomerative clustering based on the
    intersection levels of WiSARD's discriminators
    without merging them. Corresponds to single linkage.
    """

    def __init__(self, n_clusters=None, distance_threshold=None):
        if (n_clusters is None and distance_threshold is None) or (
                n_clusters is not None and distance_threshold is not None):
            raise KeyError(
                "Check parameters n_clusters and distance_threshold!")
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold

    def fit(self, X):
        if self.n_clusters is not None:
            if self.n_clusters >= len(X):
                logging.warning(
                    "Target number greater or equal to given number of clusters!")
                return set(X.keys())

        # Creating lists needed throughout the clustering process
        discr = list(X.values())
        ids = list(X.keys())
        self.ids = ids
        merges = []

        # Compute (dis)similarities between every pair of objects in the data
        # set
        distances = np.zeros((len(discr), len(discr)))
        for i in range(len(discr)):
            for j in range(len(discr)):
                if i > j:
                    distances[i][j] = discr[i].intersection_level(discr[j])
                    distances[j][i] = distances[i][j]
        logging.info("Calculated distance matrix.")

        while True:
            # Stop conditions
            if len(distances) == self.n_clusters:
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
            logging.info(
                "Highest intersection level at {} with {}%".format(
                    pos_highest, distances[first][second]))

            # Merge discriminators and delete unnecessary one
            pos_ids = (ids[first], ids[second])
            merges.append(pos_ids)
            del ids[second]
            del discr[second]

            # Recalculate distances
            for i in range(len(distances)):
                if i != first:
                    distances[i][first] = min(
                        distances[first][i], distances[second][i])
                    distances[first][i] = distances[i][first]
            logging.info("Recalculated distances.")

            # Reshape distance matrix by deleting merged rows and columns
            distances = np.delete(distances, second, 0)
            distances = np.delete(distances, second, 1)
            logging.info("Shrinked distance matrix.")

        return self._get_clusters(merges)

    def _get_clusters(self, merges):
        """
        Returns a list of clusters (as sets) containing the discriminator
        ids belonging into it, given a list of merges (as tuples).
        """
        # Step 1: Group cluster merges into sets
        cluster_groups = []

        for tup in merges:
            # No group yet
            if len(cluster_groups) == 0:
                new_cluster_group = set()
                new_cluster_group.add(tup[0])
                new_cluster_group.add(tup[1])
                cluster_groups.append(new_cluster_group)
                continue
            # Try to find matching group
            absorbed = False
            for group in cluster_groups:
                if tup[0] in group or tup[1] in group:
                    absorbed = True
                    group.add(tup[1])
                    group.add(tup[0])
                    continue
            # No match results in creating a new group
            if not absorbed:
                new_cluster_group = set()
                new_cluster_group.add(tup[0])
                new_cluster_group.add(tup[1])
                cluster_groups.append(new_cluster_group)

        # Step 2: Merge sets to final clusters
        for _ in range(len(cluster_groups)):
            # If merging happens, start all over again
            do_break = False
            for i in range(len(cluster_groups)):
                for j in range(len(cluster_groups)):
                    if i == j:
                        continue
                    if len(cluster_groups[i] & cluster_groups[j]) > 0:
                        cluster_groups[i] |= cluster_groups[j]
                        del cluster_groups[j]
                        do_break = True
                        break
                if do_break:
                    break

        # Step 3: Merge unmerged clusters/lonely discriminators
        for i in self.ids:
            found = False
            for cluster in cluster_groups:
                if i in cluster:
                    found = True
                    break
            if not found:
                cluster_groups.append({i})

        return cluster_groups


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
        if self.n_clusters is not None:
            if self.n_clusters >= len(X):
                logging.warning(
                    "Target number greater or equal to given number of clusters!")
                return set(X.keys())

        # Creating lists needed throughout the clustering process
        discr = list(X.values())
        ids = list(X.keys())
        self.ids = ids
        merges = []

        # Compute (dis)similarities between every pair of objects in the data
        # set
        distances = np.zeros((len(discr), len(discr)))
        for i in range(len(discr)):
            for j in range(len(discr)):
                if i > j:
                    distances[i][j] = discr[i].intersection_level(discr[j])
                    distances[j][i] = distances[i][j]
        logging.info("Calculated distance matrix.")

        while True:
            # Stop conditions
            if len(distances) == self.n_clusters:
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
            logging.info(
                "Highest intersection level at {} with {}%".format(
                    pos_highest, distances[first][second]))

            # Merge discriminators and delete unnecessary one
            pos_ids = (ids[first], ids[second])
            discr[first].merge(discr[second])
            merges.append(pos_ids)
            logging.info(
                "Merging discriminator {} into {} and deleting {}.".format(
                    second, first, second))
            del ids[second]
            del discr[second]

            # Reshape distance matrix by deleting merged rows and columns
            distances = np.delete(distances, second, 0)
            distances = np.delete(distances, second, 1)
            logging.info("Shrinked distance matrix.")

            # Recalculate distances
            for i in range(len(distances)):
                if i != first:
                    distances[i][first] = discr[first].intersection_level(
                        discr[i])
                    distances[first][i] = distances[i][first]
            logging.info("Recalculated distances.")

        return self._get_clusters(merges)


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
        clusters = []
        for c in np.unique(clusterer.labels_):
            cluster = set()
            for i in range(len(clusterer.labels_)):
                if c == clusterer.labels_[i]:
                    cluster.add(list(X.keys())[i])
            clusters.append(cluster)
        return clusters
