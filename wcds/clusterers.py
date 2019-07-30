import numpy as np
import copy
import logging


class AgglomerativeClustering(object):
    """
    Agglomerative clustering specially implemented 
    for WCDS and the sliding window version.
    """

    def fit(self, X, n_clusters=None, distance_threshold=None):
        """
        Receives dictionary of discriminators and merges them using
        agglomerative clustering.

        Parameters:
        -----------
            X : Dictionary of discriminators to be clusterd.
            n_clusters : The number of clusters to find. It must be None if distance_threshold is not None.
            distance_threshold : The linkage distance threshold under which, clusters will not be merged. If not None, n_clusters must be None.
        """
        if (n_clusters is None and distance_threshold is None) or (n_clusters is not None and distance_threshold is not None):
            raise KeyError("Check parameters n_clusters and distance_threshold!")
            
        # Copy given dictionary, as we just want to extract knowledge and not change real state
        X = copy.deepcopy(X)
        discr = list(X.values())
        ids = list(X.keys())
        merges = []

        # Compute (dis)similarities between every pair of objects in the data set
        distances = np.zeros((len(discr), len(discr)))
        for i in range(len(discr)):
            for j in range(len(discr)):
                if i > j:
                    distances[i][j] = discr[i].intersection_level(discr[j])
                    distances[j][i] = distances[i][j]
        logging.info("Calculated distance matrix.")

        while True:
            # Stop conditions
            if len(distances[0]) == n_clusters:
                logging.info("Number of clusters reached.")
                break
            if distance_threshold is not None:
                if np.max(distances) < distance_threshold:
                    logging.info("Nothing to merge anymore, discriminators too dissimilar.")
                    break

            # Get position of highest intersection
            pos_highest = np.unravel_index(distances.argmax(), distances.shape)
            first = pos_highest[0]
            second = pos_highest[1]
            logging.info("Highest intersection level at {} with {}%".format(pos_highest, distances[first][second]))

            # Merge and delete unnecessary discriminator
            pos_ids = (ids[first], ids[second])
            discr[first].merge(discr[second])
            merges.append(pos_ids)
            logging.info("Merging discriminator {} into {} and deleting {}.".format(second, first, second))
            del ids[second]
            del discr[second]

            # Reshape distance matrix by deleting merged rows and columns
            distances = np.delete(distances, second, 0)
            distances = np.delete(distances, second, 1)
            logging.info("Shrinked distance matrix.")

            # Recalculate distances
            for i in range(len(distances)):
                if i != first:
                    distances[i][first] = discr[first].intersection_level(discr[i])
                    distances[first][i] = distances[i][first]
            logging.info("Recalculated distances.")

        return self._get_clusters(merges)

    def _get_clusters(self, merges):
        """
        Returns a list of clusters (as sets) containing the discriminator
        id's belonging into it, given a list of merges (as tuples).
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

        return cluster_groups
