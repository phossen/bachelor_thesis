from .discriminators import *
from .neurons import *
import random
import numpy as np
import math
import json
import sys
import operator
import logging
from collections import OrderedDict


logging.basicConfig(
    filename="wcds.log",
    filemode="w",
    format='%(asctime)s %(message)s',
    level=logging.WARN)


class WiSARD(object):
    """
    The superclass of all WiSARD-like classifiers.

    This should be used as a template to any implementation, as an
    abstract class, but no method is indeed required to be overridden.
    """

    def __init__(self):
        """
        Initializes the classifier.
        """
        raise NotImplementedError("This class is abstract. Derive it.")

    def __len__(self):
        """
        Returns the number of discriminators in the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def clear(self):
        """
        Clears the discriminators.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def record(self, observation):
        """
        Records the provided observation.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def save(self, path):
        """
        Saves the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def load(self, path):
        """
        Loads the classifier.
        """
        raise NotImplementedError("This method is abstract. Override it.")


class WCDS(WiSARD):
    """
    This class implements WCDS (WiSARD for Clustering
    Data Streams) - but only the online step.
    """

    def __init__(self, omega, delta, gamma, epsilon, dimension, beta=None, µ=0,
                 discriminator_factory=SWDiscriminator, mapping="random", seed=random.randint(0, sys.maxsize)):
        """
        Constructor for WCDS.

        Parameters
        ----------
            omega : Temporal sliding window length
            delta : Number of neurons in discriminators
            gamma : Encoding resolution for binarization
            epsilon : Threshold for creating a new discriminator
            dimension : Dimension of the incoming instances
            beta : Length of the addresses, iff None is automatically assigned
            µ : Cardinality weight to tackle cluster imbalance
            discriminator_factory : Callable to create discriminators
            mapping : Maptype used in addressing, either "linear" or "random"
            seed : Used to make results replicable in random mapping
        """
        if beta is None:
            self._adjust_parameters(gamma, dimension, delta)
        else:
            self.beta = beta
            self.delta = delta
            self.gamma = gamma
            if not (beta * delta / gamma * dimension).is_integer():
                logging.CRITICAL(
                    "Parameters result in a float oversampling factor. (Beta * Delta) / (Gamma * Dimension) should be an integer.")
        self.omega = omega
        self.epsilon = epsilon
        self.µ = µ
        self.dimension = dimension
        self.mapping = mapping
        self.seed = seed
        self.discriminator_factory = discriminator_factory
        self.discriminators = OrderedDict()
        self.discriminator_id = 0  # Currently unassigned id
        self.LRU = OrderedDict()  # Least recently used
        logging.info(
            "Initialized WCDS with:\n Omega {}\n Delta {}\n Gamma {}\n Beta {}\n Epsilon {}\n Mu {}\n Dimension {}\n Seed {}\n {} mapping".format(
                self.omega,
                self.delta,
                self.gamma,
                self.beta,
                self.epsilon,
                self.µ,
                self.dimension,
                self.seed,
                self.mapping))

    def __len__(self):
        return len(self.discriminators)

    def _adjust_parameters(self, gamma, dimension, delta):
        """
        If no oversampling is used,
        the parameters have to fulfill the property:
            gamma * dimension = delta * beta
        with everything being an integer.
        This function ensures this contraint.
        """
        if delta == gamma:
            self.gamma = gamma
            self.delta = delta
            self.beta = dimension
            return
        if dimension * gamma == delta:
            self.gamma = gamma
            self.delta = delta
            self.beta = 1
            return
        logging.warn("Had to adjust parameters.")
        if dimension * gamma < delta:
            self.gamma = int(round((1. * delta) / dimension))
            self.delta = dimension * self.gamma
            self.beta = 1
            return
        if (dimension * 1. * gamma / delta).is_integer():
            self.beta = int(dimension * gamma / delta)
            self.gamma = gamma
            self.delta = delta
        else:
            self.beta = dimension * gamma // delta
            self.delta = delta
            while not (delta * self.beta / dimension).is_integer():
                self.beta += 1
            self.gamma = int(delta * self.beta / dimension)

    def record(self, observation, time):
        """
        Absorbs observation and timestamp.
        Returns the id of the discriminator
        that absorbed the observation.
        """
        logging.info(
            "Received: Observation: {} Time: {}".format(
                observation, time))

        # Delete outdated information
        lru_iter = iter(list(self.LRU.keys()))
        deleted_addr = 0
        try:
            current = next(lru_iter)
        except StopIteration:
            current = None
        if current is not None:
            while self.LRU[current] < time - self.omega:
                k, j, a = current
                del self.discriminators[k].neurons[j].locations[a]
                del self.LRU[current]
                self.discriminators[k].length = len(self.discriminators[k])
                deleted_addr += 1
                try:
                    current = next(lru_iter)
                except StopIteration:
                    break
        logging.info("Deleted {} outdated addresses.".format(deleted_addr))

        # Delete useless discriminators
        deleted_discr = self.clean_discriminators()
        logging.info("Deleted {} empty discriminators.".format(deleted_discr))

        # Calculate addressing of the observation
        addressing = self.addressing(observation)
        logging.info("Calculated addressing: {}".format(addressing))

        # Check if there is at least one discriminator
        if len(self.discriminators) > 0:
            # Calculate id and matching of best fitting discriminator
            k, best_matching = self.predict(addressing)
            logging.info(
                "Best discriminator {} matches {}%".format(
                    k, best_matching * 100))

            # If matching is too small create new discriminator
            if best_matching < self.epsilon:
                logging.info(
                    "Matching is too small. Creating new discriminator with id {}".format(
                        self.discriminator_id))
                d = self.discriminator_factory(
                    self.delta, self.discriminator_id)
                self.discriminators[self.discriminator_id] = d
                k = self.discriminator_id
                self.discriminator_id += 1
        else:
            # No discriminator yet
            logging.info(
                "No discriminators - creating a new one with id {}".format(self.discriminator_id))
            d = self.discriminator_factory(self.delta, self.discriminator_id)
            self.discriminators[self.discriminator_id] = d
            k = self.discriminator_id
            self.discriminator_id += 1

        # Absorb the current observation
        self.discriminators[k].record(addressing, time)
        for i, address in enumerate(addressing):
            try:
                del self.LRU[(k, i, address)]
            except BaseException:
                pass
            self.LRU[(k, i, address)] = time
        logging.info(
            "Absorbed observation. Current number of discriminators: {}".format(len(self)))

        return k

    def predict(self, addressing):
        """
        Predicts the best discriminator and returns highest matching.
        If two discriminators return the same matching, the first one
        is taken due to the implementation of sort.
        """
        predictions = [(d, self.discriminators[d].matching(
            addressing, self.µ)) for d in self.discriminators]
        if len(predictions) == 1:
            return predictions[0]
        predictions.sort(key=lambda x: x[1])
        k, best_matching = predictions[-1]
        confidence = predictions[-1][1] - predictions[-2][1]
        logging.info(
            "Predictions has a confidence of {}%".format(
                confidence * 100))
        return k, best_matching

    def bleach(self, threshold):
        """
        Deletes the addresses in the discriminator neurons
        that are outdated with respect to the given threshold.
        Returns the number of addresses deleted.
        """
        count = 0
        for d in self.discriminators.values():
            count += d.bleach(threshold)
        return count

    def clean_discriminators(self):
        """
        Delete all discriminators which have empty neurons only.
        Returns the number of deleted discriminators.
        """
        count = 0
        for k in list(self.discriminators.keys()):
            if not self.discriminators[k].is_useful():
                del self.discriminators[k]
                count += 1
        return count

    def addressing(self, observation):
        """
        Calculate and return the
        addressing for a given observation.
        Should be only used for numerical attributes.
        """
        binarization = np.array(
            [self._binarize(x_i, self.gamma) for x_i in observation])

        if self.mapping == "linear":
            binarization = binarization.flatten()
            binarization = np.reshape(binarization, (self.delta, self.beta))
            addressing = [tuple(b) for b in binarization]
            return addressing
        elif self.mapping == "random":
            binarization = binarization.flatten()
            mapping = list(range(self.beta * self.delta))
            random.seed(self.seed)
            random.shuffle(mapping)
            addressing = np.empty(len(mapping))
            for i, m in enumerate(mapping):
                addressing[m] = binarization[i % len(binarization)]
            addressing = np.reshape(addressing, (self.delta, self.beta))
            addressing = [tuple(b) for b in addressing]
            return addressing
        else:
            raise KeyError("Mapping has an invalid value!")

    def reverse_addressing(self, addressing):
        """
        Reverse the applied addressing
        and return an observation.
        """
        # Undo random mapping
        addressing = addressing.flatten()
        mapping = list(range(len(addressing)))
        random.seed(self.seed)
        random.shuffle(mapping)
        unmapped_matrix = np.empty(self.dimension * self.gamma)
        for i, m in enumerate(mapping):
            unmapped_matrix[i % (self.dimension * self.gamma)] = addressing[m]
        unmapped_matrix = unmapped_matrix.reshape((self.dimension, self.gamma))

        # Calculate observation
        observation = []
        for bin_ in unmapped_matrix:
            observation.append(float(sum(bin_)) / self.gamma)

        return observation

    def _bit_to_int(self, bits):
        """
        Returns integer for list of bits.
        WARNING: Does not handle too large integers.
        """
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        return out

    def _binarize(self, x, resolution):
        """
        Binarize given x using thermometer encoding.
        """
        b = list()
        b.extend([1 for _ in range(int(x * resolution))])
        b.extend([0 for _ in range(resolution - len(b))])
        return b

    def clear(self):
        self.discriminators.clear()

    def save(self, path="wcds.json"):
        """
        Saves the current configuration into a JSON file.
        """
        confg = {}

        # Parameters and meta information
        if self.omega == math.inf:
            confg["omega"] = str(self.omega)
        else:
            confg["omega"] = self.omega
        confg["delta"] = self.delta
        confg["gamma"] = self.gamma
        confg["beta"] = self.beta
        confg["epsilon"] = self.epsilon
        confg["mu"] = self.µ
        confg["dimension"] = self.dimension
        confg["mapping"] = self.mapping
        confg["seed"] = self.seed
        confg["discriminator_id"] = self.discriminator_id

        # Content of discriminators and their neurons
        discr = {}
        for d in self.discriminators.values():
            n_dict = {}
            for i, n in enumerate(d.neurons):
                n_content = []
                for key, value in zip(n.locations.keys(),
                                      n.locations.values()):
                    n_content.append(([int(x) for x in key], int(value)))
                n_dict[i] = n_content
            discr[d.id_] = n_dict
        confg["discriminators"] = discr

        # Save
        with open(path, "w") as write_file:
            json.dump(confg, write_file, indent=4)
        logging.info("Saved current configuration to {}".format(path))

    def load(self, path="wcds.json"):
        """
        Loads the configuration stored in a JSON file.
        """
        # Load
        with open(path, 'r') as read_file:
            confg = json.load(read_file)

        # Parameters and meta information
        if isinstance(confg["omega"], str):
            self.omega = math.inf
        else:
            self.omega = confg["omega"]
        self.delta = confg["delta"]
        self.gamma = confg["gamma"]
        self.beta = confg["beta"]
        self.epsilon = confg["epsilon"]
        self.µ = confg["mu"]
        self.dimension = confg["dimension"]
        self.mapping = confg["mapping"]
        self.seed = confg["seed"]
        self.discriminator_id = confg["discriminator_id"]
        self.discriminator_factory = SWDiscriminator
        self.LRU = OrderedDict()
        unordererd_lru = []

        # Content of discriminators and their neurons
        self.discriminators.clear()
        discr = confg["discriminators"]
        for key, d in zip(discr.keys(), discr.values()):
            current_discr = self.discriminator_factory(
                self.delta, key, creation_time="unknown")
            j = 0
            for n, content in zip(current_discr.neurons, d.values()):
                for c in content:
                    address = tuple(c[0])
                    time_ = int(c[1])
                    unordererd_lru.append(((key, j, address), time_))
                    n.locations[address] = time_
                j += 1
            self.discriminators[key] = current_discr

        # Sort LRU
        unordererd_lru.sort(key=lambda x: x[1])
        for value in unordererd_lru:
            self.LRU[value[0]] = value[1]

        logging.info("Loaded configuration from {}".format(path))

    def centroid(self, discr):
        """
        Approximates the centroid of a given discriminator.
        To properly work, the discriminator should not have
        been affected by a split beforehand.
        """
        # TODO: Make this a discriminator function
        # Calculate the mapped matrix
        mapped_matrix = []
        for neuron in discr.neurons:
            mean_address = tuple([0 for _ in range(self.beta)])
            for loc in neuron.locations:
                mean_address = tuple(map(operator.add, mean_address, loc))
            if len(neuron.locations) > 0:
                mean_address = tuple([x / len(neuron.locations)
                                      for x in mean_address])
            mapped_matrix.append(mean_address)
        mapped_matrix = np.array(mapped_matrix)

        # Calculate the coordinates
        coordinates = self.reverse_addressing(mapped_matrix)

        return coordinates

    def drasiw(self, discr):
        """
        Returns a list of points representing the given discriminator.
        """
        # TODO: Make this a discriminator function
        points = set()
        max_len = max([len(neuron) for neuron in discr.neurons])

        # Sampling points as often as maximum neuron length
        for _ in range(max_len):
            # Retreive random adresses
            mapped_matrix = []
            for i in range(len(discr.neurons)):
                sample = list(
                    random.choice(
                        list(
                            discr.neurons[i].locations.keys())))
                mapped_matrix.extend(sample)
            mapped_matrix = np.reshape(mapped_matrix, (self.delta, self.beta))
            # Reverse addressing
            points.add(tuple(self.reverse_addressing(mapped_matrix)))
        return list(points)
