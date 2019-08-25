from .discriminators import *
from .neurons import *
import random
import numpy as np
import math
import json
import sys
import operator
import logging


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
    Data Streams) - but only the only step.
    """

    def __init__(self, omega, delta, gamma, epsilon, dimension, µ=0,
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
            µ : Cardinality weight to tackle cluster imbalance
            discriminator_factory : Callable to create discriminators
            mapping : Maptype used in addressing, either "linear" or "random"
            seed : Used to make results replicable in random mapping
        """
        self._adjust_parameters(gamma, dimension, delta)
        self.omega = omega
        self.epsilon = epsilon
        self.µ = µ
        self.dimension = dimension
        self.mapping = mapping
        self.seed = seed
        self.discriminator_factory = discriminator_factory
        self.discriminators = {}
        self.discriminator_id = 0  # Currently unassigned id
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
        The parameters have to fulfill the property:
            gamma * dimension = delta * beta
        with everything being an integer.
        This function ensures this.
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
        deleted_addr = self.bleach(time - self.omega)
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
            k, confidence = self.predict(addressing)
            logging.info(
                "Best discriminator {} matches {}%".format(
                    k, confidence * 100))

            # If matching is too small create new discriminator
            if confidence < self.epsilon:
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
        logging.info(
            "Absorbed observation. Current number of discriminators: {}".format(
                self.__len__()))

        return k

    def predict(self, addressing):
        """
        Predicts the best discriminator and returns the confidence.
        If two discriminators return the same matching, the first one
        is taken due to the implementation of sort.
        """
        predictions = [(d, self.discriminators[d].matching(
            addressing, self.µ)) for d in self.discriminators]
        if len(predictions) == 1:
            return predictions[0]
        predictions.sort(key=lambda x: x[1])
        k, confidence = predictions[-1]
        if predictions[-1][0] == predictions[-2][0]:
            logging.warning(
                "Found two discriminators with the same matching. Choosing the first one ({}).".format(k))
        return k, confidence

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

    def centroid(self, discr):
        """
        Approximates the centroid of a given discriminator.
        To properly work, the discriminator should not have
        been affected by a split beforehand.
        """
        # Step 1: Calculate the mapped matrix
        mapped_matrix = []
        for neuron in discr.neurons:
            mean_address = tuple([0 for _ in range(self.beta)])
            for loc in neuron.locations:
                mean_address = tuple(map(operator.add, mean_address, loc))
            if len(neuron.locations) > 0:
                mean_address = tuple([x / len(neuron.locations)
                                      for x in mean_address])
            mapped_matrix.append(mean_address)
        mapped_matrix = np.array(mapped_matrix).flatten()

        # Step 2: Undo random mapping
        mapping = list(range(len(mapped_matrix)))
        random.seed(self.seed)
        random.shuffle(mapping)
        unmapped_matrix = np.zeros(len(mapped_matrix))
        for i, j in enumerate(mapping):
            unmapped_matrix[j] = mapped_matrix[i]
        unmapped_matrix = unmapped_matrix.reshape((self.dimension, self.gamma))

        # Step 3: Calculate centroid coordinates
        coordinates = []
        for i in range(len(unmapped_matrix)):
            coordinates.append(float(sum(unmapped_matrix[i]) / self.gamma))
        return coordinates

    def addressing(self, observation):
        """
        Calculate and return the
        addressing for a given observation.
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
            random.seed(self.seed)
            random.shuffle(binarization)
            binarization = np.reshape(binarization, (self.delta, self.beta))
            addressing = [tuple(b) for b in binarization]
            return addressing
        else:
            raise KeyError("Mapping has an invalid value!")

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

        with open(path, "w") as write_file:
            json.dump(confg, write_file, indent=4)
        logging.info("Saved current configuration to {}".format(path))

    def load(self, path="wcds.json"):
        """
        Loads the configuration stored in a JSON file.
        """
        with open(path, 'r') as read_file:
            confg = json.load(read_file)
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

        self.discriminators.clear()
        discr = confg["discriminators"]
        for key, d in zip(discr.keys(), discr.values()):
            current_discr = self.discriminator_factory(
                self.delta, key, creation_time="unknown")
            for n, content in zip(current_discr.neurons, d.values()):
                for c in content:
                    address = tuple(c[0])
                    time_ = int(c[1])
                    n.locations[address] = time_
            self.discriminators[key] = current_discr

        logging.info("Loaded configuration from {}".format(path))
