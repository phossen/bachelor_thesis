from .discriminators import *
from .neurons import *
from .clusterers import *
import random
import numpy as np
import json
import sys
import os
import logging


# To log more detailed change level to logging.DEBUG
logging.basicConfig(filename="wcds.log", filemode="w", format='%(asctime)s %(message)s', level=logging.ERROR)


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
    Data Streams).
    """

    def __init__(self, omega, delta, gamma, epsilon, dimension, µ=0, discriminator_factory=SWDiscriminator, mapping="random", seed=random.randint(0, sys.maxsize)):
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
        self.omega = omega
        self.delta = delta
        self.gamma = gamma
        temp = (gamma * dimension) / delta
        self.beta = int(temp) if (temp).is_integer() else self._adjust_gamma(gamma,dimension,delta)  # Length of the addresses
        self.epsilon = epsilon
        self.µ = µ
        self.dimension = dimension
        self.mapping = mapping
        self.seed = seed
        self.discriminator_factory = discriminator_factory
        self.discriminators = {}
        self.discriminator_id = 0 # Currently unassigned id
        logging.info("Initialized WCDS with:\n Omega {}\n Delta {}\n Gamma {}\n Beta {}\n Epsilon {}\n Mu {}\n Dimension {}\n Seed {}\n {} mapping".format(omega, delta, gamma, self.beta, epsilon, µ, dimension, seed, mapping))

    def __len__(self):
        return len(self.discriminators)

    def _adjust_gamma(self, gamma, dimension, delta):
        """
        The parameters have to fulfill the property:
            beta = (gamma * dimension) / delta
        with everything being an integer.
        This function is called if this is not the case
        and adjusts gamma automatically and returns new
        beta.
        """
        beta = int(round(gamma * dimension / delta))
        self.gamma = int((beta * delta) / dimension)
        logging.warning("Adjusted gamma ({}) to the clusterers needs ({}).".format(gamma, self.gamma))
        return beta

    def record(self, observation, time):
        """
        Absorbs observation and timestamp.
        Returns the id of the discriminator
        that absorbed the observation.
        """
        logging.info("Received: Observation: {} Time: {}".format(observation, time))

        # Delete outdated information
        deleted_addr = self.bleach(time - self.omega)
        logging.info("Deleted {} outdated addresses.".format(deleted_addr))

        # Delete useless discriminators
        deleted_discr = self.clean_discriminators()
        logging.info("Deleted {} empty discriminators.".format(deleted_discr))

        # Calculate addressing of the observation
        addressing = self.addressing(observation)
        logging.info("Calculating addressing for the observation.")

        # Check if there is at least one discriminator
        if len(self.discriminators) > 0:
            # Calculate id and matching of best fitting discriminator
            k, confidence = self.predict(addressing)
            logging.info("Best discriminator {} matches {}%".format(k, confidence*100))

            # If matching is too small create new discriminator
            if confidence < self.epsilon:
                logging.info("Matching is too small. Creating new discriminator with id {}".format(self.discriminator_id))
                d = self.discriminator_factory(self.delta, self.discriminator_id)
                self.discriminators[self.discriminator_id] = d
                k = self.discriminator_id
                self.discriminator_id += 1
        else:
            logging.info("No discriminators - creating a new one with id {}".format(self.discriminator_id))
            d = self.discriminator_factory(self.delta, self.discriminator_id)
            self.discriminators[self.discriminator_id] = d
            k = self.discriminator_id
            self.discriminator_id += 1

        # Absorb the current observation
        self.discriminators[k].record(addressing, time)
        logging.info("Absorbed observation. Current number of discriminators: {}".format(self.__len__()))

        return k

    def predict(self, addressing):
        """
        Predicts the best discriminator and returns the confidence.
        If two discriminators return the same matching, the first one
        is taken due to the implementation of sort.
        """
        predictions = [(d, self.discriminators[d].matching(addressing, self.µ)) for d in self.discriminators]
        if len(predictions) == 1:
            return predictions[0]
        predictions.sort(key=lambda x: x[1])
        k, confidence = predictions[-1]
        if predictions[-1][0] == predictions[-2][0]:
            logging.warn("Found two discriminators with the same matching. Choosing the first one ({}).".format(k))
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
        for d in self.discriminators.values():
            if not d.is_useful():
                del d
                count += 1
        return count

    def addressing(self, observation):
        """
        Calculate and return the
        addressing for a given observation.
        """
        binarization = np.array([self._binarize(x_i, self.gamma) for x_i in observation])

        if self.mapping == "linear":
            binarization = binarization.flatten()
            binarization = np.reshape(binarization, (self.delta, self.beta))
            tuple_addressing = [tuple(b) for b in binarization]
            return tuple_addressing
        elif self.mapping == "random":
            binarization = binarization.flatten()
            random.seed(self.seed)
            random.shuffle(binarization)
            binarization = np.reshape(binarization, (self.delta, self.beta))
            tuple_addressing = [tuple(b) for b in binarization]
            return tuple_addressing
        else:
            raise KeyError("Mapping has an invalid value!")

    def _binarize(self, x, resolution):
        """
        Binarize given x using thermometer encoding.
        """
        b = tuple()
        b += (1,) * (int(x * resolution))
        b += (0,) * (resolution - len(b))
        return b

    def clear(self):
        self.discriminators.clear()

    def save(self, path="wcds.json"):
        """
        Saves the current configuration into a JSON file.
        """
        confg = {}
        confg["omega"] = self.omega
        confg["delta"] = self.delta
        confg["gamma"] = self.gamma
        confg["beta"] = self.beta
        confg["epsilon"] = self.epsilon
        confg["mu"] = self.µ
        confg["dimension"] = self.dimension
        confg["mapping"] = self.mapping
        confg["seed"] = self.seed

        discr = {}
        for d in self.discriminators.values():
            n_dict = {}
            for n in d.neurons:
                for i in range(len(n.locations)):
                    n_content = []
                    for key, value in zip(n.locations.keys(), n.locations.values()):
                        address = [str(i) for i in list(key)]
                        timestamp = int(value)
                        new_tup = tuple(("".join(address), timestamp))
                        n_content.append(new_tup)
                    n_dict[i] = n_content
            discr[d.id_] = n_dict
        confg["discriminators"] = discr

        with open(path, "w") as write_file:
            json.dump(confg, write_file, indent=4)
        logging.info("Saved current configuration to {}".format(path))
