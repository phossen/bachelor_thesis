from .discriminators import *
from .neurons import *
import random
import numpy as np


class WiSARD(object):
    """
    The superclass of all WiSARD-like classifiers.

    This should be used as a template to any implementation, as an
    abstract class, but no method is indeed required to be overriden.
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
        Record the provided observation.
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
    This class implements WCDS,
    a stream clusterer based on WiSARD.
    """

    def __init__(
            self,
            omega,
            delta,
            gamma,
            epsilon,
            dimension,
            µ=0,
            discriminator_factory=SWDiscriminator,
            mapping="random",
            seed=1234):
        """
        Constructor for the WCDS stream clusterer.

        Parameters
        ----------
            omega : Temporal sliding window length
            delta : Number of neurons in discriminators
            gamma : Encoding resolution
            epsilon : Threshold for creating a new discriminator
            dimension : Dimension of the incoming instances
            µ : Cardinality weight to tackle cluster imbalance
            discriminator_factory : Callable to create Discriminators
            mapping : Maptype used in addressing, either "linear" or "random"
            seed : Used to make results replicable in random mapping
        """
        self.omega = omega
        self.delta = delta
        self.gamma = gamma
        temp = gamma*dimension/delta
        self.beta = int(temp) if (temp).is_integer() else self._adjust_gamma(gamma,dimension,delta)  # Length of the addresses
        self.epsilon = epsilon
        self.µ = µ
        self.dimension = dimension
        self.mapping = mapping
        self.seed = seed
        self.discriminator_factory = discriminator_factory
        self.discriminators = []

    def __len__(self):
        return len(self.discriminators)

    def _adjust_gamma(self, gamma, dimension, delta):
        """
        The parameters have to fulfill the property:
            beta = (gamma * dimension) / delta
        This function is called if this is not the case
        and adjusts gamma automatically and returns new
        beta.
        """
        beta = int(round(gamma * dimension / delta))
        self.gamma = int((beta * delta) / dimension)
        print("WARNING! Adjusted gamma ({}) to the clusterers needs ({}).".format(gamma, self.gamma))
        return beta

    def record(self, observation, time, verbose=0):
        """
        Absorbs observation and time.
        If verbosity is 1, prints information.
        """
        if verbose: print("Received: Observation: {}, Time: {}".format(observation, time))

        # Delete outdated information
        deleted_addr = self.bleach(time - self.omega)
        if verbose: print("Deleted {} outdated addresses.".format(deleted_addr))

        # Delete useless discriminators
        deleted_discr= self.clean_discriminators()
        if verbose: print("Deleted {} empty discriminators.".format(deleted_discr))

        # Calculate addressing of the observation
        addressing = self.addressing(observation)
        if verbose: print("Calculating addressing for the observation.")

        # Check if there is at least one discriminator
        if len(self.discriminators) > 0:
            # Choose index of best discriminator
            k = np.argmax([d.matching(addressing, self.µ)
                           for d in self.discriminators])
            if verbose: print("Choosing best discriminator: {} with matching: {}".format(k, self.discriminators[k].matching(addressing, self.µ)))

            # If matching is too small create new discriminator
            if self.discriminators[k].matching(
                    addressing, self.µ) < self.epsilon:
                if verbose: print("No appropriate discriminator - creating a new one")
                d = self.discriminator_factory(self.delta)
                self.discriminators.append(d)
                k = len(self.discriminators) - 1
        else:
            if verbose: print("No discriminators - creating a new one.")
            d = self.discriminator_factory(self.delta)
            self.discriminators.append(d)
            k = len(self.discriminators) - 1

        # Learn the current observation
        self.discriminators[k].record(addressing, time)
        if verbose: print("Absorbed observation.\nCurrent number of discriminators: ", self.__len__())

    def predict(self, observation):
        """
        Predicts index of best discriminator and returns the confidence.
        """
        addressing = self.addressing(observation)
        k = np.argmax([d.matching(addressing, self.µ)
                       for d in self.discriminators])
        return k, self.discriminators[k].matching(addressing, self.µ)

    def bleach(self, threshold):
        """
        Deletes the addresses in the discriminator neurons
        that are outdated with respect to the given threshold.
        Returns the number of addresses deleted.
        """
        count = 0
        for d in self.discriminators:
            count += d.bleach(threshold)
        return count

    def clean_discriminators(self):
        """
        Delete all discriminators which have empty neurons only.
        Returns the number ofe deleted discriminators.
        """
        count = 0
        for d in self.discriminators:
            if not d.is_useful():
                del d
                count += 1
        return count

    def addressing(self, observation):
        """
        Calculate and return the
        addressing for a given observation.
        """
        binarization = np.array(
            [self._binarize(x_i, self.gamma) for x_i in observation])

        if self.mapping == "linear":
            return np.reshape(binarization, (self.delta, self.beta))
        elif self.mapping == "random":
            # Create mapping
            mapping = list(range(self.dimension * self.gamma))
            random.seed(self.seed)
            random.shuffle(mapping)
            # Apply mapping
            linear_binarization = np.reshape(
                binarization, self.dimension * self.gamma)
            linear_addressing = np.array(
                [linear_binarization[i] for i in mapping])
            np.reshape(linear_addressing, (self.delta, self.beta))
            # Cast to tuple to allow it to be dict key
            tuple_addressing = [
                tuple(
                    (linear_addressing[i],)) for i in range(
                    len(linear_addressing))]
            return tuple_addressing
        else:
            raise KeyError("Mapping has an invalid value!")

    def _binarize(self, x, resolution):
        """
        Binarize given x using thermometer binarization.
        """
        b = tuple()
        for i in range(resolution):
            if x * resolution >= i:
                b += (1,)
            else:
                b += (0,) * (resolution - len(b))
                break
        return b

    def clear(self):
        self.discriminators.clear()
