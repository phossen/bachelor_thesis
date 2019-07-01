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

    def record(self, observation, class_):
        """
        Record the provided observation, relating it to the given class.
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
            omega : Defines temporal sliding window length
            delta : Number of neurons in discriminators
            gamma : Controls encoding resolution
            epsilon : Threshold for creating new discriminator
            dimension : Dimension of the incoming instances
            µ : Cardinality weight to tackle cluster imbalance
            discriminator_factory : Callable to create Discriminators
            mapping : Maptype used in addressing, either "linear" or "random"
            seed : Used to make results replicable in random mapping
        """
        self.omega = omega
        self.delta = delta
        self.gamma = gamma
        self.beta = int(
            gamma *
            dimension /
            delta) if (
            gamma *
            dimension /
            delta).is_integer() else self._adjust_gamma(
            gamma,
            dimension,
            delta)  # Length of the addresses
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
        print("WARNING! Adjusted gamma to the clusterers needs: ", self.gamma)
        return beta

    def record(self, observation, time, verbose=0):
        """
        Absorbs observation and time.
        """
        # Delete outdated information
        self.bleach(time - self.omega)

        # Delete useless discriminators
        self.clean_discriminators()

        # Calculate addressing of the observation
        addressing = self.addressing(observation)

        # Check if there is at least one discriminator
        if len(self.discriminators) > 0:
            # Choose index of best discriminator
            k = np.argmax([d.matching(addressing, self.µ)
                           for d in self.discriminators])

            # If matching is too small create new discriminator
            if self.discriminators[k].matching(
                    addressing, self.µ) < self.epsilon:
                d = self.discriminator_factory(self.delta)
                self.discriminators.append(d)
                k = len(self.discriminators) - 1
        else:
            d = self.discriminator_factory(self.delta)
            self.discriminators.append(d)
            k = len(self.discriminators) - 1

        # Learn the current observation
        self.discriminators[k].record(addressing, time)

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
        """
        for d in self.discriminators:
            d.bleach(threshold)

    def clean_discriminators(self):
        """
        Delete all discriminators which have empty neurons only.
        """
        for d in self.discriminators:
            if not d.is_useful():
                del d

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
                b += (0,)
        return b

    def clear(self):
        self.discriminators.clear()
