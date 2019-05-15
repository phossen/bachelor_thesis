import numpy as np
from collections import defaultdict
from discriminator import Discriminator


class WiSARD():
    def __init__(self, omega, delta, beta, gamma, epsilon, µ=0):
        """
        This class implements a WiSARD classifier.
        """
        self.omega = omega  # When information is outdated
        self.delta = delta  # Number of neurons in discriminators
        self.beta = beta  # Length of the addresses
        self.gamma = gamma  # Controls encoding resolution
        self.epsilon = epsilon  # Threshold for creating new discriminator
        self.µ = µ  # Cardinality weight
        self.discriminators = []

    def classify(self, dataset):
        """
        Classifies the given dataset.
        The shape of the dataset has to be:
        (data, time)
        """
        for x, t in dataset:
            # Delete outdated information
            self.delete_old_information(t)

            # Delete useless discriminators
            self.clean_discriminators()

            # Check if there is at least one discriminator
            if len(self.discriminators) > 0:
                # Choose best discriminator
                k = np.argmax([d.matching(self.addressing(x), self.µ)
                               for d in self.discriminators])

                # If matching is too small create new discriminator
                if self.discriminators[k].matching(
                        self.addressing(x), self.µ) < self.epsilon:
                    d = Discriminator(self.delta)
                    self.discriminators.append(d)
                    k = len(self.discriminators) - 1
            else:
                d = Discriminator(self.delta)
                self.discriminators.append(d)
                k = len(self.discriminators) - 1

            # Learn the current x
            for a, j in zip(self.addressing(x), range(self.delta)):
                self.discriminators[k].neurons[j].addresses[tuple(a)] = t

    def delete_old_information(self, t):
        """
        Deletes the addresses in the discriminator neurons
        that are outdated with respect to omega and t.
        """
        d_indices, n_indices, addresses = self._get_old_information(t)
        for d, n, address in zip(d_indices, n_indices, addresses):
            del self.discriminators[d].neurons[n].addresses[address]

    def _get_old_information(self, t):
        """
        Returns three lists of the indices of discriminators
        and neurons and the addresses that are outdated with
        respect to omega and t.
        """
        d_indices, n_indices, addresses = [], [], []
        for d in range(len(self.discriminators)):
            for n in range(self.delta):
                for address in self.discriminators[d].neurons[n].addresses.keys(
                ):
                    if self.discriminators[d].neurons[n].addresses[address] < t - self.omega:
                        d_indices.append(d)
                        n_indices.append(n)
                        addresses.append(address)

        return d_indices, n_indices, addresses

    def clean_discriminators(self):
        """
        Delete all discriminators which have empty neurons only.
        """
        for d in self.discriminators:
            if not d.is_useful():
                del d

    def addressing(self, x):
        """
        Calculate and return the
        addressing for a given x.
        """
        address = [self._binarize(x_i) for x_i in x]
        return np.reshape(address, (self.delta, self.beta))

    def _binarize(self, x):
        """
        Binarize given x.
        """
        b = tuple()
        for i in range(self.gamma):
            if x * self.gamma >= i:
                b += (1,)
            else:
                b += (0,)
        return b
