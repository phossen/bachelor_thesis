import numpy as np
from .neurons import *


class Discriminator(object):
    """
    The superclass of all WiSARD discriminators. Some of the
    methods are abstract and need to be overridden.
    """

    def __init__(self, no_neurons, neuron_factory):
        """
        Initialize the discriminator. neuron_factory needs
        to be a callable that creates a neuron.
        """
        self.neuron_factory = neuron_factory
        self.no_neurons = no_neurons
        self.neurons = [self.neuron_factory() for _ in range(self.no_neurons)]

    def __len__(self):
        """
        Returns the length of the discriminator.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def record(self, observation):
        """
        Record the provided observation.

        The observation is expected to be a list of addresses, each of which
        will be recorded by its respective neuron.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def bleach(self, threshold):
        """
        Bleach the discriminator by bleaching each of its neurons.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def clear(self):
        """
        Clears the discriminator by clearing all of its neurons.
        """
        for neuron in self.neurons:
            neuron.clear()

    def matching(self, observation):
        """
        Calculate the matching between an observation and this discriminator.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def intersection_level(self, dscrmntr):
        """
        Returns the intersection level between this
        and the given discriminator.
        """
        raise NotImplementedError("This class is abstract. Derive it.")


class HitDiscriminator(Discriminator):
    """
    The default WiSARD discriminator.
    """

    def __init__(self, no_neurons, neuron_factory=SetNeuron):
        super().__init__(no_neurons, neuron_factory)

    def __len__(self):
        """
        Length is defined through the total number of
        addresses in the discriminator's neurons.
        """
        return sum([len(n) for n in self.neurons])

    def record(self, observation):
        for address, neuron in zip(observation, self.neurons):
            neuron.record(address)

    def bleach(self):
        for n in self.neurons:
            n.bleach()

    def matching(self, observation, relative=False):
        """
        Returns how similar the observation is to the stored knowledge.

        The return value is the sum of the answers of each neuron to its
        respective address. This value can be normalized by being divided
        by the number of neurons.
        """
        if relative:
            return float(sum(neuron.is_set(address) for address, neuron in zip(
                observation, self.neurons))) / self.no_neurons
        return sum(neuron.is_set(address)
                   for address, neuron in zip(observation, self.neurons))

    def intersection_level(self, dscrmntr):
        """
        The intersection between two discriminators is defined as
        the mean intersection of its neurons.
        """
        return np.mean([na.intersection_level(nb)
                        for na, nb in zip(self.neurons, dscrmntr.neurons)])


class FrequencyDiscriminator(Discriminator):
    """
    A discriminator using neurons that counts the number
    of RAM writes.
    """

    def __init__(self, no_neurons, neuron_factory=DictNeuron):
        super().__init__(no_neurons, neuron_factory)

    def __len__(self):
        """
        Length is defined through the total number of
        addresses in the discriminator's neurons.
        """
        return sum([len(n) for n in self.neurons])

    def record(self, observation):
        for address, neuron in zip(observation, self.neurons):
            neuron.record(address)

    def bleach(self):
        for n in self.neurons:
            n.bleach()

    def matching(self, observation, relative=True):
        inv_total = 1. / self.no_neurons if relative else 1
        match = 0
        for address, neuron in zip(observation, self.neurons):
            if neuron.is_set(address):
                match += neuron.count(address) * inv_total

        return match

    def intersection_level(self, dscrmntr):
        """
        The intersection between two discriminators is defined as
        the mean intersection of its neurons.
        """
        return np.mean([na.intersection_level(nb)
                        for na, nb in zip(self.neurons, dscrmntr.neurons)])


class SWDiscriminator(Discriminator):
    """
    This class implements a sliding window discriminator.
    """

    def __init__(self, no_neurons, neuron_factory=SWNeuron):
        super().__init__(no_neurons, neuron_factory)

    def __len__(self):
        """
        Returns the length of the discriminator,
        which is defined as the multiplication
        of the length of all its neurons.
        """
        length = 1
        for neuron in self.neurons:
            length *= len(neuron)
        return length

    def record(self, observation, time):
        for address, neuron in zip(observation, self.neurons):
            neuron.record(address, time)

    def bleach(self, threshold):
        """
        Dump all outdated data.
        """
        for neuron in self.neurons:
            neuron.bleach(threshold)

    def matching(self, observation, µ=0):
        """
        Returns the matching score between this
        discriminator and a given observation.

        The higher µ, the more precise discriminators
        are used.
        """
        match = 0
        for i in range(self.no_neurons):
            if self.neurons[i].is_set(observation[i]):
                match += 1

        return ((1 / self.no_neurons) * match) / \
            ((self.__len__())**(µ / self.no_neurons))

    def intersection_level(self, dscrmntr):
        """
        Calculates the intersection levek of this and
        a given descriminator the way described in the
        WCDS Paper.
        """
        d_union = 0
        d_intersection = 0
        for i in range(self.no_neurons):
            d_union += len(set(self.neurons[i].addresses.keys()).union(
                set(dscrmntr.neurons[i].addresses.keys())))
            d_intersection += len(set(self.neurons[i].addresses.keys()).intersection(
                set(dscrmntr.neurons[i].addresses.keys())))

        distance = (1 + d_intersection) / (1 + d_union)
        return distance

    def is_useful(self):
        """
        Returns whether this discriminator is useful,
        which is only the case if not all of its neurons
        are empty.
        """
        for n in self.neurons:
            if len(n) != 0:
                return True
        return False
