from .neurons import *
import numpy as np
import random
import logging


class Discriminator(object):
    """
    The superclass of all WiSARD discriminators. Some of the
    methods are abstract and need to be overridden.
    """

    def __init__(self, no_neurons, id_, neuron_factory):
        """
        Initializes the discriminator. neuron_factory needs
        to be a callable that creates a neuron.
        A discriminator is identified by its id.
        """
        self.neuron_factory = neuron_factory
        self.no_neurons = no_neurons
        self.neurons = [self.neuron_factory() for _ in range(self.no_neurons)]
        self.id_ = id_

    def __len__(self):
        """
        Returns the length of the discriminator.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def get_id(self):
        """
        Returns the discriminators unique id.
        """
        return self.id_

    def record(self, observation):
        """
        Record the provided observation.

        The observation is expected to be a list of addresses, each of which
        will be recorded by its respective neuron.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def bleach(self, threshold: int):
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
        raise NotImplementedError("This method is abstract. Override it.")

    def merge(self, dscrmntr):
        """
        Merges the given into this discriminator.
        """
        raise NotImplementedError("This method is abstract. Override it.")


class HitDiscriminator(Discriminator):
    """
    The default WiSARD discriminator.
    """

    def __init__(self, no_neurons, id_, neuron_factory=SetNeuron):
        super().__init__(no_neurons, id_, neuron_factory)

    def __len__(self):
        """
        Length is defined as the total number of
        addresses in the discriminator's neurons.
        """
        return sum([len(n) for n in self.neurons])

    def record(self, observation):
        for address, neuron in zip(observation, self.neurons):
            neuron.record(address)

    def bleach(self):
        for neuron in self.neurons:
            neuron.bleach()

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
    A discriminator using neurons that count
    the frequency of addresses written.
    """

    def __init__(self, no_neurons, id_, neuron_factory=DictNeuron):
        super().__init__(no_neurons, id_, neuron_factory)

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
        for neuron in self.neurons:
            neuron.bleach()

    def matching(self, observation, relative=True):
        """
        Calculates the matching between the given observation
        and this discriminator by summing up the frequencies of
        the matching addresses.
        """
        inv_total = 1. / self.no_neurons if relative else 1
        match = 0
        for address, neuron in zip(observation, self.neurons):
            if neuron.is_set(address):
                match += neuron.count(address)
        return match * inv_total

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

    def __init__(self, no_neurons, id_, neuron_factory=SWNeuron,
                 creation_time=None):
        super().__init__(no_neurons, id_, neuron_factory)
        self.creation_time = None

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

    def record(self, observation, time_):
        if self.creation_time is None:
            self.creation_time = time_
        for address, neuron in zip(observation, self.neurons):
            neuron.record(address, time_)

    def bleach(self, threshold):
        """
        Dump all outdated data.
        Returns number of deleted addresses.
        """
        count = 0
        for neuron in self.neurons:
            count += neuron.bleach(threshold)
        return count

    def matching(self, observation, µ=0):
        """
        Returns the matching score between this
        discriminator and a given observation.

        The higher µ, the more precise discriminators
        are used.
        """
        match = 0
        for address, neuron in zip(observation, self.neurons):
            if neuron.is_set(address):
                match += 1
        return ((1. / self.no_neurons) * match) / \
            ((self.__len__())**(float(µ) / self.no_neurons))

    def intersection_level(self, dscrmntr):
        """
        Calculates the intersection level of this and
        a given discriminator the way described in the
        WCDS Paper.
        """
        intersec = 0
        for i in range(self.no_neurons):
            intersec += self.neurons[i].intersection_level(dscrmntr.neurons[i])
        return intersec
        # TODO: Compare to version of WCDS paper
        #d_union = 0
        #d_intersection = 0
        # for i in range(self.no_neurons):
        #    d_union += len(set(self.neurons[i].locations.keys()).union(
        #        set(dscrmntr.neurons[i].locations.keys())))
        #    d_intersection += len(set(self.neurons[i].locations.keys()).intersection(
        #        set(dscrmntr.neurons[i].locations.keys())))
        # return (1 + d_intersection) / (1 + d_union)

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

    def merge(self, dscrmntr):
        """
        Merges the given discriminator into
        this discriminator.
        """
        for i in range(len(self.neurons)):
            intersection = set(
                self.neurons[i].locations.keys()) & set(
                dscrmntr.neurons[i].locations.keys())
            self.neurons[i].locations = {
                **self.neurons[i].locations,
                **dscrmntr.neurons[i].locations}
            if len(
                    intersection) != 0:  # Need to differentiate regarding most recent timestamp
                for j in intersection:
                    self.neurons[i].locations[j] = max(
                        self.neurons[i].locations[j], dscrmntr.neurons[i].locations[j])

    def drasiw(self, seed, dimension, gamma):
        """
        Returns a list of points representing this discriminator.
        """
        points = set()
        max_len = max([len(neuron) for neuron in self.neurons])

        # Sampling points as often as maximum neuron length
        for _ in range(max_len):
            # Retreive random adresses
            mapped_matrix = []
            for i in range(len(self.neurons)):
                sample = list(
                    random.choice(
                        list(
                            self.neurons[i].locations.keys())))
                mapped_matrix.extend(sample)
            # Calculate mapping
            mapping = list(range(len(mapped_matrix)))
            random.seed(seed)
            random.shuffle(mapping)
            unmapped_matrix = np.zeros(len(mapped_matrix))
            # Apply mapping in reverse
            for i, j in enumerate(mapping):
                unmapped_matrix[j] = mapped_matrix[i]
            unmapped_matrix = unmapped_matrix.reshape((dimension, gamma))
            point = []
            # Calculate coordinates
            for i in range(len(unmapped_matrix)):
                point.append(float(sum(unmapped_matrix[i]) / gamma))
            points.add(tuple(point))
        return list(points)
