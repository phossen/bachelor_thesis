import numpy as np
from neuron import Neuron


class Discriminator():
    def __init__(self, delta):
        """
        The class implements a basic discriminator
        with the given number of neurons.
        """
        self.neurons = []  # static list
        self.delta = delta
        for _ in range(delta):
            self.neurons.append(Neuron())

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

    def matching(self, x, µ=0):
        """
        Returns the matching score between this
        discriminator and a given addressing of x.

        The higher µ, the more precise discriminators
        are used.
        """
        match = 0
        for i in range(self.delta):
            if tuple(x[i]) in self.neurons[i].addresses.keys():
                match += 1

        if self.__len__() == 0:
            return ((1 / self.delta) * match) / (1**(µ / self.delta))
        return ((1 / self.delta) * match) / \
            ((self.__len__())**(µ / self.delta))

    def distance(self, discriminator):
        """
        Calculates and returns the difference between
        this discriminator and a given one.
        """
        d_union = 0
        d_intersection = 0
        for i in range(self.delta):
            d_union += len(set(self.neurons[i].addresses.keys()).union(
                set(discriminator.neurons[i].addresses.keys())))
            d_intersection += len(set(self.neurons[i].addresses.keys()).intersection(
                set(discriminator.neurons[i].addresses.keys())))

        distance = (1 + d_intersection) / (1 + d_union)
        return distance

    def is_useful(self):
        """
        Returns whether the discriminator is
        useful(if not all his neurons are empty).
        """
        length = 0
        for n in self.neurons:
            length += len(n)

        if length <= 0:
            return False
        return True
