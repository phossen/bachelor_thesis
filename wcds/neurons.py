import numpy as np
import collections as cl
import logging


class Neuron(object):
    """
    The superclass of all WiSARD neurons. Some of the
    methods are abstract and have to be overridden.
    """

    def __init__(self, address_length=None):
        """
        Initializes the neuron.
        """
        self.locations = []
        self.address_length = address_length
        if address_length is None:
            self.max_entries = None
        else:
            self.max_entries = 2**address_length

    def __len__(self):
        """
        Returns how many RAM locations are written.
        """
        return len(self.locations)

    def record(self, address):
        """
        Records the given address.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def is_set(self, address):
        """
        Returns true iff the location being addressed is written.
        """
        return address in self.locations

    def bleach(self, threshold):
        """
        Bleaches RAM entries based on a defined model.
        """
        raise NotImplementedError("This method is abstract. Override it.")

    def remove(self, address):
        """
        Deletes the given address out the RAM.
        """
        try:
            del self.locations[address]
        except BaseException:
            pass

    def clear(self):
        """
        Clears all RAM locations.
        """
        self.locations.clear()

    def intersection_level(self, neuron):
        """
        Returns the intersection level between two neurons.
        """
        raise NotImplementedError("This method is abstract. Override it.")


class SetNeuron(Neuron):
    """
    This is the simplest neuron which just records
    whether a RAM location was written.
    """

    def __init__(self, address_length=None):
        super().__init__(address_length)
        self.locations = set()

    def record(self, address):
        self.locations.add(address)

    def remove(self, address):
        try:
            self.locations.remove(address)
        except BaseException:
            pass

    def intersection_level(self, neuron):
        """
        Returns the amount of locations written in both neurons.

        Considering a & b the intersection of the locations written in both
        neurons and a | b their union, this method returns (a & b)/(a | b).
        """
        len_intrsctn = float(len(self.locations & neuron.locations))
        len_union = float(len(self.locations | neuron.locations))
        return len_intrsctn / len_union


class DictNeuron(Neuron):
    """
    A basic neuron based on Python's dict().
    It counts how often an address was written.
    """

    def __init__(self, address_length=None):
        super().__init__(address_length)
        self.locations = cl.defaultdict(int)

    def record(self, address, intensity=1):
        if address in self.locations:
            self.locations[address] += intensity
        else:
            self.locations[address] = intensity

    def bleach(self, threshold):
        """
        If location was written more than threshold times,
        reduce by the threshold.
        Otherwise delete the location.
        Returns the number of deleted addresses.
        """
        count = 0
        for address in list(self.locations.keys()):
            if self.locations[address] > threshold:
                self.locations[address] -= threshold
            else:
                del self.locations[address]
                count += 1
        return count

    def intersection_level(self, neuron):
        """
        Returns the amount of locations written in both neurons.

        Considering a & b the intersection of the locations written in both
        neurons and a | b their union, this method returns (a & b)/(a | b).
        """
        len_intrsctn = float(
            len(self.locations.keys() & neuron.locations.keys()))
        len_union = float(len(self.locations.keys() | neuron.locations.keys()))
        return len_intrsctn / len_union

    def count(self, address):
        """
        Returns how many times the location being addressed was written.
        """
        return self.locations[address]


class SWNeuron(Neuron):
    """
    This neuron uses a sliding window model and
    saves the last time an address was written.
    """

    def __init__(self, address_length=None):
        super().__init__(address_length)
        self.locations = cl.OrderedDict()

    def record(self, address, time):
        self.locations[address] = time

    def bleach(self, threshold):
        """
        Clears the locations recorded before the time threshold.
        Returns the number of deleted addresses.
        """
        count = 0
        for address, time in zip(list(self.locations.keys()),
                                 list(self.locations.values())):
            if time < threshold:
                del self.locations[address]
                count += 1
            else:
                break  # end because Dict is orderd
        return count

    def intersection_level(self, neuron):
        """
        Returns the amount of locations written in both neurons.

        Considering a & b the intersection of the locations written in both
        neurons and a | b their union, this method returns (a & b)/(a | b).
        """
        len_intrsctn = float(
            len(self.locations.keys() & neuron.locations.keys()))
        len_union = float(len(self.locations.keys() | neuron.locations.keys()))
        return len_intrsctn / len_union
