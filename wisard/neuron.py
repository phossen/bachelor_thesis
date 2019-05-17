import numpy as np


class Neuron():
    def __init__(self):
        """
        This class implements a basic neuron.
        Its dictionary saves addresses and their
        LRU in form of addressing(x_i) : t
        """
        self.addresses = dict()

    def __len__(self):
        """
        Returns the number of the addresses in the neuron.
        """
        return len(self.addresses)

    def update(self, address, t):
        """
        Adds or updates an address and it's LRU time.
        """
        self.addresses[address] = t

    def clear(self):
        """
        Clears the neurons addresses.
        """
        self.addresses.clear()

    def delete(self, address):
        """
        Delete the given address.
        """
        del self.addresses[address]
