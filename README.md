# Dealing with Concept Drifts via Weightless Neural Networks

This repository contains the code for my bachelor thesis.
The goal is to deal with concept drifts in WiSARD, in particular investigating the role of incremental update and unlearning for different data streams and understanding the effect of the algorithm's parameters.

Parts of this repository are based on the implementation of Douglas O. Cardoso: https://bitbucket.org/douglascardoso/ecmlpkdd2016/src/master/


Possible Changes:
* Add static type checking to prevent errors and facilitate debugging
* Add neuron types
* Add discriminator types
* Add wisard types
* Add `__str__` method to all object types
* Use multiprocessing to optimize performance (addressing, bleach, predict, clear, ...)
* Implement receiving batches of the stream at once
* Implement `load()` in WCDS
