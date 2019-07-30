# Dealing with Concept Drifts via Weightless Neural Networks

This repository contains the code for my bachelor thesis.
The Goal is to deal with concept drifts in WiSARD, in particular investigating the role of incremental update and unlearning for different data streams and understanding the effect of the algorithm's parameters.

Parts of this repository are based on the implementation of Douglas O. Cardoso: https://bitbucket.org/douglascardoso/ecmlpkdd2016/src/master/


Possible Changes:
* Add static type checking to prevent errors and facilitate debugging
* Add neuron types
* Add discriminator types
* Add wisard types

* Code visualisierung, Visualizing epsilon Area discriminators
* use multiprocessing; improve code regarding efficency
* include getting batches at same time?
* logging module, make logging optional due to performance
* Save more efficient by using integers
* WCDS: Logging, logger format, Multiprocessing, test wrong mapping, import clustering, implement load, use pickle instead of json (to json in every layer?)
* autopep8
* git add, commit, push
