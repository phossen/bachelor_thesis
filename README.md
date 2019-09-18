# Dealing with Concept Drifts via Weightless Neural Networks

This repository contains the code for my bachelor thesis.
The goal is to deal with concept drifts via WiSARD for Clustering Data Streams (WCDS), in particular investigating the role of incremental update and unlearning for different data streams and understanding the effect of the algorithm's parameters.


## Requirements
This package needs Python 3.6+ and a few external packages. Most of them should be covered by Anaconda.


## Usage
The alogrithms of the WCDS package can be imported by `from wcds.wcds import WCDS` and `from wcds.clusterers import *`.

The file `DiscrVisual.py` is used to visualize the effect of the algorithm's parameters.

The performed experiments are in the Experiments Jupyter Notebook.


## References
Parts of this repository are based on the implementation of Douglas O. Cardoso: https://bitbucket.org/douglascardoso/ecmlpkdd2016/src/master/
