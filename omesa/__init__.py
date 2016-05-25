"""Experiment, storage and interpretation module for machine learning research.

A small framework for reproducible machine learning research that largely
builds on top of scikit-learn. Its goal is to make common research procedures
fully automated, optimized, and well recorded.
"""

from omesa import experiment
from omesa import containers
from omesa import featurizer

__author__ = 'Chris Emmery'
__contrb__ = ('Akos Kadar', 'Mike Kestemont', 'Florian Kunneman',
              'Janneke van de Loo', 'Ben Verhoeven')
__license__ = 'GPLv3'
__version__ = '0.3.0a0'

__all__ = ['experiment', 'containers', 'featurizer']
