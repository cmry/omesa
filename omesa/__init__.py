"""The Main Thing.

Omesa is a framework that makes standard text mining research protocols less
script heavy.

Examples
--------

One of the examples provided is that of n-gram classification of Wikipedia
documents. In this experiment, we are provided with n-gram that features
10 articles about Machine Learning, and 10 random other articles. To run Omesa
for this, the following configuration is used:

>>> from omesa.experiment import Experiment
>>> from omesa.featurizer import Ngrams

>>> conf = {
...     "name": "gram_experiment",
...     "train_data": ["./omesa/examples/n_gram.csv"],
...     "has_header": True,
...     "features": [Ngrams(level='char', n_list=[3])],
...     "text_column": 1,
...     "label_column": 0,
...     "folds": 10,
...     "save": ("log")
... }

>>> Experiment(conf)
"""

# pylint:    disable=E0603

__author__ = 'Chris Emmery'
__contrb__ = ('Ákos Kádár', 'Mike Kestemont', 'Florian Kunneman',
              'Janneke van de Loo', 'Ben Verhoeven')
__license__ = 'MIT'

import omesa.experiment
import omesa.processor
