"""20newsgroups experiment."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC

# for as long as it's not yet pip installable
import sys
sys.path.append('../')
# -----

try:
    from omesa.experiment import Experiment
    from omesa.featurizer import Ngrams
    from omesa.containers import CSV
except ImportError:
    exit("Could not load omesa. Please update the path in this file.")


def loader(subset, emax=None):
    categories = ['comp.graphics', 'sci.space']
    tset = fetch_20newsgroups(subset=subset, categories=categories,
                              shuffle=True, random_state=42)

    for text, label in zip(tset.data, tset.target):
        if emax is None:
            yield text, label
        elif emax:
            yield text, label
            emax -= 1
        elif emax is 0:
            break

Experiment({
    "project": "unit_tests",
    "name": "20_news",
    "train_data": loader('train'),
    "test_data": loader('test'),
    "lime_data": loader('test', emax=5),
    "features": [Ngrams(level='char', n_list=[3])],
    "normalizers": [MaxAbsScaler()],
    "classifiers": [
        {'clf': SVC(kernel='linear'), 'C': np.logspace(-2.0, 1.0, 50)},
        {'clf': MultinomialNB()}
    ],
    "save": ("log", "model", "db")
})
