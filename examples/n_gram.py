"""N-gram experiment."""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
import numpy as np
# for as long as it's not yet pip installable
import sys
sys.path.append('../')
# -----

try:
    from omesa.experiment import Experiment
    from omesa.featurizer import Ngrams
    from omesa.containers import CSV
except ImportError as e:
    print(e)
    exit("Could not load omesa. Please update the path in this file.")

Experiment({
    "project": "unit_tests",
    "name": "gram_experiment",
    "train_data": CSV("n_gram.csv", data=1, label=0, header=True),
    "lime_data": CSV("n_gram.csv", data=1, label=0, header=True),
    "features": [Ngrams(level='char', n_list=[3])],
    "normalizers": [MaxAbsScaler()],
    "classifiers": [
        # {'clf': SGDClassifier(n_jobs=-1, n_iter=5), 'alpha': 10.0**-np.arange(1, 7)},
        {'clf': LinearSVC(), 'C': np.logspace(-2.0, 1.0, 50)}
    ],
    "save": ("log", "model", "db")
})

Experiment({
    "project": "unit_tests",
    "name": "gram_experiment_bayes",
    "train_data": CSV("n_gram.csv", data=1, label=0, header=True),
    "lime_data": CSV("n_gram.csv", data=1, label=0, header=True),
    "features": [Ngrams(level='char', n_list=[3])],
    "normalizers": [MaxAbsScaler()],
    "classifiers": [
        {'clf': MultinomialNB()}
    ],
    "save": ("log", "model", "db")
})
