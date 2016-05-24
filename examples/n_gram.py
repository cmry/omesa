"""N-gram experiment."""

import numpy as np
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
    from omesa.containers import CSV, Pipe
except ImportError as e:
    print(e)
    exit("Could not load omesa. Please update the path in this file.")

Experiment(
    project="unit_tests",
    name="gram_experiment",
    train_data=CSV("n_gram.csv", data="intro", label="label"),
    lime_data=CSV("n_gram.csv", data="intro", label="label"),
    features=[Ngrams(level='char', n_list=[3])],
    pipeline=[
        Pipe('scaler', MaxAbsScaler()),
        Pipe('clf', SVC(kernel='linear'),
             parameters={'C': np.logspace(-2.0, 1.0, 10)}),
    ],
    save=("log", "model", "db")
)

Experiment(
    project="unit_tests",
    name="gram_experiment_bayes",
    train_data=CSV("n_gram.csv", data="intro", label="label"),
    lime_data=CSV("n_gram.csv", data="intro", label="label"),
    features=[Ngrams(level='char', n_list=[3])],
    pipeline=[
        Pipe('clf', MultinomialNB())
    ],
    save=("log", "model", "db")
)
