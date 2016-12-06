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
    from omesa.containers import Pipe, CSV
    from omesa.components import Vectorizer, Evaluator
except ImportError as e:
    print(e)
    exit("Could not load omesa. Please update the path in this file.")

Experiment(
    project="unit_tests",
    name="gram_experiment",
    data=[CSV("n_gram.csv", data="intro", label="label")],
    pipeline=[
        Vectorizer(features=[Ngrams(level='char', n_list=[3])]),
        Pipe('scaler', MaxAbsScaler()),
        Pipe('clf', SVC(kernel='linear'),
             parameters={'C': np.logspace(-2.0, 1.0, 10)}),
        Evaluator(scoring='f1', average='micro',
                  lime_docs=CSV("n_gram.csv", data="intro", label="label")),
    ],
    save=("log", "model", "db")
)

Experiment(
    project="unit_tests",
    name="gram_experiment_multic",
    data=[CSV("n_gram.csv", data="intro", label="label"),
          CSV("n_gram2.csv", data="intro", label="label")],
    pipeline=[
        Vectorizer(features=[Ngrams(level='char', n_list=[3])]),
        Pipe('scaler', MaxAbsScaler()),
        Pipe('clf', SVC(kernel='linear'),
             parameters={'C': np.logspace(-2.0, 1.0, 10)}),
        Evaluator(scoring='f1_weighted', average='weighted',
                  lime_docs=CSV("n_gram.csv", data="intro", label="label")),
    ],
    save=("log", "model", "db")
)
