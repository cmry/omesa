"""20newsgroups experiment."""

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD

# for as long as it's not yet pip installable
import sys
sys.path.append('../')
# -----

try:
    from omesa.experiment import Experiment
    from omesa.featurizer import Ngrams, WordEmbeddings
    from omesa.containers import Pipe
    from omesa.components import Vectorizer, Evaluator
except ImportError as e:
    print(e)
    exit("Could not load omesa. Please update the path in this file.")


def loader(subset, emax=None, categories=['comp.graphics', 'sci.space']):
    """Loader wrapper for 20news set."""
    tset = fetch_20newsgroups(subset=subset, categories=categories,
                              shuffle=True, random_state=42)

    for text, label in zip(tset.data, tset.target):
        if emax is None:
            yield text, tset.target_names[label]
        elif emax:
            yield text, tset.target_names[label]
            emax -= 1
        elif emax is 0:
            break

Experiment(
    project="unit_tests",
    name="20_news_grams",
    data=loader('train'),
    pipeline=[
        Vectorizer(features=[Ngrams(level='char', n_list=[3])]),
        Pipe('scaler', MaxAbsScaler()),
        Pipe('clf', MultinomialNB()),
        Evaluator(scoring='f1', average='micro',
                  lime_docs=[dat[0] for dat in loader('test', emax=5)])
    ],
    save=("model", "db")
)

Experiment(
    project="unit_tests",
    name="20_news_grams_mc",
    data=loader('train', categories=['comp.graphics', 'sci.space', 'alt.atheism']),
    pipeline=[
        Vectorizer(features=[Ngrams(level='char', n_list=[3])]),
        Pipe('clf', MultinomialNB()),
        Evaluator(scoring='f1_weighted', average='weighted',
                  lime_docs=[dat[0] for dat in loader('test', emax=5)])
    ],
    save=("model", "db")
)

Experiment(
    project="unit_tests",
    name="20_news_emb",
    data=loader('train', categories=['comp.graphics', 'sci.space', 'alt.atheism']),
    lime_data=[dat[0] for dat in loader('test', emax=5)],
    pipeline=[
        Vectorizer(features=[WordEmbeddings(lang='nl')]),
        Pipe('clf', SVC(kernel='linear', probability=True)),
        Evaluator(scoring='f1_weighted', average='weighted',
                  lime_docs=[dat[0] for dat in loader('test', emax=5)])
    ],
    save=("model", "db")
)
