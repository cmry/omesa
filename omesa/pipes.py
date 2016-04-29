"""Vectorizer and optimization."""

from copy import deepcopy
from operator import itemgetter
from multiprocessing import Pool

import numpy as np
from sklearn import pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC

from .featurizer import Featurizer


class Vectorizer(object):
    """Small text mining vectorizer.

    The purpose of this class is to provide a small set of methods that can
    operate on the data provided in the Experiment class. It can load data
    from an iterator or .csv, and guides that data along a set of modules such
    as the feature extraction, tf*idf function, SVD, etc. It can be controlled
    through a settings dict that is provided in conf.

    Parameters
    ----------
    conf : dict
        Configuration dictionary passed to the experiment class.

    Attributes
    ----------

    featurizer : class
        Environment class (might be replace by Featurizer) from Omesa.

    hasher : class
        DictVectorizer class from sklearn.

    normalizers : class
        TfidfTransformer class from sklearn.

    decomposers : class
        TruncatedSVD class from sklearn.

    conf : dict

    """

    def __init__(self, conf=None, featurizer=None, normalizers=None,
                 decomposers=None):
        """Start pipeline modules."""
        if conf:
            self.conf = conf
        if not featurizer:
            self.featurizer = Featurizer(conf['features'],
                                         conf.get('preprocessor'),
                                         conf.get('parser'))

        self.hasher = DictVectorizer()
        self.encoder = LabelEncoder()
        self.normalizers = conf.get('normalizers', normalizers)
        self.decomposers = conf.get('decompositer', decomposers)

    def transform(self, data, fit=False):
        """Send the data through all applicable steps to vectorize."""
        p = Pool(processes=self.conf.get('n_jobs', None))
        D, y = zip(*p.map(self.featurizer.transform, data))
        func = 'transform' if not fit else 'fit_transform'

        # NOTE: these _can't_ be put in p.map because `fit` overwrites in iter
        X = getattr(self.hasher, func)(D)
        y = getattr(self.encoder, func)(y) if len(set(y)) != 1 else ''

        # TODO: this could be moved to grid to also search over these settings
        if self.normalizers:
            for norm in self.normalizers:
                norm.copy = False
                X = getattr(norm, func)(X)
        if self.decomposers:
            for dcmp in self.decomposers:
                X = getattr(dcmp, func)(X, copy=False)

        if len(y):
            return X, y
        else:
            return X


class Optimizer(object):
    """Current placeholder for grid methods. Should be fleshed out.

    Parameters
    ----------
    classifiers : dict, optional, default None
        Dictionary where the key is a initiated model class, and the values
        are a dictionary with parameter settings in a {string: array} format,
        same as used in the scikit-learn pipeline. So for example, we provide:
        {LinearSVC(class_weight='balanced'): {'C': np.logspace(-3, 2, 6)}}.
        Note that pipeline requires some namespace (like clf__C), but the class
        handles that already.

    conf : dict, optinal, default None
        Configuration dictionary used by the Experiment class wrapper.
    """

    def __init__(self, conf=None, classifiers=None, scoring='f1'):
        """Initialize optimizer with classifier dict and scoring, or conf."""
        std_clf = [{'clf': LinearSVC(class_weight='balanced'),
                    'C': np.logspace(-3, 2, 6)}]
        if not classifiers:
            classifiers = std_clf
        if not conf.get('classifiers'):
            conf['classifiers'] = std_clf

        self.scores = {}
        self.met = conf.get('scoring', scoring)
        self.conf = conf if conf else classifiers

    def best_model(self):
        """Choose best parameters of trained classifiers."""
        score_sum = {}
        highest_score = 0
        for scores, estim in self.scores.values():
            best = sorted(scores, key=itemgetter(1), reverse=True)[0]
            score = best.mean_validation_score
            score_sum[score] = estim
            if score > highest_score:
                highest_score = score
        print("\n\n Best scores: {0}".format(
            {str(y).split('(')[2][7:]: round(x, 3)
             for x, y in score_sum.items()}))
        return highest_score, score_sum[highest_score]

    def choose_classifier(self, X, y, seed):
        """Choose a classifier based on settings."""

        for grid in deepcopy(self.conf['classifiers']):
            clf = grid.pop('clf')
            clf.probability = True
            clf.random_state = seed
            grid = {'clf__' + k: v for k, v in grid.items()}
            print("\n", "Clf: ", str(clf))
            print("\n", "Grid: ", grid)
            grid = GridSearchCV(pipeline.Pipeline([('clf', clf)]),
                                scoring=self.met, param_grid=grid,
                                n_jobs=self.conf.get('n_jobs', -1))

            print("\n Starting Grid Search...")
            grid.fit(X, y)
            print(" done!")

            self.scores[clf] = (grid.grid_scores_, grid.best_estimator_)

        score, clf = self.best_model()
        self.scores['best'] = score

        return clf
