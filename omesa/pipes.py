"""Piplines and optimization."""

# pylint:       disable=E1103,E1101,E0611,C0103,C0325,C0330,W0141,R0914

from .featurizer import Featurizer
from .data import Dataloader
from .logger import Reporter

import numpy as np
from sklearn import pipeline
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle


class Pipeline(object):
    """Small text mining research pipeline.

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

    handle : class
        LabelHandler instance defined earlier in this file.

    hasher : class
        DictVectorizer class from sklearn.

    tfdif : class
        TfidfTransformer class from sklearn.

    svd : class
        TruncatedSVD class from sklearn.

    conf : dict
        Configuration dictionary as passed to the experiment class.
    """

    def __init__(self, conf):
        """Start pipeline modules."""
        self.loader = Dataloader(conf)
        self.reporter = Reporter(conf['name'])
        self.featurizer = Featurizer(conf['features'],
                                     conf.get('preprocessor'),
                                     conf.get('parser'))

        self.hasher = DictVectorizer()
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = None
        self.conf = conf

    def train(self, data):
        """Send the training data through all applicable steps."""
        # TODO: consider removing print statements
        D, y = zip(*[(v, label) for label, v in
                     self.featurizer.transform(self.loader.load_data(data))])
        print("\n Hashing features...")
        X = self.hasher.fit_transform(D)
        print(" done!")

        if 'tfidf' in self.conf.get('settings', ''):
            print("\n Tf*idf transformation...")
            X = self.tfidf.fit_transform(X)
            print(" done!")
        else:
            X = MaxAbsScaler(copy=False).fit_transform(X)

        return X, y

    def test(self, data):
        """Send the test data through all applicable steps."""
        # same steps as pipe_train
        Di, yi = zip(*[(v, label) for label, v in
                       self.featurizer.transform(
                            self.loader.load_data(data, test=True))])
        Xi = self.hasher.transform(Di)

        if 'tfidf' in self.conf.get('settings', ''):
            Xi = self.tfidf.transform(Xi, copy=False)
        else:
            Xi = MaxAbsScaler(copy=False).fit_transform(Xi)

        if 'svd' in self.conf.get('settings', ''):
            Xi = self.svd.transform(Xi)

        return Xi, yi


class Grid(Pipeline):
    """Current placeholder for grid methods. Should be fleshed out."""

    def split_dev(self, X, y, seed=42):
        """Split dev if should be hold out (test_set here is X)."""
        X_dev, y_dev = None, None
        if 'hold_grid' in self.conf.get('settings', ''):
            X_dev, X, y_dev, y = train_test_split(X, y, test_size=0.8,
                                                  random_state=seed,
                                                  stratify=y)
        else:
            X, y = shuffle(X, y, random_state=seed)
        return X, y, X_dev, y_dev

    def choose_classifier(self, X, y, seed):
        """Choose a classifier based on settings."""
        conf = self.conf
        X, y, X_dev, y_dev = self.split_dev(X, y, seed)

        # apply SVD once
        if conf.get('components'):
            try:
                assert 'svd' in conf.get('settings')
            except AssertionError:
                exit("ERROR! You specified components but no SVD in settings.")
            n_comp = conf['components']
            self.log.post('svd', (n_comp, ))
            self.pipe.svd = TruncatedSVD(n_components=n_comp)
            X_dev = self.pipe.svd.fit_transform(X_dev)
            print(" done!")

        if 'grid' in conf.get('settings', ''):

            # will only run LinearSVC for now
            user_grid = conf.get('parameters')
            param_grid = {'clf__C': np.logspace(-3, 2, 6)} if not \
                user_grid else user_grid
            steps = [('clf', conf.get('classifier',
                                      LinearSVC(class_weight='balanced')))]

            # incorporate SVD into GridSearch
            if 'svd' in conf.get('settings') and not conf.get('components'):
                param_grid.update({'svd__n_components': [50, 100, 500, 1000]})
                steps = [('svd', TruncatedSVD())] + steps

            pipe = pipeline.Pipeline(steps)
            print("\n", "Grid: ", param_grid)
            grid = GridSearchCV(pipe, scoring='f1_micro', param_grid=param_grid,
                                n_jobs=-1)

            print("\n Starting Grid Search...")
            # grid search approximated on dev
            if X_dev and y_dev:
                grid.fit(X_dev, y_dev)
            else:
                grid.fit(X, y)
            print(" done!")

            p = self.reporter.grid(grid.grid_scores_)
            clf = grid.best_estimator_

            if 'svd' in conf.get('settings') and not conf.get('components'):
                comp = p['svd__n_components']
                self.pipe.svd = TruncatedSVD(n_components=comp)

            if 'svd' in conf.get('settings'):
                X = self.pipe.svd.transform(X)

            return X, y, clf

        elif not conf.get('classifier'):
            clf = LinearSVC(random_state=seed, C=5)
        elif conf.get('classifier'):
            clf = conf['classifier']
            clf.random_state = seed
        else:
            clf.probability = True
            clf.random_state = seed

        return X, y, clf
