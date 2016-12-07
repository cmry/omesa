"""Vectorizer and optimization.
"""

# pylint:       disable=E1135,E1101

from multiprocessing import Pool
from operator import itemgetter
from time import time

import numpy as np

from sklearn import pipeline
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split

from .featurizer import Featurizer
from .containers import Pipe, _chain


class Vectorizer(object):
    """Small text mining vectorizer.

    The purpose of this class is to provide a small set of methods that can
    operate on the data provided in the Experiment class. It can load data
    from an iterator or .csv, and guides that data along a set of modules such
    as the feature extraction, tf*idf function, SVD, etc. It can be controlled
    through a settings dict that is provided in conf.

    Parameters
    ----------
    features : list of Featurizer classes.

    preprocessor: Preprocessor class.

    parser: Parser class.
    """

    def __init__(self, features, preprocessor=None, parser=None, n_jobs=None):
        """Start pipeline modules."""
        self.featurizer = Featurizer(features, preprocessor, parser)
        self.hasher = DictVectorizer()
        self.encoder = LabelEncoder()
        self.n_jobs = n_jobs

    def _vectorize(self, data, func):
        """Send the data through all applicable steps to vectorize."""
        if isinstance(data, list) and hasattr(data[0], 'source'):
            data = _chain(data)

        if self.n_jobs != 1:
            p = Pool(processes=self.n_jobs)
            D, y = zip(*p.map(self.featurizer.transform, data))
            p.close()
            p.join()
            del p
        else:
            D, y = zip(*map(self.featurizer.transform, data))

        # NOTE: these _can't_ be put in p.map because `fit` overwrites in iter
        X = getattr(self.hasher, func)(D)
        y = getattr(self.encoder, func)(y) if len(set(y)) != 1 else ''

        if len(y):
            return X, y
        else:
            return X

    def fit_transform(self, data):
        """Adhere to sklearn API."""
        return self._vectorize(data, func='fit_transform')

    def transform(self, data):
        """Adhere to sklearn API."""
        return self._vectorize(data, func='transform')


class Evaluator(object):

    def __init__(self, **kwargs):
        """Set all relevant settings, run experiment.

        Parameters
        ----------
        test_data : list of, or single iterator
            Example: [CSV("/somedir/test.csv", label=1, text=2)]
            This works similar to the train_data. However, when a test set is
            provided, the performance of the model will also be measured on this
            test data. Omesa will dump a classification report for you.

        test_size : float
            Example: 0.3
            As opposed to a test FILE, one can also provide a test proportion,
            after which a certain amount of instances will be held out from the
            training data to test on.

        scoring: string

        detailed_train: bool

        cv: int

        proportions: int

        """
        self.__dict__.update(kwargs)
        self.__dict__['cv'] = self.__dict__.get('cv', 5)
        self.res = {}
        self.scores = {}

    def _run_proportions(self, sets, exp):

        """Repeats run and proportionally increases the amount of data."""
        X, Xi, y, yi = sets
        self.res['prop'] = {}

        for i in range(1, exp.batches):
            prop = (1 / exp.batches) * (exp.batches - i)
            exp.log.slice((1 - prop, ))

            Xp, _, yp, _ = train_test_split(X, y, train_size=prop, stratify=y)
            clff = self.grid_search(Xp, yp, self.scoring, exp.seed).fit(Xp, yp)

            tres = cross_val_predict(clff, Xp, yp, cv=5, n_jobs=-1)

            # FIXME: this is going to break with any other metric
            tscore = metrics.f1_score(yp, tres, average=self.average)
            score = metrics.f1_score(yi, clff.predict(Xi), average=self.average)

            print("\n Result: {0}".format(score))
            self.res['prop'].update(
                {1 - prop: {'train': tscore, 'test': score}})

    def best_model(self):
        """Choose best parameters of trained classifiers."""
        # FIXME: I think this can be deprecated
        score_sum, highest_score = {}, 0
        for score, estim in self.scores.values():
            score_sum[score] = estim
            if score > highest_score:
                highest_score = score

        print("\n #--------- Grid Results -----------")
        print("\n Best scores: {0}".format(
            {str(dict(y.steps)['clf'].__class__.__name__): round(x, 3)
             for x, y in score_sum.items()}))

        return highest_score, score_sum[highest_score]

    def grid_search(self, pln, X, y, seed):
        """Choose a classifier based on settings."""
        clfs, pipes = [], []

        for pipe in pln:
            pipe.check(seed)
            if pipe.idf == 'clf':
                clfs.append(pipe)
            else:
                pipes.append(pipe)

        for i, clf in enumerate(clfs):
            grid = {pipe.idf + '__' + k: v for pipe in pipes
                    for k, v in pipe.parameters.items()}
            grid.update({'clf__' + k: v for k, v in clf.parameters.items()})

            print("\n #-------- Classifier {0} ----------".format(i + 1))
            print("\n", "Clf: ", str(clf.skobj))
            print("\n", "Grid: ", grid)

            grid = GridSearchCV(
                pipeline.Pipeline([(pipe.idf, pipe.skobj) for pipe in pipes] +
                                  [('clf', clf.skobj)]),
                scoring=self.scoring, param_grid=grid,
                n_jobs=-1 if not hasattr(pipe.skobj, 'n_jobs') else 1)

            print("\n Starting Grid Search...")
            grid.fit(X, y)
            print(" done!")

            self.scores[clf] = (grid.best_score_, grid.best_estimator_)

        score, clf = self.best_model()
        self.scores['best'] = score

        return clf

    def evaluate(self, exp):
        """Split data, fit, transfrom features, tf*idf, svd, report."""
        t1 = time()

        exp.seed = 42
        exp.nj = -1
        exp.test_size = 0.3 if not hasattr(exp, 'test_size') else exp.test_size
        np.random.RandomState(exp.seed)

        # report features
        if hasattr(exp.pln[0], 'features'):
            exp.log.head(exp.pln.features, exp.name, exp.seed)

        # stream data to features
        X, y = exp.vec.fit_transform(exp.data)

        # if no test data, split
        if not hasattr(self, 'test_data'):
            X, Xi, y, yi = train_test_split(
                X, y, test_size=exp.test_size, stratify=y)
        else:
            Xi, yi = exp.vec.transform(self.test_data)

        av = self.average
        # grid search and fit best model choice
        exp.pln = self.grid_search(exp.pln, X, y, exp.seed)
        print("\n Training model...")
        exp.pln.fit(X, y)
        print(" done!")

        labs = exp.vec.encoder.classes_
        exp.log.data('sparse', 'train', X)

        # if user wants to report more than best score, do another CV on train
        # if hasattr(self, 'detailed_train'):
        sco = cross_val_predict(exp.pln, X, y, cv=self.cv, n_jobs=exp.nj)
        self.res['train'] = exp.log.report('train', y, sco, av, labs)

        exp.log.data('sparse', 'test', Xi, dump=True)
        res = exp.pln.predict(Xi)
        self.res['test'] = exp.log.report('test', yi, res, av, labs)

        if hasattr(self, 'proportions'):
            self._run_proportions((X, Xi, y, yi), exp)

        print("\n # ------------------------------------------ \n")
        t2 = time()
        dur = round(t2 - t1, 1)
        self.res['dur'] = dur
        print("\n Experiment took {0} seconds".format(dur))

        exp.store()
        print("\n" + '-' * 10, "\n")
