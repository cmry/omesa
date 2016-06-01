"""Experiment wrapper code.
"""

# License:      GPLv3

from time import time

import numpy as np

from sklearn import metrics
from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import train_test_split

from .logger import _Logger as Logger
from .components import Vectorizer, Optimizer
from .containers import Pipeline


class Experiment(object):
    """Full experiment wrapper.

    Calls several sklearn modules in the Pipeline class and reports on the
    classifier performance. For this, the class uses a configuration
    dictionary. The full list of options for this is listed under attributes.

    Parameters
    ----------
    project : string
        The project name functions as a hook to for example call the best
        performing set of parameters out of a series of experiments on the same
        data.

    name : string
        Same as the above. This will function as a hook to save your model,
        features and Omesa config under one name.

    train_data : list of, or single iterator
        Example: [CSV("/somedir/train.csv", label=1, text=2),
                  CSV("/somedir/train2.csv", label=3, text=5]
        The data on which the experiment will train. If the location of a .csv
        is provided, it will open these up and create an iterator for you.
        Alternatively, you can provide your own iterators or iterable
        structures providing instances of the data. If only training data is
        provided, the experiment will evaluate in a tenfold setting by default.

    test_data : list of, or single iterator
        Example: [CSV("/somedir/test.csv", label=1, text=2)]
        This works similar to the train_data. However, when a test set is
        provided, the performance of the model will be measured on this test
        data only. Omesa will dump a classification report for you.

    test_proportion : float
        Example: 0.3
        As opposed to a test FILE, one can also provide a test proportion,
        after which a certain amount of instances will be held out from the
        training data to test on.

    features : list of featurizer classes
        Example: [Ngrams()]
        These can be features imported from omesa.featurizer or can be any
        class you create your self. As long as it adheres to a fit / transform
        structure and returns a feature dictionary per instance as can be
        provided to, for example, the sklearn FeatureHasher.

    backbone : class from backbone
        Example: Spacy()
        The backbone is used as an all-round NLP toolkit for tagging, parsing
        and in general annotating the text that is provided to the experiment.
        If you wish to utilize features that need for example tokens, lemmas or
        POS tags, they can be parsed during loading. Please be advised that
        it's more convenient to do this yourself beforehand.

    classifier : class with sklearn API
        Example: GaussianNB()
        Used to switch the classifier used in the experiment. By default, an
        SVM with low parameter settings is used if you do NOT want to use grid
        search. In any other case, you can provide other sklearn classifiers
        that can be set to output probabilities.

    save : tuple of strings
        Example: ("log", model", "db", "man", "json", "pickle")
        Save the output of the log, or dump the entire model with its
        classification method and pipeline wrapper for new data instances.
    """

    def __init__(self, **kwargs):
        """Set all relevant classes, run experiment (currently)."""
        self.conf = kwargs
        self.log = Logger(self.conf['name'])
        self.vec = Vectorizer(self.conf)
        self.opt = Optimizer(self.conf)
        self.clf = None
        self.clf_unfit = None
        self.res = {}
        self._run(self.conf)

    def save(self):
        """Save desired Experiment data."""
        if self.conf.get('save'):
            if 'log' in self.conf['save']:
                self.log.save()
            if 'features' in self.conf['save']:
                self.log.echo(" Feature saving has not been implemented yet!")
            if 'model' in self.conf['save']:
                Pipeline(self).save()

    def _run_proportions(self, sets, av, seed, pr=None):
        """Repeats run and proportionally increases the amount of data."""
        X, Xi, y, yi = sets
        pr = self.conf.get('proportions', pr)
        self.res['prop'], tmp, conf = {}, self.opt, self.conf
        for i in range(1, pr):
            self.opt, prop = Optimizer(conf), (1 / pr) * (pr - i)
            self.log.slice((1 - prop, ))
            Xp, _, yp, _ = train_test_split(X, y, test_size=prop, stratify=y)
            clff = self.opt.choose_classifier(Xp, yp, seed).fit(Xp, yp)
            tres = cross_val_predict(clff, Xp, yp, cv=5, n_jobs=-1)
            tscore = metrics.f1_score(yp, tres, average=av)
            score = metrics.f1_score(yi, clff.predict(Xi), average=av)
            print("\n Result: {0}".format(score))
            self.res['prop'].update(
                {1 - prop: {'train': tscore, 'test': score}})
        self.opt = tmp

    def _run(self, conf):
        """Split data, fit, transfrom features, tf*idf, svd, report."""
        t1 = time()
        seed = 42
        np.random.RandomState(seed)

        # report features
        self.log.head(conf['features'], conf['name'], seed)

        # stream data to sparse features

        X, y = self.vec.fit_transform(conf['train_data'])
        # split off test data
        if not conf.get('test_data'):
            X, Xi, y, yi = train_test_split(
                X, y, test_size=conf.get('test_proportion', 0.1), stratify=y)

        # grid search and fit best model choice
        self.clf = self.opt.choose_classifier(X, y, seed)
        print("\n Training model...")
        self.clf.fit(X, y)
        print(" done!")

        av = 'binary' if len(set(y)) == 2 else 'micro'
        labs = self.vec.encoder.classes_
        self.log.data('sparse', 'train', X)
        # if user wants to report more than best score, do another CV on train
        if conf.get('detailed_train', True):
            res = cross_val_predict(self.clf, X, y, cv=5, n_jobs=-1)
            self.res['train'] = self.log.report('train', y, res, av, metrics,
                                                labs)

        # report performance
        if conf.get('test_data'):
            Xi, yi = self.vec.transform(conf['test_data'])

        self.log.data('sparse', 'test', Xi, dump=True)
        res = self.clf.predict(Xi)
        self.res['test'] = self.log.report('test', yi, res, av, metrics, labs)

        if conf.get('proportions'):
            self._run_proportions((X, Xi, y, yi), av, seed)

        print("\n # ------------------------------------------ \n")
        t2 = time()
        dur = round(t2 - t1, 1)
        self.res['dur'] = dur
        print("\n Experiment took {0} seconds".format(dur))

        self.save()
        print("\n" + '-' * 10, "\n")
