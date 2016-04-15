# -*- coding: utf-8 -*-

"""Experiment wrapper code."""

# License:      MIT
# pylint:       disable=E1103,E1101,E0611,C0103,C0325,C0330,W0141,E0401,R0903

from time import time
from types import GeneratorType

import numpy as np
from sklearn import metrics
from sklearn.cross_validation import cross_val_predict

from .logger import Logger
from .pipes import Vectorizer, Optimizer
from .io import Pipeline

from os import getcwd


class Experiment(object):
    """Full experiment wrapper.

    Calls several sklearn modules in the Pipeline class and reports on the
    classifier performance. For this, the class uses a configuration
    dictionary. The full list of options for this is listed below:

    conf = {
        "experiment_name": {

        The experiment name functions as a hook to for example call the best
        performing set of parameters out of a series of experiments on the same
        data.
        ---

            "name": "experiment_name",

            Same as the above. This will function as a hook to save your model,
            features and Omesa config under one name.
            ---

            "train_data": [CSV("/somedir/train.csv", label=1, text=2),
                           CSV("/somedir/train2.csv", label=3, text=5],

            The data on which the experiment will train. If the location of a
            .csv is provided, it will open these up and create an iterator for
            you. Alternatively, you can provide your own iterators or iterable
            structures providing instances of the data. If only training data
            is provided, the experiment will evaluate in a tenfold setting by
            default.
            ---

            "test_data": [CSV("/somedir/test.csv", label=1, text=2)], # either

            This works similar to the train_data. However, when a test set is
            provided, the performance of the model will be measured on this
            test data only. Omesa will dump a classification report for you.
            ---

            "test_proportion": 0.3,                                    # or

            As opposed to a test FILE, one can also provide a test proportion,
            after which a certain amount of instances will be held out from the
            training data to test on.
            ---

            "features": [Ngrams()],

            These can be features imported from omesa.featurizer or can be
            any class you create your self. As long as it adheres to a fit /
            transform structure and returns a feature dictionary per instance
            as can be provided to, for example, the sklearn FeatureHasher.
            ---

            "backbone": Spacy(),                   # or Frog() - optional

            The backbone is used as an all-round NLP toolkit for tagging,
            parsing and in general annotating the text that is provided to the
            experiment. If you wish to utilize features that need for example
            tokens, lemmas or POS tags, they can be parsed during loading.
            Please be advised that it's more convenient to do this yourself
            beforehand.
            ---

            "classifier": GaussianNB()             # SVC by default - optional

            Used to switch the classifier used in the experiment. By default,
            an SVM with low parameter settings is used if you do NOT want to
            use grid search. In any other case, you can provide other sklearn
            classifiers that can be set to output probabilities.
            ---

            "setting": "grid",                     # optional

            Currently only can be set to grid. If this is done, the classifier
            of choice will be an SVM, for which the experiment will try to
            optimize the parameter settings through CVGridSearch. At some
            point, it might be useful to be able to provide the parameters
            yourself (for other classifiers for example).
            ---

            "components": 200,                     # n for SVD - optional

            Simply the n_components for TruncatedSVD. If not included, SVD
            will not run for your experiment.
            ---

            "save": ("log", model")                # include whichever

            Save the output of the log, or dump the entire model with its
            classification method and pipeline wrapper for new data instances.
            In development: save the features for a certain setting.
            ---
        }
    }

    Parameters
    ----------
    conf :
        Dictionary where key is a setting parameter and value the value. The
        options are pretty broad so they are explained in the class docstring.

    cold : boolean, optional, default False
        If true, will not immediately run the experiment after calling the
        class. Generally we assume that one immediately wants to run on call.
    """

    def __init__(self, conf, cold=False):
        """Set all relevant classes, run experiment (currently)."""
        self.conf = conf
        self.log = Logger(conf['name'])
        self.vec = Vectorizer(conf)
        self.opt = Optimizer(conf)
        self.clf = None
        if not cold:
            self.run(conf)

    def save(self):
        """Save desired Experiment data."""
        if self.conf.get('save'):
            if 'log' in self.conf['save']:
                self.log.save()
            if 'features' in self.conf['save']:
                self.log.echo(" Feature saving has not been implemented yet!")
            if 'model' in self.conf['save']:
                Pipeline(self).save()

    def run(self, conf):
        """Split data, fit, transfrom features, tf*idf, svd, report."""
        t1 = time()
        seed = 42
        np.random.RandomState(seed)

        # report features
        self.log.post('head', ('\n'.join([str(c) for c in conf['features']]),
                               conf['name'], seed))

        X, y = self.vec.fit_transform(conf['train_data'])
        self.log.loop('sparse', ('train', X.shape))

        X, y, self.clf = self.opt.choose_classifier(X, y, seed)
        print("\n Training model...")
        self.clf.fit(X, y)
        print(" done!")

        # report performance
        if not conf.get('test_data'):
            res = cross_val_predict(self.clf, X, y, cv=5, n_jobs=-1)
            self.log.post('cr', (metrics.classification_report(y, res),))
        else:
            print("\n Fetching test data...")
            Xi, yi = self.vec.transform(conf['test_data'])
            self.log.loop('sparse', ('test', Xi.shape))
            self.log.dump('sparse')
            print(" done!")

            res = self.clf.predict(Xi)
            yi = list(yi)
            self.log.post('cr', (metrics.classification_report(yi, res),))

        self.save()
        t2 = time()
        print("\n Experiment took {0} seconds".format(round(t2-t1, 1)))
        print("\n" + '-'*10, "\n")
