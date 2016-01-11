# -*- coding: utf-8 -*-

"""Experiment wrapper code."""

# pylint:       disable=E1103,E1101,E0611,C0103,C0325,C0330,W0141

from . import environment as env

import csv
import numpy as np
import pickle
import sys
from collections import Counter, OrderedDict
from copy import deepcopy
from operator import itemgetter
from sklearn import metrics
from sklearn import pipeline
from sklearn.utils import shuffle
from sklearn.cross_validation import cross_val_predict, train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import SVC, LinearSVC
from time import time


class LabelHandler(object):

    """
    Tiny label handler class.

    Accepts a dictionary of label names as keys and a [count, conversion] list
    as value. `Count` is the number that has to be grouped under the
    `conversion` label.

    Parameters
    ----------
    labs: dict
        Keys are original labels, value is a list of [count, conversion].

    Examples
    --------
    In [1]: d = {'Label1': [1, 'Label2'],
    ...          'Label3': ['Label4']
    ...     }

    In [2]: [LabelHandler(d).check(x) for x in ('Label1', 'Label1', 'Label3')]
    Out[2]: ['Label2', 'Label4']
    """

    def __init__(self, labs):
        """Copy the original counts to keep them intact across train / test."""
        self.labs = deepcopy(labs)

    def check(self, label):
        """Check if label has count, otherwise return label. If zero, none."""
        if self.labs.get(label) and self.labs[label][0]:
            self.labs[label][0] = self.labs[label][0] - 1
            if len(self.labs[label]) > 1:
                return self.labs[label][1]
            else:
                return label
        if not any([x[0] for x in self.labs.values()]):
            return 'break'


class Log(object):

    """
    Provides feedback to the user and can store settings in a log file.

    Class holds a log string that can be formatted according to the used
    components and is used to list settings that are provided to the
    experiment. Aside from flat printing, it can iteratively store certain
    lines that are reused (such as loading in multiple datasets). The save
    function makes sure the self.log items are saved according to their logical
    order.

    Parameters
    ----------
    fn: str
        File name of the logfile (and the experiment).

    Attributes
    ----------
    fn: str
        File name.

    log: dict
        Keys are short names for each step, values are strings with .format
        placeholders. Formatting is sort of handled by the strings as well.

    buffer: list
        Used to stack lines in a loop that can be written to the log line once
        the loop has been completed.
    """

    def __init__(self, fn):
        """Set log dict. Empty buffer."""
        self.fn = fn + '.log'
        self.log = {
            'head':   "\n---- Shed ---- \n\n Config: \n" +
                      "\t {0} \n\tname: {1} \n\tseed: {2} " +
                      "\n\t \n",
            # 'read':   "\n Reading from {0}... Acquired {1} from data.\n ",
            'sparse': "\n Sparse {0} shape: {1}",
            'svd':    "\n Fitting SVD with {0} components...",
            'rep':    "\n\n---- {0} Results ---- \n" +
                      "\n Distribution: {1}" +
                      "\n Accuracy @ baseline: \t {2}" +
                      "\n Reporting on class {3}",
            'grid':   "\n Model with rank: {0} " +
                      "\n Mean validation score: {1:.3f} (std: {2:.3f}) " +
                      "\n Parameters: {3} \n",
            'tfcv':   "\n Tf-CV Result: {0}",
            'f1sc':   "\n F1 Result: {0}",
            'cr':     "\n Performance on test set: \n\n{0}"
        }
        self.buffer = []

    def echo(self, *args):
        """Replacement for a print statement. Legacy function."""
        message = ' '.join([str(x) for x in args])
        print(message)

    def loop(self, k, v):
        """Print and store line to buffer."""
        line = self.log[k].format(*v)
        print(line)
        self.buffer.append(line)

    def dump(self, k):
        """Dump buffer to log."""
        self.log[k] = ''.join(self.buffer)
        self.buffer = []

    def post(self, k, v):
        """Print and store line to log."""
        line = self.log[k].format(*v)
        print(line)
        self.log[k] = line

    def save(self):
        """Save log."""
        with open(self.fn, 'w') as f:
            o = ['head', 'sparse', 'svd', 'rep', 'grid', 'tfcv', 'f1sc', 'cr']
            f.write(' '.join([v for v in OrderedDict(sorted(self.log.items(),
                              key=lambda i: o.index(i[0]))).values()]))


class Pipeline(object):

    """
    Small text mining research pipeline.

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
        Environment class (might be replace by Featurizer) from shed.

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
        backbone = conf.get('backbone', 'fallback')
        self.featurizer = env.Environment(conf['name'], backbone=backbone)
        self.handle = LabelHandler(conf.get('label_selection'))
        self.hasher = DictVectorizer()
        self.tfidf = TfidfTransformer(sublinear_tf=True)
        self.svd = None
        self.conf = conf

    def load_csv(self, data):
        """Iterate through csv files."""
        csv.field_size_limit(sys.maxsize)
        for d in data:
            reader = csv.reader(open(d, 'r'))
            for i, x in enumerate(reader):
                if self.conf.get("has_header") and not i:
                    continue
                yield x

    def load_data(self, data):
        """Load from given datasets provided amount of instances."""
        conf = self.conf
        i_text, i_label = conf['text_column'], conf['label_column']
        i_ann, i_feats = conf.get('ann_column'), conf.get('feature_columns')

        # so that data can also be an iterable
        loader = self.load_csv(data) if data[0][-4:] == '.csv' else data
        for x in loader:
            label = self.handle.check(x[i_label]) if self.handle.labs \
                else x[i_label]
            if label == 'break':
                break
            ann, feats = ('' if not v else x[v] for v in [i_ann, i_feats])
            ann = [x.split('\t') for x in ann.split('\n')]
            if label and x[i_text]:
                yield (label, x[i_text], ann, feats)
        self.handle = LabelHandler(conf.get('label_selection'))

    def train(self, data, features):
        """Send the training data through all applicable steps."""
        D, y = self.featurizer.transform(self.load_data(data), features)
        print("\n Hashing features...")
        X = self.hasher.fit_transform(D)
        print(" done!")

        if 'tfidf' in self.conf.get('settings'):
            print("\n Tf*idf transformation...")
            X = self.tfidf.fit_transform(X)
            print(" done!")
        else:
            X = MaxAbsScaler(copy=False).fit_transform(X)

        return X, y

    def test(self, data):
        """Send the test data through all applicable steps."""
        # same steps as pipe_train
        Di, yi = self.featurizer.transform(self.load_data(data))
        Xi = self.hasher.transform(Di)

        if 'tfidf' in self.conf.get('settings'):
            Xi = self.tfidf.transform(Xi, copy=False)
        else:
            Xi = MaxAbsScaler(copy=False).fit_transform(Xi)

        if 'svd' in self.conf.get('settings'):
            Xi = self.svd.transform(Xi)

        return Xi, yi


class Model(object):

    """
    Shell for experiment model storing and handling.

    Parameters
    ----------
    pipe : class
        Instance of Pipeline with fitted models.

    clf : class
        Classifier that adheres to the sklearn type (with a predict function).
    """

    def __init__(self, pipe, clf):
        """Set the pipeline for transformation and clf for classification."""
        self.pipeline = pipe
        self.clf = clf

    def classify(self, data):
        """Given a data iterator, return a (label, probability) tuple."""
        self.pipeline.conf['label_column'] = 0
        self.pipeline.conf['text_column'] = 1
        self.pipeline.handle.labs = None
        v, _ = self.pipeline.test(data)
        enc = dict(map(reversed, self.pipeline.shed.featurizer.labels.items()))
        return [enc[l] for l in self.clf.predict(v)], self.clf.predict_proba(v)


class Experiment(object):

    """
    Full experiment wrapper.

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
            features and shed environment under one name.
            ---

            "train_data": ["/somedir/train.csv", "/somedir/train2.csv"],

            The data on which the experiment will train. If the location of a
            .csv is provided, it will open these up and create an iterator for
            you. Alternatively, you can provide your own iterators or iterable
            structures providing instances of the data. If only training data
            is provided, the experiment will evaluate in a tenfold setting by
            default.
            ---

            "test_data": ["/somedir/test.csv"],    # either

            This works similar to the train_data. However, when a test set is
            provided, the performance of the model will be measured on this
            test data only. Shed will dump a classification report for you.
            ---

            "test_proportion": 0.3,                # or

            As opposed to a test FILE, one can also provide a test proportion,
            after which a certain amount of instances will be held out from the
            training data to test on.
            ---

            "features": [Ngrams()],

            These can be features imported from shed.featurizer or can be
            any class you create your self. As long as it adheres to a fit /
            transform structure and returns a feature dictionary per instance
            as can be provided to, for example, the sklearn FeatureHasher.
            ---

            "text_column": 2,                      # index

            Index integer of the .csv or iterator where the text is located.
            ---

            "label_column": 0,                     # index

            Index integer of the .csv or iterator where the label is located.
            ---

            "ann_column": 3,                       # list (token, lemma, POS)

            Index integer of the .csv or iterator where the annotations are
            provided. Currently it assumes that these are per instance a list
            of, for every word, (token, lemma, POS). Frog and spaCy are
            implemented to provide these for you.
            ---

            "feature_columns": [1],                # if .csv contains features

            If you have columns in your .csv or iterator that should serve as
            features (meta-data) for example, you can add a multitude of their
            indices in this setting.
            ---

            "label_selection": {                   # optional
                'Label1': (100, 'GroupA'),
                'Label2': (100, 'GroupB'),
                'Label3': (-1,  'GroupC'),
                'Label4': (100)
            },

            A dict of {label: (amount, new_label)} pairs. With this, you can
            somewhat control how many of a certain label you wish to include
            in loading your data (due to memory constraints for example). If
            you want all, you can just put -1. The new_label part of the tuple
            can be used in the rare case you might want to convert the labels
            at load time. This is pure convenience, you should have probably
            already done this before providing the data, though.
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
        self.log = Log(conf['name'])
        self.pipe = Pipeline(conf)
        if not cold:
            self.run(conf)

    def report(self, t, y):
        """Report baseline, and label distribution."""
        maj_class = Counter(y).most_common(1)[0][0]
        baseline = [maj_class for _ in y]
        dist = Counter(y).most_common(10)
        self.log.post('rep', (
            t, dist, round(metrics.accuracy_score(y, baseline), 3),
            dist[1][0]))
        return dist[1][0]

    def grid_report(self, grid_scores, n_top=1):
        """Post gridsearch report."""
        top_scores = sorted(grid_scores, key=itemgetter(1),
                            reverse=True)[:n_top]
        for i, score in enumerate(top_scores):
            self.log.loop('grid', (i + 1, score.mean_validation_score,
                                   np.std(score.cv_validation_scores),
                                   score.parameters))
        self.log.dump('grid')
        return top_scores[0].parameters

    def choose_classifier(self, X, y, seed):
        """Choose a classifier based on settings."""
        conf = self.conf

        # split dev (test_set here is X)
        if conf.get('settings'):
            X_dev, X, y_dev, y = train_test_split(
                X, y, test_size=0.8, random_state=666)

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

        if 'grid' in conf.get('settings'):

            param_grid = {'svc__kernel': ['rbf', 'linear'],
                          # 'svc__gamma': [1e-1, 1e-2, 1e-3, 1e-4],
                          'svc__gamma': np.logspace(-3, 2, 3),
                          'svc__C': np.logspace(0.1, 3, 6)}
            steps = [('svc', SVC(random_state=seed, cache_size=80000))]

            # incorporate SVD into GridSearch
            if 'svd' in conf.get('settings') and not conf.get('components'):
                param_grid.update({'svd__n_components': [50, 100, 500, 1000]})
                steps = [('svd', TruncatedSVD())] + steps

            pipe = pipeline.Pipeline(steps)
            clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)

            print("\n Starting Grid Search...")
            clf.fit(X_dev, y_dev)
            print(" done!")

            p = self.grid_report(clf.grid_scores_)
            clf = SVC(random_state=seed, gamma=p['svc__gamma'],
                      kernel=p['svc__kernel'], C=p['svc__C'],
                      cache_size=150000, probability=True)

            if 'svd' in conf.get('settings') and not conf.get('components'):
                comp = p['svd__n_components']
                self.pipe.svd = TruncatedSVD(n_components=comp)

            if 'svd' in conf.get('settings'):
                X = self.pipe.svd.transform(X)

            return X, y, clf

        elif not conf.get('classifier'):
            clf = LinearSVC(random_state=seed, C=10)
        elif conf.get('classifier'):
            clf = conf['classifier']
            clf.random_state = seed
        else:
            clf.probability = True
            clf.random_state = seed

        return X, y, clf

    def save(self, clf):
        """Save desired Experiment data."""
        if self.conf.get('save'):
            if 'log' in self.conf['save']:
                self.log.save()
            if 'features' in self.conf['save']:
                self.log.echo(" Feature saving has not been implemented yet!")
            if 'model' in self.conf['save']:
                pickle.dump(Model(self.pipe, clf),
                            open(self.conf['name'] + '.pickle', 'wb'))

    def run(self, conf):
        """Split data, fit, transfrom features, tf*idf, svd, report."""
        t1 = time()
        seed = 666
        np.random.RandomState(seed)

        # report features
        self.log.post('head', ('\n'.join([str(c) for c in conf['features']]),
                               conf['name'], seed))

        X, y = self.pipe.train(conf['train_data'], conf['features'])
        X, y = shuffle(X, y, random_state=666)
        self.log.loop('sparse', ('train', X.shape))

        X, y, clf = self.choose_classifier(X, y, seed)
        print("\n Training model...")
        clf.fit(X, y)
        print(" done!")

        # report performance
        if not conf.get('test_data'):
            res = cross_val_predict(clf, X, y, cv=10, n_jobs=-1)
            self.log.post('cr', (metrics.classification_report(y, res),))
        else:
            print("\n Fetching test data...")
            Xi, yi = self.pipe.test(conf['test_data'])
            self.log.loop('sparse', ('test', Xi.shape))
            self.log.dump('sparse')
            print(" done!")

            res = clf.predict(Xi)
            yi = list(yi)
            self.log.post('cr', (metrics.classification_report(yi, res),))

        self.save(clf)
        t2 = time()
        print("\n Experiment took {0} seconds".format(round(t2-t1, 1)))
        print("\n" + '-'*10, "\n")
