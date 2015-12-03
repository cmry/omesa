"""Main experiment code."""

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
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
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
            'svd':    "\n Fitting SVD...",
            'rep':    "\n\n---- {0} Results ---- \n" +
                      "\n Distribution: {1}" +
                      "\n Accuracy @ baseline: \t {2}" +
                      "\n Reporting on class {3}",
            'grid':   "\n Model with rank: {0} " +
                      "\n Mean validation score: {1:.3f} (std: {2:.3f}) " +
                      "\n Parameters: {3} \n",
            'tfcv':   "\n Tf-CV Result: {0}",
            'f1sc':   "\n Performance on test set: \n{0}"
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
            o = ['head', 'sparse', 'svd', 'rep', 'grid', 'tfcv', 'f1sc']
            f.write(' '.join([v for v in OrderedDict(sorted(self.log.items(),
                              key=lambda i: o.index(i[0]))).values()]))

class Pipeline(object):

    def __init__(self, conf):
        """Start pipeline modules."""
        backbone = conf.get('backbone', 'fallback')
        self.shed = env.Environment(conf['name'], backbone=backbone)
        self.handle = LabelHandler(conf.get('label_selection'))
        self.hasher = DictVectorizer()
        self.tfidf = TfidfTransformer()
        self.svd = TruncatedSVD(n_components=conf.get('components'))
        self.conf = conf

    def load_csv(self, data):
        """Iterates through csv files."""
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
            if label and x[i_text]:
                yield (label, x[i_text], ann, feats)
        self.handle = LabelHandler(conf.get('label_selection'))

    def train(self, data, features):
        """Send the training data through all applicable steps."""
        D, y = self.shed.fit_transform([self.load_data(data),
                                        self.load_data(data)], features)
        X = self.hasher.fit_transform(D)
        X_tf = self.tfidf.fit_transform(X)

        if self.conf.get('components'):
            X_tf = self.svd.fit_transform(X_tf, y)

        return X_tf, y

    def test(self, data):
        """Send the test data through all applicable steps."""
        # same steps as pipe_train
        Di, yi = self.shed.transform(self.load_data(data))
        Xi = self.hasher.transform(Di)
        Xi_tf = self.tfidf.transform(Xi)

        if self.conf.get('components'):
            Xi_tf = self.svd.transform(Xi_tf)

        return Xi_tf, yi


class MiniModel(object):

    def __init__(self, pipeline, clf):
        self.pipeline = pipeline
        self.clf = clf

    def classify(self, data):
        self.pipeline.conf['label_column'] = 0
        self.pipeline.conf['text_column'] = 1
        self.pipeline.handle.labs = None
        v, y = self.pipeline.test(data)
        enc = dict(map(reversed, self.pipeline.shed.featurizer.labels.items()))
        return [enc[l] for l in self.clf.predict(v)], self.clf.predict_proba(v)


class Experiment(object):

    """
    Full experiment wrapper.

    Calls several sklearn modules in the Pipeline class and reports on the
    classifier performance.
    """

    def __init__(self, conf):
        """Set all relevant classes, run experiment (currently)."""
        self.conf = conf
        self.log = Log(conf['name'])
        self.pipe = Pipeline(conf)
        self.experiment(conf)

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

    def choose_classifier(self, seed):
        """Chooses a classifier based on settings."""
        if self.conf.get('setting') == 'grid':
            param_grid = {'svc__kernel': ['rbf', 'linear'],
                          'svc__gamma': [1e-2, 1e-3, 1e-4],
                          'svc__C': [1, 10, 100, 1000, 2000]}
            pipe = pipeline.Pipeline([
                ('svc', SVC(random_state=seed, cache_size=80000))
            ])
            clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1)
        elif not self.conf.get('classifier'):
            clf = SVC(random_state=seed, gamma=1e-2, kernel='linear', C=1,
                      cache_size=150000, probability=True)
        else:
            clf.probability = True
            clf.random_state = seed

        return clf

    def save(self, clf):
        """Saves desired Experiment data."""
        if self.conf.get('save'):
            if 'log' in self.conf['save']:
                self.log.save()
            if 'features' in self.conf['save']:
                self.log.echo(" Feature saving has not been implemented yet!")
            if 'model' in self.conf['save']:
                self.pipe.shed.save()
            if 'mini' in self.conf['save']:
                pickle.dump(MiniModel(self.pipe, clf),
                            open(self.conf['name'] + '.pickle', 'wb'))

    def experiment(self, conf):
        """Split data, fit, transfrom features, tf*idf, svd, report."""
        seed = 666
        np.random.RandomState(seed)

        # report features
        self.log.post('head', ('\n'.join([str(c) for c in conf['features']]),
                               conf['name'], seed))

        X, y = self.pipe.train(conf['train_data'], conf['features'])
        self.log.loop('sparse', ('train', X.shape))

        if conf.get('test_data'):
            Xi, yi = self.pipe.test(conf['test_data'])
            self.log.loop('sparse', ('test', Xi.shape))
            self.log.dump('sparse')

        clf = self.choose_classifier(seed)
        clf.fit(X, y)

        # report performance
        if conf.get('setting') == 'grid':
            self.grid_report(clf.grid_scores_)
        elif not conf.get('test_data'):
            f1_scorer = metrics.make_scorer(metrics.f1_score)
            score = np.average(cross_val_score(clf, X, y, cv=10,
                                               scoring=f1_scorer,
                                               n_jobs=-1))
            self.log.post('tfcv', (score,))
        else:
            res = clf.predict(Xi)
            self.log.post('f1sc', metrics.classification_report(yi, res))

        self.save(clf)
