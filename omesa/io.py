"""Data handling functions."""

import csv
import sys
import json
import numpy as np
from inspect import isclass, isgenerator
from .tools import serialize_sk as sr

# pylint:       disable=R0903,R0913,W0141


class Pipeline(object):
    """Shell for experiment pipeline storing and handling.

    Parameters
    ----------
    vec : class
        Instance of Vectorizer with fitted pipes.

    clf : class
        Classifier that adheres to the sklearn type (with a predict function).
    """

    def __init__(self, exp):
        """Set the pipeline for transformation and clf for classification."""
        self.vec = exp.vec
        self.clf = exp.clf

    def save(self):
        """bla."""
        print(" Saving experiment...")
        top = self.vec.__dict__
        print(top)
        ser = sr.data_to_json(top)
        print(" done!")
        if 'db' in top['conf']['save']:
            pass
        else:
            json.dump(ser, open(top['conf']['name'] + '.json', 'w'))

    def load(self):
        pass

    def classify(self, data):
        """Given a data iterator, return a (label, probability) tuple."""
        self.pipeline.conf['label_column'] = 0
        self.pipeline.conf['text_column'] = 1
        # self.pipeline.loader.handle.labs = None
        v, _ = self.pipeline.test(data)
        # FIXME: this is like a java call
        enc = dict(map(reversed, self.pipeline.featurizer.labels.items()))
        return [enc[l] for l in self.clf.predict(v)], self.clf.predict_proba(v)


class CSV:
    """Quick and dirty csv loader.

    Parameters
    ----------
    label : integer
        Index integer of the .csv where the label is located.

    text : integer
        Index integer of the .csv where the text is located.

    parse : integer, optional, default None
        Index integer of the .csv where the annotations are provided. Currently
        it assumes that these are per instance a list of, for every word,
        (token, lemma, POS). Frog and spaCy are implemented to provide these
        for you.

    features : list of integers, optional, default None
        If you have columns in your .csv othat should serve as features
        (meta-data) for example, you can add a multitude of their indices in
        this setting.

    header : boolean, optional, default False
        If the file has a header, you can skip it by setting this to true.

    selection : dict, optional, default None
        A dict of {label: integer amount} pairs. With this, you can somewhat
        control how many of a certain label you wish to include in loading your
        data (due to memory constraints for example). If you want all, you can
        just put -1.
    """

    def __init__(self, csv_dir, label, data, parse=None, features=None,
                 header=False, selection=None):
        """Set configuration and label handler."""
        csv.field_size_limit(sys.maxsize)
        self.file = csv.reader(open(csv_dir, 'r'))

        if header:
            self.file.__next__()
        if isinstance(features, (int, type(None))):
            features = [features]

        self.idx = list(filter(None.__ne__, [label, data, parse] + features))
        self.selection = {} if not selection else selection

    def __iter__(self):
        """Standard iter method."""
        return self

    def __next__(self):
        """Iterate through csv file."""
        row = self.file.__next__()
        if self.selection.get(self.idx[0]):
            if self.selection[self.idx[0]]:
                self.selection[self.idx[0]] -= 1
                return tuple(row[i] for i in self.idx)
        else:
            return tuple(row[i] for i in self.idx)
