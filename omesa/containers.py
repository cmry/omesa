"""Data handling containers."""

import csv
import json
import pickle
import sys
from types import GeneratorType

from .tools import serialize_sk as sr


class Pipeline(object):
    """Shell for experiment pipeline storing and handling.

    Parameters
    ----------
    exp : class, optional, default None
        Instance of Experimen with fitted pipes. If not supplied, name and
        source should be set.

    name : str, optional, default None
        Name that the pipeline should be saved/loaded under/from.

    source : tuple, optional, default None
        Tuple with storage options, can be "man" (manual json serialization),
        "json" (for jsonpickle, requires this package), "db" (for database
        storage, requires blitzdb).
    """

    def __init__(self, exp=None, name=None, source=None):
        """Set the pipeline for transformation and clf for classification."""
        if not exp:
            assert name
        self.vec = exp.vec if exp else None
        self.clf = exp.clf if exp else None
        self.res = exp.res if exp else None
        self.hook = self.vec.conf['name'] if not name else name
        self.serialize = None
        self.storage = self.vec.conf['save'] if not source else source
        if 'db' in self.storage:
            from .database import Database, Experiment
            self.db = Database()
            self.data = Experiment
        if 'json' in self.storage:
            import jsonpickle
            self.serialize = jsonpickle
        # FIXME: jsonpickle should be preferred, doesn't currently work though
        elif 'man' in self.storage or 'db' in self.storage:
            self.serialize = sr
            # self.hook += '_man'

    def _make_top(self):
        """Top level experiment representation.

        Generates a top-level representation of an experiment. This stores
        JSON native information ONLY, and is used for the experiment table in
        the front-end, as deserializing a lot of experiments will be expensive
        in terms of loading times."""
        # TODO: check if this can't be handled in front-end
        top = {'name': self.hook, 'vec': self.vec, 'clf': self.clf,
               'clf_name': self.clf.__dict__['steps'][0][1].__class__.__name__,
               'project': self.vec.conf.get('project', '-')}
        for n in ('train', 'test'):
            try:
                top.update({n + '_data':
                            self.vec.__dict__['conf'][n + '_data'].source})
            except Exception as e:
                top.update({n + '_data': 'split'})
        top.update({'features': ','.join([x.__str__() for x in
                                          self.vec.featurizer.helpers]),
                    'res': self.res,
                    'test_score': self.res['test']['score'],
                    'dur': self.res['dur']})
        return top

    def save(self):
        """Save experiment and classifier in format specified."""
        print(" Saving experiment...")
        top = self._make_top()

        fl = self.hook
        if self.serialize:
            serialized = self.serialize.encode(top)

        if any([x in self.storage for x in ('man', 'json')]) and serialized:
            json.dump(serialized, open(self.hook + '.json', 'w'))
        if 'pickle' in self.storage:
            for t in ('train', 'test'):
                c = top['conf']['{0}_data'].format(t)
                c = '' if isinstance(c, GeneratorType) else c
            pickle.dump(top, open(fl + '.pickle', 'wb'))
        if 'db' in self.storage:
            doc = self.data(json.loads(serialized))
            self.db.save(doc)

    def load(self):
        """Load experiment and classifier from source specified."""
        if any([x in self.storage for x in ('man', 'json')]):
            mod = self.serialize.decode(json.load(open(self.hook + '.json')))
        if 'pickle' in self.storage:
            mod = pickle.load(open(self.hook + '.pickle', 'rb'))
        if 'db' in self.storage:
            mod = self.db.fetch(self.data, {'name': self.hook})
            mod = self.serialize.decode(json.dumps(dict(mod)))
        self.clf = mod['clf']
        self.vec = mod['vec']
        self.res = mod['res']

    def classify(self, data):
        """Given a data point, return a (label, probability) tuple."""
        X, _ = self.vec.transform(data)
        X = X.todense().reshape((1, -1))
        # LinearSVC no predict proba?
        return self.clf.predict(X)  # , self.clf.predict_proba(X)


class CSV:
    """Quick and dirty csv loader.

    Parameters
    ----------
    text : integer
        Index integer of the .csv where the text is located.

    label : integer
        Index integer of the .csv where the label is located.

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

    def __init__(self, csv_dir, data, label, parse=None, features=None,
                 header=False, selection=None):
        """Set configuration and label handler."""
        csv.field_size_limit(sys.maxsize)
        self.source = csv_dir
        self.file = csv.reader(open(csv_dir, 'r'))

        if header:
            self.file.__next__()
        if isinstance(features, (int, type(None))):
            features = [features]

        self.idx = list(filter(None.__ne__, [data, label, parse] + features))
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
