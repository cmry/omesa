"""Data handling containers.
"""

import csv
import json
import pickle
import sys
from types import GeneratorType
from os import getcwd

from .tools import serialize_sk as sr
from .tools import lime_eval as le

try:
    from .database import Database, Configuration, Vectorizer, \
                          Classifier, Results, Table
except ImportError as e:
    print("Database could not be loaded, functionality disabled.")


class Pipeline(object):
    """Shell for experiment pipeline storing and handling.

    Parameters
    ----------
    exp : class, optional, default None
        Instance of Experimen with fitted pipes. If not supplied, name and
        source should be set.

    name : str, optional, default None
        Name that the pipeline should be saved/loaded under/from.

    out : tuple, optional, default None
        Tuple with storage options, can be "json" (json serialization),
        or "db" (for database storage, requires blitzdb).
    """

    def __init__(self, exp=None, name=None, out=None):
        """Set the pipeline for transformation and clf for classification."""
        if not exp:
            assert name

        self.vec = exp.vec if exp else None
        self.cnf = exp.vec.conf if exp else None
        self.clf = exp.clf if exp else None
        self.res = exp.res if exp else None

        self.hook = self.vec.conf['name'] if not name else name
        self.storage = self.vec.conf['save'] if not out else out

        if 'db' in self.storage:
            self.db = Database()

    def _make_tab(self):
        """Tabular level experiment representation.

        Generates a table-level representation of an experiment. This stores
        JSON native information ONLY, and is used for the experiment table in
        the front-end, as deserializing a lot of experiments will be expensive
        in terms of loading times.
        """
        tab = {'project': self.cnf.get('project', '-'), 'name': self.hook,
               'clf': self.clf.__dict__['steps'][0][1].__class__.__name__,
               'clf_full': str(self.clf.__dict__['steps'][0][1])}

        for n in ('train', 'test', 'lime'):
            try:
                tab.update({n + '_data': self.cnf[n + '_data'].source})
            except Exception as e:
                tag = 'split' if n == 'test' else self.cnf[n + '_data']
            try:
                tab.update({n + '_data_path': self.cnf[n + '_data'].path,
                            n + '_data_repr': self.cnf[n + '_data'].__dict__})
            except Exception as e:
                if tag is not 'split':
                    tag = '-'
                tab.update({n + '_data_path': tag})
                tab.update({n + '_data_repr': tag})

        if not self.cnf.get('lime_protect') and tab.get('lime_data'):
            # FIXME: replace with multi-labelled case
            labs = self.vec.encoder.inverse_transform([0, 1])
            lime = le.LimeEval(self.clf, self.vec, labs)
            exps = lime.load_omesa(tab['lime_data_repr'])
            lime_list = []
            for exp in exps:
                expl, prb, cln = lime.unwind(exp)
                lime_list.append({'expl': expl, 'prb': prb, 'cln': cln})
            tab.update({'lime_data_comp': lime_list})

        tab.update({'features': ','.join([x.__str__() for x in
                                          self.vec.featurizer.helpers]),
                    'test_score': self.res['test']['score'],
                    'dur': self.res['dur'],
                    })
        return tab

    def save(self):
        """Save experiment and classifier in format specified."""
        print(" Saving experiment...")
        tab = self._make_tab()
        fl = self.hook

        try:  # purge double vec instance
            self.vec.conf = self.hook
        except AttributeError:  # when loading
            pass

        if any([x in self.storage for x in ('json', 'pickle')]):
            top = {'name': self.hook, 'cnf': self.cnf, 'vec': self.vec,
                   'clf': self.clf, 'res': self.res, 'tab': tab}
        if 'json' in self.storage:
            serialized = sr.encode(top)
            json.dump(serialized, open(self.hook + '.json', 'w'))
        if 'pickle' in self.storage:
            for t in ('train', 'test', 'lime'):
                c = top['conf']['{0}_data'].format(t)
                c = '' if isinstance(c, GeneratorType) else c
            pickle.dump(top, open(fl + '.pickle', 'wb'))
        if 'db' in self.storage:
            top = {Configuration: self.cnf, Vectorizer: self.vec,
                   Classifier: self.clf, Results: self.res, Table: tab}
            for doc, bind in top.items():
                js = json.loads(sr.encode(bind))
                js['name'] = self.hook
                self.db.save(doc(js))

    def load(self):
        """Load experiment and classifier from source specified."""
        if any([x in self.storage for x in ('man', 'json')]):
            mod = sr.decode(json.load(open(self.hook + '.json')))
        if 'pickle' in self.storage:
            mod = pickle.load(open(self.hook + '.pickle', 'rb'))
        if 'db' in self.storage:
            mod = {'clf': Classifier, 'vec': Vectorizer, 'res': Results,
                   'cnf': Configuration}
            for k, doc in mod.items():
                mod[k] = sr.decode(json.dumps(dict(
                    self.db.fetch(doc, {'name': self.hook}))))
        self.clf = mod['clf']
        self.vec = mod['vec']
        if 'db' in self.storage:
            self.vec.conf = mod['cnf']
        self.res = mod['res']

    def classify(self, data):
        """Given a data point, return a (label, probability) tuple."""
        X = self.vec.transform(data)
        # X = X.todense().reshape((1, -1))
        # FIXME: some clfs like LinearSVC have no predict proba
        return self.clf.predict(X), self.clf.predict_proba(X)


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
        self.path = getcwd() + ("/" if not csv_dir.startswith('/') else '') + \
            csv_dir
        self.file = csv.reader(open(csv_dir, 'r'))
        self.header = header

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
