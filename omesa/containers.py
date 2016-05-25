"""Data handling containers.
"""

# pylint:       disable=W0234

import csv
import json
import pickle
import sys
from types import GeneratorType
from os import getcwd

from .tools import serialize_sk as sr

try:
    from .database import Database, Configuration, Vectorizer, \
                          Classifier, Results, Table
except ImportError as e:
    print(e)
    print("Database could not be loaded, functionality disabled.")


class Pipeline(object):
    """Shell for experiment pipeline storing and handling.

    Parameters
    ----------
    exp : class, optional, default None
        Instance of Experimen with fitted pipes. If not supplied, name and
        store should be set.

    name : str, optional, default None
        Name that the pipeline should be saved/loaded under/from.

    store : tuple, optional, default None
        Tuple with storage options, can be "json" (json serialization),
        or "db" (for database storage, requires blitzdb).
    """

    def __init__(self, exp=None, name=None, store=None):
        """Set the pipeline for transformation and clf for classification."""
        if not exp:
            assert name

        self.vec = exp.vec if exp else None
        self.cnf = exp.vec.conf if exp else None
        self.clf = exp.clf if exp else None
        self.res = exp.res if exp else None

        self.hook = self.vec.conf['name'] if not name else name
        self.storage = self.vec.conf['save'] if not store else store

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
            except (AttributeError, KeyError, TypeError):
                tag = 'split' if n == 'test' else self.cnf.get(n + '_data', [])
            try:
                tab.update({n + '_data': self.cnf[n + '_data'].source,
                            n + '_data_path': self.cnf[n + '_data'].path,
                            n + '_data_repr': self.cnf[n + '_data'].__dict__})
            except (AttributeError, KeyError, TypeError):
                tab.update({n + '_data': tag,
                            n + '_data_path': tag,
                            n + '_data_repr': tag})

        if not self.cnf.get('lime_protect') and tab.get('lime_data'):
            from .tools import lime_eval as le
            labs = self.vec.encoder.classes_
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
                    'dur': self.res['dur']})
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
            sr.encode(top, open(self.hook + '.json', 'w'))
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
        if 'json' in self.storage:
            # FIXME: try to recursively solve imports if that works
            mod = sr.decode(open(self.hook + '.json'))
        if 'pickle' in self.storage:
            mod = pickle.load(open(self.hook + '.pickle', 'rb'))
        if 'db' in self.storage:
            mod = {'clf': Classifier, 'vec': Vectorizer, 'res': Results,
                   'cnf': Configuration}
            for k, doc in mod.items():
                mod[k] = self.db.get_component(doc, self.hook)
        self.clf = mod['clf']
        self.vec = mod['vec']
        if 'db' in self.storage:
            self.vec.conf = mod['cnf']
        self.res = mod['res']

    def classify(self, data, best_only=False):
        """Given instance(s) return list with (label, probabilities).

        Parameters
        ----------
        data : value or list
            Can be one or multiple data instances (strings for example).

        best_only : bool, optional, default False
            If set to True, returns probabilties for highest only.

        Returns
        -------
        self.clf.predict : list
            List with one or more tuples with (label, array of probabilities).
        """
        if isinstance(data, str):
            data = [data]
        X = self.vec.transform(data)
        try:
            prob_d = [{self.vec.encoder.inverse_transform(i): p
                       for i, p in enumerate(pl)}
                      for pl in self.clf.predict_proba(X)]
            if best_only:
                return sorted(prob_d.items(), key=lambda x: x[1])[-1]
            return prob_d
        except AttributeError:
            pass
        return self.vec.encoder.inverse_transform(self.clf.predict(X))


class CSV(object):
    """Quick and dirty csv loader.

    Parameters
    ----------
    text : integer or string
        Index integer of the .csv where the text is located. If string is
        provided instead, will look for its index in the header.

    label : integer or string
        Index integer of the .csv where the label is located. If string is
        provided instead, will look for its index in the header.

    parse : integer, optional, default None
        Index integer of the .csv where the annotations are provided. If string
        is provided instead, will look for its index in the header. Currently
        it assumes that these are per instance a list of, for every word,
        (token, lemma, POS). Frog and spaCy are implemented to provide these
        for you.

    features : list of integers or strings, optional, default None
        If you have columns in your .csv that should serve as features
        (meta-data) for example, you can add a multitude of their indices in
        this setting in integer format. If string is provided instead, will
        look for its index in the header.

    no_header : boolean, optional, default False
        If the file has no header, and integer values are provided as column
        indices, set to True if it has to be included.

    selection : dict, optional, default None
        A dict of {label: integer amount} pairs. With this, you can somewhat
        control how many of a certain label you wish to include in loading your
        data (due to memory constraints for example). If you want all, you can
        just put -1.
    """

    def __init__(self, csv_dir, data, label, parse=None, features=None,
                 no_header=False, selection=None):
        """Set configuration and label handler."""
        csv.field_size_limit(sys.maxsize)
        self.source = csv_dir
        self.path = getcwd() + ("/" if not csv_dir.startswith('/') else '') + \
            csv_dir
        self.file = csv.reader(open(csv_dir, 'r'))
        self.no_header = no_header

        if not no_header or isinstance(data, str):
            head = self.file.__next__()

        if isinstance(features, (int, type(None))):
            features = [features]

        self.idx = list(filter(None.__ne__, [data, label, parse] + features))
        self.selection = {} if not selection else selection

        if isinstance(data, str):
            self.idx = [head.index(ind) for ind in self.idx]

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


class Pipe(object):
    """Pipe in pipeline wrapper.

    Used to cleanly handle pipeline components that adhere to the scikit-learn
    API, meaning they have a fit/transform method.

    Parameters
    ----------
    idf : string
        Name representation used in the pipeline. Only 'clf' should be used to
        selection of multiple classifiers.

    skobj : class
        Classifier, normalizer, or decompostion class that adheres to the
        scikit-learn API.

    parameters : dict
        With {'parameter name': values}. If an iterator or array-like object is
        provided as parameter, these will by default be used as combinations to
        apply grid search on.
    """

    def __init__(self, idf, skobj, parameters=False):
        """Pipe initialization."""
        self.idf = idf
        self.skobj = skobj
        self.parameters = parameters if parameters else {}

    def check(self, seed):
        """Check if correct params are set for objects."""
        try:
            self.skobj.copy = False
        except AttributeError:
            pass

        try:
            self.skobj.probability = True
            self.skobj.random_state = seed
        except AttributeError:
            pass
