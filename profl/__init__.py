"""
The Main Thing
=====

The idea of this module is to provide an interface for loading data, training
and testing models, storing well performing model versions with their
associated data and feature combinations, and the ability to load these all
back in again to test on new data. As such, it will combine an interactive
set-up with a passive one.

Examples
-----

Say that we are starting session in which we would like to train on some data.
We need a config name, a list of data, a label field we want to predict from
it, and what kind of features we whish to use for this.

    >>> import profl
    >>> from os import getcwd

    >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

    >>> from profl.featurizer import *
    >>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

    >>> data_conf = make(name='bayes_age_v1', data=data, target_label='age',
                         features=features)

The config will make sure that whatever model we store can be retrieved under
the same name with exactly the same configuration, without having to re-load
data and featurizers on it. Therefore, every parameter is optional except for
the `name`, and the make function will always return an AMiCA configuation
class object. After, the object can be either trained, tested or dumped. If
your config is a new one, model_conf['state'] will return 0, and 1 if it has
been trained. Training will just consist of calling a classifier and its
parameters:

    >>> from profl.models import NaiveBayes
    >>> model = NaiveBayes(...(1))
    >>> nvb = model.train(data_conf)

Given this, the model can either be dumped for later, or tested:

    >>> test_data = [getcwd()+'/data/data.csv']
    >>> report = nvb.test(test_data)

Please note that there is no n-fold cross-validation in the test() module, as
it requires the model to train multiple times. For this, one would want to do
the following:

    >>> from profl.models import fold
    >>> report = fold(model, data_conf, f=10)

We now have a classification report stored in res, from which we can extract
the desired scores:

    >>> report.fscore()
    0.54321

If we're satisfied with the results, we can store the whole thing as a pickle
object:

    >>> nvb.save()

Later, it should be retrievable as a classifier with the make function:

    >>> query = 'this is some text that we received as input'
    >>> model = make('bayes_age_v1')
    >>> model.classify(query)
    age = 21-100, confidence = 9001%

...

(1) params go here

If your model does not exist yet, and you just want to quickly train on a toy
dataset, you can specify `dev=True`, which will not require any parameters to
be set except for a name.
"""

from .datareader import Datareader
from .featurizer import *
from os import path

__all__ = ['models']

features = [
    SimpleStats(),
    Ngrams(),
    # FuncWords(),
    # LiwcCategories(),
    # SentimentFeatures(),
    # TokenPCA()
]


class Env:

    def __init__(self, name, dev=None):
        # env
        self.name = name
        self.dir = path.dirname(path.realpath(__file__))
        self.dev = dev
        # modules
        self.reader = None
        self.featurizer = None
        self.model = None

    def load(self, data=['./profl/data/test3.csv'], target_label='age',
             max_n=None, shuffle=True, rnd_seed=666, reset=False):
        print("Setting reader...", end='')
        if not self.reader:
            self.reader = Datareader(data=data, max_n=max_n, shuffle=shuffle,
                                     rnd_seed=rnd_seed, label=target_label)
        if reset:
            self.reader.file_list = data
        print(" done!")

    # def preprocess():
    #    pass

    def fit_transform(self, features, fit=True):
        if not self.reader:
            raise ValueError("There's not data to fit, please 'load' first.")

        print("Loading data...", end='')
        labels, raw, frog = self.reader.load(dict_format=True)
        print(" succes!")

        print("Creating features...", end='')
        self.featurizer = Featurizer(raw, frog, features)
        space = self.featurizer.fit_transform()
        print(" transformed!")

        return space, labels

    def train(self, model, space, labels):
        self.model = model
        self.model.fit(space, labels)

    def test(self, space, labels):
        if not self.model:
            raise EnvironmentError("There is no trained model to test.")
        self.featurizer.fit
        self.model.predict(instance)
