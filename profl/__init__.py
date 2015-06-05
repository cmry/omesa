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

    >>> from profl import make
    >>> from os import getcwd

    >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

    >>> data_conf = make(name='bayes_age_v1', data=data, target_label='age',
                         features=['pca', 'liwc'])

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
from .featurizer import Featurizer

__all__ = ['models']


def make(name, data=['./profl/data/test3.csv'], dev=None, target_label='age',
         features={'liwc': {}, 'token_pca': {'dimensions': 2,'max_tokens': 10}}, 
         max_n=None, shuffle=True, rnd_seed=666):

    print("::: loading datasets :::")
    reader = Datareader(max_n=max_n, shuffle=shuffle, rnd_seed=rnd_seed,
                        label=target_label)
    labels, raw, frog = reader.load(data, dict_format=True)

    print("::: creating features :::")
    featurizer = Featurizer(raw, frog, features)
    space = featurizer.fit_transform()
    print(space)
    # config = {k: v for k, v in args}
    # return config, space
