"""
The Main Thing.

profl is currently used to conduct Author Profiling experiments. This text
mining task usually relies on custom language features. Constructing these
by hand can be a time-consuming task. Therefore, this module aims to make
loading and featurizing existing, as well as new data a bit easier. It is
specifically intended for Dutch, but just replacing the Frog module with an
language-specific tagger (from NLTK for example) would make it broadly usable.

Examples
--------
Say that we are starting session in which we would like to train on some
data. We need a config name, a list of data, and what kind of features we
whish to extract from for this.

    >>> import profl
    >>> from os import getcwd

    >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

    >>> from profl.featurizer import *
    >>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

    >>> env = profl.Profiler(name='bayes_age_v1')
    >>> loader = env.load(data=data, target_label='age')
    >>> space, labels = env.fit_transform(loader, features)

The `env` config `name` will make sure that whatever model we store can be
retrieved under the same name with exactly the same configuration, without
having to re-load data and featurizers on it. Therefore, every parameter is
optional except for the `name`, and the make function will always return an
AMiCA configuation class object. After, the object can be either trained,
tested or dumped. If your config is a new one, env.model should return
None. Training will just consist of either calling a classifier and its
parameters, or providing one from another module (currently only sklearn).

    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> env.train(clf, space, labels)

Given this, the model can either be dumped for later, or tested:

    >>> test_data = [getcwd()+'/data/data.csv']
    >>> loader = env.load(data=test_data, target_label='age')
    >>> tspace, tlabels = env.fit_transform(loader, features, fit=False)
    >>> env.test(tspace, tlabels)

Conceptual ----------------------------------------------------------------

Please note that there is no n-fold cross-validation in the test() module,
as it requires the model to train multiple times. For this, one would want
to do the following:

    >>> report = env.fold(model, space, labels, f=10)

We now have a classification report stored in res, from which we can
extract the desired scores:

    >>> report.fscore()
    0.54321

If we're satisfied with the results, we can store the whole thing as a
pickle object:

    >>> env.save()

Later, it should be retrievable as a classifier with the make function:

    >>> query = 'this is some text that we received as input'
    >>> env = profl.Env('bayes_age_v1')
    >>> model.predict(list(query))
    age = 21-100, confidence = 9001%

---------------------------------------------------------------------------

If your model does not exist yet, and you just want to quickly train on a
toy dataset, you can call each function without optinal parameters.

Have fun,
Chris

"""

from .environment import Pipeline

__author__ = 'Chris Emmery'
__contrb__ = 'Mike Kestemont, Florian Kunneman, Ben Verhoeven' \
             'Janneke van de Loo'
__license__ = 'BSD 3-Clause'
