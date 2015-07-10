"""
Namespace environment.

The idea of this module is to provide an interface for loading data, training
and testing models, storing well performing model versions with their
associated data and feature combinations, and the ability to load these all
back in again to test on new data.
"""

from .datareader import Datareader
from .featurizer import _Featurizer, Ngrams
from os import path


class Profiler:

    """
    Starts the profiling environment and initiates its namespace.

    Parameters
    ----------
    name : string
        The namespace under which you want to save the existing configuration.

    Attributes
    ----------

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
    """

    def __init__(self, name):
        """Set environment variables."""
        self.name = name
        self.dir = path.dirname(path.realpath(__file__))
        self.reader = None
        self.featurizer = None
        self.model = None

    def load(self, data=['./profl/data/test3.csv'], target_label='age',
             proc=None, max_n=None, shuffle=True, rnd_seed=666):
        """
        Wrapper for the data loader.

        If no arguments are provided, will just extract from some small test
        set. This can be used during development.

        Parameters
        ----------
        data : list of strings
            List with document directories to be loaded.

        proc : string or function, [None (default), 'text', 'label', 'both', \
                                    function]
            If you want any label or text conversion, you can indicate this
            here with a string, or either supply your own to apply to the row
            object. These are constructed as list[label, text, frog].

            'text':
                Apply a generic normalization and preprocessing process to the
                input data.
            'label':
                Do a general categorization based on the labels that is
                provided if need be. Age will for example be splitted into
                several clases.
            'both':
                All of the above.
            function:
                Specify your own function by which you want to edit the row.

        max_n : int, optional, default False
            Maximum number of data instances *per dataset* user wants to work
            with.

        shuffle : bool, optional, default True
            If the order of the dataset should be randomized.

        rnd_seed : int, optional, default 666
            A seed number used for reproducing the random order.

        target_label : str, optional, default age header
            Name of the label header row that should be retrieved. If not set,
            the second column will be asummed to be a label column.

        Returns
        -------
        loader : generator
            The loader iteratively yields a preprocessed data instance with
            (label, raw, frog).

        Examples
        --------
        Loading some data:
        >>> import profl
        >>> from os import getcwd

        >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

        >>> from profl.featurizer import *
        >>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

        >>> env = profl.Profiler(name='bayes_age_v1')
        >>> loader = env.load(data=data, max_n=2000, target_label='age')
        """
        self.reader = Datareader(data=data, proc=proc, max_n=max_n,
                                 shuffle=shuffle, rnd_seed=rnd_seed,
                                 label=target_label)
        loader = self.reader.load
        return loader

    def fit_transform(self, loader, features=Ngrams(), fit=True):
        """
        Fit and transform data with the provided features, fit is optional.

        Parameters
        ----------
        loader : generator
            The loader should iteratively yield a preprocessed data instance
            with (label, raw, frog).

        features : list of class instances, optional, default Ngrams
            Featurizer helper class instances and parameters found in
            featurizer.py.

        fit : bool, optional, default True
            In case your fitted data has no relation to what was retrieved from
            the loader, and you need to seperately load and transform these,
            you can opt for skipping the fitting operation.

        Returns
        -------
        space : numpy array of shape [n_samples, n_features]
            Matrix with feature space.

        labels : list of shape [n_labels]
            List of labels for data instances.

        """
        self.featurizer = _Featurizer(features)
        if fit:
            self.featurizer.fit(loader())
        space = self.featurizer.transform(loader())
        labels = self.featurizer.labels
        return space, labels

    def train(self, model, space, labels):
        self.model = model
        self.model.fit(space, labels)

    def test(self, space, labels):
        if not self.model:
            raise EnvironmentError("There is no trained model to test.")
        res = self.model.predict(space)
        return res
