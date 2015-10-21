"""
Namespace environment.

The idea of this module is to provide an interface for loading data, training
and testing models, storing well performing model versions with their
associated data and feature combinations, and the ability to load these all
back in again to test on new data.
"""

from .datareader import Datareader
from .featurizer import Featurizer, Ngrams
from .processor import Processor
from os import path
import pickle

# Author:       Chris Emmery
# License:      BSD 3-Clause
# pylint:       disable=E1103


class Pipeline:

    """
    Starts the framework's environment and initiates its namespace.

    Parameters
    ----------
    name : string
        The namespace under which you want to save the existing configuration.

    Attributes
    ----------

    reader : Datareader class
        Located in datareader.py

    featurizer : Featurizer class
        Located in featurizer.py

    model : Model class
        Sklearn / any other external class.

    Examples
    --------
    Typical experiment:
    >>> import shed
    >>> from os import getcwd

    >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

    >>> from shed.featurizer import *
    >>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

    >>> env = shed.Pipeline(name='bayes_age_v1')
    >>> loader = env.load(data=data, target_label='age')
    >>> space, labels = env.fit_transform(loader(), features)
    """

    def __init__(self, name, backbone='frog'):
        """Set environment variables."""
        self.name = name
        self.dir = path.dirname(path.realpath(__file__))
        self.reader = None
        self.hook = backbone
        self.backbone = Processor(backbone)
        self.featurizer = None
        self.model = None

    def load(self, data=['./shed/data/test3.csv'], proc=None,
             max_n=None, skip=range(0, 0), shuffle=True, rnd_seed=666,
             target_label='age', meta=[]):
        r"""
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

        skip : range, optional, default False
            Range of indices that need to be skipped in the data. Can be used
            for splitting as in tenfold cross-validation, whilst retaining the
            iterative functionality and therefore keeping memory consumption low.

        shuffle : bool, optional, default True
            If the order of the dataset should be randomized.

        rnd_seed : int, optional, default 666
            A seed number used for reproducing the random order.

        target_label : str, optional, default age header
            Name of the label header row that should be retrieved. If not set,
            the second column will be asummed to be a label column.

        meta : list of str, optional, default empty
            If you'd like to extract features from the dataset itself, this can
            be used to specify the headers or the indices in which these are
            located. Include 'file' if you want the filename to be a feature.

        Returns
        -------
        loader : generator
            The loader iteratively yields a preprocessed data instance with
            (label, raw, frog).

        Examples
        --------
        Loading some data:
        >>> import shed
        >>> from os import getcwd

        >>> data = [getcwd()+'/data/data.csv', getcwd()+'/data/data2.csv']

        >>> from shed.featurizer import *
        >>> features = [SimpleStats(), Ngrams(level='pos'), FuncWords()]

        >>> env = shed.Pipeline(name='bayes_age_v1')
        >>> loader = env.load(data=data, max_n=2000, target_label='age')
        """
        if not self.backbone:
            self.backbone = Processor(self.hook)
        self.reader = Datareader(self.backbone, data=data,
                                 proc=proc, max_n=max_n, skip=skip,
                                 shuffle=shuffle, rnd_seed=rnd_seed,
                                 label=target_label, meta=meta)
        loader = self.reader.load()
        return loader

    def fit(self, loader, features=Ngrams()):
        """
        Fit the provided features to cover the training data.

        Parameters
        ----------
        loader : generator
            The loader should iteratively yield a preprocessed training data
            instance with (label, raw, frog).

        features : list of class instances, optional, default Ngrams
            Featurizer helper class instances and parameters found in
            featurizer.py.
        """
        print("Starting fitting ...")
        self.featurizer = Featurizer(features)
        self.featurizer.fit(loader)
        print("done!")

    def transform(self, loader):
        """
        Transform the test data according to the fitted features.

        Parameters
        ----------
        loader : generator
            The loader should iteratively yield a preprocessed testing data
            instance with (label, raw, frog).

        Returns
        -------
        space : numpy array of shape [n_samples, n_features]
            Matrix with feature space.

        labels : list of shape [n_labels]
            List of labels for data instances.
        """
        print("Starting transforming ...")
        if not self.featurizer:
            raise EnvironmentError("Data is not fitted yet.")
        space = self.featurizer.transform(loader)
        labels = self.featurizer.labels
        print("done!")
        return space, labels

    def fit_transform(self, loader, features=Ngrams()):
        """Shorthand for fit and transform methods."""
        self.fit(loader, features)
        space, labels = self.transform(loader)
        return space, labels

    def train(self, model, space, labels):
        """Small wrapper to fit a sklearn syntax compatible classifier."""
        self.model = model
        self.model.fit(space, labels)

    def test(self, space):
        """Small wrapper to test a sklearn syntax compatible classifier."""
        if not self.model:
            raise EnvironmentError("There is no trained model to test.")
        res = self.model.predict(space)
        return res

    def classify(self, text, v=None):
        """Small wrapper to quickly transform and predict a text instance."""
        if not v or not self.backbone:
            self.backbone = Processor(self.hook, wake=True)
            v = [('', text, self.backbone.parse(text), '')]
        x, _ = self.transform(v)
        return (self.model.predict(x), self.model.predict_proba(x))

    def save(self):
        """Save the environment to a folder in model."""
        self.backbone = None
        file_loc = self.dir + '/models/' + self.name + '.pickle'
        pickle.dump(self, open(file_loc, 'wb'))
