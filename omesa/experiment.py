"""Experiment wrapper code.
"""

# License:      GPLv3
# pylint:       disable=E1135,E1101

from .logger import _Logger as Logger
from .containers import Pipeline


class Experiment(object):
    """Full experiment wrapper.

    Experimental wrapper that uses data and pipeline objects to evaluate
    performance and stores all reports to the back-end or just simply reports
    them.

    Parameters
    ----------
    project : string
        The project name functions as a hook to for example call the best
        performing set of parameters out of a series of experiments on the same
        data.

    name : string
        Same as the above. This will function as a hook to save your model,
        features and Omesa config under one name.

    data : list of, or single iterator
        Example: [CSV("/somedir/train.csv", label=1, text=2),
                  CSV("/somedir/train2.csv", label=3, text=5]
        The data on which the experiment will train. If the location of a .csv
        is provided, it will open these up and create an iterator for you.
        Alternatively, you can provide your own iterators or iterable
        structures providing instances of the data.

    pipeline : list of pipeline elements
        Example: [('vec', Vectorizer()),
                  ('clf', GaussianNB()),
                  ('eval', Evaluator())]
        Same functionality as the scikit-learn Pipeline object. However, at the
        end it should include Omesa's evaluation module.
        This list should contain classes that all have a fit and transform
        module and should include at least one Vectorizer (can be omesa's) or
        any other, and a classfier object.

    save : tuple of strings
        Example: ("log", model", "db", "man", "json", "pickle")
        Save the output of the log, or dump the entire model with its
        classification method and pipeline wrapper for new data instances.

    n_jobs: int, optional, default 1
        Controls the amount of jobs cross_validation is run in. Switch this to
        any other number if the classifier used does not support multi-
        threading.
    """

    def __init__(self, **kwargs):
        """Set all relevant classes."""
        self.__dict__.update(kwargs)
        self.__dict__['pln'] = self.__dict__.pop('pipeline')
        self.log = Logger(self)
        self.vec = self.pln.pop(0)
        self.eva = self.pln.pop()
        self.eva.evaluate(self)

    def store(self):
        """Save desired Experiment data."""
        if hasattr(self, 'save'):
            if 'log' in self.save:
                self.log.save()
            if 'features' in self.save:
                self.log.echo(" Feature saving has not been implemented yet!")
            if 'model' in self.save:
                Pipeline(self).save()
