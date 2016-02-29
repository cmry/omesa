"""Data handling functions."""

import csv
import sys
from copy import deepcopy

# pylint:       disable=E1103,W0512,R0903,C0103


class LabelHandler(object):
    """Tiny label handler class.

    Accepts a dictionary of label names as keys and a [count, conversion] list
    as value. `Count` is the number that has to be grouped under the
    `conversion` label.

    Parameters
    ----------
    labs: dict
        Keys are original labels, value is a list of [count, conversion].

    Examples
    --------
    In [1]: d = {'Label1': [1, 'Label2'],
    ...          'Label3': ['Label4']
    ...     }

    In [2]: [LabelHandler(d).check(x) for x in ('Label1', 'Label1', 'Label3')]
    Out[2]: ['Label2', 'Label4']
    """

    def __init__(self, labs):
        """Copy the original counts to keep them intact across train / test."""
        self.labs = deepcopy(labs)

    def check(self, label, test):
        """Check if label has count, otherwise return label. If zero, none."""
        if self.labs.get(label) and self.labs[label][0]:
            if not test:  # always grab full test
                self.labs[label][0] = self.labs[label][0] - 1
            if len(self.labs[label]) > 1:
                return self.labs[label][1]
            else:
                return label
        if not len(self.labs):
            return label
        if not any([x[0] for x in self.labs.values()]):
            return 'break'


class Dataloader(object):
    """Quick and dirty data loader."""

    def __init__(self, conf):
        """Set configuration and label handler."""
        self.conf = conf
        self.handle = LabelHandler(conf.get('label_selection', {}))

    def load_csv(self, data):
        """Iterate through csv files."""
        csv.field_size_limit(sys.maxsize)
        for d in data:
            reader = csv.reader(open(d, 'r'))
            for i, x in enumerate(reader):
                if self.conf.get("has_header") and not i:
                    continue
                yield x

    def load_data(self, data, test=False):
        """Load from given datasets provided amount of instances."""
        conf = self.conf
        i_text, i_label = conf['text_column'], conf['label_column']
        i_ann, i_feats = conf.get('ann_column'), conf.get('feature_columns')

        # so that data can also be an iterable
        loader = self.load_csv(data) if data[0][-4:] == '.csv' else data
        for x in loader:
            label = self.handle.check(x[i_label], test)
            if label == 'break':
                break
            ann, feats = ('' if not v else x[v] for v in [i_ann, i_feats])
            ann = [x.split('\t') for x in ann.split('\n')]
            if label is not None and x[i_text]:
                yield (label, x[i_text], ann, feats)
        self.handle = LabelHandler(conf.get('label_selection', {}))
