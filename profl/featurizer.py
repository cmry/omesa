import numpy as np
import os
import operator
import re
from .utils import liwc
from .utils import find_ngrams, freq_dict
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
import pickle

# Authors: Chris Emmery, Mike Kestemont
# Contributors: Ben Verhoeven, Florian Kunneman, Janneke van de Loo
# License: BSD 3-Clause


def identity(x):
    return x


class Featurizer:

    """
    Parameters
    -----

    raws : list
        The raw data comes in an array where each entry represents a text
        instance in the data file.

    frogs : list
        The frog data ...

    features : dict
        Subset any of the entries in the following dictionary:

    Notes
    -----
    For an explanation regarding the frog features, please refer either to
    utils.frog.extract_tags or http://ilk.uvt.nl/frog/.
    """

    def __init__(self, raws, frogs, features, preprocess):
        self.frog = frogs
        self.raw = raws
        self.helpers = features

    def loop_helpers(self, func):
        features = {}
        for helper in self.helpers:
            if helper.name in ['token_pca']:
                func(helper, features)
            else:
                for raw, frog in zip(self.raw, self.frog):
                    func(helper, raw, frog)
                    # left off here, check below scope
            if func == self.func_fit:
                helper.close_fit()
            else:
                features[helper.name] = np.array(helper.instances)
        submatrices = [features[ft] for ft in sorted(features.keys())]
        X = np.hstack(submatrices)
        return X

    def func_fit(self, helper, raw, frog):
        helper.fit(raw, frog)

    def func_transform(self, helper, raw, frog):
        if not helper.feats:
            raise ValueError('There are no features for ' + helper.name + ' to\
                               transform the data with. You probably did \
                              not fit before transforming.')
        helper.transform(raw, frog)

    def fit(self):
        return self.loop_helpers(self.func_fit)

    def transform(self):
        return self.loop_helpers(self.func_transform)

    def fit_transform(self):
        self.fit()
        return self.transform()


class Ngrams:

    """
    Calculate token ngram frequencies.

    nlist : list with n's one wants to ADD

    max_feats : limit on how many features will be generated
    """

    def __init__(self, level='token', n_list=[2], max_feats=None):
        self.name = level+'_ngram'
        self.feats = {}
        self.instances = []
        self.n_list = n_list
        self.max_feats = max_feats
        self.level = level
        self.i = 0 if level == 'token' else 2

    def close_fit(self):
        self.feats = [i for i, j in sorted(self.feats.items(), reverse=True,
              key=operator.itemgetter(1))][:self.max_feats]

    # bug: fit errors if run two times with reinitiating profl
    def fit(self, raw, frog):
        inst = raw if self.level == 'char' else frog
        needle = list(inst) if self.level == 'char' else inst[self.i]
        for n in self.n_list:
            self.feats.update(freq_dict([self.level+"-"+"_".join(item) for
                                         item in find_ngrams(needle, n)]))

    def transform(self, raw_data, frog_data):
        inst = raw_data if self.level == 'char' else frog_data

        dct = {}
        needle = list(inst) if self.level == 'char' else inst[self.i]
        for n in self.n_list:
            dct.update(freq_dict([self.level+"-"+"_".join(item) for item
                                  in find_ngrams(needle, n)]))
        self.instances.append([dct.get(f, 0) for f in self.feats])


class FuncWords:

    """
    Function Word Featurizer
    ======
    Computes relative frequencies of function words according to Frog data,
    and adds the respective frequencies as a feature.

    Parameters
    -----
    None

    Attributes
    -----
    name : string
        String representation of the featurizer.

    feats : list
        List with the function words that occur in the training set.

    Notes
    -----
    Implemented by: Ben Verhoeven
    Quality check: Chris Emmery
    """

    def __init__(self):
        self.name = 'func_words'
        self.feats = {}
        self.instances = []

    def func_freq(self, frogstring):
        """
        Function Word frequencies
        =====
        Return a frequency dictionary of the function words in the text.
        Input is a string of frog output. Selects based on relevant functors
        the words that are function words from this input.

        Parameters
        -----
        frogstring : list
            List with Frogged data elements, example:
            ['zijn', 'zijn', 'WW(pv,tgw,mv)', '43'], ['?', '?', 'LET()', '43']

        Returns
        -----
        freq_dict(tokens): Counter
            Frequency dictionary with the function words from the training set.
        """
        functors = {'VNW': 'pronouns', 'LID': 'determiners',
                    'VZ': 'prepositions', 'BW': 'adverbs', 'TW': 'quantifiers',
                    'VG': 'conjunction'}
        tokens = [item[0] for item in frogstring if item[2].split('(')[0]
                  in functors]
        return freq_dict(tokens)

    def close_fit(self):
        self.feats = self.feats.keys()

    def fit(self, raw, frog):
        self.feats.update(self.func_freq(frog))

    def transform(self, raw, frog):
        func_dict = self.func_freq(frog)
        self.instances.append([func_dict.get(f, 0) for f in self.feats])


class TokenPCA():

    """
    Tryout: transforms unigram counts to PCA matrix
    """

    def __init__(self, dimensions=3, max_tokens=10):
        self.name = 'token_pca'
        self.pca = PCA(n_components=dimensions)
        self.vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False,
                                          max_features=max_tokens)

    def fit(self, raw_data, frog_data):
        X = self.vectorizer.fit_transform(raw_data).toarray()
        self.pca.fit(X)
        return self

    def transform(self, raw_data, frog_data):
        X = self.vectorizer.transform(raw_data).toarray()
        return self.pca.transform(X)


class LiwcCategories():

    """
    Compute relative frequencies for the LIWC categories.
    """

    def __init__(self):
        self.name = 'liwc'
        self.feats = {}
        self.instances = []

    def fit(self, raw, frog):
        self.feats = liwc.liwc_nl_dict.keys()
        return self

    def transform(self, raw, frog):
        # tok_data = [dat.split() for dat in raw_data]  # adapt to frog words
        liwc_dict = liwc.liwc_nl(frog)  # TODO: token index
        self.instances.append([liwc_dict[f] for f in self.feats])


class SentimentFeatures():

    """
    Calculates four features related to sentiment: average polarity, number of
    positive, negative and neutral words. Counts based on the Duoman and
    Pattern sentiment lexicons.

    Based on code by Cynthia Van Hee, Marjan Van de Kauter, Orphee De Clercq
    """

    def __init__(self):
        self.lexiconDict = pickle.load(open('profl/sentilexicons.cpickle',
                                            'r'))
        self.instances = []

    def fit(self, raw, frog):
        return self

    def calculate_sentiment(self, instance):
        """
        Calculates four features for the input instance.
        instance is a list of word-pos-lemma tuples that represent a token.
        """
        polarity_score = 0.0
        token_dict = OrderedDict({
            r'SPEC\(vreemd\)': ('f', 'f'),
            r'BW\(\)': ('b', 'b'),
            r'N\(': ('n', 'n'),
            r'TWS\(\)': ('i', 'i'),
            r'ADJ\(': ('a', 'a'),
            r'WW\((od|vd).*(,prenom|,vrij)': ('a', 'v'),
            r'WW\((od|vd).*,nom': ('n', 'v'),
            r'WW\(inf,nom': ('n', 'v'),
            r'WW\(': ('v', 'v')
        })
        for token in instance:
            word, pos, lemma, sent_index = token
            for regx, param in token_dict.items():
                if re.search(regx, token):
                    if (word.lower(), param[0]) in self.lexiconDict or \
                       (lemma.lower(), param[1]) in self.lexiconDict:
                        polarity_score += self.lexiconDict[token]
                    break
                    # note: might still want to get the token numbers here
        return polarity_score

    def transform(self, raw, frog):
        self.instances.append(self.calculate_sentiment(frog))
