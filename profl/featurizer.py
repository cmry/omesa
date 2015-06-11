import numpy as np
import os
import operator
from .utils import liwc
from .utils import find_ngrams, freq_dict
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import cPickle

# Authors: Chris Emmery, Mike Kestemont
# Contributors: Ben Verhoeven, Florian Kunneman, Janneke van de Loo
# License: BSD 3-Clause


def identity(x):
    return x


class Featurizer:
    """
    Parameters
    -----

    raw : list
        The raw data comes in an array where each entry represents a text
        instance in the data file.

    frogs : list
        The frog data ...

    features : dict
        Subset any of the entries in the following dictionary:

        features = {
            'simple_stats': {}
            'token_ngrams': {'n_list': bla, 'max_feats': bla}
            'token_pca':    {'dimensions': 2, 'max_tokens': 10}
            ... PLEASE ADD YOURS! -c-
        }

    Notes
    -----
    For an explanation regarding the frog features, please refer either to
    utils.frog.extract_tags or http://ilk.uvt.nl/frog/.
    """
    def __init__(self, raws, frogs, features):

        self.frog = frogs
        self.raw = raws
        self.helpers = features

    def loop_helpers(self, func):
        features = {}
        for h in self.helpers:
            func(h, features)
        submatrices = [features[ft] for ft in sorted(features.keys())]
        X = np.hstack(submatrices)
        return X

    def func_transform(self, h, features):
        if not h.feats:
            raise ValueError('There are no features for ' + h.name + ' to \
                              transform the data with. You probably did \
                              not fit before transforming.')
        features[h.name] = h.transform(self.raw, self.frog)

    def func_fit_transform(self, h, features):
        h.fit(self.raw, self.frog)
        features[h.name] = h.transform(self.raw, self.frog)

    def transform(self):
        self.loop_helpers(self.func_transform)

    def fit_transform(self):
        self.loop_helpers(self.func_fit_transform)


class BlueprintFeature:

    def __init__(self):
        # etc.
        pass

    def fit(self, raw, frog):
        # get feature types
        pass

    def some_function(self, input_vector):
        # do some stuff to input_vector
        pass

    def transform(self, raw, frog):
        instances = []
        for input_vector in raw:
            your_feature_vector = self.some_function(input_vector)
            instances.append(your_feature_vector)
        return instances

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class SimpleStats:

    def __init__(self):
        pass

    def fit(self):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        self.fit()
        self.transform()


class Ngrams:
    """
    Calculate token ngram frequencies.

    nlist : list with n's one wants to ADD

    max_feats : limit on how many features will be generated
    """
    def __init__(self, level='token', n_list=[2], max_feats=None):
        self.feats = {}
        self.n_list = n_list
        self.max_feats = max_feats
        self.level = level
        self.i = 0 if level == 'token' else 2

    def fit(self, raw_data, frog_data):
        data = raw_data if self.level == 'char' else frog_data
        for inst in data:
            needle = list(inst) if self.level == 'char' else zip(inst)[self.i]
            for n in self.n_list:
                self.feats.update(freq_dict([self.level+"-"+"_".join(item) for
                                             item in find_ngrams(needle, n)]))

    def transform(self, raw_data, frog_data):
        instances = []
        for inst in frog_data:
            dct = {}
            needle = list(inst) if self.level == 'char' else zip(inst)[self.i]
            for n in self.n_list:
                dct.update(freq_dict([self.level+"-"+"_".join(item) for item
                                      in find_ngrams(needle, n)]))
            instances.append([dct.get(f, 0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data, n_list, max_feats=None):
        self.fit(raw_data, frog_data, n_list, max_feats=max_feats)
        return self.transform(raw_data, frog_data)


class FuncWords:
    """
    Computes relative frequencies of function words.
    """
    def __init__(self):
        self.feats = None

    def func_words(self, frogstring):
        """
        Return a frequency dictionary of the function words in the text.
        Input is a string of frog output.
        """
        # Define the POS tags that comprise function words
        functors = {'VNW': 'pronouns', 'LID': 'determiners',
                    'VZ': 'prepositions', 'BW': 'adverbs', 'TW': 'quantifiers',
                    'VG': 'conjunction'}
        # Make a list of all tokens where the POS tag is in the functors list
        tokens = [item[0] for item in frogstring
                  if item[2].split('(')[0] in functors]
        return tokens

    def fit(self, raw_data, frog_data):
        feats = {}
        for inst in frog_data:
            feats.update(freq_dict(self.func_words(inst)))
        self.feats = feats.keys()
        #print self.feats
        
    def transform(self, raw_data, frog_data):
        instances = []
        for inst in frog_data:
            func_dict = self.func_words(inst)
            instances.append([func_dict.get(f,0) for f in self.feats])
        return np.array(instances)
        
    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class TokenPCA():
    """
    Tryout: transforms unigram counts to PCA matrix
    """
    def __init__(self, **kwargs):
        # set params
        self.dimensions = kwargs['dimensions']
        self.max_tokens = kwargs['max_tokens']
        # init fitters:
        self.pca = PCA(n_components=self.dimensions)
        self.vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False,
                                          max_features=self.max_tokens)

    def fit(self, raw_data, frog_data):
        X = self.vectorizer.fit_transform(raw_data).toarray()
        self.pca.fit(X)
        return self

    def transform(self, raw_data, frog_data):
        X = self.vectorizer.transform(raw_data).toarray()
        return self.pca.transform(X)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data)
        return self.transform(raw_data)


class LiwcCategories():
    """
    Compute relative frequencies for the LIWC categories.
    """
    def __init__(self, **kwargs):
        pass

    def fit(self, raw_data, frog_data):
        self.feats = liwc.liwc_nl_dict.keys()
        return self

    def transform(self, raw_data, frog_data):
        instances = []
        tok_data = [dat.split() for dat in raw_data]  # adapt to frog words
        for inst in tok_data:
            liwc_dict = liwc.liwc_nl(inst)
            instances.append([liwc_dict[f] for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)

class SentimentFeatures():
    """
    Calculates four features related to sentiment: average polarity, number of positive, negative and neutral words.
    Counts based on the Duoman and Pattern sentiment lexicons.
    
    Based on code by Cynthia Van Hee, Marjan Van de Kauter, Orphee De Clercq
    """
    def __init__(self):
        print(os.getcwd())
        self.lexiconDict = cPickle.load(open('profl/sentilexicons.cpickle','r'))

    def fit(self, raw_data, frog_data):
        return self

    def update_values(self, token, polarityScore, posTokens, negTokens, neutTokens):
        """Updates all feature values based on a token of which polarity is extracted from the lexicon"""
        token_polarity = self.lexiconDict[token]
        polarityScore += token_polarity
        if token_polarity > 0:
            posTokens += 1
        elif token_polarity < 0:
            negTokens += 1
        elif token_polarity == 0:
            neutTokens += 1
        return polarityScore, posTokens, negTokens, neutTokens

    def calculate_sentiment(self, instance):
        """
        Calculates four features for the input instance.
        instance is a list of word-pos-lemma tuples that represent a token.
        """
        polarityScore = 0.0
        posTokens = 0.0
        negTokens = 0.0
        neutTokens = 0.0
        for token in instance:
            word, pos, lemma, sent_index = token
            word = word.lower()
            lemma = lemma.lower()
            if pos == 'SPEC(vreemd)':
                if (word, 'f') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'f'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'f') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'f'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos == 'BW()':
                if (word, 'b') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'b'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'b') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'b'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('N('):
                if (word, 'n') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'n'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'n') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'n'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos == 'TSW()':
                if (word, 'i') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'i'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'i') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'i'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('ADJ(nom'):
                if (word, 'a') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'a'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'a') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'a'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('ADJ('):
                if (word, 'a') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'a'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'a') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'a'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('WW(od') or pos.startswith('WW(vd'):
                if ',nom,' in pos:
                    if (word, 'n') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'n'), polarityScore, posTokens, negTokens, neutTokens)
                    elif (lemma, 'v') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'v'), polarityScore, posTokens, negTokens, neutTokens)
                elif ',prenom,' in pos:
                    if (word, 'a') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'a'), polarityScore, posTokens, negTokens, neutTokens)
                    elif (lemma, 'v') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'v'), polarityScore, posTokens, negTokens, neutTokens)
                elif ',vrij,' in pos:
                    if (word, 'a') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'a'), polarityScore, posTokens, negTokens, neutTokens)
                    elif (lemma, 'v') in self.lexiconDict:
                        polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'v'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('WW(inf,nom'):
                if (word, 'n') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'n'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'v') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'v'), polarityScore, posTokens, negTokens, neutTokens)
            elif pos.startswith('WW('):
                if (word, 'v') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((word, 'v'), polarityScore, posTokens, negTokens, neutTokens)
                elif (lemma, 'v') in self.lexiconDict:
                    polarityScore, posTokens, negTokens, neutTokens = self.update_values((lemma, 'v'), polarityScore, posTokens, negTokens, neutTokens)
        # Normalize the sentiment feature scores
        totalTokens = len(instance)
        if totalTokens == 0:
            print(instance)
        polarityScore = polarityScore # In contrast with previous code by Van Hee, Van de Kauter & De Clercq, this score is not normalised
        posTokens = posTokens/totalTokens
        negTokens = negTokens/totalTokens
        neutTokens = neutTokens/totalTokens
        return polarityScore, posTokens, negTokens, neutTokens

    def transform(self, raw_data, frog_data):
        instances = []
        for instance in frog_data:
            instances.append(self.calculate_sentiment(instance))
        print(instances)
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)
