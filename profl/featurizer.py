"""Text feature extraction module.

This module contains several helper classes for extracting textual features
used in Text Mining applications, partly based on instances parsed with Frog.
It also includes a wrapper class to cleverly hanlde this within the profl
environment.

"""

import numpy as np
import operator
import re
from .utils import liwc, freq_dict
from .utils import preproc_netlog as pnet
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict, Counter
import pickle

# Author:       Chris Emmery
# Contributors: Mike Kestemont, Ben Verhoeven, Florian Kunneman,
#               Janneke van de Loo
# License:      BSD 3-Clause


class _Featurizer:

    """Wrapper for looping feature extractors in fit and transform operations.

    Calls helper classes which extract different features from text data. Given
    a list of initialized feature extractor classes, correctly streams or dumps
    instances along these classes. Also provides an interface to fit and
    transform methods.

    Examples
    --------
    During training with a full space and a generator:
    >>> loader = reader.load  # assumes that this is a generator
    >>> features = [Ngrams(level='char', n_list=[1,2])]
    >>> ftr = _Featurizer(features)
    >>> ftr.fit(loader())
    X, Y = ftr.transform(loader()), ftr.labels

    During testing with only one instance:
    >>> new_data = 'this is some string to test'
    >>> tex, tey = ftr.transform(new_data), ftr.labels

    Notes
    -----
    For an explanation regarding the frog features, please refer either to
    utils.frog.extract_tags or http://ilk.uvt.nl/frog/.
    """

    def __init__(self, features):
        """Initialize the wrapper and set the provided features to a var.

        Parameters
        ----------
        features : list
            List of initialized feature extractor classes. The classes can be
            found within this module.
        """
        self.labels = []
        self.helpers = features
        self.space_based = ['TokenPCA', 'Doc2Vec', 'L-LDA']
        self.X = []
        self.Y = []

    def loop_helpers(self, stream, func):
        """Call all the helpers to extract features.

        Parameters
        ----------
        stream : generator
            Yields an instance with (label, raw, frog).
        func : function
            Function object from etiher the fit or transform method.

        Returns
        -------
        X : numpy array of shape [n_samples, n_features]
            Training data returns when applying the transform function.
        """
        for label, raw, frog in stream:
            for helper in self.helpers:
                if helper.name in self.space_based:
                    self.X.append(raw)
                    self.Y.append(label)
                else:
                    func(helper, raw, frog)
            if func == self.func_transform:
                self.labels.append(label)
        submatrices = []
        for helper in self.helpers:
            if func == self.func_fit:
                if helper.name not in self.space_based:
                    helper.close_fit()
                else:
                    helper.fit(self.X, self.Y)
            if func == self.func_transform:
                submatrices.append(helper.instances)
        if func == self.func_transform:
            X = np.hstack(submatrices)
            return X

    @staticmethod
    def empty_inst(helper, lo):
        """Declare an empty matrix if there is none yet."""
        try:
            helper.instances.ndim
        except AttributeError:
            helper.instances = np.empty((0, len(lo)))

    def func_fit(self, helper, raw, frog):
        helper.fit(raw, frog)

    def func_transform(self, helper, raw, frog):
        helper.transform(raw, frog)

    def fit(self, stream):
        """Fit the extractors according to their requirements."""
        return self.loop_helpers(stream, self.func_fit)

    def transform(self, stream):
        """Transform an instance according to the fitted extractors."""
        return self.loop_helpers(stream, self.func_transform)


class Ngrams:

    """
    Calculate ngram frequencies.

    Extract for

    n_list : list with n's one wants to ADD

    max_feats : limit on how many features will be generated

    Notes
    -----
    Initial code: Ben Verhoeven
    Refactoring: Chris Emmery
    """

    def __init__(self, level='token', n_list=[2], max_feats=None):
        self.name = level+'_ngram'
        self.feats = {}
        self.instances = None
        self.n_list = n_list
        self.max_feats = max_feats
        self.level = level
        self.i = 0 if level == 'token' else 2

    def find_ngrams(self, input_list, n):
        """
        Calculate n-grams from a list of tokens/characters with added begin and
        end items. Based on the implementation by Scott Triglia http://locally
        optimal.com/blog/2013/01/20/elegant-n-gram-generation-in-python/
        """
        inp = [''] + input_list + ['']
        return zip(*[inp[i:] for i in range(n)])

    def close_fit(self):
        self.feats = [i for i, j in sorted(self.feats.items(), reverse=True,
                      key=operator.itemgetter(1))][:self.max_feats]

    def fit(self, raw, frog):
        inst = raw if self.level == 'char' else frog
        needle = list(inst) if self.level == 'char' else [x[self.i] for x in inst]
        for n in self.n_list:
            self.feats.update(freq_dict([self.level+"-"+"_".join(item) for
                                         item in self.find_ngrams(needle, n)]))

    def transform(self, raw_data, frog_data):
        inst = raw_data if self.level == 'char' else frog_data
        dct = {}
        needle = list(inst) if self.level == 'char' else [x[self.i] for x in inst]
        for n in self.n_list:
            dct.update(freq_dict([self.level+"-"+"_".join(item) for item
                                  in self.find_ngrams(needle, n)]))
        Featurizer.empty_inst(self, self.feats)
        self.instances = np.append(self.instances,
                                   [[dct.get(f, 0) for f in self.feats]],
                                   axis=0)


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
        self.instances = None

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
        Featurizer.empty_inst(self, self.feats)
        self.instances = np.append(self.instances,
                               [[func_dict.get(f, 0) for f in self.feats]],
                               axis=0)


class TokenPCA():

    """
    Tryout: transforms unigram counts to PCA matrix

    Notes
    -----
    Implemented by: Mike Kestemont
    Quality check: Chris Emmery
    """

    def __init__(self, dimensions=100, max_tokens=1000):
        self.name = 'token_pca'
        self.pca = PCA(n_components=dimensions)
        self.vectorizer = TfidfVectorizer(analyzer=self.identity, use_idf=False,
                                          max_features=max_tokens)
        self.feats = None
        self.instances = None

    def identity(self, x):
        return x

    def close_fit(self):
        pass

    def fit(self, raw_data, frog_data):
        X = self.vectorizer.fit_transform(raw_data).toarray()
        self.pca.fit(X)
        self.feats = True
        return self

    def transform(self, raw_data, frog_data):
        X = self.vectorizer.transform(raw_data).toarray()
        self.instances = self.pca.transform(X)


class LiwcCategories():

    """
    Compute relative frequencies for the LIWC categories.

    Notes
    -----
    Implemented by: Ben Verhoeven
    """

    def __init__(self):
        self.name = 'liwc'
        self.feats = {}
        self.instances = None

    def close_fit(self):
        pass

    def fit(self, raw, frog):
        self.feats = liwc.liwc_nl_dict.keys()
        return self

    def transform(self, raw, frog):
        liwc_dict = liwc.liwc_nl([f[0] for f in frog])  # TODO: token index
        Featurizer.empty_inst(self, self.feats)
        self.instances = np.append(self.instances,
                               [[liwc_dict[f] for f in self.feats]], axis=0)

class SentimentFeatures():

    """
    Calculates four features related to sentiment: average polarity, number of
    positive, negative and neutral words. Counts based on the Duoman and
    Pattern sentiment lexicons.

    Notes
    -----
    Based on code by Cynthia Van Hee, Marjan Van de Kauter, Orphee De Clercq

    Implemented by: Chris Emmery
    """

    def __init__(self):
        self.name = 'sentiment'
        self.lexiconDict = pickle.load(open('./profl/data/'\
                                            'sentilexicons.cpickle', 'rb'))
        self.instances = None

    def close_fit(self):
        pass

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
            word, lemma, pos, sent_index = token
            for regx, param in token_dict.items():
                if re.search(regx, pos):
                    if (word, param[0]) in self.lexiconDict:
                        polarity_score += self.lexiconDict[(word, param[0])]
                    elif (lemma, param[1]) in self.lexiconDict:
                        polarity_score += self.lexiconDict[(lemma, param[1])]
                    break
                    # note: might still want to get the token numbers here
        return polarity_score

    def transform(self, raw, frog):
        Featurizer.empty_inst(self, '1')
        self.instances = np.append(self.instances,
                                  [[self.calculate_sentiment(frog)]], axis=0)


class SimpleStats:

    """
    Notes
    -----
    Code by: Janneke van de Loo
    Implemented by: Chris Emmery
    """

    def __init__(self, text=(), regex_punc=None, regex_word=None, regex_caps=None):
        self.name = 'simple_stats'
        self.regex_punc = r'[\!\?\.\,\:\;\(\)\"\'\-]' if not \
                          regex_punc else regex_punc
        self.regex_word = r'^[a-zA-Z\-0-9]*[a-zA-Z][a-zA-Z\-0-9]*$' if not \
                          regex_word else regex_word
        self.regex_caps = r'^[A-Z\-0-9]*[A-Z][A-Z\-0-9]*$' if not \
                          regex_caps else regex_caps
        self.feats = None
        self.instances = None

    def close_fit(self):
        pass

    def fit(self, raw, frog):
        self.feats = True

    def preprocess(self, text):
        text = pnet.restore_html_symbols(text)
        text = pnet.replace_netlog_tags(text)
        text = pnet.replace_url_email(text)
        text = pnet.replace_emoticons(text)
        return text

    def only_alph(self, floodings):
        return [fl for fl in floodings if re.search(r'^[a-zA-Z]+$', fl[1])]

    def only_punc(self, floodings):
        return [fl for fl in floodings if re.search(self.regex_punc, fl[1])]

    def avg_fl_len(self, floodings):
        if floodings:
            avg_len = np.mean([len(fl) for fl, char in floodings])
        else:
            avg_len = 0
        return avg_len

    def flooding_stats(self, text):
        vector = []
        fl = pnet.floodings(text)
        fl_alph = self.only_alph(fl)
        fl_punc = self.only_punc(fl)
        vector.append(len(fl))
        vector.append(len(fl_alph))
        vector.append(len(fl_punc))
        vector.append(self.avg_fl_len(fl))
        vector.append(self.avg_fl_len(fl_alph))
        vector.append(self.avg_fl_len(fl_punc))
        return vector

    def num_punc_seqs(self, text):
        regex_punc_seq = self.regex_punc+'+'
        return len(re.findall(regex_punc_seq, text))

    def num_num_seqs(self, text):
        regex_num_seq = r'[0-9]+'
        return len(re.findall(regex_num_seq, text))

    def char_type_stats(self, text):
        vector = []
        vector.append(self.num_punc_seqs(text))
        vector.append(self.num_num_seqs(text))
        return vector

    def num_emoticons(self, text):
        return len(re.findall(r'_EMOTICON_', text))

    def get_words(self, tokens):
        return [tok for tok in tokens if re.search(self.regex_word, tok)]

    def avg_word_len(self, words):
        avg = np.mean([len(w) for w in words])
        return avg if str(avg) != 'nan' else 0.0

    def num_allcaps_words(self, words):
        return sum([1 for w in words if re.search(self.regex_caps, w)])

    def num_startcap_words(self, words):
        return sum([1 for w in words if re.search(r'^[A-Z]', w)])

    def num_urls(self, tokens):
        return sum([1 for tok in tokens if tok == '_URL_'])

    def num_photos(self, tokens):
        return sum([1 for tok in tokens if tok == '_PHOTO_'])

    def num_videos(self, tokens):
        return sum([1 for tok in tokens if tok == '_VIDEO_'])

    def text_based_feats(self, text):
        vector = []
        text = self.preprocess(text)
        vector.extend(self.flooding_stats(text))
        vector.extend(self.char_type_stats(text))
        vector.append(self.num_emoticons(text))
        return vector

    def token_based_feats(self, tokens):
        vector = []
        words = self.get_words(tokens)
        vector.append(self.avg_word_len(words))
        vector.append(self.num_allcaps_words(words))
        vector.append(self.num_startcap_words(words))
        vector.append(self.num_urls(tokens))
        vector.append(self.num_photos(tokens))
        vector.append(self.num_videos(tokens))
        return vector

    def avg_sent_length(self, sent_nums):
        sent_len_dict = Counter(sent_nums)
        sent_lengths = [val for key, val in sent_len_dict.items()]
        avg_len = np.mean(sent_lengths)
        return avg_len

    def transform(self, raw, frog):
        # TODO: have to remove completely empty rows because they might
        # introduce errors within this function. upd: think this is fixed?
        fts = self.text_based_feats(raw) + \
              self.token_based_feats([f[0] for f in frog]) + \
             [self.avg_sent_length([f[3] for f in frog if len(frog) > 3])]
        Featurizer.empty_inst(self, fts)
        self.instances = np.append(self.instances, [fts], axis=0)
