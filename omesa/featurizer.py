# -*- coding: utf-8 -*-

"""Text feature extraction module.

This module contains several helper classes for extracting textual features
used in Text Mining applications, partly based on instances parsed with parse.
It also includes a wrapper class to cleverly hanlde this within the Omesa
framework.
"""

import re
import json
import pickle
from collections import OrderedDict, Counter
from urllib.parse import urlencode

import numpy as np


class Featurizer(object):
    """Wrapper for looping feature extractors in fit and transform operations.

    Calls helper classes which extract different features from text data. Given
    a list of initialized feature extractor classes, correctly streams or dumps
    instances along these classes. Also provides an interface to fit and
    transform methods.

    Parameters
    ----------
    features : list
        List of initialized feature extractor classes. The classes can be
        found within this module.

    Attributes
    ----------
    helper : list of classes
        Store for the provided features.

    X : list of lists of shape [n_samples, n_words]
        All data instances used by space_based featurizer helpers.

    Y : list of labels
        Labels for X.

    Examples
    --------
    Note: this is just for local use only.

    During training with a full space and a generator:
    >>> loader = reader.load  # assumes that this is a generator
    >>> features = [Ngrams(level='char', n_list=[1,2])]
    >>> ftr = _Featurizer(features)
    >>> ftr.fit(loader())
    >>> X, Y = ftr.transform(loader()), ftr.labels

    During testing with only one instance:
    >>> new_data = 'this is some string to test'
    >>> tex, tey = ftr.transform(new_data), ftr.labels

    Notes
    -----
    For an explanation regarding the parse features, please refer either to
    utils.parse.extract_tags or http://ilk.uvt.nl/parse/.
    """

    def __init__(self, features, preprocessor=False, parser=False):
        """Initialize the wrapper and set the provided features to a var."""
        self.helpers = features
        self.preprocessor = preprocessor
        self.parser = parser
        self.head = []

    def transform(self, instance):
        """Call all the helpers to extract features.

        Parameters
        ----------
        instance : tuple
            Containing at least (raw) and optionally (parse, meta).

        Returns
        -------
        v : dict
            Feature vector where key, value = feature, value.
        label : str
        """
        if isinstance(instance, str):
            instance = tuple([instance])
        raw, label, parse, meta = instance + (None,) * (4 - (len(instance)))

        text = self.preprocessor.clean(raw) if self.preprocessor else raw
        if not parse and self.parser:
            parse = self.parser.parse(raw if self.parser.raw else text)

        v = {}
        for helper in self.helpers:
            v.update(helper.transform(text, parse))
        if meta:
            for name, value in meta:
                v.update({"meta_" + name: value})

        return v, label


class Ngrams(object):
    """Calculate n-gram frequencies.

    Can either be applied on token, POS or character level. The transform
    method dumps a feature dictionary that can be used for feature hashing.

    Parameters
    ----------
    n_list : list of integers
        Amount of grams that have to be extracted, can be multiple. Say that
        uni and bigrams have to be extracted, n_list has to be [1, 2].

    max_feats : integers
        Limits how many features will be generated.

    Examples
    --------
    Token-level uni and bigrams with a maximum of 2000 feats per n:

    >>> ng = Ngrams(level='token', n_list=[1, 2], max_feats=2000)
    >>> ng.transform('this is text')
    ... {'this': 1, 'is': 1, 'text': 1, 'this is': 1, 'is text': 1}

    Notes
    -----
    Implemented by: Chris Emmery
    """

    def __init__(self, level='token', n_list=None):
        """Set parameters for N-gram extraction."""
        self.name = level+'_ngram'
        self.n_list = [2] if not n_list else n_list
        self.level = level
        self.row = 0 if level is 'token' else 2
        self.index, self.counter = 0, 0

    def __str__(self):
        """Report on feature settings."""
        return '''NGrams(level={0}, n_list={1})'''.format(self.level,
                                                          self.n_list)

    @staticmethod
    def find_ngrams(input_list, n):
        """Magic n-gram function.

        Calculate n-grams from a list of tokens/characters with added begin and
        end items. Based on the implementation by Scott Triglia.
        """
        inp = [''] * n + input_list + [''] * n
        return zip(*[inp[i:] for i in range(n)])

    def transform(self, raw, parse=None):
        """Given a document, return level-grams as Counter dict."""
        if self.level == 'char':
            needle = list(raw)
        elif self.level == 'text':
            needle = raw.split()
        elif self.level == 'token' or self.level == 'pos':
            # FIXME: parses are not handled well
            needle = parse[self.row] if parse else raw.split()
            if self.level == 'pos' and not parse:
                raise EnvironmentError("There's no POS annotation.")

        c = Counter()
        for n in self.n_list:
            c += Counter([self.level+"-"+"_".join(item) for
                          item in self.find_ngrams(needle, n)])
        return c


class FuncWords(object):
    """Extract function word frequencies.

    Computes relative frequencies of function words according to parse data,
    and adds the respective frequencies as a feature.

    Notes
    -----
    Implemented by: Chris Emmery
    Dutch functors: Ben Verhoeven
    """

    def __init__(self, lang='en'):
        """Set parameters for function word extraction."""
        self.name = 'func_words'

        if lang == 'en':
            raise NotImplementedError
        elif lang == 'nl':
            self.functors = {
                'VNW': 'pronouns', 'LID': 'determiners', 'VZ': 'prepositions',
                'BW': 'adverbs', 'TW': 'quantifiers', 'VG': 'conjunction'}

    def transform(self, _, parse):
        """Extract frequencies for fitted function word possibilites."""
        tokens = [item[0] for item in parse if item[2].split('(')[0]
                  in self.functors]
        return Counter(tokens)


class APISent(object):
    """Sentiment features using API tools.

    Interacts with web and therefore needs urllib3. Might be _very_ slow,
    use with caution and prefrably store features.

    Parameters
    ----------
    mode : string, optional, default 'deep'
        Can be either 'deep' for Twitter-based neural sentiment (py2, boots
        local server instance), or 'nltk' for the text-processing.com API.

    Examples
    --------
    >>> sent = APISent()
    >>> sent.transform("you're gonna have a bad time")
    ... 0.030120761495050809
    >>> sent = APISent(mode='nltk')
    >>> sent.transform("you're gonna have a bad time")
    ...

    Notes
    -----
    Implemented by: Chris Emmery
    Deep sentiment: https://github.com/xiaohan2012/twitter-sent-dnn
    NLTK API: http://text-processing.com
    """

    def __init__(self, mode='deep'):
        """Load poolmanager and set API location."""
        from urllib3 import PoolManager
        self.name = 'apisent'
        self.mode = mode
        self.pool = PoolManager()

    def __str__(self):
        """String representation for APISent."""
        return '''
        feature:    {0}
        mode:       {1}
        '''.format(self.name, self.mode)

    def transform(self, raw, _):
        """Return a dictionary of feature values."""
        if self.mode == 'deep':
            jsf = json.dumps({'text': raw})
            header = {'content-type': 'application/json'}
            request = "http://localhost:6667/api"
            r = self.pool.request('POST', request, headers=header, body=jsf)
            out = {'deepsent': float(r.data.decode('utf-8'))}
        elif self.mode == 'nltk':
            qf = urlencode({'text': raw})
            request = "http://text-processing.com/api/sentiment/"
            r = self.pool.request('POST', request, body=qf)
            try:
                out = json.loads(r.data.decode('utf-8'))["probability"]
            except ValueError:
                exit("SentAPI threw unexpected response, " +
                     "probably reached rate limit.")
        return out


class DuSent(object):
    """Lexicon based sentiment features.

    Calculates four features related to sentiment: average polarity, number of
    positive, negative and neutral words. Counts based on the Duoman and
    Pattern sentiment lexicons.

    Notes
    -----
    Implemented by: Chris Emmery

    Based on code by Cynthia Van Hee, Marjan Van de Kauter, Orphée De Clercq
    """

    def __init__(self):
        """Load the sentiment lexicon."""
        self.name = 'sentiment'
        self.lexiconDict = pickle.load(
            open(__file__.split('featurizer.py')[0] +
                 '/data/sentilexicons.cpickle', 'rb'))

    def __str__(self):
        """Class string representation."""
        return '''
        feature:   %s
        ''' % (self.name)

    def calculate_sentiment(self, instance):
        """Calculate four features for the input instance.

        Instance is a list of word-pos-lemma tuples that represent a token.
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
        for parse in instance:
            try:
                word, lemma, pos, _ = parse
            except ValueError:
                exit("ERROR: DuSent relies on Frogged data!")
            for regx, param in token_dict.items():
                if re.search(regx, pos):
                    if (word, param[0]) in self.lexiconDict:
                        polarity_score += self.lexiconDict[(word, param[0])]
                    elif (lemma, param[1]) in self.lexiconDict:
                        polarity_score += self.lexiconDict[(lemma, param[1])]
                    break
                    # FIXME: reinclude the token numbers here
        return polarity_score

    def transform(self, _, parse):
        """Get the sentiment belonging to the words in the parse string."""
        return {self.name: self.calculate_sentiment(parse)}


class SimpleStats(object):
    r"""Word and token based features.

    Parameters
    ----------
    text : boolean, optional, default True
        Text-based features to be extracted, includes:

        - Total amount of flooding, and individually punctuation and
          alphanumeric stats.
        - Frequency of punctuation and number sequences.
        - Emoticon frequencies.

    token : boolean, optional, default True
        Token-based features to be extracted, includes:

        - Word lengths.
        - Number of all CAPITAL words.
        - Number Of Start Capital Words.
        - Occurence of URLs.
        - Occurence of links to pictures.
        - Occurence of links to videos.
        - Every feature listed above.

    sentence_lenth : boolean, optional, default True
        Add the sentence length as a feature.

    Examples
    --------
    All features:
    >>> SimpleStats()

    Only text features:
    >>> SimpleStats(token=False, sentence_length=False)

    Notes
    -----
    Implemented by: Chris Emmery
    Features by: Janneke van de Loo
    """

    def __init__(self, text=True, token=True, sentence_length=True):
        """Initialize all parameters to extract simple stats."""
        self.name = 'simple_stats'
        self.v = {}
        self.text, self.token, self.stl = text, token, sentence_length

    @staticmethod
    def avg(iterb):
        """Average length of iter."""
        return np.mean([len(fl) for fl, _ in iterb]) if iterb else 0

    def text_based_feats(self, raw):
        """Include features that are based on the raw text."""
        r_punc = r'[\!\?\.\,\:\;\(\)\"\'\-]'
        flood, flood_alph, flood_punc = [], [], []

        for fl in re.findall(r"((.)\2{2,})", raw):
            flood.append(len(fl[0]))
            if re.search(r'^[a-zA-Z]+$', fl[1]):
                flood_alph.append(len(fl[0]))
            if re.search(r_punc, fl[1]):
                flood_punc.append(len(fl[0]))
        # print(fl, fl_alph, fl_punc)
        av = (np.mean(flood), np.mean(flood_alph), np.mean(flood_punc))

        self.v.update({'flood_norm_len': len(flood),
                       'flood_alph_len': len(flood_alph),
                       'flood_punc_len': len(flood_punc),
                       'flood_norm_avg': av[0] if str(av[0]) != 'nan' else 0,
                       'flood_alph_avg': av[1] if str(av[1]) != 'nan' else 0,
                       'flood_punc_avg': av[2] if str(av[2]) != 'nan' else 0,
                       'num_punc': len(re.findall(r_punc + '+', raw)),
                       'num_num': len(re.findall(r'[0-9]+', raw)),
                       'num_emots': len(re.findall(r'_EMOTICON_', raw))})

    def token_based_feats(self, tokens):
        """Include features that are based on certain tokens."""
        stats = {'word_len': 0, 'cap_words': 0, 'start_cap': 0,
                 'num_urls': 0, 'num_phots': 0, 'num_vids': 0}
        r_cap = r'[A-Z\-0-9]*[A-Z][A-Z\-0-9]*$'

        for token in tokens:
            stats['word_len'] += len(token)
            stats['cap_words'] += len(''.join(re.findall(r_cap, token)))
            stats['start_cap'] += len(re.findall(r'[A-Z][a-z]', token))

            # needs token parser
            if token == '__URL__':
                stats['num_urls'] += 1
            elif token == '__PHOTO__':
                stats['num_phots'] += 1
            elif token == '__VIDEO__':
                stats['num_vids'] += 1

        self.v.update(stats)

    @staticmethod
    def avg_sent_length(sentence_indices):
        """Calculate average sentence length."""
        words_per_sent = Counter(sentence_indices)
        return np.mean([val for _, val in words_per_sent.items()])

    def transform(self, raw, parse):
        """Transform given instance into simple text features."""
        if self.text:
            self.text_based_feats(raw)
        try:
            if self.token:
                assert parse
                self.token_based_feats([p[0] for p in parse])
            if self.stl:
                assert parse
                self.avg_sent_length([p[3] for p in parse])
        except AssertionError:
            exit("SimpleStats - No parses were found to extract token or " +
                 "sentence features from. Please provide or disable features.")
        return self.v


class Readability(object):
    """Get readability-related features.

    Notes
    -----
    Implemented by: Chris Emmery
    Attributes by: Tom De Smedt
    """

    def __init__(self):
        """Initialize empty class variables."""
        self.name = 'readability'
        self.diacritics = \
            u"àáâãäåąāæçćčςďèéêëēěęģìíîïīłįķļľņñňńйðòóôõöøþřšťùúûüůųýÿўžż"
        self.punctuation = ".,;:!?()[]{}`''\"@#$^&*+-|=~_"
        self.flooding = re.compile(r"((.)\2{2,})", re.I)  # ooo, xxx, !!!, ...
        self.emoticons = set((
            '*)', '*-)', '8)', '8-)', '8-D', ":'''(", ":'(", ':(', ':)',
            ':-(', ':-)', ':-.', ':-/', ':-<', ':-D', ':-O', ':-P', ':-S',
            ':-[', ':-b', ':-c', ':-o', ':-p', ':-s', ':-|', ':/', ':3', ':>',
            ':D', ':O', ':P', ':S', ':[', ':\\', ':]', ':^)', ':b', ':c',
            ':c)', ':o', ':o)', ':p', ':s', ':{', ':|', ':}', ";'(", ';)',
            ';-)', ';-]', ';D', ';]', ';^)', '<3', '=(', '=)', '=-D', '=/',
            '=D', '=]', '>.>', '>:)', '>:/', '>:D', '>:P', '>:[', '>:\\',
            '>:o', '>;]', 'X-D', 'XD', 'o.O', 'o_O', 'x-D', 'xD', u'\xb0O\xb0',
            u'\xb0o\xb0', u'\u2665', u'\u2764', '^_^', '-_-'
        ))
        self.emoji = set((
            u'\U0001f44c', u'\U0001f44d', u'\U0001f47f', u'\U0001f495',
            u'\U0001f499', u'\U0001f49a', u'\U0001f49b', u'\U0001f49c',
            u'\U0001f600', u'\U0001f601', u'\U0001f602', u'\U0001f603',
            u'\U0001f604', u'\U0001f605', u'\U0001f606', u'\U0001f607',
            u'\U0001f608', u'\U0001f60a', u'\U0001f60b', u'\U0001f60c',
            u'\U0001f60d', u'\U0001f60e', u'\U0001f60f', u'\U0001f610',
            u'\U0001f612', u'\U0001f613', u'\U0001f614', u'\U0001f615',
            u'\U0001f61b', u'\U0001f61c', u'\U0001f61d', u'\U0001f61e',
            u'\U0001f61f', u'\U0001f620', u'\U0001f621', u'\U0001f622',
            u'\U0001f625', u'\U0001f626', u'\U0001f627', u'\U0001f629',
            u'\U0001f62a', u'\U0001f62b', u'\U0001f62c', u'\U0001f62d',
            u'\U0001f62e', u'\U0001f62f', u'\U0001f633', u'\U0001f636',
            u'\U0001f63b', u'\U0001f63f', u'\U0001f640', u'\u2764\ufe0f',
            u'\u263a', u'\ud83d', u'\ude09'
        ))
        self.url = re.compile(r"https?://[^\s]+")
        self.ref = re.compile(r"@[a-z0-9_./]+", flags=re.I)

    def transform(self, raw, _):
        """Add each metric to the feature vector."""
        # TODO: add stuff here
        raw = raw + _
        return NotImplementedError
