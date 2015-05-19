# standard lib:
import numpy as np
from collections import Counter

# liwc imports:
import utils.liwc as liwc

# function word imports:
from utils import find_ngrams, freq_dict

# sklearn imports:
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

def identity(x):
    return x

class Featurizer:

    def __init__(self, data=dict(), state='train', features=[], target_label='gender'):
        """
        :data: a dict, created by the datareader
        :state: str can be either test or train
        :features: list of features calls by string
        """

        if 'frog' in data.keys():
            self.frog = data['frogs']
        self.raw = data['texts']
        self.helpers = [v for k, v in FEATURES.items() if k in features]
        self.state = state

        # construct feature_families by combining the given features with their indices, 
        # omits the use of an OrderedDict

    def transform(self):
        features = {}
        for helper in self.helpers:
            h = helper().fit(self.raw, self.frog)
            features[h.name] = h.transform(self.raw, self.frog)
        submatrices = [features[ft] for ft in sorted(features.keys())]
        X = np.hstack(submatrices)
        return X

    def fit(self):
        pass

    def selection(self):
        pass


class BlueprintFeature:

    def __init__(self):
        # any global features
        pass

    def fit(self, data):
        # get feature types
        pass

    def some_function(self, input_vector):
        # do some stuff to input_vector
        pass

    def transform(self, data):
        instances = []
        for input_vector in data:
            your_feature_vector = some_function(input_vector)
            instances.append(your_feature_vector)
        return instances

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


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


class TokenNgrams:
    """
    Calculate token ngram frequencies.
    Uses \n newline as token.
    """
    def __init__(self, n_list=[1]):
        self.feats = None
        self.n_list = n_list
        self.name = 'token_ngrams'

    def fit(self, raw_data, frog_data):
        feats = {}
        token_pattern = u'(?u)\b\w\w+\b'
        for inst in raw_data:
            #tokens = tokenize(inst)
            #feats.update(ngrams(tokens,self.n))
            pass
        self.feats = feats

    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        return np.array(instances)

     def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class CharNgrams:
    """
    Computes frequencies of char ngrams
    """
    def __init__(self, n_list=[3]):
        self.feats = None
        self.n_list = n_list
        self.name = 'char_ngrams'

    def fit(self, raw_data, frog_data):
        feats = {}
        for inst in raw_data:
            for n in self.n_list:
                feats.update(freq_dict(find_ngrams(inst, n)))
        self.feats = feats.keys()

    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        for inst in raw_data:
            char_dict = {}
            for n in self.n_list:
                char_dict.update(freq_dict(find_ngrams(inst, n)))
            instances.append([char_dict.get(f,0) for f in self.feats])
        return np.array(instances)

    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


# class POSTagger:

#     def __init__(self):
#         pass

#     def fit(self):
#         pass

#     def transform(self):
#         pass

#     def fit_transform(self):
#         self.fit()
#         self.transform()
    

# if __name__ == '__main__':
#     data = Datasheet.load('csi_reviews_10.csv',headers=False)
    
#     lijstje = [p[-1] for p in data]
#     #print func_words(a,b)
#     fw = FunctionWords()
#     fw.fit(lijstje)
#     X = fw.transform(lijstje)
#     print X

class FuncWords:
    """
    Computes relative frequencies of function words.
    """
    def __init__(self):
        self.feats = None
        self.name = 'function_words'

    @staticmethod
    def func_words(frogstring):
        """
        Return a frequency dictionary of the function words in the text.
        Input is a string of frog output.
        """
        # Define the POS tags that comprise function words
        functors = {'VNW':'pronouns', 'LID':'determiners', 'VZ':'prepositions', 'BW':'adverbs', 
                    'TW':'quantifiers', 'VG':'conjunction', }
        # Make a list of all tokens where the POS tag is in the functors list
        tokens = [item[0] for item in frogstring if item[2].split('(')[0] in functors]
        # Return the frequency dictionary of this token list
        return tokens
    
    def fit(self,raw_data, frog_data):
        feats = {}
        for inst in frog_data:
            feats.update(freq_dict(func_words(inst)))
        self.feats = feats.keys()
        #print self.feats
        
    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        for inst in frog_data:
            func_dict = func_words(inst)
            instances.append([func_dict.get(f,0) for f in self.feats])
        return np.array(instances)
        
    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


class TokenPCA():
    """
    Tryout: transforms unigram counts to PCA matrix
    """
    def __init__(self, dimensions=2, max_tokens=10):
        # set params
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.name = "token_pca"
        # init fitters:
        self.pca = PCA(n_components=self.dimensions)
        self.vectorizer = TfidfVectorizer(analyzer=identity, use_idf=False, max_features=self.max_tokens)

    def fit(self, data):
        X = self.vectorizer.fit_transform(data).toarray()
        self.pca.fit(X)
        return self

    def transform(self, data):
        X = self.vectorizer.transform(data).toarray()
        return self.pca.transform(X)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class LiwcCategories():
    """
    Compute relative frequencies for the LIWC categories.
    """
    def __init__(self):
        self.feats = None
        self.name = "liwc"
        
    def fit(self, raw_data, frog_data):
        self.feats = liwc.liwc_nl_dict.keys()
        return self
    
    def transform(self, raw_data, frog_data):
        if self.feats == None:
            raise ValueError('There are no features to transform the data with. You probably did not "fit" before "transforming".')
        instances = []
        tok_data = raw_data.split() #adapt to frog words
        for inst in tok_data:
            liwc_dict = liwc.liwc_nl(inst)
            instances.append([liwc_dict[f] for f in self.feats])
        return np.array(instances)
        
    def fit_transform(self, raw_data, frog_data):
        self.fit(raw_data, frog_data)
        return self.transform(raw_data, frog_data)


FEATURES = {
    'simple_stats': SimpleStats,
    'token_ngrams': TokenNgrams,
    'char_ngrams': CharNgrams,
    #'post_tags': POSTagger,
    'function_words': FuncWords,
    'liwc': LiwcCategories,
    'pca': TokenPCA
}

