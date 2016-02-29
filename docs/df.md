# Omesa + Your Pipeline - Data To Features in 5 Minutes

The package was originally developed to be used as an easy data-to-features
wrapper, with as few dependencies as possible. For this purpose, the
`Environment` class was built, which allows minimal use of Omesa within an
existing framework. An example of its use can be seen below.


## Preparing Settings

Say that we are starting session in which we would like to train on some data.
We need a config name, a list of data, and what kind of features we wish to
extract from for this. First we import Omesa, and the `featurizer` classes
we want to use. After, the feature classes can be initialized with the relevant
parameters, and we provide the directory to our data. Finally, the
`omesa.Environment` is called with a name that all work can be saved under:

``` python
import omesa
from omesa.featurizer import Ngrams

features = [Ngrams(level='char', n_list=[3]),
            Ngrams(level='token', n_list=[1, 2])]

data = ['/dir/to/data/data.csv', 'dir/to/data/data2.csv']

om = omesa.Environment(name='ngram_bayes')
```


## Data To Features

Now we can `transform` our data to `X, y`. Given that we quickly want to
test if it works, we provide a maximum amount of instances with the `max_n`
argument.

``` python
loader = om.load(data=data, target_label='category', max_n=10000)
X, y = om.transform(loader, features)
```

`X` is returned as a list of sparse dictionary representations for the
features. To transform these into a sparse matrix, it is currently advised
to use either the `FeatureHasher` or `DictVectorizer` from `sklearn`, which
can be found [here](scikit-learn.org/stable/modules/feature_extraction.html).

``` python
X = DictVectorizer.fit_transform(X)
```


## Own Pipeline

From there on, you can do whatever you wish with `X`, such as a common `sklearn`
classification operation. In the end, the model can be stored by calling
`om.train` rather than the `fit` of the classifier, like so:

``` python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
om.train(clf, X, y)
```
The last line will make sure that the trained model is stored in `om.model`.
To test the model, you can just do:

``` python
from sklearn.metrics import accuracy_score
print(accuracy_score(om.test(clf, X), y))
```


## Saving Environment

To save your model, you can:

``` python
om.save()
```
Which will store it in a pickle under the name that was given with the
initiation of `omesa.Environment`. If you ever wish to implement it in a demo
of some sorts, just call it under the same name again:

``` python
om = omesa.Environment('ngram_bayes')
mod = om.load()
prediction_labels, confidence_scores = mod.classify(text)
```
