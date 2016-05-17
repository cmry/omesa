# Omesa + Your Pipeline - Data To Features in 5 Minutes

The package was originally developed to be used as an easy data-to-features
wrapper, with as few dependencies as possible. For this purpose, the
`Vectorizer` class was built, which allows minimal use of Omesa within an
existing framework. An example of its use can be seen below.


## Preparing Settings

Say that we are starting session in which we would like to train on some data.
We need a config name, a list of data, and what kind of features we wish to
extract from for this. First we import Omesa, and the `featurizer` classes
we want to use. After, the feature classes can be initialized with the relevant
parameters, and we provide the directory and info to open our data

``` python
from omesa.containers import CSV
from omesa.featurizer import Ngrams


features = [Ngrams(level='char', n_list=[3]),
            Ngrams(level='token', n_list=[1, 2])]

data = CSV('/dir/to/data/data.csv', data=1, label=0, header=True)
```


## Data To Features

Now we can `transform` our data to `X, y`:

``` python
from omesa.pipes import Vectorizer

vec = Vectorizer(features)
X, y = vec.transform(data)
```

`X` is returned as a sparse matrix, and `y` a list of labels.


## Own Pipeline

From there on, you can do whatever you wish with `X`, such as a common `sklearn`
classification operation.

``` python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X, y)
```


## Saving for Deployment

To save your model, you can do:

``` python
from omesa.containers import Pipeline

pl = Pipeline(name='my_experiment', source='json')
pl.save(vectorizer=vec, classifier=clf)
```

In a demo, this could be loaded in again by using:

``` python
from omesa.containers import Pipeline
from sklearn.naive_bayes import GaussianNB

pl = Pipeline(name='my_experiment', source='json')
pl.load()

pl.classify('raw text')
... [label], (0.12231, 0.87769)
```
