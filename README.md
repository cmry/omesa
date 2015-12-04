# shed
A small framework for reproducible Text Mining research that largely builds on top of [scikit-learn](http://scikit-learn.org/stable/). Its goal is to make common research procedures fully automated, optimized, and well recorded. To this end it features:

  - Exhaustive search over best features, pipeline options, to classifier optimization.
  - Flexible wrappers to plug in your tools and features of choice.
  - Completely sparse pipeline through hashing - from data to feature space.
  - Record of all settings and fitted parts of the entire experimen, promoting reproducability.
  - Dump an easily deployable version of the final model for plug-and-play demos.

Read the documentation at [readthedocs](shd.readthedocs.org).

## Important Note

This repository is currently in development, so don't expect any stable functionality until this part is removed. :)

## Experiment

...

## `Environment`

shed was originally developped to be used as an easy data-to-feature-space wrapper, with as few dependencies as possible. For this purpose, the `Environment` class was built, which allows minimal use of shed within an existing framework. An example of its use can be seen below.

### Example

Say that we are starting session in which we would like to train on some data. We need a config name, a list of data, and what kind of features we wish to extract from for this. First we import `shed`, and the `featurizer` classes we want to use. After, the feature classes can be initialized with the relevant parameters, and we provide the directory to our data. Finally, the `shed.Environment` is called with a name that all workcan be saved under:

``` python
import shed
from shed.featurizer import Ngrams

features = [Ngrams(level='char', n_list=[3]),
            Ngrams(level='token', n_list=[1, 2])]

data = ['/dir/to/data/data.csv', 'dir/to/data/data2.csv']

shd = shed.Environment(name='ngram_bayes')
```

Now we can `fit_transform` our data to `X, y`. Given that we quickly want to test if it works, we provide a maximum amount of instances with the `max_n` argument.

``` python
loader = shd.load(data=data, target_label='category', max_n=10000)
X, y = shd.fit_transform(loader, features)
```
From there on, you can do whatever you wish with `X`, such as a common `sklearn` classification operation. In the end, the model can be stored by calling `shd.train` rather than the `fit` of the classifier, like so:

``` python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
shd.train(clf, X, y)
```
The last line will make sure that the trained model is stored in `shd.model`. To test the model, you can just do:

``` python
from sklearn.metrics import accuracy_score
print(accuracy_score(shd.test(clf, X), y))
```

To save your model, you can:

``` python
shd.save()
```
Which will store it in a pickle under the name that was given with the initiation of `shed.Environment`. If you ever wish to implement it in a demo of some sorts, just call it under the same name again:

``` python
shd = shed.Environment('ngram_bayes')
mod = shd.load()
prediction_labels, confidence_scores = mod.classify(text)
```

## Adding own Features

Here's an example of the most minimum word frequency feature class:

``` python
class SomeFeaturizer:

    def __init__(self, some_params):
        """Set parameters for SomeFeaturizer."""
        self.name = 'hookname'
        self.some_params = some_params
    
    def transform(self, raw, parse):
        """Return a dictionary of feature values."""
        return Counter([x for x in raw])
```

This returns a `{word: frequency}` dict per instance that can easily be transformed into a sparse matrix.
