# shed
A small framework for reproducible Text Mining research that largely builds
on top of [scikit-learn](http://scikit-learn.org/stable/). Its goal is to make
common research procedures quick to set up, structed according to best
practices, optimized, and well recorded. To this end it features:

  - Optional exhaustive search over best features, pipeline options, and
    classifier parameters.
  - Flexible wrappers to plug in your tools and features of choice.
  - Completely sparse pipeline through hashing - from data to feature space.
  - Record of all settings and fitted components of the entire experiment,
    promoting reproducibility.
  - Dump an easily deployable version of the final model for plug-and-play
    demos.

## Important Note

This repository is currently in development, stable functionality is not
guaranteed as long as this message shows.

## Dependencies

`shed` currently heavily relies on `numpy`, `scipy` and `sklearn`. To use the
[Frog](https://languagemachines.github.io/frog/) wrapper as a Dutch back-end, we
strongly recommend using [LaMachine](https://proycon.github.io/LaMachine/). For
English, there is a [spaCy](https://spacy.io/) wrapper available.

## `shed` Only - End-To-End In 2 Minutes

With the end-to-end `Experiment` pipeline and a configuration dictionary,
several experiments or set-ups can be ran and evaluated with a very minimal
piece of code. One of the test examples provided is that of
[n-gram classification]('https://www.github.com/cmry/shed/examples/n_gram.py')
of Wikipedia documents. In this experiment, we are provided with a toy set
[`n_gram.csv`]('https://www.github.com/cmry/shed/examples/n_gram.csv') that
features 10 articles about Machine Learning, and 10 random other articles. To
run `shed` on this, the following configuration is used:

``` python
from shed.experiment import Experiment
from shed.featurizer import Ngrams

conf = {
    "gram_experiment": {
        "name": "gram_experiment",
        "train_data": ["./n_gram.csv"],
        "has_header": True,
        "features": [Ngrams(level='char', n_list=[3])],
        "text_column": 1,
        "label_column": 0,
        "folds": 10,
        "save": ("log")
    }
}

for experiment, configuration in conf.items():
    Experiment(configuration)
```

This will ten-fold cross validate performance on the `.csv`, selecting text
and label columns and indicating a header is present in the `.csv` document.
We provide the `Ngrams` function and parameters to be used as features, and
store the log. The output is as follows:

``` yml
---- Shed ----

 Config:

        feature:   char_ngram
        n_list:    [3]

	name: gram_experiment
	seed: 111

 Sparse train shape: (20, 1287)

 Tf-CV Result: 0.8
```

## `shed` + Your Pipeline - Data To Features in 5 Minutes

The package was originally developed to be used as an easy data-to-features
wrapper, with as few dependencies as possible. For this purpose, the
`Environment` class was built, which allows minimal use of `shed` within an
existing framework. An example of its use can be seen below.

Say that we are starting session in which we would like to train on some data.
We need a config name, a list of data, and what kind of features we wish to
extract from for this. First we import `shed`, and the `featurizer` classes
we want to use. After, the feature classes can be initialized with the relevant
parameters, and we provide the directory to our data. Finally, the
`shed.Environment` is called with a name that all workcan be saved under:

``` python
import shed
from shed.featurizer import Ngrams

features = [Ngrams(level='char', n_list=[3]),
            Ngrams(level='token', n_list=[1, 2])]

data = ['/dir/to/data/data.csv', 'dir/to/data/data2.csv']

shd = shed.Environment(name='ngram_bayes')
```

Now we can `transform` our data to `X, y`. Given that we quickly want to
test if it works, we provide a maximum amount of instances with the `max_n`
argument.

``` python
loader = shd.load(data=data, target_label='category', max_n=10000)
X, y = shd.transform(loader, features)
```

`X` is returned as a list of sparse dictionary representations for the
features. To transform these into a sparse matrix, it is currently advised
to use either the `FeatureHasher` or `DictVectorizer` from `sklearn`, which
can be found [here](scikit-learn.org/stable/modules/feature_extraction.html).

``` python
X = DictVectorizer.fit_transform(X)
```

From there on, you can do whatever you wish with `X`, such as a common `sklearn`
classification operation. In the end, the model can be stored by calling
`shd.train` rather than the `fit` of the classifier, like so:

``` python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
shd.train(clf, X, y)
```
The last line will make sure that the trained model is stored in `shd.model`.
To test the model, you can just do:

``` python
from sklearn.metrics import accuracy_score
print(accuracy_score(shd.test(clf, X), y))
```

To save your model, you can:

``` python
shd.save()
```
Which will store it in a pickle under the name that was given with the
initiation of `shed.Environment`. If you ever wish to implement it in a demo
of some sorts, just call it under the same name again:

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

This returns a `{word: frequency}` dict per instance that can easily be
transformed into a sparse matrix.

## Acknowledgements

Part of the work on `shed` was carried out in the context of the
[AMiCA](http://www.amicaproject.be/) (IWT SBO-project 120007) project, funded
by the government agency for Innovation by Science and Technology (IWT).
