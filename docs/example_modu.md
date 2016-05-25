# Omesa Deployed - Modular Implementation from Storage

This example describes some of the advantages and cases to using Omesa for
deploying Machine Learning experiments. The primary goal of Omesa is making
experiments transparent, reproducible, and efficient (both in storage as well
as speed). As such, principles such as general storage formats, pipeline
modularity, and drop-in deployment play an important role.

### General Storage Format

The primary advantage of storing models with Omesa, apart from the front-end
functionality in the [web app]('example_web.md') is that the pipeline is stored
in a general format. This has a few advantages over common methods to store
python models, such as [`pickle`](https://docs.python.org/3/library/pickle.html).
Pickle is incredibly convenient, but can be easy to corrupt, is not very
transparent, and has [compatibility issues](https://bugs.python.org/issue6137).
By using JSON, Omesa makes sure that every model is both version and language
agnostic, providing short summaries of the conducted experiments in text, and
the ability to easily interpret and relate configuration to performance.

From the file alone, relevant observations can be drawn such as model
configuration and tuned parameters:

``` json
"classifiers": [{
  "C": {
    "py/numpy.ndarray": {
      "dtype": "float64",
      "values": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    }
  },
  "clf": {
    "py/class": {
      "attr": {
        "class_weight": "balanced",
        "penalty": "l2",
        "max_iter": 1000,
        "tol": 0.0001,
        "C": 1.0,
        "dual": true,
        "intercept_scaling": 1,
        "multi_class": "ovr",
        "loss": "squared_hinge",
        "verbose": 0,
        "fit_intercept": true,
        "random_state": null
      },
      "name": "LinearSVC",
      "mod": "sklearn.svm.classes"
    }
  }
}],
```

Feature combinations and their parameters:

``` json
"features": [{
  "py/class": {
    "attr": {
      "n_list": [3],
      "row": 2,
      "level": "char",
      "name": "char_ngram",
      "counter": 0,
      "index": 0
    },
    "name": "Ngrams",
    "mod": "omesa.featurizer"
  }
}],
```

The fitted feature hasher and its vocab:

``` json
"hasher": {
  "py/class": {
    "attr": {
      "dtype": {
        "py/numpy.type": "float64"
      },
      "sort": true,
      "feature_names_": ["char-\b_I_B", "char-\t_\t_\t", ...,],
      "vocabulary_": {
        "char- _(_E": 2883,
        "char-__\n_L": 31897,
        "char-?_ _]": 19221,
        "char-'_ _G": 6702,
        "char-D_ _ ": 21627,
        "char-C_A_n": 21191,
        "char-d_}_,": 35553,
        ...
      }
    }
  }
}
```

And for example a small overview of experiment meta-data:

``` json
"tab": {
  "train_data_repr": "-",
  "clf_full": "MultinomialNB()",
  "lime_data_repr": "-",
  "test_score": 0.8565072302558397,
  "dur": 6.0,
  "features": "NGrams(level=char, n_list=[3])",
  "test_data_path": "split",
  "test_data_repr": "split",
  "lime_data_path": "-",
  "train_data_path": "-",
  "project": "unit_tests",
  "clf": "MultinomialNB",
  "name": "20_news"
},
```

No need to load the experiment in python and unpickle to understand its
configuration, and no worries about incompatibilities between `2.x` and `3.x`.
Even when using another programming language, the individual data can still
be used to for example fit a feature hasher, or configure a classifier. Of
course, the flat text representation can be decoded using Omesa, returning it
to its original python objects ready for use. This allows for modularity when
constructing pipelines, effective database calls the ablility to load an entire
pipeline for deployment, which will be discussed in the following sections.

> More information about JSON storage for sklearn-like pipelines used in Omesa
> can be found in [this blog series](https://cmry.github.io/notes/serialize).

### Deployment

For a full I/O installment, there is no need to muck around with loading
single components from the database. One can just simply import a full pipeline
for classification like so:

``` python
from omesa.containers import Pipeline

pl = Pipeline(name='omesa_exp', store='db')
pl.load()

pl.classify(["some raw text"], best_only=True)

# output -----
[(1, 0.912273)]
```

Despite the fact that this might suffice for simple demos, actual (distributed)
high-load applications might require only using the vectorizer at times, or
just applying the classifier to a batch of vectors. It might also be the case
that several classifiers have been trained on (partly) the same vector
representation, just with a different target. In that case, loading the full
pipeline multiple times for shared tasks generates unneccesary overhead, and
modularity should be preferred.


### Modularity

As we've seen, rather than having to load an entire pipeline
including dependencies and unwind its parts as one would have to do with
pickle, Omesa allows a particular part of the pipeline to be decoded. A
standard loading procedure would look something like:

```python
import json
from omesa.tools import serialize_sk as sr

mod = json.load(open('omesa_exp.json'))
vec = sr.decode(mod['vec'])

vec.transform("some raw text")
```

Even though the experiment might have some classifier package as dependencies,
as the deserialization step happens *after* selection we also don't need to worry
about these being installed on each machine, and can load the vectorizer in
isolation. Even better, using the database to store models
(`save=('model', 'db')`) allows for partial *retrieval* of the modules so that
the full file does need to be in memory. Like so:

```python
from omesa.database import Database, Vectorizer
from omesa.tools import serialize_sk as sr

db = Database()
vec = db.get_component(Vectorizer, 'omesa_exp')

vec.transform("some raw text")
```
