# Omesa Only - End-To-End In 2 Minutes

With the end-to-end `Experiment` pipeline and a configuration dictionary,
several experiments or set-ups can be ran and evaluated with a very minimal
piece of code. One of the test examples provided is that of
[n-gram classification](https://github.com/cmry/omesa/blob/master/examples/n_gram.py)
of Wikipedia documents. In this experiment, we are provided with a toy set
[n_gram.csv](https://github.com/cmry/omesa/blob/master/examples/n_gram.csv) that
features 10 articles about Machine Learning, and 10 random other articles. To
run the experiment, the following configuration is used:

``` python
from omesa.experiment import Experiment
from omesa.featurizer import Ngrams

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
store the log.

## Ouput

The log file will be printed during run time, as well as stored in the
script's directory. The output of the current experiment is as follows:

``` yml
---- Omesa ----

 Config:

        feature:   char_ngram
        n_list:    [3]

	name: gram_experiment
	seed: 111

 Sparse train shape: (20, 1287)

 Tf-CV Result: 0.8
```
