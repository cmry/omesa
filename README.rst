Omesa
=====

.. image:: https://travis-ci.org/cmry/omesa.svg?branch=master
    :target: https://travis-ci.org/cmry/omesa
    :alt: Travis-CI

.. image:: https://readthedocs.org/projects/omesa/badge/?version=latest
    :target: http://omesa.readthedocs.org/en/latest/?badge=latest
    :alt: Docs

.. image:: https://landscape.io/github/cmry/omesa/master/landscape.svg?style=flat
    :target: https://landscape.io/github/cmry/omesa/master
    :alt: Landscape

.. image:: https://img.shields.io/badge/python-3.5-blue.svg
    :alt: Python 3.4

.. _scikit-learn: http://scikit-learn.org/stable/
.. _readthedocs: http://omesa.readthedocs.org/

A small framework for easily deployable, reproducible machine learning research that largely builds
on top the scikit-learn_ API. Its goal is to make common research procedures templated, optimized,
and well recorded. To this end it features:

- Flexible wrappers to plug in your tools and features of choice.
- Sparse and multi-threaded pipeline through hashing and pooling - from data to feature space.
- Storage of all settings and fitted parts of the entire experiment, promoting reproducibility.
- Dump an easily deployable version of the final model for plug-and-play demos.
- Visual overview in an web-environment for comparison and intepretation of experiments.

Read the documentation at readthedocs_.

Important Note
''''''''''''''

This repository is currently in alpha development, and is therefore not stable.

Why use Omesa?
''''''''''''''

Omesa can be used in various scenarios:

- As an end-to-end template-to-results framework.
- As several fast classes to aid your machine learning workflow.
- As a storage and overview for your experiments.

As such, you can vary between restricted options and little code, and many options (thus more code)
that require some restricted storage format. Either way, Omesa aids in structuring and comparing
experiments.

Front-end Preview
'''''''''''''''''''

.. _dev: https://github.com/cmry/omesa/tree/dev
.. _lime: https://github.com/marcotcr/lime

The front-end provides visualization and comparison of model performance. Currently only the
'Results' section works, preview below:

.. image:: http://www.cmry.nl/dump/omesa.png
    :alt: Front

.. image:: http://www.cmry.nl/dump/omesa_prop.png
    :alt: Front Prop

To test, do the following:

.. code-block:: shell

    $ cd /dir/to/omesa/examples
    $ python3 n_gram.py
    $ cd ../front
    $ python3 ./app.wsgi

And follow the ``localhost`` link that is shown to access the web app. Please
note that this part can be quite unstable. Bug reports are welcome.


Dependencies
''''''''''''

.. _Frog: https://languagemachines.github.io/frog/
.. _LaMachine: https://proycon.github.io/LaMachine/
.. _spaCy: https://spacy.io/

Omesa currently heavily relies on ``numpy``, ``scipy`` and ``sklearn``. To use
Frog_ as a Dutch back-end, we strongly recommend using LaMachine_. For
English, there is a spaCy_ wrapper available.

Omesa Only - End-To-End In 2 Minutes
------------------------------------

With the end-to-end ``Experiment`` pipeline and a configuration dictionary,
several experiments or set-ups can be ran and evaluated with a very minimal
piece of code. One of the test examples provided is that of n-gram
classification of Wikipedia documents. In this experiment, we are provided with
a toy set n_gram.csv that features 10 articles about Machine Learning, and 10
random other articles. To run the experiment, the following configuration is used:

Example
'''''''

.. _`n-gram classification`: https://github.com/cmry/omesa/blob/master/examples/n_gram.py
.. _`n_gram.csv`: https://github.com/cmry/omesa/blob/master/examples/n_gram.csv

With the end-to-end ``Experiment`` pipeline and a configuration dictionary,
several experiments or set-ups can be ran and evaluated with a very minimal
piece of code. One of the test examples provided is that of `n-gram classification`_
of Wikipedia documents. In this experiment, we are provided with a toy set
`n_gram.csv`_ that features 10 articles about Machine Learning, and 10 random
other articles. To run the experiment, the following configuration is used:

.. code-block:: python

        from omesa.experiment import Experiment
        from omesa.featurizer import Ngrams
        from omesa.containers import Pipe
        from omesa.components import Vectorizer, Evaluator
        from sklearn.naive_bayes import MultinomialNB

    Experiment(
        project="unit_tests",
        name="20_news_grams",
        data=[CSV("n_gram.csv", data="intro", label="label")],
        pipeline=[
            Vectorizer(
                features=[
                    Ngrams(level='char', n_list=[3])
                ]),
            Pipe('clf', MultinomialNB()),
            Evaluator(scoring='f1', average='micro',
                      lime_docs=CSV("n_gram.csv", data="intro", label="label")),
        ],
        save=("model", "db")
    )

This will cross validate performance on the ``.csv``, selecting text
and label columns and indicating a header is present in the ``.csv`` document.
We provide the ``Ngrams`` function and parameters to be used as features, and
store the log.

Output
''''''

The log file will be printed during run time, as well as stored in the
script's directory. A sample from the output of the current experiment is as
follows:

.. code-block:: shell

    ---- Omesa ----

     Config:

            feature:   char_ngram
            n_list:    [3]

    	name: gram_experiment
    	seed: 42

     Sparse train shape: (20, 1301)

     Performance on test set:

                 precision    recall  f1-score   support

             DF       0.83      0.50      0.62        10
             ML       0.64      0.90      0.75        10

    avg / total       0.74      0.70      0.69        20


     Experiment took 0.2 seconds

    ----------

Adding own Features
-------------------

Here's an example of the most minimum word frequency feature class:

.. code-block:: python

    class SomeFeaturizer(object):

        def __init__(self, some_params):
            """Set parameters for SomeFeaturizer."""
            self.name = 'hookname'
            self.some_params = some_params

        def transform(self, raw, parse):
            """Return a dictionary of feature values."""
            return Counter([x for x in raw])

This returns a ``{word: frequency}`` dict per instance that can easily be
transformed into a sparse matrix.

Acknowledgements
----------------

.. _AMiCA: http://www.amicaproject.be/

Part of the work on Omesa was carried out in the context of the
AMiCA_ (IWT SBO-project 120007) project, funded by the government agency for
Innovation by Science and Technology (IWT).
