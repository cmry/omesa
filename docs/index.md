# Omesa
A small framework for reproducible Text Mining research that largely builds
on top of [scikit-learn](http://scikit-learn.org/stable/). Its goal is to make
common research procedures quick to set up, structured according to best
practices, optimized, well recorded, and easily interpretable. To this end it
features:

  - Web front-end and stand-alone database to overview experiments and
    interpret their performance.
  - Flexible wrappers to plug in your tools and features of choice.
  - Sparse and multi-threaded feature extraction.
  - Optional exhaustive search over best features, pipeline options, and
    classifier parameters.
  - Record of all settings and fitted components of the entire experiment,
    promoting reproducibility.
  - Dump an easily deployable version of the final model for plug-and-play
    demos.

---

## Important Note

*This repository is currently in development, stable functionality is not
guaranteed as long as this message is showing.*

---

## Getting Started

We offer three quick examples to demonstrate the functionality:

- **2 minutes**: using [Omesa only](example_simple.md) for simple text
  classification.
- **5 minutes**: integrating Omesa for [data-to-features](example_df.md).
- **5 minutes**: viewing experiment performance via the [web app](example_web.md).

---

## Dependencies

Omesa currently heavily relies on `numpy`, `scipy` and `sklearn`. When using
the web app, `bottle`, `blitzdb`, `plotly` and
[lime](https://github.com/marcotcr/lime) are added dependencies. Currently
these need to be installed by hand, later they will become standards. To use the
[Frog](https://languagemachines.github.io/frog/) wrapper as a Dutch back-end, we
strongly recommend using [LaMachine](https://proycon.github.io/LaMachine/). For
English, there is a [spaCy](https://spacy.io/) wrapper available.

---

## Acknowledgements

Part of the work on Omesa was carried out in the context of the
[AMiCA](http://www.amicaproject.be/) (IWT SBO-project 120007) project, funded
by the government agency for Innovation by Science and Technology (IWT).
