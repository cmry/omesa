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

---

## Important Note

*This repository is currently in development, stable functionality is not
guaranteed as long as this message is showing.*

---

## Getting Started

We offer two quick examples to demonstrate the functionality:

- **2 minutes**: using [Shed only](shed.md) for text classification.
- **5 minutes**: integrating Shed for [data-to-features](df.md).

---

## Dependencies

Shed currently heavily relies on `numpy`, `scipy` and `sklearn`. To use the
[Frog](https://languagemachines.github.io/frog/) wrapper as a Dutch back-end, we
strongly recommend using [LaMachine](https://proycon.github.io/LaMachine/). For
English, there is a [spaCy](https://spacy.io/) wrapper available.

---

## Acknowledgements

Part of the work on Shed was carried out in the context of the
[AMiCA](http://www.amicaproject.be/) (IWT SBO-project 120007) project, funded
by the government agency for Innovation by Science and Technology (IWT).
