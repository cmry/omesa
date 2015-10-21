# Toolboard

Toolboard is a virtual environment which can be used to install all dependencies
used in shed. It currently forks the functionality of [LaMachine](https://github.com/proycon/LaMachine).

## Packages
There is a number of packages that shed (and pretty much the majority of text
mining research in Python) relies on. The following of which will be currently
be installed in the virtual environment:

  - numpy
  - scipy
  - sklearn
  - tqdm

## LaMachine
LaMachine is a software distribution of NLP software developed by the Language
Machines research group and CLST (Radboud University Nijmegen), as well as TiCC
(Tilburg University). It has been adapted to suit the current needs (Frog) of
shed.

Pre-installed software:
- [Timbl](http://ilk.uvt.nl/timbl) - Tilburg Memory Based Learner
- [Ucto](http://ilk.uvt.nl/ucto) - Tokenizer
- [Frog](http://ilk.uvt.nl/frog) - Frog is an integration of memory-based natural language processing (NLP) modules developed for Dutch.
- [Mbt](http://ilk.uvt.nl/mbt) - Memory-based Tagger
- [FoLiA-tools](http://proycon.github.io/folia) - Command line tools for working with the FoLiA format
- *C++ libraries* - [ticcutils](http://ilk.uvt.nl/ticcutils), [libfolia](http://proycon.github.io/folia)
- *Python bindings* - [python-ucto](https://github.com/proycon/python-ucto), [python-frog](https://github.com/proycon/python-frog), [python-timbl](https://github.com/proycon/python-timbl)

# Installation (Linux only)


# Alternatives

Get [LaMachine](https://github.com/proycon/LaMachine) for Frog (it's probably
more stable to begin with), and use the environment to install the dependencies
written in `dependencies.txt`.
