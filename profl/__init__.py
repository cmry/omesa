"""
The Main Thing.

profl is currently used to conduct Author Profiling experiments. This text
mining task usually relies on custom language features. Constructing these
by hand can be a time-consuming task. Therefore, this module aims to make
loading and featurizing existing, as well as new data a bit easier. It is
specifically intended for Dutch, but just replacing the Frog module with an
language-specific tagger (from NLTK for example) would make it broadly usable.

For help, refer to the docstring of the Profiler class (?? profl.Profiler).

Have fun,
Chris

"""

from .environment import Profiler

__author__ = 'Chris Emmery'
__contrb__ = 'Mike Kestemont, Ben Verhoeven, Florian Kunneman,' \
             'Janneke van de Loo'
__license__ = 'BSD 3-Clause'
