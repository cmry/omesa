"""N-gram experiment."""

# for as long as it's not yet pip installable
import sys
sys.path.append('/home/emmery/git/shed/')
# -----

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
        "save": ("log")
    }
}

for f, c in conf.items():
    Experiment(c)
