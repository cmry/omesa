"""N-gram experiment."""

# for as long as it's not yet pip installable
import sys
sys.path.append('../')
# -----

try:
    from omesa.experiment import Experiment
    from omesa.featurizer import Ngrams
    from omesa.io import CSV
except ImportError:
    exit("Could not load omesa. Please update the path in this file.")

conf = {
    "gram_experiment": {
        "name": "gram_experiment",
        "train_data": CSV("n_gram.csv", label=0, data=1, header=True),
        "has_header": True,
        "features": [Ngrams(level='char', n_list=[3])],
        "text_column": 1,
        "label_column": 0,
        "save": ("log", "model", "db")
    }
}

for f, c in conf.items():
    Experiment(c)
