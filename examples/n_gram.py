"""N-gram experiment."""

from shed.experiment import Experiment
from shed.featurizer import Ngrams


conf = {
    "experiment_name": {
        "name": "gram_experiment",
        "train_data": ["./n_gram.csv"],
        "has_header": True,
        "features": [Ngrams(n_list=[1])],
        "text_column": 1,
        "label_column": 0,
        "save": ("log")
    }
}

for f, c in conf.items():
    Experiment(c)
