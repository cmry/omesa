"""
Container of datasets to be passed on to a featurizer
Can convert .csv files into a dataset with a chosen key
Envisioned for training:
    - can return combined datasets with any combinations of keys
    - works with a specified number of lines

Example:
reader = Datareader(max_n = 1000)
reader.add_dataset('blogs.csv', dataset_key = 'blogs')
reader.add_dataset('csi.csv', dataset_key = 'csi')
csi = reader.get_data('csi')
blogs_csi = reader.combine_datasets(['blogs','csi'])
instance = reader.process_raw_text(string)
"""

import random as rnd
import csv


class Datareader:
    """
    Object which handles reading and parsing the codecamp-csv-files
    """
    def __init__(self, max_n=False, shuffle=True, rnd_seed=99):
        # set params:
        self.max_n = max_n  # number of lines
        self.shuffle = shuffle  # boolean
        self.rnd_seed = rnd_seed
        # init dataset container as dict
        self.datasets = {}

    def add_dataset(self, filepath="path.csv", dataset_name="facebook", max_n=None, shuffle=None):
        # make sure that global settings for shuffling etc. are used,
        # if a user leaves these params unspecified.
        if not max_n:
            max_n = self.max_n
        if not shuffle:
            shuffle = self.shuffle
        # load:
        dataset = load_data_linewise(filename=filepath, max_n=max_n, shuffle=shuffle)
        # check for presence of frog-column:
        if 'frogs' in dataset.keys():  # set frogstrings in write format -> [[token,lemma,postag,sentenceindex],[token,lemma,...]]
            dataset['frogs'] = [decode_frogstring_train(string) for string in dataset['frogs']]
        self.datasets[dataset_name] = dataset

    def get_dataset(self, dataset_name):
        return self.datasets[dataset_name]

    def combine_datasets(self, dataset_names=[], shuffle=None, rnd_seed=None):
        """
        returns a combination of 1+ datasets, in dict-format, which one key per column
        will return e.g. {"age"=[all age labels], "gender"=[all gender labels]}
        keys = a list of names, specifiyng which datasets you wish to combine
        if no keys are specified, the entire dataset will be returned
        if reshuffle: will shuffle
        """
        if not shuffle:
            shuffle = self.shuffle
        if not dataset_names:  # return all of
            dataset_names = self.datasets.keys()
        # combine the lines of the datasets
        combined_lines = []
        for dataset_name in dataset_names:
            combined_lines.extend(dataset_2_rows(self.datasets[dataset_name]))
        # shuffle the lines if necessary
        if shuffle:
            rnd.seed(rnd_seed)
            rnd.shuffle(combined_lines)
        return rows_2_dataset(combined_lines)


def load_data_linewise(random_state=1066, filename="dummy.csv", max_n=10000, shuffle=True):
    """
    Main script to load data from the Amica-csv-files.
    All other parsing script are deprecated now.
    """
    rows = []
    with open(filename, 'r') as F:
        csv_reader = csv.reader(F)
        if csv.Sniffer().has_header(F.read(10)):
            next(csv_reader, None)  # skip header
        for i, line in enumerate(csv_reader):
            if max_n and i >= max_n:
                break
            rows.append(line)

    # shuffle the dataset:
    if shuffle:
        rnd.seed(random_state)
        rnd.shuffle(rows)
    if max_n:
        rows = rows[:max_n]
    dataset = rows_2_dataset(rows)
    return dataset


def rows_2_dataset(rows):
    """
    Converts 2D-list of items in row to a dict-based structure,
    with one feature column for each key.
    """
    # initialize the dataset
    fields = "user_id age gender loc_country loc_region loc_city education pers_big5 pers_mbti texts".split()
    if len(rows[0]) > len(fields):  # i.e. a column with frogged data is included
        fields.append("frogs")
    dataset = {k: [] for k in fields}
    # write rows to the dataset
    for row in rows:
        for category, val in zip(fields, row):
            dataset[category].append(val)
    return dataset


def dataset_2_rows(dataset):
    """
    Converts a dict-based structure (one feature column for each key)
    to a 2D-list of rows
    """
    # format of a dataset {'texts'=[], 'user_id'=[], ...}. will be converted in an instance per line
    fields = "user_id age gender loc_country loc_region loc_city education pers_big5 pers_mbti texts".split()
    if len(dataset.keys()) > len(fields):
        fields.append('frogs')
    return zip(*[dataset[field] for field in fields])


def decode_frogstring_train(frogstring):
    lines = frogstring.split("\n")
    decoded = []
    for line in lines:
        # add a tuple with (token,lemma,postag,sentence index)
        decoded.append(line.split("\t"))
    return decoded


def process_raw_text(text):
    """
    function to convert raw text into an instance list with frog column
    can be used for processing new inputs in a demo setting
    returns a list with values
    """
    # initialize list
    instance = [False] * 9  # empty fields
    instance.append(text)
    # add frogged text
    fr = frogger.Frogger([text])
    fr.frog_docs()
    fr.decode_frogstrings()
    instance.append(fr.decoded_frogstrings[0])
    return instance


def extract_tags(document, tags):
    """
    Function to extract a list of tags from a frogged document
    Document is the frogged column of a single instance
    Tags is a list with any of 'token', 'lemma', 'postag', or 'sentence'
    (can just be one of them)
    """
    tagdict = {'token': 0, 'lemma': 1, 'postag': 2, 'sentence': 3}
    taglists = [[token[tagdict[tag]] for token in document] for tag in tags]
    if len(taglists) > 1:
        return zip(*[taglists])
    else:
        return taglists[0]
