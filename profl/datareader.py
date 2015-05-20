import random as rnd
import csv


class Datareader:
    """
    Datareader
    ======

    Container of datasets to be passed on to a featurizer. Can convert .csv
    files into a dataset with a chosen key. Envisioned for training:
    - can return combined datasets with any combinations of keys
    - works with a specified number of lines

    Parameters
    ------

    max_n : int, optional, default = False
        Maximum number of data instances user wants to work with.

    shuffle : bool, optional, default = True
        If the order of the dataset should be randomized.

    rnd_seed : int, optional, default = 99
        A seed number used for reproducing the random order.

    Attributes
    -----

    datasets : dict
        Dictionary where key is the name of a dataset, and the value its rows.
        Will not be filled if data is streamed.

    headers : list
        List of headers / labels if they were provided with the datasets.

    Examples
    -----

    Interactive:

    >>> reader = Datareader(max_n=1000)
    >>> reader.add_dataset('blogs.csv', dataset_key = 'blogs')
    >>> reader.add_dataset('csi.csv', dataset_key = 'csi')
    >>> csi = reader.get_data('csi')
    >>> blogs_csi = reader.combine_datasets(['blogs','csi'])
    >>> instance = reader.process_raw_text(string)

    Generator:

    >>> reader = Datareader(max_n=1000)
    >>> data = ['~/Documents/data1.csv', '~/Documents/data2.csv']
    >>> for line in reader.stream_data(data):
    >>>     # do something with line
    """

    def __init__(self, max_n=False, shuffle=True, rnd_seed=99):
        self.max_n = max_n
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed
        # init dataset container as dict
        self.datasets = {}
        self.headers = None

    def add_dataset(self, filepath, dataset_name):
        #  removed any parameters overlapping with global class variables,
        #  here and in all other class functions, they do not make any
        #  sense -> no one wants to shuffle one set, but not the other.
        #
        #  Also, got rid of a lot of optional parameters that would introduce
        #  bugs.
        #
        # -c-

        dataset = self.load_data_linewise(filename=filepath)

        # check for presence of frog-column:
        # set frogstrings in write format ->
        # [[token,lemma,postag,sentenceindex], [token,lemma,...]]
        if 'frogs' in dataset.keys():
            dataset['frogs'] = [self.decode_frogstring_train(string) for string
                                in dataset['frogs']]
        self.datasets[dataset_name] = dataset

    def get_dataset(self, dataset_name):    # why?? can't you just call
                                            # class.datasets[dataset_name] ?
                                            # -c-
        return self.datasets[dataset_name]

    def combine_datasets(self, dataset_names):
        """
        returns a combination of 1+ datasets, in dict-format, which one key per
        column will return e.g. {"age"=[all age labels], "gender"=[all gender
        labels]} keys = a list of names, specifiyng which datasets you wish to
        combine if no keys are specified, the entire dataset will be returned
        if reshuffle: will shuffle
        """
        if type(dataset_names) != list or len(dataset_names) < 2:
            return
        combined_lines = [row for name in dataset_names for row in
                          self.dataset_2_rows(self.datasets[name])]
        if self.shuffle:
            rnd.seed(self.rnd_seed)
            rnd.shuffle(combined_lines)
        return self.rows_2_dataset(combined_lines)

    def load_data_linewise(self, filename="./data/example.csv"):
        """
        Main script to load data from the Amica-csv-files.
        All other parsing script are deprecated now.
        """
        rows, head = [], False
        with open(filename, 'r') as F:
            csv_reader = csv.reader(F)
            if csv.Sniffer().has_header(F.read(10)):
                head = True
            for i, line in enumerate(csv_reader):
                if head and i == 0:
                    self.headers = line
                if self.max_n and i >= self.max_n:
                    break
                rows.append(line)

        # shuffle the dataset:
        if self.shuffle:
            rnd.seed(self.rnd_seed)
            rnd.shuffle(rows)
        if self.max_n:
            rows = rows[:self.max_n]
        dataset = self.rows_2_dataset(rows)
        return dataset

    def rows_2_dataset(self, rows):
        """
        Converts 2D-list of items in row to a dict-based structure,
        with one feature column for each key.
        """
        # initialize the dataset
        if len(rows[0]) > len(self.headers):  # i.e. a column w. frog data
            self.headers.append("frogs")
        dataset = {k: [] for k in self.headers}
        # write rows to the dataset
        for row in rows:
            for category, val in zip(self.headers, row):
                dataset[category].append(val)
        return dataset

    def dataset_2_rows(self, dataset):
        """
        Converts a dict-based structure (one feature column for each key)
        to a 2D-list of rows
        """
        # format of a dataset {'texts'=[], 'user_id'=[], ...}.
        # will be converted in an instance per line
        if len(dataset.keys()) > len(self.headers):
            self.headers.append('frogs')
        return zip(*[dataset[field] for field in self.headers])

    def decode_frogstring_train(self, frogstring):
        lines = frogstring.split("\n")
        decoded = []
        for line in lines:
            # add a tuple with (token,lemma,postag,sentence index)
            decoded.append(line.split("\t"))
        return decoded

    def process_raw_text(self, text):
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

    def extract_tags(self, document, tags):
        """
        Function to extract a list of tags from a frogged document
        Document is the frogged column of a single instance
        Tags is a list with any of 'token', 'lemma', 'postag', or 'sentence'
        (can just be one of them)
        """
        tagdict = {'token': 0, 'lemma': 1, 'postag': 2, 'sentence': 3}
        taglists = [[token[tagdict[tag]] for token in document]
                    for tag in tags]
        if len(taglists) > 1:
            return zip(*[taglists])
        else:
            return taglists[0]
