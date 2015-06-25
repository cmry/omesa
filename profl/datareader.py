import random as rnd
import sys
import csv

from .utils import frog

# Authors: Chris Emmery, Florian Kunneman
# License: BSD 3-Clause


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
    data : list of strings
        List with document directories to be loaded.

    proc : string or function, [None (default), 'text', 'label', 'both', \
                                function]
        If you want any label or text conversion, you can indicate this
        here with a string, or either supply your own to apply to the row
        object. These are constructed as list[label, text, frog].

        'text':
            Apply a generic normalization and preprocessing process to the
            input data.
        'label':
            Do a general categorization based on the labels that is
            provided if need be. Age will for example be splitted into
            several clases.
        'both':
            All of the above.
        function:
            Specify your own function by which you want to edit the row.

    max_n : int, optional, default False
        Maximum number of data instances *per dataset* user wants to work with.

    shuffle : bool, optional, default True
        If the order of the dataset should be randomized.

    rnd_seed : int, optional, default 99
        A seed number used for reproducing the random order.

    label : str, optional, default 2nd header
        Name of the label header row that should be retrieved. If not set, the
        second column will be asummed to be a label column.

    Attributes
    -----
    datasets : dict
        Dictionary where key is the name of a dataset, and the value its rows.
        Will not be filled if data is streamed.

    headers : list
        List of headers / labels if they were found present in the datasets. If
        not, standard AMiCA list is provided. Might introduce bugs.

    Examples
    -----
    >>> data = ['~/Documents/data1.csv', '~/Documents/data2.csv']
    >>> reader = Datareader(data, max_n=1000)
    >>> labels, raw, frog = reader.load(data, dict_format=True)

    Notes
    -----
    Interactive use has been deprecated in this version.
    """
    def __init__(self, data, proc, max_n, shuffle, rnd_seed, label):

        self.file_list = data
        self.proc = proc
        self.max_n = max_n
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed
        self.label = label
        self.headers = "user_id age gender loc_country loc_region \
                       loc_city education pers_big5 pers_mbti text".split()
        self.datasets = {}

        rnd.seed(self.rnd_seed)

    def load(self, dict_format=False):
        """
        Raw data loader
        =====
        This is now the main way to load in your .csv files. It will check
        which labels are present in the input data, and will isolate any
        specified label. Please note that frog data **must** be under a 'frogs'
        header, otherwise it won't parse it.

        Parameters
        -----
        dict_format : bool, optional, default False
            Set to True if the datasets should be divided in a dictionary where
            their key is the filename and the value the data matrix.

        Returns
        -----
        labels : list
            Labels for each of the instances in raw, labels should be
            maintained in the same position for e.g. sklearn evaluation.
        raw : list
            The raw data comes in an array where each entry represents a text
            instance in the data file.
        frogs : list
            The frog data, list is empty if no data is found.
        """
        data = [self.preprocess(row) for file_name in self.file_list
                for row in self.load_data_linewise(file_name)]
        if self.shuffle:
            rnd.shuffle(data)
        labels, raw, frogs = zip(*data)
        return list(labels), list(raw), list(frogs)

    def label_convert(self, label_field):
        if self.label == 'age':
            age = {range(15):      'child',
                   range(15, 18):  'teen',
                   range(18, 21):  'post-teen',
                   range(21, 26):  'young adult',
                   range(26, 100): 'adult'}
            for r in age.keys():
                if int(label_field) in r:
                    return age[r]

    def preprocess(self, row):
        """
        Text and label preprocessor
        =====

        Parameters
        -----
        row : list
            Row pruned to include only the selected label, raw data and decoded
            frog data.

        Returns
        -----
        """
        if self.proc == 'label' or self.proc == 'both':
            new_label = self.label_convert(row[0])
            row[0] = new_label
        if self.proc == 'text' or self.proc == 'both':
            new_text = row[1].lower()
            row[1] = new_text
        if self.proc and type(self.proc) != str:
            row = self.proc(row)
        return row

    def extract_row(self, line):
        """
        Data extractor
        =====
        Fetches required data from data files. Handles frog data correctly.

        Parameters
        -----
        line : list
            List with .csv frow.

        Returns
        -----
        rows : list
            Row pruned to include only the selected label, raw data and decoded
            frog data.

        """
        label_data = line[self.headers.index(self.label)]
        text_data = line[self.headers.index('text')]
        try:
            head = self.headers.index('frogs')
            frog_data = frog.decode_frogstring_train(line[head])
        except ValueError:
            frog_data = []
        row = [label_data, text_data, frog_data]
        return row

    def check_header(self, filename):
        """
        Header checker
        =====
        Sniffs if a .csv file has a header.

        Parameters
        -----
        filename : str
            Directory of a .csv file to be sniffed.

        Returns
        -----
        has_header : bool
            True if a header was sniffed in the file.
        """
        with open(filename, 'r') as sniff_file:
            has_header = True if csv.Sniffer().has_header(
                    sniff_file.read(200)) else False
        return has_header

    def load_data_linewise(self, filename):
        """
        Csv reader
        =====
        Reads a csv file by pathname, extracts headers and returns matrix.

        Parameters
        -----
        filename : str
            Directory of a .csv file to be stored into a list.

        Returns
        -----
        rows : list
            List of lists where each row is an instance and column a label
            entry or text data.

        """
        csv.field_size_limit(sys.maxsize)
        rows, has_header = [], self.check_header(filename)
        # check if provided file has a label
        with open(filename, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            for i, line in enumerate(csv_reader):
                # if there's a header, set it, if no label, 2nd header = label
                if has_header and i == 0:
                    self.headers = line
                    if not self.label:
                        self.label = self.headers[1]
                # stop if we reached max_n
                elif self.max_n and i >= self.max_n:
                    break
                else:
                    row = self.extract_row(line)
                    if row[0]:  # if label
                        rows.append(row)
        return rows
