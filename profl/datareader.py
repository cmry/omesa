import random as rnd
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
    >>> reader = Datareader(max_n=1000)
    >>> data = ['~/Documents/data1.csv', '~/Documents/data2.csv']
    >>> dataset = reader.load(data, dict_format=True)

    Notes
    -----
    Interactive use has been deprecated in this version.
    """
    def __init__(self, max_n=False, shuffle=True, rnd_seed=99, label=None):

        self.max_n = max_n
        self.shuffle = shuffle
        self.rnd_seed = rnd_seed
        self.label = label
        self.headers = "user_id age gender loc_country loc_region \
                       loc_city education pers_big5 pers_mbti texts".split()
        self.datasets = {}

        rnd.seed(self.rnd_seed)

    def load(self, file_list, dict_format=False):
        """
        Raw data loader
        =====
        This is now the main way to load in your .csv files. It will check
        which labels are present in the input data, and will isolate any
        specified label. Please note that frog data **must** be under a 'frogs'
        header, otherwise it won't parse it.

        Parameters
        -----
        file_list : list of strings
            List with document directories to be loaded.

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
        data = [row for filename in file_list for row
                in self.load_data_linewise(filename)]
        if self.shuffle:
            rnd.shuffle(data)
        labels, raw, frogs = zip(*data)
        return list(labels), list(raw), list(frogs)

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
            frog_data = [frog.decode_frogstring_train(string)
                         for string in
                         line[self.headers.index('frogs')]]
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
<<<<<<< HEAD
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
                    rows.append(self.extract_row(line))
        return rows
=======
        Dataset appender
        ===
        Function to add a new dataset to the file reader object

        Parameters
        -----
        filepath : str
            file of the dataset
        dataset_name : str
            key for the dataset

        """
        dataset = self.handle_data(filename=filepath)
        # check for presence of frog-column:
        # set frogstrings in write format ->
        # [[token,lemma,postag,sentenceindex], [token,lemma,...]]
        if 'frogs' in dataset.keys():
            dataset['frogs'] = [self.decode_frogstring_train(string) for string
                                in dataset['frogs']]
        self.datasets[dataset_name] = dataset

    def combine_datasets(self, dataset_names):
        """
        Dataset combiner
        =====
        returns a combination of 1+ datasets, in dict-format, which one key 
        per column will return e.g. {"age"=[all age labels], 
        "gender"=[all gender labels]} 

        Parameters
        -----
        dataset_names : a list of names, specifiyng which datasets you wish to
            combine 
            if no keys are specified, the entire dataset will be returned

        Returns
        -----
        combi: dict of lists as combination of required datasets

        """  

        if type(dataset_names) != list or len(dataset_names) < 2:
            return
        combined_lines = [row for name in dataset_names for row in
                          self.dataset_2_rows(self.datasets[name])]
        if self.shuffle:
            rnd.shuffle(combined_lines)
        combi = self.rows_2_dataset(combined_lines)
        return combi

    # Data & Frog operations --------------------------------------------------

    def rows_2_dataset(self, rows):
        """
        Rows converter
        =====
        Converts 2D-list of items in row to a dict-based structure,
        with one feature column for each key.

        Parameters
        -----
        rows : list of instances

        Returns
        -----
        dataset : dict of lists
            The rows are converted from an instance per line to a list per 
            instance column. 
        """       

        dataset = {k: [] for k in self.headers}
        # write rows to the dataset
        for row in rows:
            for category, val in zip(self.headers, row):
                dataset[category].append(val)
        return dataset

    def dataset_2_rows(self, dataset):
        """
        Dataset converter
        =====
        Converts a dict-based structure (one feature column for each key)
        to a 2D-list of rows

        Parameters
        -----
        dataset : dict of lists
            The format of a dataset is {'texts'=[], 'user_id'=[], ...}. 

        Returns
        -----
        rows : list
            The dataset is converted from a list per dict key to an instance 
            per line. 
        """       
        
        rows =  zip(*[dataset[field] for field in self.headers])
        return rows

    def decode_frogstring_train(self, frogstring):
        """
        Decoder of frogged data in the Amica csv-files
        =====
        function to convert frogged lines into a list with tokens as
        [token, lemma, postag, sentence]

        Parameters
        -----
        frogstring : the frogged data of a single document

        Returns
        -----
        decoded : list 
            The frogstring as list of lists
        """       

        lines = frogstring.split("\n")
        decoded = []
        for line in lines:
            # add a tuple with (token,lemma,postag,sentence index)
            decoded.append(line.split("\t"))
        return decoded

    def process_raw_text(self, text):
        """
        Extractor of frog tags
        =====
        function to convert raw text into an instance list with frog column
        can be used for processing new inputs in a demo setting
        returns a list with values

        Parameters
        -----
        text : a raw string of characters

        Returns
        -----
        instance : list 
            The 'row' of one instance, containing all metadata fields that
            are present in the Amica csv-files, as well as the text and 
            frogged column
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
        Extractor of frog tags
        =====
        Function to extract a list of tags from a frogged document
        Document is the frogged column of a single document
        Tags is a list with any of 'token', 'lemma', 'postag', or 'sentence'
        (can just be one of them)

        Parameters
        -----
        document : list of tuples corresponding to all tokens in a single 
            text. A token is a list with the fields 'token', 'lemma', 'postag'
            and 'sentence'.  
        tags: list of tags to return from a document. Options:
            - token
            - lemma
            - postag
            - sentence

        Returns
        -----
        extracted_tags : list 
            Sequence of the tokens in the document, with the selected tags
            if the value of 'tags' is '[token, postag]', the output list will 
            be '[[token, postag], [token, postag], ...]' 
        """

        tagdict = {'token': 0, 'lemma': 1, 'postag': 2, 'sentence': 3}
        taglists = [[token[tagdict[tag]] for token in document]
                    for tag in tags]
        if len(taglists) > 1:
            extracted_tags = zip(*[taglists])
        else:
            extracted_tags = taglists[0]

        return extracted_tags
>>>>>>> master
